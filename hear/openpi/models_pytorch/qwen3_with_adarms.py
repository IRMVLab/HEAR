# coding=utf-8
import torch
from torch import nn
from typing import Optional, Tuple, Union, List
from collections.abc import Sequence

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3Config,
    Qwen3Attention,
    Qwen3MLP,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

logger = logging.get_logger(__name__)

def _gated_residual(x, y, gate):
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate

class Qwen3RMSNormWithAdaRMS(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = hidden_size
        self.cond_dim = cond_dim

        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            if self.dense.bias is not None:
                nn.init.zeros_(self.dense.bias)
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.dense = None

    def _norm(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            return normed_inputs * self.weight, None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift

        return normed_inputs, gate

class Qwen3DecoderLayerWithAdaRMS(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)

        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None
        self.input_layernorm = Qwen3RMSNormWithAdaRMS(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.post_attention_layernorm = Qwen3RMSNormWithAdaRMS(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
        self.attention_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        adarms_cond: Optional[torch.Tensor] = None,
        external_past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)


        active_past_key_values = past_key_values

        if external_past_key_values is not None:
            k_tens, v_tens = external_past_key_values
            if k_tens is None or v_tens is None:
                raise ValueError("external_past_key_values provided but (k,v) contains None.")


            try:
                temp_cache = DynamicCache(config=getattr(self.self_attn, "config", None))
            except TypeError:
                temp_cache = DynamicCache()


            temp_cache.update(k_tens, v_tens, self.layer_idx, cache_kwargs={})

            active_past_key_values = temp_cache

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=active_past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # ===================

        hidden_states = _gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class Qwen3ModelWithAdaRMS(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayerWithAdaRMS(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None
        self.norm = Qwen3RMSNormWithAdaRMS(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def prepare_inputs_for_layers(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return {
            "hidden_states": hidden_states,
            "causal_mask_mapping": causal_mask_mapping,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "use_cache": use_cache,
            **kwargs
        }

    def run_single_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        causal_mask_mapping: dict,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache],
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        use_cache: bool = False,
        output_attentions: bool = False,
        adarms_cond: Optional[torch.Tensor] = None,
        external_past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        decoder_layer = self.layers[layer_idx]
        attention_mask = causal_mask_mapping[decoder_layer.attention_type]

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            adarms_cond=adarms_cond,
            external_past_key_values=external_past_key_values,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        attentions = layer_outputs[1] if output_attentions else None

        return hidden_states, attentions

    def finalize_outputs(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        all_self_attns: Optional[Tuple[torch.Tensor]] = None,
        adarms_cond: Optional[torch.Tensor] = None,
    ):
        """
        Apply final norm and package outputs.
        Accepts accumulated hidden states/attentions to package them into the final result.
        """
        hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)

        # Add the final hidden state to all_hidden_states if tracking
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        external_past_key_values: Optional[Union[Cache, Sequence[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Prepare
        prepared = self.prepare_inputs_for_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = prepared["hidden_states"]
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # 2. Run Layers
        for layer_idx in range(len(self.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)


            layer_external_kv = None
            if external_past_key_values is not None:


                try:
                    layer_external_kv = external_past_key_values[layer_idx]
                except Exception:
                    layer_external_kv = None
            # ========================================

            hidden_states, layer_attn = self.run_single_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask_mapping=prepared["causal_mask_mapping"],
                position_ids=prepared["position_ids"],
                past_key_values=prepared["past_key_values"],
                cache_position=prepared["cache_position"],
                position_embeddings=prepared["position_embeddings"],
                use_cache=prepared["use_cache"],
                output_attentions=output_attentions,
                adarms_cond=adarms_cond,
                external_past_key_values=layer_external_kv,
                **{k: v for k, v in prepared.items() if k not in [
                    "hidden_states", "causal_mask_mapping", "position_ids",
                    "past_key_values", "cache_position", "position_embeddings", "use_cache"
                ]}
            )

            if output_attentions:
                all_self_attns += (layer_attn,)

        # 3. Finalize
        final_outputs = self.finalize_outputs(
            hidden_states,
            past_key_values=prepared["past_key_values"] if use_cache else None,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            adarms_cond=adarms_cond
        )

        if not return_dict:
            return tuple(v for v in [final_outputs.last_hidden_state, final_outputs.past_key_values, final_outputs.hidden_states, final_outputs.attentions] if v is not None)

        return final_outputs

class Qwen3ForCausalLMWithAdaRMS(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3ModelWithAdaRMS(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        adarms_cond: Optional[torch.Tensor] = None,
        external_past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            adarms_cond=adarms_cond,
            external_past_key_values=external_past_key_values,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + (outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

SplitQwen3Model = Qwen3ModelWithAdaRMS
SplitQwen3ForCausalLM = Qwen3ForCausalLMWithAdaRMS

__all__ = [
    "Qwen3RMSNormWithAdaRMS",
    "Qwen3DecoderLayerWithAdaRMS",
    "Qwen3ModelWithAdaRMS",
    "Qwen3ForCausalLMWithAdaRMS",
    "SplitQwen3Model",
    "SplitQwen3ForCausalLM",
]
