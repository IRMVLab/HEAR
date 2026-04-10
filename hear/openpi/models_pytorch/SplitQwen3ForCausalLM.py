import torch
import torch.nn as nn
from typing import Optional, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.generation import GenerationMixin


class SplitQwen3Model(Qwen3Model):


    def prepare_inputs_for_layers(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
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
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs
    ):

        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"layer_idx must be between 0 and {len(self.layers)-1}, got {layer_idx}")

        decoder_layer = self.layers[layer_idx]
        attention_mask = causal_mask_mapping[decoder_layer.attention_type]


        layer_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            **kwargs,
        }


        if hasattr(decoder_layer, 'input_layernorm') and hasattr(decoder_layer.input_layernorm, 'dense'):
            layer_kwargs['adarms_cond'] = adarms_cond

        hidden_states = decoder_layer(**layer_kwargs)

        return hidden_states

    def run_layers_range(
        self,
        start_layer: int,
        end_layer: int,
        hidden_states: torch.Tensor,
        causal_mask_mapping: dict,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache],
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        use_cache: bool = False,
        **kwargs
    ):

        for layer_idx in range(start_layer, end_layer):
            hidden_states = self.run_single_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                **kwargs,
            )
        return hidden_states

    def finalize_outputs(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):


        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:


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
        for layer_idx in range(len(self.layers)):
            hidden_states = self.run_single_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                causal_mask_mapping=prepared["causal_mask_mapping"],
                position_ids=prepared["position_ids"],
                past_key_values=prepared["past_key_values"],
                cache_position=prepared["cache_position"],
                position_embeddings=prepared["position_embeddings"],
                use_cache=prepared["use_cache"],
                **{k: v for k, v in prepared.items() if k not in [
                    "hidden_states", "causal_mask_mapping", "position_ids",
                    "past_key_values", "cache_position", "position_embeddings", "use_cache"
                ]}
            )


        return self.finalize_outputs(
            hidden_states=hidden_states,
            past_key_values=prepared["past_key_values"],
            use_cache=prepared["use_cache"],
        )


class SplitQwen3ForCausalLM(Qwen3ForCausalLM):


    def __init__(self, config):


        super(Qwen3ForCausalLM, self).__init__(config)


        self.model = SplitQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


        self.post_init()
