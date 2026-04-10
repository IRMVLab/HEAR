# coding=utf-8
from typing import Optional, Union, List, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerCausalLMOutputWithPast,
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
    _get_feat_extract_output_lengths,
    load_balancing_loss_func,
    create_causal_mask,
)
from openpi.models_pytorch.audio_history_encoder import AudioHistoryTemporalEncoder

import pathlib
import re


def _read_audio_history_defaults() -> tuple[int, float]:
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "training" / "config.py"
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return 0, 1.0

    def _find(name: str, fallback):
        m = re.search(rf"^{name}\s*:\s*[^=]*=\s*([0-9.+-eE]+)\s*$", text, flags=re.MULTILINE)
        if not m:
            return fallback
        try:
            return type(fallback)(m.group(1))
        except (TypeError, ValueError):
            return fallback

    window_default = _find("AUDIO_HISTORY_WINDOW_DEFAULT", 0)
    scale_default = _find("AUDIO_HISTORY_SCALE_DEFAULT", 1.0)
    return int(window_default), float(scale_default)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SplitQwen3OmniMoeThinkerTextModel(Qwen3OmniMoeThinkerTextModel):
    def prepare_inputs_for_layers(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache


        # manual_gc_with_cache = bool(self.training and self.config.gradient_checkpointing and use_cache)
        # manual_gc_with_cache = bool(self.training and use_cache)
        manual_gc_with_cache = use_cache
        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and (not manual_gc_with_cache) and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "text_position_ids": text_position_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "use_cache": use_cache,
            "manual_gc_with_cache": manual_gc_with_cache,
            **kwargs,
        }

    def _layer_forward_return_kv(
        self,
        decoder_layer: nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.LongTensor,
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        input_shape = hidden_states.shape[:-1]
        head_dim = decoder_layer.self_attn.head_dim
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = decoder_layer.self_attn.q_norm(
            decoder_layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = decoder_layer.self_attn.k_norm(
            decoder_layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = decoder_layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            decoder_layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else decoder_layer.self_attn.attention_dropout,
            scaling=decoder_layer.self_attn.scaling,
            sliding_window=decoder_layer.self_attn.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = decoder_layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        mlp_out = decoder_layer.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out

        return hidden_states, key_states, value_states

    def run_single_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.LongTensor,
        past_key_values: Optional[Cache],
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        manual_gc_with_cache: bool = False,
        use_cache: bool = False,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        **kwargs,
    ):
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(self.layers)})")

        decoder_layer = self.layers[layer_idx]


        if manual_gc_with_cache:

            def _ckpt(hs, am, cos, sin):
                pe = (cos, sin)
                out_hs, k, v = self._layer_forward_return_kv(
                    decoder_layer,
                    layer_idx,
                    hs,
                    am,
                    text_position_ids,
                    cache_position,
                    pe,
                    **kwargs,
                )
                return out_hs, k, v

            cos, sin = position_embeddings
            out_hs, k, v = checkpoint(_ckpt, hidden_states, attention_mask, cos, sin, use_reentrant=False)
            return out_hs, (k, v)


        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        if isinstance(layer_outputs, tuple):
            out_hs = layer_outputs[0]
        else:
            out_hs = layer_outputs
        return out_hs, None

    def finalize_outputs(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        all_self_attns: Optional[Tuple[torch.Tensor, ...]] = None,
        all_router_logits: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        hidden_states = self.norm(hidden_states)
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
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prepared = self.prepare_inputs_for_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = prepared["hidden_states"]
        manual_gc_with_cache = prepared["manual_gc_with_cache"]
        use_cache = prepared["use_cache"]

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None


        collected_kv = [] if (manual_gc_with_cache and use_cache) else None

        for layer_idx in range(len(self.layers)):
            hidden_states, layer_kv = self.run_single_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=prepared["attention_mask"],
                text_position_ids=prepared["text_position_ids"],
                past_key_values=prepared["past_key_values"],
                cache_position=prepared["cache_position"],
                position_embeddings=prepared["position_embeddings"],
                manual_gc_with_cache=manual_gc_with_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                **kwargs,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                if visual_pos_masks is not None:
                    hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])

            if collected_kv is not None:
                collected_kv.append(layer_kv)

        final_past = tuple(collected_kv) if collected_kv is not None else prepared["past_key_values"]

        final_outputs = self.finalize_outputs(
            hidden_states=hidden_states,
            past_key_values=final_past,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            all_router_logits=all_router_logits,
        )

        if not return_dict:
            return (final_outputs.last_hidden_state, final_outputs.past_key_values, final_outputs.hidden_states)

        return final_outputs


class SplitQwen3OmniMoeThinkerForConditionalGeneration(Qwen3OmniMoeThinkerForConditionalGeneration):
    def __init__(self, config):
        super(Qwen3OmniMoeThinkerForConditionalGeneration, self).__init__(config)

        self.audio_tower = Qwen3OmniMoeAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = SplitQwen3OmniMoeThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.num_experts = config.text_config.num_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

        default_history_window, default_history_scale = _read_audio_history_defaults()

        self.audio_history_window = int(default_history_window)
        self.audio_history_enabled = bool(self.audio_history_window and self.audio_history_window > 0)
        self.audio_history_temporal = None
        self.audio_history_scale = None
        if self.audio_history_enabled:
            history_heads = getattr(config, "audio_history_heads", None)
            if history_heads is None:
                history_heads = min(8, config.audio_config.encoder_attention_heads)
            history_layers = getattr(config, "audio_history_layers", 2)
            history_ffn_dim = getattr(config, "audio_history_ffn_dim", config.audio_config.output_dim * 4)
            history_dropout = getattr(config, "audio_history_dropout", 0.0)
            history_gc = bool(getattr(config, "audio_history_gradient_checkpointing", True))
            self.audio_history_temporal = AudioHistoryTemporalEncoder(
                hidden_size=config.audio_config.output_dim,
                num_layers=history_layers,
                num_heads=history_heads,
                ffn_dim=history_ffn_dim,
                dropout=history_dropout,
                gradient_checkpointing=history_gc,
            )
            self.audio_history_scale = nn.Parameter(torch.tensor(float(default_history_scale)))
        self.post_init()

    def _prepare_audio_inputs(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor],
        audio_feature_lengths: Optional[torch.LongTensor],
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        if feature_attention_mask is None:
            if audio_feature_lengths is None:
                raise ValueError("feature_attention_mask or audio_feature_lengths must be provided.")
            seq_len = input_features.shape[-1]
            feature_attention_mask = (
                torch.arange(seq_len, device=input_features.device)[None, :] < audio_feature_lengths[:, None]
            )
        feature_lengths = feature_attention_mask.sum(dim=1)
        if feature_lengths.dtype != torch.long:
            feature_lengths = feature_lengths.to(dtype=torch.long)
        flat_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        return flat_features, feature_lengths

    def _encode_audio_features(
        self,
        audio_tower: nn.Module,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor],
        audio_feature_lengths: Optional[torch.LongTensor],
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        flat_features, feature_lengths = self._prepare_audio_inputs(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_outputs = audio_tower(
            flat_features,
            feature_lens=feature_lengths,
        )
        return audio_outputs.last_hidden_state, feature_lengths

    def _split_audio_features(
        self,
        audio_features: torch.FloatTensor,
        feature_lengths: torch.LongTensor,
    ) -> tuple[tuple[torch.Tensor, ...], list[int], torch.LongTensor]:
        output_lengths = _get_feat_extract_output_lengths(feature_lengths)
        total_length = int(output_lengths.sum().item())
        if audio_features.shape[0] != total_length:
            raise ValueError(
                f"Audio features length {audio_features.shape[0]} != expected {total_length}."
            )
        output_lengths_list = output_lengths.tolist()
        if output_lengths_list:
            splits = torch.split(audio_features, output_lengths_list, dim=0)
        else:
            splits = ()
        return splits, output_lengths_list, output_lengths

    def _build_history_from_current(
        self,
        current_splits: tuple[torch.Tensor, ...],
        output_lengths_list: list[int],
    ) -> torch.FloatTensor | None:
        if self.audio_history_temporal is None:
            return None
        batch_size = len(output_lengths_list)
        if batch_size == 0:
            return None
        history_len = max(1, int(self.audio_history_window))
        max_len = max(output_lengths_list)
        hidden_size = current_splits[0].shape[1] if current_splits else 0
        history_tensor = current_splits[0].new_zeros((batch_size, history_len, max_len, hidden_size))
        history_mask = torch.zeros(
            (batch_size, history_len, max_len),
            dtype=torch.bool,
            device=history_tensor.device,
        )
        for i, seq in enumerate(current_splits):
            seq_len = seq.shape[0]
            if seq_len == 0:
                continue
            history_tensor[i, :, :seq_len, :] = seq.unsqueeze(0).expand(history_len, -1, -1)
            history_mask[i, :, :seq_len] = True
        history_agg = self.audio_history_temporal(history_tensor, history_mask)
        history_flat = torch.cat(
            [history_agg[i, :output_lengths_list[i], :] for i in range(batch_size)],
            dim=0,
        )
        return history_flat

    def _build_history_from_inputs(
        self,
        history_input_features: torch.FloatTensor,
        history_feature_attention_mask: Optional[torch.LongTensor],
        history_audio_feature_lengths: Optional[torch.LongTensor],
        current_output_lengths: list[int],
    ) -> torch.FloatTensor | None:
        if self.audio_history_temporal is None:
            return None
        batch_size, history_len, mel_bins, seq_len = history_input_features.shape
        flat_features = history_input_features.reshape(batch_size * history_len, mel_bins, seq_len)
        flat_mask = None
        flat_lengths = None
        if history_feature_attention_mask is not None:
            if history_feature_attention_mask.shape != (batch_size, history_len, seq_len):
                raise ValueError("history_feature_attention_mask must match [batch, history, seq_len].")
            flat_mask = history_feature_attention_mask.reshape(batch_size * history_len, seq_len)
        if history_audio_feature_lengths is not None:
            if history_audio_feature_lengths.shape != (batch_size, history_len):
                raise ValueError("history_audio_feature_lengths must match [batch, history].")
            flat_lengths = history_audio_feature_lengths.reshape(batch_size * history_len)
        history_features, history_feature_lens = self._encode_audio_features(
            audio_tower=self.audio_tower,
            input_features=flat_features,
            feature_attention_mask=flat_mask,
            audio_feature_lengths=flat_lengths,
        )
        history_splits, history_lengths_list, _ = self._split_audio_features(
            audio_features=history_features,
            feature_lengths=history_feature_lens,
        )
        if not history_lengths_list:
            return None
        max_len = max(max(history_lengths_list), max(current_output_lengths))
        hidden_size = history_features.shape[1]
        history_tensor = history_features.new_zeros((batch_size, history_len, max_len, hidden_size))
        history_mask = torch.zeros(
            (batch_size, history_len, max_len),
            dtype=torch.bool,
            device=history_features.device,
        )
        idx = 0
        for i in range(batch_size):
            for j in range(history_len):
                seq = history_splits[idx]
                idx += 1
                seq_len = seq.shape[0]
                if seq_len == 0:
                    continue
                history_tensor[i, j, :seq_len, :] = seq
                history_mask[i, j, :seq_len] = True
        history_agg = self.audio_history_temporal(history_tensor, history_mask)
        history_flat = torch.cat(
            [history_agg[i, :current_output_lengths[i], :] for i in range(batch_size)],
            dim=0,
        )
        return history_flat

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        history_override_mask = getattr(self, "_history_feature_attention_mask", None)
        history_input_features = None
        history_feature_attention_mask = None
        history_audio_feature_lengths = None
        if input_features.dim() == 4:
            history_input_features = input_features
            input_features = input_features[:, -1, :, :]
            history_len = history_input_features.shape[1]
            batch_size = history_input_features.shape[0]
            if feature_attention_mask is not None:
                if feature_attention_mask.dim() == 3:
                    history_feature_attention_mask = feature_attention_mask
                    feature_attention_mask = feature_attention_mask[:, -1, :]
                elif feature_attention_mask.dim() == 2:
                    if history_override_mask is not None:
                        if history_override_mask.shape != (
                            batch_size,
                            history_len,
                            feature_attention_mask.shape[-1],
                        ):
                            raise ValueError(
                                "history_feature_attention_mask must match [batch, history, seq_len]."
                            )
                        history_feature_attention_mask = history_override_mask
                    else:
                        history_feature_attention_mask = (
                            feature_attention_mask.unsqueeze(1).expand(-1, history_len, -1).contiguous()
                        )
                else:
                    raise ValueError("feature_attention_mask must be 2D or 3D.")
            if audio_feature_lengths is not None:
                if audio_feature_lengths.dim() == 2:
                    history_audio_feature_lengths = audio_feature_lengths
                    audio_feature_lengths = audio_feature_lengths[:, -1]
                elif audio_feature_lengths.dim() == 1:
                    history_audio_feature_lengths = (
                        audio_feature_lengths.unsqueeze(1).expand(-1, history_len).contiguous()
                    )
                else:
                    raise ValueError("audio_feature_lengths must be 1D or 2D.")

        current_features, current_feature_lens = self._encode_audio_features(
            audio_tower=self.audio_tower,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

        if not self.audio_history_enabled or self.audio_history_window <= 0:
            return current_features

        current_splits, current_lengths_list, _ = self._split_audio_features(
            audio_features=current_features,
            feature_lengths=current_feature_lens,
        )
        if history_input_features is not None:
            history_features = self._build_history_from_inputs(
                history_input_features=history_input_features,
                history_feature_attention_mask=history_feature_attention_mask,
                history_audio_feature_lengths=history_audio_feature_lengths,
                current_output_lengths=current_lengths_list,
            )
        else:
            history_features = self._build_history_from_current(
                current_splits=current_splits,
                output_lengths_list=current_lengths_list,
            )
        if history_features is None:
            return current_features
        history_features = history_features.to(dtype=current_features.dtype)
        scale = self.audio_history_scale if self.audio_history_scale is not None else 1.0
        return current_features + history_features * scale

    def encode_multimodal_inputs(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)


            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale
                visual_pos_masks = video_mask
            else:

                image_mask = visual_pos_masks
                visual_pos_masks = video_mask | image_mask

                visual_embeds_multiscale_joint = ()
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(visual_embeds_multiscale, video_embeds_multiscale):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
                visual_embeds_multiscale = visual_embeds_multiscale_joint
                # --- end ---

        return inputs_embeds, visual_embeds_multiscale, visual_pos_masks

    def forward(self, **kwargs):

        history_feature_attention_mask = kwargs.pop("history_feature_attention_mask", None)
        if history_feature_attention_mask is not None:
            self._history_feature_attention_mask = history_feature_attention_mask
        try:
            return Qwen3OmniMoeThinkerForConditionalGeneration.forward(self, **kwargs)
        finally:
            if history_feature_attention_mask is not None:
                self._history_feature_attention_mask = None


class SplitQwen3OmniMoeForConditionalGeneration(Qwen3OmniMoeForConditionalGeneration):
    def __init__(self, config):
        super(Qwen3OmniMoeForConditionalGeneration, self).__init__(config)

        self.thinker = SplitQwen3OmniMoeThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.enable_talker()
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.thinker(*args, **kwargs)
