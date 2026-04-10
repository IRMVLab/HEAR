from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):


    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        use_joint_attention: bool = False,
        use_cross_attention: bool = False,
        use_kv_cache_transfer: bool = True,
        use_gradient_checkpointing: bool = True,
        freeze_vlm_layers: int = 0,
    ):

        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()


        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152


        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None


        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        vlm_config_hf._attn_implementation = "eager"


        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
            _attn_implementation="eager",
        )


        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None


        self.use_joint_attention = use_joint_attention
        self.use_cross_attention = use_cross_attention
        self.use_kv_cache_transfer = use_kv_cache_transfer
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.freeze_vlm_layers = freeze_vlm_layers


        mode_count = sum([use_cross_attention, use_kv_cache_transfer, use_joint_attention])
        if mode_count > 1:
            raise ValueError(
                "Choose exactly one mode: use_cross_attention, use_kv_cache_transfer, or use_joint_attention."
            )


        if not use_joint_attention and use_cross_attention:


            self.vlm_to_expert_proj = nn.Linear(
                vlm_config.width,
                action_expert_config.width,
                bias=False
            )


        self.to_bfloat16_for_selected_params(precision)


        if freeze_vlm_layers > 0:
            for i, layer in enumerate(self.paligemma.language_model.layers):
                if i < freeze_vlm_layers:
                    for param in layer.parameters():
                        param.requires_grad = False


    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):

        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")


        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]


        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):

        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):

        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):


        if adarms_cond is None:
            adarms_cond = [None, None]


        if inputs_embeds[1] is None:

            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None


        elif inputs_embeds[0] is None:

            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None


        else:

            if not self.use_joint_attention:


                use_gc = self.use_gradient_checkpointing and self.training


                prefix_seq_len = inputs_embeds[0].shape[1]


                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        # 4D mask: [batch, num_heads, seq_len, seq_len] [8, 1, 1028, 1028]

                        prefix_attention_mask = attention_mask[:, :, :prefix_seq_len, :prefix_seq_len]
                    elif attention_mask.dim() == 2:
                        # 2D mask: [batch, seq_len]
                        prefix_attention_mask = attention_mask[:, :prefix_seq_len]
                    else:
                        prefix_attention_mask = None
                else:
                    prefix_attention_mask = None


                if position_ids is not None:
                    prefix_position_ids = position_ids[:, :prefix_seq_len]
                else:
                    prefix_position_ids = None


                def run_vlm_forward(inputs_embeds_0, prefix_attention_mask, prefix_position_ids, adarms_cond_0):
                    aligemma.language_model.forward(
                        inputs_embeds=inputs_embeds_0,
                        attention_mask=prefix_attention_mask,
                        position_ids=prefix_position_ids,
                        past_key_values=None,
                        use_cache=self.use_kv_cache_transfer or use_cache,
                        adarms_cond=adarms_cond_0,
                    )


                # if use_gc:

                #     prefix_outputs = torch.utils.checkpoint.checkpoint(
                #         run_vlm_forward,
                #         inputs_embeds[0],       # [8, 968, 2048]
                #         prefix_attention_mask,
                #         prefix_position_ids,
                #         adarms_cond[0] if adarms_cond is not None else None,


                #     )
                # else:

                prefix_outputs = run_vlm_forward(
                    inputs_embeds[0],
                    prefix_attention_mask,
                    prefix_position_ids,
                    adarms_cond[0] if adarms_cond is not None else None,
                )

                prefix_output = prefix_outputs.last_hidden_state
                vlm_past_key_values = prefix_outputs.past_key_values if self.use_kv_cache_transfer else None


                suffix_seq_len = inputs_embeds[1].shape[1]
                total_seq_len = prefix_seq_len + suffix_seq_len


                if self.use_kv_cache_transfer:


                    if attention_mask is not None:
                        if attention_mask.dim() == 4:
                            batch_size = attention_mask.shape[0]
                            num_heads = attention_mask.shape[1]


                            new_mask = torch.full(
                                (batch_size, num_heads, suffix_seq_len, total_seq_len),
                                torch.finfo(attention_mask.dtype).min,
                                device=attention_mask.device,
                                dtype=attention_mask.dtype
                            )


                            new_mask[:, :, :, :prefix_seq_len] = 0


                            new_mask[:, :, :, prefix_seq_len:] = \
                                attention_mask[:, :, prefix_seq_len:, prefix_seq_len:]

                            suffix_attention_mask = new_mask
                        elif attention_mask.dim() == 2:

                            vlm_mask = torch.ones(
                                attention_mask.shape[0], prefix_seq_len,
                                device=attention_mask.device,
                                dtype=attention_mask.dtype
                            )
                            suffix_mask = attention_mask[:, prefix_seq_len:]
                            suffix_attention_mask = torch.cat([vlm_mask, suffix_mask], dim=-1)
                        else:
                            suffix_attention_mask = None
                    else:
                        suffix_attention_mask = None


                    if position_ids is not None:

                        suffix_position_ids = position_ids[:, prefix_seq_len:]
                    else:
                        suffix_position_ids = None


                    def run_expert_forward_with_kv(inputs_embeds_1, suffix_mask, suffix_pos_ids,
                                                   vlm_kv, adarms_cond_1):
                           return self.gemma_expert.model.forward(
                            inputs_embeds=inputs_embeds_1,
                            attention_mask=suffix_mask,
                            position_ids=suffix_pos_ids,
                            past_key_values=vlm_kv,
                            use_cache=False,
                            adarms_cond=adarms_cond_1,
                        )


                    if use_gc:


                        suffix_outputs = torch.utils.checkpoint.checkpoint(
                            run_expert_forward_with_kv,
                            inputs_embeds[1],
                            suffix_attention_mask,
                            suffix_position_ids,
                            vlm_past_key_values,
                            adarms_cond[1] if adarms_cond is not None else None,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        suffix_outputs = run_expert_forward_with_kv(
                            inputs_embeds[1],
                            suffix_attention_mask,
                            suffix_position_ids,
                            vlm_past_key_values,
                            adarms_cond[1] if adarms_cond is not None else None,
                        )

                    suffix_output = suffix_outputs.last_hidden_state


                elif self.use_cross_attention:

                    if attention_mask is not None:
                        if attention_mask.dim() == 4:

                            suffix_attention_mask = attention_mask[:, :, prefix_seq_len:total_seq_len, prefix_seq_len:total_seq_len]
                        elif attention_mask.dim() == 2:
                            # 2D mask
                            suffix_attention_mask = attention_mask[:, prefix_seq_len:total_seq_len]
                        else:
                            suffix_attention_mask = None
                    else:
                        suffix_attention_mask = None


                    if position_ids is not None:
                        suffix_position_ids = position_ids[:, prefix_seq_len:total_seq_len] # [8, 60]

                        suffix_position_ids = suffix_position_ids - suffix_position_ids[:, 0:1]
                    else:
                        suffix_position_ids = None


                    vlm_context = self.vlm_to_expert_proj(prefix_output)  # [batch, prefix_len, expert_width]
                    vlm_context_len = vlm_context.shape[1]
                    combined_seq_len = vlm_context_len + suffix_seq_len


                    if attention_mask is not None and attention_mask.dim() == 4:
                        batch_size = attention_mask.shape[0]
                        num_heads = attention_mask.shape[1]


                        new_mask = torch.full(
                            (batch_size, num_heads, combined_seq_len, combined_seq_len),
                            torch.finfo(attention_mask.dtype).min,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )


                        new_mask[:, :, :vlm_context_len, :vlm_context_len] = \
                            attention_mask[:, :, :vlm_context_len, :vlm_context_len]


                        new_mask[:, :, vlm_context_len:, :vlm_context_len] = 0


                        new_mask[:, :, vlm_context_len:, vlm_context_len:] = \
                            attention_mask[:, :, prefix_seq_len:total_seq_len, prefix_seq_len:total_seq_len]

                        suffix_attention_mask = new_mask
                    elif attention_mask is not None and attention_mask.dim() == 2:

                        vlm_mask = torch.ones(
                            attention_mask.shape[0], vlm_context_len,
                            device=attention_mask.device,
                            dtype=attention_mask.dtype
                        )
                        suffix_mask = attention_mask[:, prefix_seq_len:total_seq_len]
                        suffix_attention_mask = torch.cat([vlm_mask, suffix_mask], dim=-1)
                    else:
                        suffix_attention_mask = None


                    if position_ids is not None:

                        vlm_position_ids = position_ids[:, :vlm_context_len]


                        expert_position_ids = position_ids[:, prefix_seq_len:total_seq_len]

                        expert_position_ids = expert_position_ids - expert_position_ids[:, 0:1] + vlm_context_len


                        combined_position_ids = torch.cat([vlm_position_ids, expert_position_ids], dim=1)
                    else:
                        combined_position_ids = None


                    combined_inputs_embeds = torch.cat([vlm_context, inputs_embeds[1]], dim=1)


                    def run_expert_forward(combined_embeds, suffix_mask, combined_pos_ids, adarms_cond_1):
                        self.gemma_expert.model.forward(
                            inputs_embeds=combined_embeds,
                            attention_mask=suffix_mask,
                            position_ids=combined_pos_ids,
                            past_key_values=None,
                            use_cache=False,
                            adarms_cond=adarms_cond_1,
                        )


                    if use_gc:
                        suffix_outputs = torch.utils.checkpoint.checkpoint(
                            run_expert_forward,
                            combined_inputs_embeds,
                            suffix_attention_mask,
                            combined_position_ids,
                            adarms_cond[1] if adarms_cond is not None else None,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        suffix_outputs = run_expert_forward(
                            combined_inputs_embeds,
                            suffix_attention_mask,
                            combined_position_ids,
                            adarms_cond[1] if adarms_cond is not None else None,
                        )


                    suffix_output = suffix_outputs.last_hidden_state[:, vlm_context_len:, :]


                else:


                    if attention_mask is not None:
                        if attention_mask.dim() == 4:
                            suffix_attention_mask = attention_mask[:, :, prefix_seq_len:total_seq_len, prefix_seq_len:total_seq_len]
                        elif attention_mask.dim() == 2:
                            suffix_attention_mask = attention_mask[:, prefix_seq_len:total_seq_len]
                        else:
                            suffix_attention_mask = None
                    else:
                        suffix_attention_mask = None


                    if position_ids is not None:
                        suffix_position_ids = position_ids[:, prefix_seq_len:total_seq_len]

                        suffix_position_ids = suffix_position_ids - suffix_position_ids[:, 0:1]
                    else:
                        suffix_position_ids = None


                    def run_expert_forward_simple(inputs_embeds_1, suffix_mask, suffix_pos_ids, adarms_cond_1):
                                 return self.gemma_expert.model.forward(
                            inputs_embeds=inputs_embeds_1,
                            attention_mask=suffix_mask,
                            position_ids=suffix_pos_ids,
                            past_key_values=None,
                            use_cache=False,
                            adarms_cond=adarms_cond_1,
                        )


                    if use_gc:
                        suffix_outputs = torch.utils.checkpoint.checkpoint(
                            run_expert_forward_simple,
                            inputs_embeds[1],
                            suffix_attention_mask,
                            suffix_position_ids,
                            adarms_cond[1] if adarms_cond is not None else None,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        suffix_outputs = run_expert_forward_simple(
                            inputs_embeds[1],
                            suffix_attention_mask,
                            suffix_position_ids,
                            adarms_cond[1] if adarms_cond is not None else None,
                        )

                    suffix_output = suffix_outputs.last_hidden_state

                prefix_past_key_values = None


            else:


                models = [self.paligemma.language_model, self.gemma_expert.model]
                num_layers = self.paligemma.config.text_config.num_hidden_layers


                use_gradient_checkpointing = (
                    hasattr(self.gemma_expert.model, "gradient_checkpointing")
                    and self.gemma_expert.model.gradient_checkpointing
                    and self.training
                ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)


                if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    if not self.gemma_expert.model.gradient_checkpointing:
                        print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                        self.gemma_expert.model.gradient_checkpointing = True
                    use_gradient_checkpointing = True


                if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                    print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                    print(f"Model training mode: {self.training}")
                    print(
                        f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                    )
                    if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                        print(
                            f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                        )
                    self._debug_gc_printed = True


                def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):

                    models = [self.paligemma.language_model, self.gemma_expert.model]


                    query_states = []
                    key_states = []
                    value_states = []
                    gates = []

                    for i, hidden_states in enumerate(inputs_embeds):
                        layer = models[i].layers[layer_idx]

                        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                        gates.append(gate)


                        input_shape = hidden_states.shape[:-1]  # [batch, seq_len]
                        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)  # [batch, seq_len, num_heads, head_dim]


                        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                        query_states.append(query_state)
                        key_states.append(key_state)
                        value_states.append(value_state)


                    query_states = torch.cat(query_states, dim=2)  # [batch, heads, combined_seq, head_dim]
                    key_states = torch.cat(key_states, dim=2)
                    value_states = torch.cat(value_states, dim=2)


                    dummy_tensor = torch.zeros(
                        query_states.shape[0],
                        query_states.shape[2],
                        query_states.shape[-1],
                        device=query_states.device,
                        dtype=query_states.dtype,
                    )
                    cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)

                    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, unsqueeze_dim=1
                    )

                    batch_size = query_states.shape[0]
                    scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling


                    att_output, _ = modeling_gemma.eager_attention_forward(
                        self.paligemma.language_model.layers[layer_idx].self_attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling,
                    )

                    head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)


                    outputs_embeds = []
                    start_pos = 0
                    for i, hidden_states in enumerate(inputs_embeds):
                        layer = models[i].layers[layer_idx]
                        end_pos = start_pos + hidden_states.shape[1]


                        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])


                        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                        after_first_residual = out_emb.clone()


                        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])


                        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                            out_emb = out_emb.to(dtype=torch.bfloat16)


                        out_emb = layer.mlp(out_emb)


                        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                        outputs_embeds.append(out_emb)
                        start_pos = end_pos

                    return outputs_embeds


                for layer_idx in range(num_layers):
                    if use_gradient_checkpointing:

                        inputs_embeds = torch.utils.checkpoint.checkpoint(
                            compute_layer_complete,
                            layer_idx,
                            inputs_embeds,
                            attention_mask,
                            position_ids,
                            adarms_cond,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:

                        inputs_embeds = compute_layer_complete(
                            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                        )


                def compute_final_norms(inputs_embeds, adarms_cond):
                    outputs_embeds = []
                    for i, hidden_states in enumerate(inputs_embeds):
                        out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                        outputs_embeds.append(out_emb)
                    return outputs_embeds


                if use_gradient_checkpointing:
                    outputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                    )
                else:
                    outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

                prefix_output = outputs_embeds[0]
                suffix_output = outputs_embeds[1]
                prefix_past_key_values = None


        return [prefix_output, suffix_output], prefix_past_key_values
