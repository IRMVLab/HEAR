from typing import Any, Literal
import logging
import os

import torch
from functools import partial
from torch import nn
from openpi.models_pytorch.SplitQwen3OmniMoeThinkerForConditionalGeneration import (
    SplitQwen3OmniMoeForConditionalGeneration,
    SplitQwen3OmniMoeThinkerForConditionalGeneration,
)
from openpi.models_pytorch.qwen3_with_adarms import (
    Qwen3ForCausalLMWithAdaRMS,
    SplitQwen3ForCausalLM,
)

from collections.abc import Mapping
try:
    from transformers.feature_extraction_utils import BatchFeature
    batch_feature_types = (BatchFeature,)
except Exception:
    batch_feature_types = ()

class Qwen3WithExpertModel(nn.Module):

    def __init__(
        self,
        total_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        use_joint_attention: bool = False,
        use_kv_cache_transfer: bool = True,
    ):

        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.precision = precision
        model_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

        def _resolve_mimi_num_code_groups() -> int | None:
            override = getattr(total_config, "audio_codec_num_quantizers", None)
            if override is not None:
                return int(override)

            model_path = getattr(total_config, "audio_codec_path", None)
            if not model_path:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                candidates = []
                env_path = os.environ.get("HEAR_AUDIO_CODEC_PATH")
                if env_path:
                    candidates.append(env_path)
                candidates.extend(
                    [
                        os.path.join(repo_root, "models", "mimi"),
                        os.path.join(repo_root, "mimi"),
                    ]
                )
                hf_home = os.environ.get("HF_HOME")
                if hf_home:
                    candidates.append(os.path.join(hf_home, "hub", "models--kyutai--mimi"))
                for path in candidates:
                    if path and os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                        model_path = path
                        break

            if model_path and os.path.isdir(model_path):
                try:
                    from transformers import MimiConfig
                    return int(MimiConfig.from_pretrained(model_path).num_quantizers)
                except Exception:
                    return None

            return None

        enable_qwen3omni_talker = bool(getattr(total_config, "enable_qwen3omni_talker", False))
        if enable_qwen3omni_talker:
            qwen3omni_config = None
            num_code_groups = None
            if getattr(total_config, "qwen3omni_path", None):
                from transformers import Qwen3OmniMoeConfig

                qwen3omni_config = Qwen3OmniMoeConfig.from_pretrained(total_config.qwen3omni_path)
                if getattr(total_config, "audio_codec_backend", "mimi") == "mimi":
                    num_code_groups = _resolve_mimi_num_code_groups() or 32
                    qwen3omni_config.talker_config.num_code_groups = num_code_groups
                    if hasattr(qwen3omni_config.talker_config, "code_predictor_config"):
                        qwen3omni_config.talker_config.code_predictor_config.num_code_groups = num_code_groups
                    if hasattr(qwen3omni_config, "code2wav_config"):
                        qwen3omni_config.code2wav_config.num_quantizers = num_code_groups

            self.qwen3omni = SplitQwen3OmniMoeForConditionalGeneration.from_pretrained(
                total_config.qwen3omni_path,
                config=qwen3omni_config,
                dtype=model_dtype,
                # device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                # attn_implementation="flash_attention_2",
            )
            enable_talker = bool(getattr(total_config, "enable_audio_output", True))
            if hasattr(self.qwen3omni, "has_talker") and self.qwen3omni.has_talker and not enable_talker:
                self.qwen3omni.disable_talker()
        else:
            self.qwen3omni = SplitQwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                total_config.qwen3omni_path,
                dtype=model_dtype,
                # device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                # attn_implementation="flash_attention_2",
            )

        qwen3omni_thinker = self.qwen3omni.thinker if hasattr(self.qwen3omni, "thinker") else self.qwen3omni
        history_window = getattr(total_config, "audio_history_window", None)
        if history_window is not None and hasattr(qwen3omni_thinker, "audio_history_window"):
            qwen3omni_thinker.audio_history_window = history_window
        history_scale = getattr(total_config, "audio_history_scale", None)
        if (
            history_scale is not None
            and hasattr(qwen3omni_thinker, "audio_history_scale")
            and qwen3omni_thinker.audio_history_scale is not None
        ):
            qwen3omni_thinker.audio_history_scale.data.fill_(float(history_scale))
        history_gc = getattr(total_config, "audio_history_gradient_checkpointing", None)
        if history_gc is not None and hasattr(qwen3omni_thinker, "audio_history_temporal"):
            if qwen3omni_thinker.audio_history_temporal is not None:
                qwen3omni_thinker.audio_history_temporal.set_gradient_checkpointing(bool(history_gc))


        from transformers import AutoConfig

        qwen3_config = AutoConfig.from_pretrained(total_config.qwen3_path)


        qwen3_config.adarms_cond_dim = qwen3_config.hidden_size if use_adarms[1] else None
        qwen3_config.use_adarms = use_adarms[1]


        self.qwen3_expert = Qwen3ForCausalLMWithAdaRMS(qwen3_config)


        pretrained_state_dict = SplitQwen3ForCausalLM.from_pretrained(
            total_config.qwen3_path,
            dtype=model_dtype,
            # device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            # attn_implementation="flash_attention_2",
        ).state_dict()


        self.qwen3_expert.load_state_dict(pretrained_state_dict, strict=False)


        if precision == "bfloat16":
            self.qwen3_expert = self.qwen3_expert.to(dtype=torch.bfloat16)


        self.qwen3_expert.model.embed_tokens = None


        self.use_joint_attention = use_joint_attention
        self.use_kv_cache_transfer = use_kv_cache_transfer


        mode_count = sum([use_kv_cache_transfer, use_joint_attention])
        if mode_count > 1:
            raise ValueError("Choose exactly one mode: use_kv_cache_transfer or use_joint_attention.")
        if mode_count == 0:
            raise ValueError("Enable at least one mode: use_kv_cache_transfer or use_joint_attention.")


    def _ensure_gradient_checkpointing(self):

        if getattr(self, "_is_gc_enabled_runtime", False):
            return
        if not self.training:
            return

        logging.info("Enabling gradient checkpointing for Qwen3 modules.")

        non_reentrant_checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        count = 0

        def _apply_gc_to_modules(module_list):
            nonlocal count
            for layer in module_list:
                layer.gradient_checkpointing = True
                # Ensure HF modeling layers have the checkpoint function set.
                layer._gradient_checkpointing_func = non_reentrant_checkpoint_func
                layer.gradient_checkpointing_func = non_reentrant_checkpoint_func
                count += 1

        # ==========================================

        # ==========================================
        qwen3omni_thinker = self.qwen3omni.thinker if hasattr(self.qwen3omni, "thinker") else self.qwen3omni
        if hasattr(qwen3omni_thinker, "config"):

            qwen3omni_thinker.config.gradient_checkpointing = True
            qwen3omni_thinker.config.use_cache = True


            if hasattr(qwen3omni_thinker, "audio_tower") and hasattr(qwen3omni_thinker.audio_tower, "layers"):
                _apply_gc_to_modules(qwen3omni_thinker.audio_tower.layers)

            if hasattr(qwen3omni_thinker, "visual") and hasattr(qwen3omni_thinker.visual, "blocks"):
                _apply_gc_to_modules(qwen3omni_thinker.visual.blocks)


            if hasattr(qwen3omni_thinker, "model") and hasattr(qwen3omni_thinker.model, "layers"):

                qwen3omni_thinker.model.gradient_checkpointing = True
                for layer in qwen3omni_thinker.model.layers:
                    layer.gradient_checkpointing = False
                logging.info("Qwen3Omni thinker text uses manual checkpointing with cache transfer.")


        if hasattr(self.qwen3omni, "talker"):
            talker = self.qwen3omni.talker
            if hasattr(talker, "model") and hasattr(talker.model, "layers"):
                talker.model.gradient_checkpointing = True
                _apply_gc_to_modules(talker.model.layers)
            if (
                hasattr(talker, "code_predictor")
                and hasattr(talker.code_predictor, "model")
                and hasattr(talker.code_predictor.model, "layers")
            ):
                talker.code_predictor.model.gradient_checkpointing = True
                _apply_gc_to_modules(talker.code_predictor.model.layers)

        if (
            hasattr(self.qwen3omni, "code2wav")
            and hasattr(self.qwen3omni.code2wav, "pre_transformer")
            and hasattr(self.qwen3omni.code2wav.pre_transformer, "layers")
        ):
            self.qwen3omni.code2wav.pre_transformer.gradient_checkpointing = True
            _apply_gc_to_modules(self.qwen3omni.code2wav.pre_transformer.layers)

        # ==========================================

        # ==========================================
        if hasattr(self.qwen3_expert, "config"):
            self.qwen3_expert.config.gradient_checkpointing = True

        if hasattr(self.qwen3_expert, "model") and hasattr(self.qwen3_expert.model, "layers"):
            _apply_gc_to_modules(self.qwen3_expert.model.layers)

        logging.info("Configured gradient checkpointing for %s layers.", count)
        self._is_gc_enabled_runtime = True

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: list[torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        labels: torch.LongTensor | None = None,
        return_vlm_loss: bool = False,
    ):
        if not self.use_kv_cache_transfer:
            raise NotImplementedError("The current implementation only supports use_kv_cache_transfer.")
        if inputs_embeds is None or len(inputs_embeds) != 2:
            raise ValueError("inputs_embeds must be a two-item list: [vlm_inputs, expert_inputs].")

        if adarms_cond is None:
            adarms_cond = [None, None]
        elif len(adarms_cond) != 2:
            raise ValueError("adarms_cond must contain exactly two elements: [vlm_cond, expert_cond].")

        if isinstance(attention_mask, dict):
            if "full_attention" not in attention_mask:
                raise ValueError("attention_mask must contain the 'full_attention' key.")
            attention_mask_full = attention_mask["full_attention"]
        else:
            attention_mask_full = attention_mask


        vlm_text_loss = None


        if self.use_kv_cache_transfer:

            self._ensure_gradient_checkpointing()

            def _slice_suffix_attention_mask(full_mask, prefix_len, suffix_len, total_len):
                if full_mask is None:
                    return None
                if full_mask.dim() != 4:
                    raise ValueError("KV cache transfer requires a 4D full_attention mask.")
                q_end = prefix_len + suffix_len
                if full_mask.shape[2] < q_end:
                    raise ValueError("The query dimension of full_attention is smaller than prefix+suffix.")
                if full_mask.shape[-1] < total_len:
                    raise ValueError("The key dimension of full_attention is smaller than prefix+suffix.")
                return full_mask[:, :, prefix_len:q_end, :total_len].contiguous()

            vlm_inputs_raw = inputs_embeds[0]
            if isinstance(vlm_inputs_raw, (Mapping,) + batch_feature_types):
                vlm_inputs = dict(vlm_inputs_raw)
            elif isinstance(vlm_inputs_raw, torch.Tensor):
                vlm_inputs = {"inputs_embeds": vlm_inputs_raw}
            else:
                raise TypeError("inputs_embeds[0] must be a Mapping or a Tensor.")


            if labels is not None:
                if "input_ids" not in vlm_inputs:
                    raise ValueError("vlm_inputs must include input_ids when labels are provided.")
                target_labels = labels.to(dtype=torch.long, device=vlm_inputs["input_ids"].device)
                if target_labels.shape != vlm_inputs["input_ids"].shape:
                    raise ValueError(
                        f"labels shape {tuple(target_labels.shape)} does not match input_ids shape "
                        f"{tuple(vlm_inputs['input_ids'].shape)}."
                    )
                vlm_inputs["labels"] = target_labels

            device_type = "cpu"
            for value in vlm_inputs.values():
                if isinstance(value, torch.Tensor):
                    device_type = value.device.type
                    break
            autocast_enabled = self.precision == "bfloat16" and device_type == "cuda"

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):

                vlm_use_cache = True if self.use_kv_cache_transfer else bool(use_cache)
                prefix_outputs = self.qwen3omni(
                    **vlm_inputs,
                    past_key_values=None,
                    use_cache=vlm_use_cache,
                    output_hidden_states=True,
                )


                # k_tensor = prefix_outputs.past_key_values[0][0]

                # print(f"\n[Gradient Check] KV Cache Status:")
                # print(f"  -> Is None? {k_tensor is None}")
                # print(f"  -> Requires Grad? {k_tensor.requires_grad}")
                # print(f"  -> Has grad_fn? {k_tensor.grad_fn}")

                # if k_tensor.grad_fn is None:


                # else:


                if hasattr(prefix_outputs, "loss") and prefix_outputs.loss is not None:
                    vlm_text_loss = prefix_outputs.loss

                prefix_output = prefix_outputs.hidden_states
                prefix_last_hidden = prefix_outputs.hidden_states[-1]
                prefix_seq_len = prefix_last_hidden.shape[1]
                vlm_past_key_values = prefix_outputs.past_key_values


                # print(f"[Debug] kv shape: {vlm_past_key_values[0][0].shape}, prefix_seq_len: {prefix_seq_len}")

                suffix_inputs = inputs_embeds[1]
                if not isinstance(suffix_inputs, torch.Tensor):
                    raise TypeError("inputs_embeds[1] must be a Tensor.")
                suffix_seq_len = suffix_inputs.shape[1]
                total_seq_len = prefix_seq_len + suffix_seq_len

                suffix_attention_mask = _slice_suffix_attention_mask(
                    attention_mask_full, prefix_seq_len, suffix_seq_len, total_seq_len
                )
                suffix_position_ids = (
                    position_ids[:, prefix_seq_len:prefix_seq_len + suffix_seq_len]
                    if position_ids is not None
                    else None
                )

                expert_dtype = self.qwen3_expert.model.layers[0].self_attn.q_proj.weight.dtype
                expert_adarms_cond = (adarms_cond[1] if adarms_cond is not None else None)
                if expert_adarms_cond is not None:
                    expert_adarms_cond = expert_adarms_cond.to(dtype=expert_dtype)
                if suffix_attention_mask is not None:
                    suffix_attention_mask = suffix_attention_mask.to(dtype=expert_dtype)

                def run_expert_forward_with_kv(
                    inputs_embeds_1,
                    suffix_mask,
                    suffix_pos_ids,
                    vlm_kv,
                    adarms_cond_1,
                ):
                    attention_mask_arg = {"full_attention": suffix_mask} if suffix_mask is not None else None

                    return self.qwen3_expert.model(
                        inputs_embeds=inputs_embeds_1,
                        attention_mask=attention_mask_arg,
                        position_ids=suffix_pos_ids,
                        past_key_values=None,
                        external_past_key_values=vlm_kv,
                        use_cache=False,
                        adarms_cond=adarms_cond_1,
                    )

                suffix_outputs = run_expert_forward_with_kv(
                    suffix_inputs,
                    suffix_attention_mask,
                    suffix_position_ids,
                    vlm_past_key_values,
                    expert_adarms_cond,
                )

                suffix_output = suffix_outputs.last_hidden_state
                prefix_past_key_values = vlm_past_key_values


        if return_vlm_loss:

            return ([prefix_output, suffix_output], vlm_text_loss), prefix_past_key_values

        return [prefix_output, suffix_output], prefix_past_key_values
