from __future__ import annotations
from typing import Optional, Tuple
import torch
import copy
import cv2
import json
__all__ = [
    "expand_qwen3_position_ids",
    "split_attention_and_position",
]
import logging
import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image
import os
from openpi.models_pytorch.qwen3_pytorch import Qwen3WithExpertModel
from openpi.models_pytorch.audio_codec import AudioCodecConfig, SimpleAudioCodec, AudioCodePredictor, MimiAudioCodec
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from transformers import Qwen3OmniMoeProcessor, AutoTokenizer
from qwen_omni_utils import process_mm_info
from openpi.models import tokenizer as _tokenizer
import scipy.io.wavfile as wavfile

def _slice_attention_mask(
    attention_mask: Optional[torch.Tensor], start: int, length: int
) -> Optional[torch.Tensor]:
    if attention_mask is None or length <= 0:
        return None
    end = start + length
    if attention_mask.dim() == 4:
        return attention_mask[:, :, start:end, start:end]
    return attention_mask[:, start:end]


def _collapse_attention_mask(mask_slice: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask_slice is None:
        return None
    if mask_slice.dim() == 2:
        return mask_slice
    diag = torch.diagonal(mask_slice[:, 0], dim1=-2, dim2=-1)
    threshold = torch.finfo(diag.dtype).min / 2
    collapsed = torch.zeros_like(diag)
    collapsed[diag > threshold] = 1
    return collapsed


def _slice_position_ids(
    position_ids: Optional[torch.Tensor], start: int, length: int
) -> Optional[torch.Tensor]:
    if position_ids is None or length <= 0:
        return None
    end = start + length
    return position_ids[:, start:end].to(dtype=torch.long)


def expand_qwen3_position_ids(position_ids_slice: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if position_ids_slice is None:
        return None
    if position_ids_slice.dim() == 3:
        return position_ids_slice
    return position_ids_slice.unsqueeze(0).expand(3, -1, -1)


def split_attention_and_position(
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    prefix_len: int,
    suffix_len: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    prefix_mask_4d = _slice_attention_mask(attention_mask, 0, prefix_len) if attention_mask is not None else None
    suffix_mask_4d = (
        _slice_attention_mask(attention_mask, prefix_len, suffix_len) if attention_mask is not None else None
    )
    prefix_mask_2d = _collapse_attention_mask(prefix_mask_4d) if prefix_mask_4d is not None else None
    suffix_mask_2d = _collapse_attention_mask(suffix_mask_4d) if suffix_mask_4d is not None else None
    prefix_pos = _slice_position_ids(position_ids, 0, prefix_len)
    suffix_pos = _slice_position_ids(position_ids, prefix_len, suffix_len)
    return prefix_mask_4d, suffix_mask_4d, prefix_mask_2d, suffix_mask_2d, prefix_pos, suffix_pos


def save_audio_debug(audio_np, file_path, sample_rate=16000):
    audio_np = np.asarray(audio_np)
    if audio_np.ndim > 2:
        raise ValueError(f"Unsupported audio rank: {audio_np.ndim}")
    max_abs = np.abs(audio_np).max(initial=0)
    if max_abs > 1:
        audio_np = audio_np / max_abs
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    wavfile.write(file_path, sample_rate, audio_int16)

def save_audio_waveform(audio_np, file_path, image_width=1024, image_height=256, line_color=(255, 255, 255)):
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    max_abs = np.max(np.abs(audio_np), initial=0.0)
    if max_abs > 0:
        audio_np = audio_np / max_abs
    points = np.linspace(0, image_width - 1, num=len(audio_np), dtype=np.float32)
    scaled = (audio_np * 0.5 + 0.5) * (image_height - 1)
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    pts = np.stack([points, scaled], axis=1).astype(np.int32)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, tuple(p0.tolist()), tuple(p1.tolist()), line_color, 1, cv2.LINE_AA)
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(file_path, img)

def get_safe_dtype(target_dtype, device_type):

    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:

    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)

    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi

    sin_input = scaling_factor[None, :] * time[:, None]

    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):


    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)

    dist = torch.distributions.Beta(alpha_t, beta_t)

    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2: raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2: raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config, total_config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        self.Qwen3_with_expert = Qwen3WithExpertModel(
            total_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.Qwen3Omni_processor = Qwen3OmniMoeProcessor.from_pretrained(total_config.qwen3omni_path)
        self.Qwen3_tokenizer = AutoTokenizer.from_pretrained(total_config.qwen3_path)
        self.Paligemma_tokenizer = _tokenizer.PaligemmaTokenizer()

        action_expert_width = self.Qwen3_with_expert.qwen3_expert.model.config.hidden_size
        self.action_in_proj = nn.Linear(32, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(32, action_expert_width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out = nn.Linear(action_expert_width, action_expert_width)

        self.audio_output_enabled = bool(getattr(total_config, "enable_audio_output", True))
        self.audio_output_weight = float(getattr(total_config, "audio_output_weight", 1.0))
        self.audio_predictor_weight = float(getattr(total_config, "audio_predictor_weight", 1.0))
        self.audio_codec_weight = float(getattr(total_config, "audio_codec_weight", 1.0))
        self.audio_recon_weight = float(getattr(total_config, "audio_recon_weight", 1.0))
        self.audio_vq_weight = float(getattr(total_config, "audio_vq_weight", 1.0))
        self.vlm_text_loss_weight = float(getattr(total_config, "vlm_text_loss_weight", 0.1))
        self.audio_codec_device = str(getattr(total_config, "audio_codec_device", "cpu")).lower()
        if self.audio_codec_device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported audio_codec_device: {self.audio_codec_device}")

        if self.audio_output_enabled:
            audio_codec = self._build_audio_codec(total_config)
            # Keep codec out of nn.Module to avoid DDP buffer sync/device mismatches.
            object.__setattr__(self, "_audio_codec", audio_codec)
            qwen3omni_thinker = (
                self.Qwen3_with_expert.qwen3omni.thinker
                if hasattr(self.Qwen3_with_expert.qwen3omni, "thinker")
                else self.Qwen3_with_expert.qwen3omni
            )
            context_dim = qwen3omni_thinker.config.text_config.hidden_size
            pred_hidden_dim = int(getattr(total_config, "audio_predictor_hidden_dim", 512))
            pred_layers = int(getattr(total_config, "audio_predictor_layers", 2))
            pred_heads = int(getattr(total_config, "audio_predictor_heads", 8))
            pred_dropout = float(getattr(total_config, "audio_predictor_dropout", 0.0))
            max_code_len = getattr(total_config, "audio_code_max_len", None)
            if max_code_len is None:
                max_code_len = int(getattr(audio_codec, "max_code_len", 1024))
            self.audio_code_predictor = AudioCodePredictor(
                context_dim=context_dim,
                codebook_size=audio_codec.codebook_size,
                num_quantizers=audio_codec.num_quantizers,
                max_code_len=int(max_code_len),
                hidden_dim=pred_hidden_dim,
                num_layers=pred_layers,
                num_heads=pred_heads,
                dropout=pred_dropout,
            )
        else:
            object.__setattr__(self, "_audio_codec", None)
            self.audio_code_predictor = None

        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = bool(getattr(total_config, "pi0_gradient_checkpointing", False))


    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

        logging.info("Enabled PI0 checkpoint wrappers; Qwen3 GC is handled in Qwen3WithExpertModel.")

    def gradient_checkpointing_disable(self):

        self.gradient_checkpointing_enabled = False

        logging.info("Disabled PI0 checkpoint wrappers; Qwen3 GC is handled in Qwen3WithExpertModel.")

    def is_gradient_checkpointing_enabled(self):

        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):

        if self.gradient_checkpointing_enabled and self.training:

            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )

        return func(*args, **kwargs)

    @property
    def audio_codec(self):
        return self.__dict__.get("_audio_codec", None)

    def _find_local_mimi_path(self) -> str | None:
        candidates = []
        env_path = os.environ.get("HEAR_AUDIO_CODEC_PATH")
        if env_path:
            candidates.append(env_path)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(str(repo_root / "models" / "mimi"))
        candidates.append(str(repo_root / "mimi"))
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(os.path.join(hf_home, "hub", "models--kyutai--mimi"))
        for path in candidates:
            if path and os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                return path
        return None

    def _resolve_audio_num_quantizers(self, total_config) -> int | None:
        override = getattr(total_config, "audio_codec_num_quantizers", None)
        if override is not None:
            return int(override)
        codec_cfg = getattr(total_config, "audio_codec_config", None)
        if isinstance(codec_cfg, dict):
            value = codec_cfg.get("num_quantizers")
            if value is not None:
                return int(value)
        return None

    def _build_audio_codec(self, total_config):
        backend = getattr(total_config, "audio_codec_backend", "mimi")
        if backend == "mimi":
            model_path = getattr(total_config, "audio_codec_path", None) or self._find_local_mimi_path()
            if model_path is None:
                raise ValueError("audio_codec_path is required for mimi codec.")
            num_quantizers = self._resolve_audio_num_quantizers(total_config)
            input_sr = int(getattr(total_config, "audio_input_sample_rate", 16000))
            local_only = bool(getattr(total_config, "audio_codec_local_files_only", True))
            return MimiAudioCodec(
                model_path=model_path,
                num_quantizers=num_quantizers,
                input_sample_rate=input_sr,
                local_files_only=local_only,
            )
        if backend == "simple":
            codec_config = self._resolve_audio_codec_config(total_config)
            return SimpleAudioCodec(codec_config)
        raise ValueError(f"Unsupported audio_codec_backend: {backend}")

    def _read_qwen3_codec_defaults(self, total_config) -> dict:
        qwen_path = getattr(total_config, "qwen3omni_path", None)
        if not qwen_path:
            return {}
        config_path = os.path.join(qwen_path, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return {}
        code2wav_cfg = cfg.get("code2wav_config", {})
        defaults = {
            "codebook_size": code2wav_cfg.get("codebook_size"),
            "codebook_dim": code2wav_cfg.get("codebook_dim")
            or code2wav_cfg.get("vector_quantization_hidden_dimension"),
            "num_quantizers": code2wav_cfg.get("num_quantizers"),
        }
        max_len = code2wav_cfg.get("max_position_embeddings")
        if max_len is not None:
            defaults["max_code_len"] = int(max_len)
        return defaults

    def _resolve_audio_codec_config(self, total_config) -> AudioCodecConfig:
        user_cfg = getattr(total_config, "audio_codec_config", None)
        if isinstance(user_cfg, AudioCodecConfig):
            return user_cfg
        cfg_dict = {}
        if isinstance(user_cfg, dict):
            cfg_dict.update(user_cfg)
        override_len = getattr(total_config, "audio_code_max_len", None)
        if override_len is not None:
            cfg_dict["max_code_len"] = int(override_len)
        qwen_defaults = self._read_qwen3_codec_defaults(total_config)
        for key, value in qwen_defaults.items():
            if value is not None and key not in cfg_dict:
                cfg_dict[key] = value
        return AudioCodecConfig.from_dict(cfg_dict)

    def _resolve_audio_codec_device(self, reference_device: torch.device) -> torch.device:
        if self.audio_codec_device == "auto":
            return reference_device
        if self.audio_codec_device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.audio_codec_device)

    def _prepare_audio_tensor(self, audio, device) -> torch.Tensor | None:
        if audio is None:
            return None
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        elif hasattr(audio, "numpy"):
            audio_tensor = torch.as_tensor(audio)
        else:
            audio_tensor = torch.tensor(audio)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(1)
        elif audio_tensor.dim() == 3 and audio_tensor.shape[1] != 1 and audio_tensor.shape[2] == 1:
            audio_tensor = audio_tensor.transpose(1, 2)
        return audio_tensor.to(device=device, dtype=torch.float32)

    def _compute_audio_loss(self, prefix_hidden, prefix_pad_masks, next_audios) -> torch.Tensor | None:
        if not self.audio_output_enabled or self.audio_codec is None or self.audio_code_predictor is None:
            return None
        codec_device = self._resolve_audio_codec_device(prefix_hidden.device)
        audio_tensor = self._prepare_audio_tensor(next_audios, codec_device)
        if audio_tensor is None:
            return None
        with torch.no_grad():
            target_codes = self.audio_codec.encode(audio_tensor)
            if isinstance(target_codes, tuple):
                target_codes = target_codes[0]
        target_codes = target_codes.detach()
        if target_codes.device != prefix_hidden.device:
            target_codes = target_codes.to(device=prefix_hidden.device, non_blocking=True)
        code_len = target_codes.shape[-1]
        logits = self.audio_code_predictor(prefix_hidden, prefix_pad_masks, code_len)
        logits = logits.to(dtype=torch.float32)
        targets = target_codes.permute(0, 2, 1).contiguous()
        ce_loss = F.cross_entropy(logits.reshape(-1, self.audio_codec.codebook_size), targets.reshape(-1))
        return self.audio_predictor_weight * ce_loss

    def _prepare_attention_masks_4d(self, att_2d_masks, *, dtype=None):

        att_2d_masks_4d = att_2d_masks[:, None, :, :]

        mask_dtype = dtype if dtype is not None else torch.float32
        zeros = torch.zeros((), dtype=mask_dtype, device=att_2d_masks_4d.device)
        neg_inf = torch.full(
            (), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=att_2d_masks_4d.device
        )
        return torch.where(att_2d_masks_4d, zeros, neg_inf)

    def _preprocess_observation(self, observation, *, train=True):

        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)


        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )


    def sample_noise(self, shape, device):

        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):

        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001

        return time.to(dtype=torch.float32, device=device)

    def embed_prefix_qwen3omni(
        self, images, img_masks, prompts, audios=None, stage_labels=None, debug=False, hist_audios=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed the prefix using Qwen3 Omni VLM.
        Args:
            stage_labels: Optional list of strings in JSON format, e.g. ['{"stage": "grasp"}', ...].
                         If provided (during training), these are appended to the conversation as assistant response.
        """
        conversations_qwen3omni = []
        device = images[0].device
        batch_size = len(prompts)
        assert images[0].shape[0] == batch_size, "Prompt batch size must match image batch size."
        if audios is None:
            audios_list = [None] * batch_size
        elif isinstance(audios, torch.Tensor):
            if audios.shape[0] != batch_size:
                raise ValueError("Audio batch size must match prompt batch size.")
            audios_list = [audios[i] for i in range(batch_size)]
        else:
            audios_list = list(audios)
            if len(audios_list) != batch_size:
                raise ValueError("Audio batch size must match prompt batch size.")

        if stage_labels is not None:
            assert len(stage_labels) == batch_size, "Stage labels batch size must match prompt batch size."

        img_masks_bool = [mask_tensor.to(torch.bool).cpu().tolist() for mask_tensor in img_masks]

        pil_images_by_cam = []
        for cam_idx, cam_tensor in enumerate(images):
            pil_images = batch_tensor_to_pil_images(
                cam_tensor,
                save_dir="./debug_data" if debug else None,
                prefix=f"camera_{cam_idx}"
            )
            pil_images_by_cam.append(pil_images)

        def _to_np(audio_item):
            if isinstance(audio_item, torch.Tensor):
                return audio_item.detach().cpu().numpy()
            if hasattr(audio_item, "numpy"):
                return audio_item.numpy()
            return np.asarray(audio_item)

        history_audio_flat = None
        history_len = None
        history_total_len = None
        if hist_audios is not None:
            if not isinstance(hist_audios, torch.Tensor):
                raise TypeError("hist_audios must be a torch.Tensor when provided.")
            if hist_audios.dim() == 2:
                if batch_size != 1:
                    raise ValueError("hist_audios must have shape [batch, history, ...].")
                hist_audios = hist_audios.unsqueeze(0)
            if hist_audios.shape[0] != batch_size:
                raise ValueError("hist_audios batch size must match prompt batch size.")
            history_len = hist_audios.shape[1]
            if history_len > 0:
                for i in range(batch_size):
                    if audios_list[i] is None:
                        audios_list[i] = hist_audios[i, -1]
                if all(audio is not None for audio in audios_list):
                    history_total_len = history_len + 1
                    history_audio_flat = []
                    for i in range(batch_size):
                        for j in range(history_len):
                            history_audio_flat.append(_to_np(hist_audios[i, j]))
                        history_audio_flat.append(_to_np(audios_list[i]))

        for i in range(batch_size):
            content = []
            for cam_idx, cam_masks in enumerate(img_masks_bool):
                if cam_masks[i]:
                    content.append({"type": "image", "image": pil_images_by_cam[cam_idx][i]})
            if not content:
                raise ValueError(f"No available camera images for batch index {i}.")

            # Keep historical audio out of the conversation to avoid placeholder mismatches.
            current_audio = audios_list[i]
            if current_audio is not None:
                audio_np = _to_np(current_audio)
                if debug:
                    audio_path = os.path.join("./debug_data", f"batch_{i:03d}.wav")
                    save_audio_debug(audio_np, audio_path)
                    waveform_path = os.path.join("./debug_data", f"batch_{i:03d}_waveform.png")
                    save_audio_waveform(audio_np, waveform_path)
                content.append({"type": "audio", "audio": audio_np})

            content.append({"type": "text", "text": prompts[i]})

            conv = [{"role": "user", "content": content}]

            if stage_labels is not None:
                conv.append({"role": "assistant", "content": stage_labels[i]})

            conversations_qwen3omni.append(conv)

        add_generation_prompt = stage_labels is None
        Qwen3Omnitext = self.Qwen3Omni_processor.apply_chat_template(
            conversations_qwen3omni,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        Qwen3Omni_audios, Qwen3Omni_images, Qwen3Omni_videos = process_mm_info(
            conversations_qwen3omni, use_audio_in_video=False
        )


        Qwen3Omni_inputs = self.Qwen3Omni_processor(
            text=Qwen3Omnitext,
            audio=Qwen3Omni_audios,
            images=Qwen3Omni_images,
            videos=Qwen3Omni_videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        if (
            history_audio_flat is not None
            and history_total_len is not None
            and "input_features" in Qwen3Omni_inputs
            and "feature_attention_mask" in Qwen3Omni_inputs
        ):
            target_seq_len = Qwen3Omni_inputs["input_features"].shape[-1]
            history_inputs = self.Qwen3Omni_processor.feature_extractor(
                history_audio_flat,
                sampling_rate=self.Qwen3Omni_processor.feature_extractor.sampling_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            history_input_features = history_inputs["input_features"].view(
                batch_size, history_total_len, *history_inputs["input_features"].shape[1:]
            )
            history_feature_attention_mask = history_inputs["attention_mask"].view(
                batch_size, history_total_len, history_inputs["attention_mask"].shape[-1]
            )

            def _pad_or_trim_last_dim(tensor, target_len):
                seq_len = tensor.shape[-1]
                if seq_len == target_len:
                    return tensor
                if seq_len > target_len:
                    return tensor[..., :target_len]
                pad_len = target_len - seq_len
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_len
                return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=-1)

            history_input_features = _pad_or_trim_last_dim(history_input_features, target_seq_len)
            history_feature_attention_mask = _pad_or_trim_last_dim(
                history_feature_attention_mask, target_seq_len
            )
            Qwen3Omni_inputs["input_features"] = history_input_features
            Qwen3Omni_inputs["history_feature_attention_mask"] = history_feature_attention_mask
        Qwen3Omni_inputs = Qwen3Omni_inputs.to(device)
        for key, value in Qwen3Omni_inputs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                Qwen3Omni_inputs[key] = value.to(torch.bfloat16)
        # xxx = Qwen3Omni_inputs["input_features"].cpu().to(torch.float32).numpy()
        # yyy = Qwen3Omni_inputs["feature_attention_mask"].cpu().to(torch.float32).numpy()
        # xxx1 = xxx[0]
        # xxx2 = xxx[1]
        # yyy1 = yyy[0]
        # yyy2 = yyy[1]


        if stage_labels is not None:
            tokenizer = self.Qwen3Omni_processor.tokenizer
            input_ids = Qwen3Omni_inputs["input_ids"]
            labels = torch.full_like(input_ids, fill_value=-100)
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            pad_token_id = tokenizer.pad_token_id
            newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
            newline_id = newline_ids[0] if newline_ids else None

            for i in range(batch_size):
                seq = input_ids[i]
                start_positions = (seq == im_start_id).nonzero(as_tuple=True)[0]
                if start_positions.numel() == 0:
                    continue
                last_start_idx = start_positions[-1].item()
                content_start = last_start_idx + 1
                if newline_id is not None:
                    cursor = content_start
                    while cursor < seq.size(0) and seq[cursor] != newline_id:
                        cursor += 1
                    if cursor < seq.size(0) and seq[cursor] == newline_id:
                        content_start = cursor + 1
                end_idx = None
                for pos in (seq == im_end_id).nonzero(as_tuple=True)[0].tolist():
                    if pos > last_start_idx:
                        end_idx = pos
                        break
                if end_idx is None or content_start >= end_idx:
                    continue
                labels[i, content_start:end_idx] = seq[content_start:end_idx]
            if pad_token_id is not None:
                labels[input_ids == pad_token_id] = -100
            Qwen3Omni_inputs["labels"] = labels

        return Qwen3Omni_inputs


    def embed_suffix(self, state, noisy_actions, timestep):

        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)


            embs.append(state_emb[:, None, :])

            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:

            time_emb = time_emb[:, None, :].expand_as(action_emb)

            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)


            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

            adarms_cond = None
        else:

            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)

            action_time_emb = action_emb

            adarms_cond = time_emb

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]

        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)

        pad_masks = torch.cat(pad_masks, dim=1)

        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None: noise = self.sample_noise(actions.shape, actions.device)
        if time is None: time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]

        x_t = time_expanded * noise + (1 - time_expanded) * actions

        u_t = noise - actions
        prompts = observation.full_prompt
        audios = observation.audio if hasattr(observation, 'audio') else None
        hist_audios = observation.hist_audio if hasattr(observation, 'hist_audio') else None
        next_audios = observation.next_audio if hasattr(observation, 'next_audio') else None
        stage_labels = self._extract_stage_labels(observation, actions.shape[0])
        Qwen3Omni_inputs = self.embed_prefix_qwen3omni(
            images, img_masks, prompts, audios, stage_labels=stage_labels, hist_audios=hist_audios
        )
        prefix_labels = Qwen3Omni_inputs.pop("labels", None)
        prefix_pad_masks = Qwen3Omni_inputs['attention_mask'].bool()  # [4, 75]
        prefix_att_masks = torch.zeros_like(prefix_pad_masks).bool()  # [4, 75]


        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        pixel_values = Qwen3Omni_inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            suffix_att_masks = suffix_att_masks.to(dtype=torch.bfloat16)
            if adarms_cond is not None:
                adarms_cond = adarms_cond.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks, dtype=suffix_embs.dtype)


        def forward_func(Qwen3Omni_inputs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, labels):
            (outputs, vlm_loss), _ = self.Qwen3_with_expert.forward(
                attention_mask={"full_attention": att_2d_masks_4d},
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[Qwen3Omni_inputs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                labels=labels,
                return_vlm_loss=True,
            )
            prefix_outputs, suffix_out = outputs
            if isinstance(prefix_outputs, (tuple, list)):
                prefix_hidden = prefix_outputs[-1]
            else:
                prefix_hidden = prefix_outputs
            return prefix_hidden, suffix_out, vlm_loss


        prefix_hidden, suffix_out, vlm_loss = self._apply_checkpoint(
            forward_func, Qwen3Omni_inputs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, prefix_labels
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]

        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)


        action_loss = F.mse_loss(u_t, v_t, reduction="mean")
        total_loss = action_loss
        if vlm_loss is not None:
            vlm_loss = vlm_loss.to(action_loss.device, dtype=action_loss.dtype)
            total_loss = total_loss + self.vlm_text_loss_weight * vlm_loss
        audio_loss = self._compute_audio_loss(prefix_hidden, prefix_pad_masks, next_audios)
        if audio_loss is not None:
            audio_loss = audio_loss.to(action_loss.device, dtype=action_loss.dtype)
            total_loss = total_loss + self.audio_output_weight * audio_loss
        return total_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prompts = observation.full_prompt
        audios = observation.audio if hasattr(observation, 'audio') else None
        hist_audios = observation.hist_audio if hasattr(observation, 'hist_audio') else None


        Qwen3Omni_inputs = self.embed_prefix_qwen3omni(
            images, img_masks, prompts, audios, hist_audios=hist_audios
        )

        prefix_pad_masks = Qwen3Omni_inputs["attention_mask"].bool().to(device)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        prefix_position_ids = prefix_position_ids.clamp_min(0)


        self.Qwen3_with_expert._ensure_gradient_checkpointing()
        device_type = prefix_pad_masks.device.type
        autocast_enabled = self.Qwen3_with_expert.precision == "bfloat16" and device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            prefix_outputs = self.Qwen3_with_expert.qwen3omni(
                **Qwen3Omni_inputs,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
            )
        prefix_past_key_values = prefix_outputs.past_key_values

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise

        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:

            expanded_time = time.expand(bsize)

            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=prefix_past_key_values,
                x_t=x_t,
                timestep=expanded_time,
            )

            x_t = x_t + dt * v_t

            time += dt

        return x_t

    @torch.no_grad()
    def predict_next_audio(self, device, observation) -> torch.Tensor:
        if not self.audio_output_enabled or self.audio_codec is None or self.audio_code_predictor is None:
            raise RuntimeError("Audio output is disabled for this model.")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prompts = observation.full_prompt
        audios = observation.audio if hasattr(observation, 'audio') else None
        Qwen3Omni_inputs = self.embed_prefix_qwen3omni(images, img_masks, prompts, audios)
        prefix_pad_masks = Qwen3Omni_inputs["attention_mask"].bool().to(device)

        device_type = prefix_pad_masks.device.type
        autocast_enabled = self.Qwen3_with_expert.precision == "bfloat16" and device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            prefix_outputs = self.Qwen3_with_expert.qwen3omni(
                **Qwen3Omni_inputs,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=True,
            )
        prefix_hidden = prefix_outputs.hidden_states[-1]
        codec_device = self._resolve_audio_codec_device(prefix_hidden.device)
        audio_tensor = self._prepare_audio_tensor(audios, codec_device)
        if audio_tensor is None:
            raise ValueError("observation.audio is required to infer target audio length.")
        code_len = self.audio_codec.code_length(audio_tensor.shape[-1])
        logits = self.audio_code_predictor(prefix_hidden, prefix_pad_masks, code_len)
        codes = logits.argmax(dim=-1).permute(0, 2, 1).contiguous()
        if codes.device != codec_device:
            codes = codes.to(codec_device)
        pred_audio = self.audio_codec.decode(codes, output_len=audio_tensor.shape[-1])
        return pred_audio.squeeze(1)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_seq_len=None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        device = suffix_embs.device
        prefix_pad_masks = prefix_pad_masks.to(device=device, dtype=torch.bool)
        suffix_pad_masks = suffix_pad_masks.to(device=device, dtype=torch.bool)
        suffix_att_masks = suffix_att_masks.to(device=device, dtype=torch.int32)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        if prefix_seq_len is None:
            prefix_seq_len = prefix_pad_masks.shape[1]
        prefix_pad_masks = prefix_pad_masks[:, :prefix_seq_len]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_seq_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        expert_dtype = self.Qwen3_with_expert.qwen3_expert.model.layers[0].self_attn.q_proj.weight.dtype
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks, dtype=expert_dtype)
        suffix_embs = suffix_embs.to(dtype=expert_dtype)
        if adarms_cond is not None:
            adarms_cond = adarms_cond.to(dtype=expert_dtype)

        prefix_offsets = prefix_pad_masks.long().sum(dim=-1, keepdim=True)
        suffix_positions = torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        position_ids = (prefix_offsets + suffix_positions).clamp_min(0)


        working_past_key_values = None
        if past_key_values is not None:
            working_past_key_values = copy.deepcopy(past_key_values)
        suffix_outputs = self.Qwen3_with_expert.qwen3_expert.model(
            inputs_embeds=suffix_embs,
            attention_mask={"full_attention": full_att_2d_masks_4d},
            position_ids=position_ids,
            past_key_values=None,
            external_past_key_values=working_past_key_values,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = suffix_outputs.last_hidden_state

        suffix_out = suffix_out[:, -self.config.action_horizon :]

        suffix_out = suffix_out.to(dtype=torch.float32)

        return self.action_out_proj(suffix_out)

    def _extract_stage_labels(self, observation, batch_size):
        if not hasattr(observation, "stage") or observation.stage is None:
            return None
        stage_data = observation.stage
        if isinstance(stage_data, torch.Tensor):
            stage_items = stage_data.detach().cpu().tolist()
        elif isinstance(stage_data, np.ndarray):
            stage_items = stage_data.tolist()
        elif isinstance(stage_data, (list, tuple)):
            stage_items = list(stage_data)
        else:
            stage_items = [stage_data]
        if len(stage_items) != batch_size:
            raise ValueError(
                f"observation.stage size mismatch: expected {batch_size}, got {len(stage_items)}"
            )
        stage_labels = []
        for stage_value in stage_items:
            if stage_value is None:
                stage_text = "unknown"
            elif isinstance(stage_value, str):
                stage_text = stage_value.strip() or "unknown"
            else:
                stage_text = str(stage_value)
            stage_labels.append(json.dumps({"stage": stage_text}))
        return stage_labels


def batch_tensor_to_pil_images(images_tensor, save_dir=None, prefix="image"):

    if images_tensor.device != torch.device('cpu'):
        images_tensor = images_tensor.cpu()
    images_tensor = images_tensor.float()

    images_tensor = (images_tensor + 1.0) / 2.0

    images_tensor = torch.clamp(images_tensor, 0.0, 1.0)

    images_tensor = (images_tensor * 255).byte()

    images_np = images_tensor.permute(0, 2, 3, 1).numpy()

    pil_images = []
    for i, img_np in enumerate(images_np):
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            pil_img = Image.fromarray(img_np, mode='L')
        else:
            pil_img = Image.fromarray(img_np, mode='RGB')

        pil_images.append(pil_img)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
            pil_img.save(save_path)

    if save_dir is not None:
        logging.info(f"Saved {len(pil_images)} images to {save_dir}")

    return pil_images
