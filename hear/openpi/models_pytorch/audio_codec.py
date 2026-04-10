from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


@dataclass
class AudioCodecConfig:
    codebook_size: int = 1024
    codebook_dim: int = 256
    num_quantizers: int = 4
    encoder_channels: tuple[int, ...] = (128, 256, 256)
    decoder_channels: tuple[int, ...] = (256, 256, 128)
    strides: tuple[int, ...] = (4, 4, 4)
    kernel_size: Optional[int] = None
    commitment_cost: float = 0.25
    max_code_len: int = 1024
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AudioCodecConfig":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**dict(data))

    def validate(self) -> None:
        if len(self.encoder_channels) != len(self.strides):
            raise ValueError("encoder_channels must match strides length.")
        if len(self.decoder_channels) != len(self.strides):
            raise ValueError("decoder_channels must match strides length.")
        if self.num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1.")
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2.")
        if self.codebook_dim < 1:
            raise ValueError("codebook_dim must be >= 1.")
        if self.max_code_len < 1:
            raise ValueError("max_code_len must be >= 1.")


class MimiAudioCodec(nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        num_quantizers: Optional[int] = None,
        input_sample_rate: Optional[int] = None,
        local_files_only: bool = True,
    ):
        super().__init__()
        from transformers import MimiModel

        self.model = MimiModel.from_pretrained(model_path, local_files_only=local_files_only)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.codebook_size = int(self.model.config.codebook_size)
        self.num_quantizers = int(num_quantizers or self.model.config.num_quantizers)
        self.max_code_len = int(self.model.config.max_position_embeddings)
        self.codec_sample_rate = int(self.model.config.sampling_rate)
        self.input_sample_rate = int(input_sample_rate or self.codec_sample_rate)

    def _ensure_device(self, device: torch.device) -> None:
        if next(self.model.parameters()).device != device:
            self.model.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _prepare_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1 and audio.shape[2] == 1:
            audio = audio.transpose(1, 2)
        if audio.dim() != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, 1, time] or [batch, time].")
        return audio

    def _resample(self, audio: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        if from_sr == to_sr:
            return audio
        target_len = max(1, int(round(audio.shape[-1] * float(to_sr) / float(from_sr))))
        return F.interpolate(audio, size=target_len, mode="linear", align_corners=False)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.LongTensor:
        audio = self._prepare_audio(audio).to(dtype=torch.float32)
        audio = self._resample(audio, self.input_sample_rate, self.codec_sample_rate)
        self._ensure_device(audio.device)
        outputs = self.model.encode(audio, num_quantizers=self.num_quantizers, return_dict=True)
        codes = getattr(outputs, "audio_codes", outputs[0])
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.LongTensor, output_len: Optional[int] = None) -> torch.Tensor:
        self._ensure_device(codes.device)
        outputs = self.model.decode(codes, return_dict=True)
        audio = getattr(outputs, "audio_values", outputs[0])
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        audio = self._resample(audio, self.codec_sample_rate, self.input_sample_rate)
        if output_len is not None:
            audio = audio[..., :output_len]
        return audio

    def code_length(self, audio_len: int) -> int:
        resampled_len = int(round(audio_len * float(self.codec_sample_rate) / float(self.input_sample_rate)))
        length = torch.tensor([resampled_len], dtype=torch.long, device=next(self.model.parameters()).device)
        return int(self.model.get_encoded_length(length).item())


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        # inputs: [B, D, T]
        bsz, dim, steps = inputs.shape
        flat = inputs.permute(0, 2, 1).contiguous().view(-1, dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)[None, :]
            - 2 * flat @ codebook.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view(bsz, steps, dim).permute(0, 2, 1).contiguous()
        vq_loss = F.mse_loss(quantized.detach(), inputs) + self.commitment_cost * F.mse_loss(
            quantized, inputs.detach()
        )
        quantized = inputs + (quantized - inputs).detach()
        return quantized, indices.view(bsz, steps), vq_loss

    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        # indices: [B, T]
        bsz, steps = indices.shape
        emb = self.codebook(indices.reshape(-1)).reshape(bsz, steps, -1)
        return emb.permute(0, 2, 1).contiguous()


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, codebook_dim, commitment_cost=commitment_cost)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        residual = inputs
        all_indices = []
        total_vq_loss = inputs.new_zeros(())
        quantized_sum = inputs.new_zeros(inputs.shape)
        for quantizer in self.quantizers:
            quantized, indices, vq_loss = quantizer(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            all_indices.append(indices)
            total_vq_loss = total_vq_loss + vq_loss
        codes = torch.stack(all_indices, dim=1)
        return quantized_sum, codes, total_vq_loss

    def decode(self, codes: torch.LongTensor) -> torch.Tensor:
        # codes: [B, Q, T]
        if codes.dim() != 3:
            raise ValueError("codes must have shape [batch, num_quantizers, steps].")
        quantized = None
        for idx, quantizer in enumerate(self.quantizers):
            emb = quantizer.decode(codes[:, idx, :])
            quantized = emb if quantized is None else quantized + emb
        return quantized


class SimpleAudioEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Iterable[int], strides: Iterable[int], kernel_size: Optional[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for ch, stride in zip(channels, strides):
            k = kernel_size or stride * 2
            pad = stride // 2
            layers.append(nn.Conv1d(prev, ch, kernel_size=k, stride=stride, padding=pad))
            layers.append(_group_norm(ch))
            layers.append(nn.GELU())
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SimpleAudioDecoder(nn.Module):
    def __init__(self, channels: Iterable[int], strides: Iterable[int], kernel_size: Optional[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        channels_list = list(channels)
        strides_list = list(strides)
        prev = channels_list[0]
        for ch, stride in zip(channels_list[1:], strides_list):
            k = kernel_size or stride * 2
            pad = stride // 2
            layers.append(nn.ConvTranspose1d(prev, ch, kernel_size=k, stride=stride, padding=pad))
            layers.append(_group_norm(ch))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass
class AudioCodecOutput:
    recon_audio: torch.Tensor
    codes: torch.LongTensor
    vq_loss: torch.Tensor
    pad_len: int
    orig_len: int


class SimpleAudioCodec(nn.Module):
    def __init__(self, config: AudioCodecConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.total_stride = 1
        for stride in config.strides:
            self.total_stride *= stride
        self.encoder = SimpleAudioEncoder(
            in_channels=1,
            channels=config.encoder_channels,
            strides=config.strides,
            kernel_size=config.kernel_size,
        )
        self.to_codebook = nn.Conv1d(self.encoder.out_channels, config.codebook_dim, kernel_size=1)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            commitment_cost=config.commitment_cost,
        )
        decoder_channels = (config.codebook_dim,) + tuple(config.decoder_channels)
        self.decoder = SimpleAudioDecoder(
            channels=decoder_channels,
            strides=config.strides,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.out_proj = nn.Conv1d(decoder_channels[-1], 1, kernel_size=3, padding=1)

    @property
    def codebook_size(self) -> int:
        return self.config.codebook_size

    @property
    def num_quantizers(self) -> int:
        return self.config.num_quantizers

    def _prepare_audio(self, audio: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1 and audio.shape[2] == 1:
            audio = audio.transpose(1, 2)
        if audio.dim() != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, 1, time] or [batch, time].")
        orig_len = audio.shape[-1]
        pad_len = (self.total_stride - (orig_len % self.total_stride)) % self.total_stride
        if pad_len:
            audio = F.pad(audio, (0, pad_len), mode="constant", value=0.0)
        return audio, orig_len, pad_len

    def encode(
        self,
        audio: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.LongTensor | tuple[torch.LongTensor, torch.Tensor, int, int]:
        audio, orig_len, pad_len = self._prepare_audio(audio)
        audio = audio.to(dtype=torch.float32)
        hidden = self.encoder(audio)
        hidden = self.to_codebook(hidden)
        quantized, codes, vq_loss = self.quantizer(hidden)
        if return_details:
            return codes, vq_loss, orig_len, pad_len
        return codes

    def decode(self, codes: torch.LongTensor, output_len: Optional[int] = None) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        recon = self.decoder(quantized)
        recon = torch.tanh(self.out_proj(recon))
        if output_len is not None:
            recon = recon[..., :output_len]
        return recon

    def forward(self, audio: torch.Tensor) -> AudioCodecOutput:
        codes, vq_loss, orig_len, pad_len = self.encode(audio, return_details=True)
        recon = self.decode(codes, output_len=orig_len)
        return AudioCodecOutput(
            recon_audio=recon,
            codes=codes,
            vq_loss=vq_loss,
            pad_len=pad_len,
            orig_len=orig_len,
        )

    def code_length(self, audio_len: int) -> int:
        padded_len = audio_len + (self.total_stride - (audio_len % self.total_stride)) % self.total_stride
        return padded_len // self.total_stride


class AudioCodePredictor(nn.Module):
    def __init__(
        self,
        context_dim: int,
        codebook_size: int,
        num_quantizers: int,
        max_code_len: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.max_code_len = max_code_len
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(max_code_len, hidden_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, codebook_size * num_quantizers)

    def _masked_mean(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)
        mask = mask.to(dtype=hidden_states.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        context_hidden: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        code_len: int,
    ) -> torch.Tensor:
        if code_len > self.max_code_len:
            raise ValueError(f"code_len {code_len} exceeds max_code_len {self.max_code_len}.")
        pooled = self._masked_mean(context_hidden, context_mask)
        pooled = pooled.to(dtype=self.context_proj.weight.dtype)
        hidden = self.context_proj(pooled)[:, None, :]
        hidden = hidden + self.pos_embed[:code_len].unsqueeze(0)
        hidden = self.layers(hidden)
        logits = self.head(hidden)
        bsz = logits.shape[0]
        logits = logits.view(bsz, code_len, self.num_quantizers, self.codebook_size)
        return logits
