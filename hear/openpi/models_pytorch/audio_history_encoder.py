from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class AudioHistoryTemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_size * 4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.gradient_checkpointing = bool(gradient_checkpointing)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _encode(self, seq: torch.Tensor, attn_mask: torch.Tensor | None, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        out = self.encoder(seq, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return self.final_norm(out)

    def _maybe_checkpoint(
        self,
        seq: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            def _ckpt(seq_in: torch.Tensor) -> torch.Tensor:
                return self._encode(seq_in, attn_mask, key_padding_mask)
            return checkpoint(_ckpt, seq, use_reentrant=False)
        return self._encode(seq, attn_mask, key_padding_mask)

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_features.ndim != 4:
            raise ValueError("history_features must be [batch, history, seq_len, hidden].")
        batch_size, history_len, seq_len, hidden_size = history_features.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"history_features last dim {hidden_size} != hidden_size {self.hidden_size}."
            )

        seq = history_features.permute(0, 2, 1, 3).reshape(batch_size * seq_len, history_len, hidden_size)

        key_padding_mask = None
        token_valid = None
        if history_mask is not None:
            if history_mask.shape != (batch_size, history_len, seq_len):
                raise ValueError("history_mask must match [batch, history, seq_len].")
            token_valid = history_mask.any(dim=1)
            key_padding_mask = ~history_mask.permute(0, 2, 1).reshape(batch_size * seq_len, history_len)

        attn_mask = None
        if history_len > 1:
            attn_mask = torch.triu(
                torch.ones(history_len, history_len, device=seq.device, dtype=torch.bool),
                diagonal=1,
            )

        if token_valid is not None:
            valid_index = token_valid.reshape(-1)
            if valid_index.any():
                seq_valid = seq[valid_index]
                key_padding_mask_valid = key_padding_mask[valid_index] if key_padding_mask is not None else None
                out_valid = self._maybe_checkpoint(seq_valid, attn_mask, key_padding_mask_valid)
                out_full = seq.new_zeros((seq.shape[0], hidden_size))
                out_full[valid_index] = out_valid[:, -1, :].to(out_full.dtype)
                out = out_full.reshape(batch_size, seq_len, hidden_size)
            else:
                out = seq.new_zeros((batch_size, seq_len, hidden_size))
            out = out * token_valid.unsqueeze(-1).to(out.dtype)
            return out

        out = self._maybe_checkpoint(seq, attn_mask, key_padding_mask)
        return out[:, -1, :].reshape(batch_size, seq_len, hidden_size)
