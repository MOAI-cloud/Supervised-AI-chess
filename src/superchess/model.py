from __future__ import annotations

from dataclasses import dataclass
import math

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from superchess.encoding import (
    LEGACY_BOARD_CHANNELS,
    PIECE_PLANES,
    POLICY_PLANES,
    POLICY_SIZE,
    piece_plane_mask,
    policy_plane_geometry,
)
from superchess.targets import value_from_logits

NUM_SQUARES = 64
MASKED_LOGIT = -1e4
"""Finite stand-in for ``-inf`` so masked logits stay NaN-free through softmax and mixed precision."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyper-parameters.

    ``input_channels`` defaults to the 18 clock-free planes because the Lichess
    evaluation dump carries no move counters; use ``20`` only for game data whose
    positions have real clocks (see :func:`superchess.training.check_input_planes`).
    """

    input_channels: int = LEGACY_BOARD_CHANNELS
    channels: int = 256
    cnn_blocks: int = 2
    transformer_layers: int = 10
    attention_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    attention_bias: bool = True
    smolgen_hidden: int = 32
    smolgen_gen: int = 256
    policy_head: str = "attention"
    policy_dim: int = 0
    value_head: str = "categorical"
    value_bins: int = 64
    value_token_dim: int = 32
    value_hidden: int = 128
    structural_mask: bool = True

    def __post_init__(self) -> None:
        if self.channels % self.attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        if self.policy_head not in ("attention", "planes"):
            raise ValueError("policy_head must be 'attention' or 'planes'")
        if self.value_head not in ("categorical", "wdl"):
            raise ValueError("value_head must be 'categorical' or 'wdl'")
        if self.value_head == "categorical" and self.value_bins < 2:
            raise ValueError("value_bins must be at least 2")

    @property
    def effective_policy_dim(self) -> int:
        return self.policy_dim if self.policy_dim > 0 else self.channels

    @property
    def uses_smolgen(self) -> bool:
        return self.smolgen_hidden > 0 and self.smolgen_gen > 0


LEGACY_MODEL_FIELDS = {
    "policy_head": "planes",
    "value_head": "wdl",
    "smolgen_hidden": 0,
    "structural_mask": False,
    "cnn_blocks": 6,
}
"""Defaults that reproduce checkpoints saved before the attention-policy/categorical-value heads."""


def model_config_from_checkpoint(payload: dict) -> ModelConfig:
    """Build a :class:`ModelConfig` from checkpoint metadata, filling legacy defaults."""
    fields = dict(payload)
    if "policy_head" not in fields:
        for name, value in LEGACY_MODEL_FIELDS.items():
            fields.setdefault(name, value)
    return ModelConfig(**fields)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(gate) * value))


class Smolgen(nn.Module):
    """Position-dependent attention logits (Lc0 "smolgen").

    Each token is compressed, the board is flattened, and two dense layers
    produce a per-head latent that a *shared* generator (owned by the model and
    passed in at call time) expands to a ``[heads, 64, 64]`` additive bias. This
    lets attention depend on global board context rather than only on pairwise
    query/key agreement and a static square prior.
    """

    def __init__(self, dim: int, heads: int, hidden: int, gen: int) -> None:
        super().__init__()
        self.heads = heads
        self.gen = gen
        self.compress = nn.Linear(dim, hidden, bias=False)
        self.dense1 = nn.Linear(NUM_SQUARES * hidden, gen)
        self.norm1 = nn.LayerNorm(gen)
        self.dense2 = nn.Linear(gen, gen * heads)
        self.norm2 = nn.LayerNorm(gen * heads)

    def forward(self, x: torch.Tensor, shared: nn.Linear) -> torch.Tensor:
        batch = x.shape[0]
        hidden = self.compress(x).reshape(batch, -1)
        hidden = self.norm1(F.silu(self.dense1(hidden)))
        hidden = self.norm2(F.silu(self.dense2(hidden)))
        return shared(hidden.reshape(batch, self.heads, self.gen)).reshape(batch, self.heads, NUM_SQUARES, NUM_SQUARES)


class SquareAttention(nn.Module):
    """Multi-head self-attention over the 64 squares with static and dynamic biases."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim, heads = config.channels, config.attention_heads
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = config.dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.square_bias = (
            nn.Parameter(torch.zeros(heads, NUM_SQUARES, NUM_SQUARES)) if config.attention_bias else None
        )
        self.smolgen = (
            Smolgen(dim, heads, config.smolgen_hidden, config.smolgen_gen) if config.uses_smolgen else None
        )

    def forward(self, x: torch.Tensor, smolgen_shared: nn.Linear | None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        bias: torch.Tensor | None = None
        if tokens == NUM_SQUARES:
            if self.square_bias is not None:
                bias = self.square_bias.unsqueeze(0)
            if self.smolgen is not None and smolgen_shared is not None:
                dynamic = self.smolgen(x, smolgen_shared)
                bias = dynamic if bias is None else bias + dynamic
        if bias is not None:
            bias = bias.to(query.dtype)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.channels)
        self.attention = SquareAttention(config)
        self.norm2 = RMSNorm(config.channels)
        self.mlp = SwiGLU(config.channels, config.channels * config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor, smolgen_shared: nn.Linear | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), smolgen_shared)
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


def _square_to_token_index() -> torch.Tensor:
    """Token index (row-major, rank 8 first) of each python-chess square."""
    squares = torch.arange(NUM_SQUARES)
    return (7 - squares // 8) * 8 + squares % 8


class AttentionPolicyHead(nn.Module):
    """Lc0-style policy head: one logit per (from, to) square pair plus promotion offsets.

    The bilinear ``query_from · key_to`` form scores complete moves instead of
    predicting 73 independent planes per square, and the result is scattered
    into the fixed ``64 x 73`` layout used by the datasets and the search.
    """

    def __init__(self, dim: int, policy_dim: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(policy_dim)
        self.query = nn.Linear(dim, policy_dim)
        self.key = nn.Linear(dim, policy_dim)
        self.promotion = nn.Linear(policy_dim, 3, bias=False)
        to_square, promotion = policy_plane_geometry()
        from_square = np.repeat(np.arange(NUM_SQUARES), POLICY_PLANES)
        flat_to = to_square.reshape(-1)
        valid = flat_to >= 0
        pair_index = from_square * NUM_SQUARES + np.where(valid, flat_to, 0)
        promo_kind = np.tile(promotion, NUM_SQUARES)
        promo_file = np.where(valid, flat_to - 56, 0).clip(0, 7)
        promo_slot = promo_file * 3 + np.clip(promo_kind - chess.KNIGHT, 0, 2)
        self.register_buffer("pair_index", torch.from_numpy(pair_index), persistent=False)
        self.register_buffer("valid", torch.from_numpy(valid), persistent=False)
        self.register_buffer("promo_mask", torch.from_numpy(promo_kind > 0), persistent=False)
        self.register_buffer("promo_index", torch.from_numpy(promo_slot), persistent=False)
        nn.init.zeros_(self.promotion.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        query = self.query(tokens)
        key = self.key(tokens)
        pair_logits = torch.matmul(query, key.transpose(1, 2)) * self.scale
        logits = pair_logits.reshape(batch, NUM_SQUARES * NUM_SQUARES).gather(1, self.pair_index.expand(batch, -1))
        promo = self.promotion(key[:, 56:64]).reshape(batch, -1)
        promo_logits = promo.gather(1, self.promo_index.expand(batch, -1)) * self.promo_mask
        logits = (logits + promo_logits).float()
        return logits.masked_fill(~self.valid, MASKED_LOGIT)


class CategoricalValueHead(nn.Module):
    """Per-square projection, flatten, MLP, ``bins`` logits over the expected score."""

    def __init__(self, dim: int, token_dim: int, hidden: int, bins: int) -> None:
        super().__init__()
        self.token = nn.Linear(dim, token_dim)
        self.dense = nn.Linear(NUM_SQUARES * token_dim, hidden)
        self.out = nn.Linear(hidden, bins)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.token(tokens)).flatten(1)
        return self.out(F.gelu(self.dense(hidden)))


class ChessCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer trunk with policy and value heads.

    ``forward`` returns ``policy`` (``[N, 4672]`` float32 logits in the dataset
    layout, structurally impossible moves set to :data:`MASKED_LOGIT`),
    ``value_logits`` (categorical bins, or WDL logits for legacy checkpoints) and
    ``value`` (scalar in ``[-1, 1]`` from the side to move).
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.GELU(),
        )
        self.cnn = nn.Sequential(*(ResidualConvBlock(config.channels) for _ in range(config.cnn_blocks)))
        self.square_embedding = nn.Parameter(torch.zeros(1, NUM_SQUARES, config.channels))
        self.transformer = nn.ModuleList(EncoderBlock(config) for _ in range(config.transformer_layers))
        self.smolgen_shared = (
            nn.Linear(config.smolgen_gen, NUM_SQUARES * NUM_SQUARES, bias=False) if config.uses_smolgen else None
        )
        self.norm = RMSNorm(config.channels)
        if config.policy_head == "attention":
            self.policy_head = AttentionPolicyHead(config.channels, config.effective_policy_dim)
        else:
            self.policy_head = nn.Linear(config.channels, POLICY_PLANES)
        if config.value_head == "categorical":
            self.value_head = CategoricalValueHead(
                config.channels, config.value_token_dim, config.value_hidden, config.value_bins
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(config.channels, config.channels),
                nn.GELU(),
                nn.Linear(config.channels, 3),
            )
        self.register_buffer("square_to_token", _square_to_token_index(), persistent=False)
        self.register_buffer("piece_plane_mask", torch.from_numpy(piece_plane_mask()), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.square_embedding, std=0.02)
        if isinstance(self.policy_head, nn.Linear):
            nn.init.zeros_(self.policy_head.bias)
        # Start every residual branch near identity so deep stacks train stably:
        # zero the closing BatchNorm gamma of each conv block (He et al., "Bag of
        # Tricks") and depth-scale the transformer residual projections (GPT-2).
        for block in self.cnn:
            nn.init.zeros_(block.net[-1].weight)
        residual_std = 0.02 / math.sqrt(2 * max(1, self.config.transformer_layers))
        for layer in self.transformer:
            nn.init.normal_(layer.attention.proj.weight, std=residual_std)
            nn.init.normal_(layer.mlp.out_proj.weight, std=residual_std)
        if self.smolgen_shared is not None:
            nn.init.zeros_(self.smolgen_shared.weight)

    def structural_mask(self, boards: torch.Tensor) -> torch.Tensor:
        """Boolean ``[N, POLICY_SIZE]`` mask of geometrically possible moves for ``boards``."""
        own = boards[:, : len(PIECE_PLANES)].flip(2).flatten(2) > 0.5
        allowed = (own.unsqueeze(-1) & self.piece_plane_mask.unsqueeze(0)).any(dim=1)
        return allowed.reshape(boards.shape[0], POLICY_SIZE)

    def forward(self, boards: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn(self.stem(boards))
        tokens = features.flatten(2).transpose(1, 2) + self.square_embedding
        for layer in self.transformer:
            tokens = layer(tokens, self.smolgen_shared)
        tokens = self.norm(tokens)
        batch = boards.shape[0]

        if isinstance(self.policy_head, AttentionPolicyHead):
            policy = self.policy_head(tokens[:, self.square_to_token])
        else:
            policy = (
                self.policy_head(tokens)
                .reshape(batch, 8, 8, POLICY_PLANES)
                .flip(1)
                .reshape(batch, POLICY_SIZE)
                .float()
            )
        if self.config.structural_mask:
            policy = policy.masked_fill(~self.structural_mask(boards), MASKED_LOGIT)

        if isinstance(self.value_head, CategoricalValueHead):
            value_logits = self.value_head(tokens[:, self.square_to_token])
            value = value_from_logits(value_logits)
        else:
            value_logits = self.value_head(tokens.mean(dim=1))
            wdl = value_logits.float().softmax(dim=-1)
            value = wdl[..., 0] - wdl[..., 2]
        return {"policy": policy, "value_logits": value_logits, "value": value}
