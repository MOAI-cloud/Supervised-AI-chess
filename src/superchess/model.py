from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from superchess.encoding import BOARD_CHANNELS, POLICY_PLANES, POLICY_SIZE

NUM_SQUARES = 64

@dataclass(frozen=True, slots=True)
class ModelConfig:
    input_channels: int = BOARD_CHANNELS
    channels: int = 256
    cnn_blocks: int = 6
    transformer_layers: int = 10
    attention_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    attention_bias: bool = True


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


class SquareAttention(nn.Module):
    """Multi-head self-attention over the 64 board squares.

    Adds an optional learned per-head ``[64, 64]`` pairwise bias so the model can
    encode board geometry (a static relative-position prior, as in T5/Swin).
    """

    def __init__(self, dim: int, heads: int, dropout: float, attention_bias: bool) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.square_bias = (
            nn.Parameter(torch.zeros(heads, NUM_SQUARES, NUM_SQUARES)) if attention_bias else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        bias = None
        if self.square_bias is not None and tokens == NUM_SQUARES:
            bias = self.square_bias.to(query.dtype).unsqueeze(0)
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
        self.attention = SquareAttention(
            config.channels, config.attention_heads, config.dropout, config.attention_bias
        )
        self.norm2 = RMSNorm(config.channels)
        self.mlp = SwiGLU(config.channels, config.channels * config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
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


class ChessCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer with policy and win/draw/loss value heads.

    The value head predicts a 3-way WDL distribution (the modern standard, and a
    natural target when distilling Stockfish evaluations). A scalar
    ``value = P(win) - P(loss)`` in ``[-1, 1]`` is also exposed so neural MCTS and
    the GUI keep their existing scalar interface.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        if config.channels % config.attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.GELU(),
        )
        self.cnn = nn.Sequential(*(ResidualConvBlock(config.channels) for _ in range(config.cnn_blocks)))
        self.square_embedding = nn.Parameter(torch.zeros(1, NUM_SQUARES, config.channels))
        self.transformer = nn.Sequential(*(EncoderBlock(config) for _ in range(config.transformer_layers)))
        self.norm = RMSNorm(config.channels)
        self.policy_head = nn.Linear(config.channels, POLICY_PLANES)
        self.value_head = nn.Sequential(
            nn.Linear(config.channels, config.channels),
            nn.GELU(),
            nn.Linear(config.channels, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.square_embedding, std=0.02)
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

    def forward(self, boards: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.cnn(self.stem(boards))
        tokens = features.flatten(2).transpose(1, 2) + self.square_embedding
        tokens = self.norm(self.transformer(tokens))
        policy = (
            self.policy_head(tokens)
            .reshape(boards.shape[0], 8, 8, POLICY_PLANES)
            .flip(1)
            .reshape(boards.shape[0], POLICY_SIZE)
        )
        wdl_logits = self.value_head(tokens.mean(dim=1))
        wdl = wdl_logits.softmax(dim=-1)
        value = wdl[..., 0] - wdl[..., 2]
        return {"policy": policy, "wdl": wdl_logits, "value": value}
