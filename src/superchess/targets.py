"""Training-target shaping shared by preprocessing, training, search, and the GUI.

Scores
------
Every Stockfish evaluation is stored as one signed integer *score* from the
side-to-move perspective: plain centipawns clamped to ``+-(MATE_SCORE - 1)``,
or ``+-(MATE_SCORE + MATE_HORIZON - n)`` for a forced mate in ``n`` moves so
that shorter mates order above longer ones and every mate orders above every
non-mate evaluation.

Expected score
--------------
Scores map to an expected game score in ``[0, 1]`` through a logistic with a
centipawn scale. The default is the Lichess win-probability model
(``1 / (1 + exp(-0.00368208 * cp))``), which is also what the DeepMind
"Grandmaster-level chess without search" work used for Stockfish targets.

Value targets
-------------
The value head is a categorical distribution over ``K`` bins of the expected
score, trained with histogram-loss (HL-Gauss) targets as recommended by
Farebrother et al. ("Stop Regressing", 2024): a Gaussian centred on the target
score is integrated over each bin. The scalar value used by MCTS is the
distribution mean rescaled to ``[-1, 1]``.

Policy targets
--------------
Multi-PV evaluations give action values for the top moves. The policy target is
a softmax over the expected score of each candidate move divided by
``policy_temperature`` (plus a tiny raw-centipawn tie-break so that faster
mates and larger material edges still order when the logistic saturates).
Moves absent from the multi-PV list receive zero probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np
import torch

MATE_SCORE = 12_000
MATE_HORIZON = 500
SCORE_CLAMP = MATE_SCORE - 1
MAX_SCORE = MATE_SCORE + MATE_HORIZON
LICHESS_WIN_SLOPE = 0.00368208
DEFAULT_CP_SCALE = 1.0 / LICHESS_WIN_SLOPE


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Hyper-parameters that turn raw scores into network targets."""

    cp_scale: float = DEFAULT_CP_SCALE
    policy_temperature: float = 0.04
    policy_cp_tiebreak: float = 2000.0
    hl_gauss_sigma: float = 0.75

    def __post_init__(self) -> None:
        if self.cp_scale <= 0:
            raise ValueError("cp_scale must be positive")
        if self.policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive")
        if self.policy_cp_tiebreak < 0:
            raise ValueError("policy_cp_tiebreak must be non-negative")
        if self.hl_gauss_sigma <= 0:
            raise ValueError("hl_gauss_sigma must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TargetConfig":
        if not payload:
            return cls()
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in names})


def mate_score(mate_in: int) -> int:
    """Signed score for a forced mate in ``mate_in`` moves (sign gives the winner)."""
    distance = min(abs(int(mate_in)), MATE_HORIZON - 1)
    magnitude = MATE_SCORE + (MATE_HORIZON - distance)
    return magnitude if mate_in >= 0 else -magnitude


def encode_score(cp: int | float | None, mate: int | None) -> int:
    """Encode a (cp | mate) evaluation into the integer score domain."""
    if mate is not None:
        return mate_score(int(mate))
    if cp is None:
        raise ValueError("evaluation has neither cp nor mate")
    return int(max(-SCORE_CLAMP, min(SCORE_CLAMP, round(float(cp)))))


def is_mate_score(score: int | float) -> bool:
    return abs(float(score)) >= MATE_SCORE


def score_to_expected(score: float, cp_scale: float = DEFAULT_CP_SCALE) -> float:
    """Expected game score in ``[0, 1]`` for one side-to-move score."""
    x = float(score) / cp_scale
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def expected_to_score(expected: float, cp_scale: float = DEFAULT_CP_SCALE) -> float:
    """Inverse of :func:`score_to_expected`, clamped to the finite score range."""
    p = min(1.0 - 1e-9, max(1e-9, float(expected)))
    return max(-SCORE_CLAMP, min(SCORE_CLAMP, cp_scale * math.log(p / (1.0 - p))))


def value_to_cp(value: float, cp_scale: float = DEFAULT_CP_SCALE) -> int:
    """Map a network value in ``[-1, 1]`` (``2 * expected - 1``) to integer centipawns."""
    return int(round(expected_to_score((float(value) + 1.0) / 2.0, cp_scale)))


def scores_to_expected(scores: torch.Tensor, cp_scale: float) -> torch.Tensor:
    return torch.sigmoid(scores.to(torch.float32) / cp_scale)


def value_bin_centers(bins: int, device: torch.device | None = None) -> torch.Tensor:
    return (torch.arange(bins, device=device, dtype=torch.float32) + 0.5) / bins


def hl_gauss_targets(expected: torch.Tensor, bins: int, sigma_ratio: float) -> torch.Tensor:
    """HL-Gauss categorical targets over ``bins`` uniform bins of ``[0, 1]``.

    ``expected`` has shape ``[N]``; the result has shape ``[N, bins]`` and rows
    sum to one (the Gaussian is truncated to ``[0, 1]`` and renormalised).
    """
    expected = expected.to(torch.float32).clamp(0.0, 1.0).unsqueeze(1)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=expected.device, dtype=torch.float32)
    sigma = sigma_ratio / bins
    cdf = torch.special.ndtr((edges.unsqueeze(0) - expected) / sigma)
    probs = cdf[:, 1:] - cdf[:, :-1]
    total = (cdf[:, -1:] - cdf[:, :1]).clamp_min(1e-12)
    return probs / total


def value_targets(scores: torch.Tensor, bins: int, config: TargetConfig) -> torch.Tensor:
    """Categorical value targets for side-to-move ``scores`` of shape ``[N]``."""
    return hl_gauss_targets(scores_to_expected(scores, config.cp_scale), bins, config.hl_gauss_sigma)


def outcome_value_targets(results: torch.Tensor, bins: int, config: TargetConfig) -> torch.Tensor:
    """Categorical targets for game results in ``{-1, 0, 1}`` (side-to-move perspective)."""
    expected = (results.to(torch.float32).clamp(-1.0, 1.0) + 1.0) / 2.0
    return hl_gauss_targets(expected, bins, config.hl_gauss_sigma)


def value_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Scalar value in ``[-1, 1]`` from categorical value logits ``[N, bins]``."""
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    centers = value_bin_centers(logits.shape[-1], device=logits.device)
    return 2.0 * (probs * centers).sum(dim=-1) - 1.0


def policy_targets(
    policy_scores: torch.Tensor,
    valid: torch.Tensor,
    config: TargetConfig,
) -> torch.Tensor:
    """Soft policy targets ``[N, K]`` from per-move scores and a validity mask.

    Rows with no valid move return all zeros.
    """
    scores = policy_scores.to(torch.float32)
    logits = scores_to_expected(scores, config.cp_scale) / config.policy_temperature
    if config.policy_cp_tiebreak > 0:
        logits = logits + scores / config.policy_cp_tiebreak
    logits = logits.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(logits, dim=1)
    return torch.nan_to_num(probs, nan=0.0)


def best_target_moves(policy_indices: torch.Tensor, policy_scores: torch.Tensor) -> torch.Tensor:
    """Index of the highest-scored valid move per row (first one on ties)."""
    valid = policy_indices >= 0
    scores = policy_scores.to(torch.float32).masked_fill(~valid, float("-inf"))
    best = scores.argmax(dim=1, keepdim=True)
    return policy_indices.gather(1, best).squeeze(1)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Row-wise Shannon entropy in nats; zeros contribute nothing."""
    probs = probs.to(torch.float32)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


# ---------------------------------------------------------------------------
# Legacy shard support: reconstruct raw scores from the old WDL / soft-policy
# arrays so existing datasets keep working without re-preprocessing.
# ---------------------------------------------------------------------------

LEGACY_DEFAULTS = {
    "value_scale": 400.0,
    "wdl_scale": 380.0,
    "wdl_draw_margin": 100.0,
    "policy_temperature": 1.0,
}


def reconstruct_legacy_scores(
    wdl: np.ndarray,
    policy_indices: np.ndarray,
    policy_probs: np.ndarray,
    legacy_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct approximate side-to-move scores from legacy transforms.

    Unsaturated WDL values recover the value score up to floating-point error.
    Policy log-ratios recover score differences, but their absolute anchor is
    unknown when value and policy came from different-depth snapshots. The
    value score supplies that approximate anchor. Saturation loses mate detail.
    Returns ``(scores[N], policy_scores[N, K])``.
    """
    params = {**LEGACY_DEFAULTS, **(legacy_config or {})}
    win = np.clip(np.asarray(wdl, dtype=np.float64)[:, 0], 1e-12, 1.0 - 1e-7)
    scores = params["wdl_draw_margin"] + params["wdl_scale"] * np.log(win / (1.0 - win))
    saturated = np.asarray(wdl, dtype=np.float32)[:, 0] >= 1.0 - 1e-7
    scores = np.where(saturated, float(MATE_SCORE), np.clip(scores, -SCORE_CLAMP, SCORE_CLAMP))
    loss_saturated = np.asarray(wdl, dtype=np.float32)[:, 2] >= 1.0 - 1e-7
    scores = np.where(loss_saturated, float(-MATE_SCORE), scores)

    probs = np.asarray(policy_probs, dtype=np.float64)
    valid = np.asarray(policy_indices) >= 0
    safe = np.where(valid, np.clip(probs, 1e-300, None), 1.0)
    reference = np.where(valid, safe, -np.inf).max(axis=1, keepdims=True)
    reference = np.where(np.isfinite(reference), reference, 1.0)
    delta = params["value_scale"] * params["policy_temperature"] * np.log(safe / reference)
    policy_scores = scores[:, None] + delta
    limit = np.where(np.abs(scores) >= MATE_SCORE, float(MAX_SCORE), float(SCORE_CLAMP))[:, None]
    policy_scores = np.clip(policy_scores, -limit, limit)
    policy_scores = np.where(valid, policy_scores, 0.0)
    return scores.astype(np.float32), policy_scores.astype(np.float32)
