"""Dataset identities, frozen split manifests, and integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from superchess.encoding import (
    FULL_BITPACKED_BOARD_BYTES,
    LEGACY_PACKED_BOARD_BYTES,
    PACKED_BOARD_BYTES,
    POLICY_SIZE,
    board_from_packed,
    legal_policy_indices,
)

DATASET_SCHEMA = "superchess-evals-dataset-v1"
SPLITS = ("train", "validation", "test")
_BYTE_COUNTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def position_key(packed: np.ndarray) -> bytes:
    """Group clock/en-passant variants without altering the model's input planes."""
    identity = np.asarray(packed, dtype=np.uint8)[: LEGACY_PACKED_BOARD_BYTES - 8].tobytes()
    return hashlib.blake2b(identity, digest_size=16, person=b"s-chess-group-v1").digest()


def split_for_key(key: bytes, validation_fraction: float, test_fraction: float, seed: int) -> str:
    payload = str(seed).encode("ascii") + b":" + key
    hashed = int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"superchess-split").digest(), "big")
    if hashed < int(test_fraction * 2**64):
        return "test"
    if hashed < int((test_fraction + validation_fraction) * 2**64):
        return "validation"
    return "train"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manifest_identity(metadata: dict[str, Any]) -> str:
    content = {name: metadata[name] for name in ("dataset_schema", "config", "shards")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_manifest(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / "metadata.json"
    if not path.exists():
        if any((data_dir / split).exists() for split in SPLITS):
            raise ValueError(f"dataset {data_dir} has split directories but no completed manifest")
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "dataset_schema" not in metadata:
        return None
    if metadata["dataset_schema"] != DATASET_SCHEMA or metadata.get("status") != "complete":
        raise ValueError(f"unsupported or incomplete dataset manifest: {path}")
    if manifest_identity(metadata) != metadata.get("dataset_id"):
        raise ValueError(f"dataset manifest identity mismatch: {path}")
    return metadata


def manifest_shards(data_dir: Path, metadata: dict[str, Any], split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown dataset split: {split}")
    result: list[Path] = []
    for entry in metadata["shards"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] not in SPLITS or not relative.name.startswith("shard-") or relative.suffix != ".npz":
            raise ValueError(f"unsafe shard path in manifest: {relative}")
        path = data_dir / relative
        if entry["split"] != relative.parts[0]:
            raise ValueError(f"shard split mismatch in manifest: {relative}")
        if entry["split"] == split:
            if path.is_symlink() or path.parent.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"missing or modified dataset shard: {path}")
            result.append(path)
    if len(result) != len(set(result)):
        raise ValueError("duplicate shard paths in dataset manifest")
    actual = set((data_dir / split).glob("shard-*.npz"))
    if set(result) != actual:
        raise ValueError(f"unlisted or missing shards in {data_dir / split}")
    return result


def verify_shard_checksums(data_dir: Path, metadata: dict[str, Any], paths: list[Path]) -> None:
    entries = {entry["path"]: entry for entry in metadata["shards"]}
    for path in paths:
        entry = entries[str(path.relative_to(data_dir))]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"dataset shard checksum mismatch: {path}")


def _audit_arrays(arrays: dict[str, np.ndarray], legacy_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    from superchess.targets import MAX_SCORE, reconstruct_legacy_scores

    boards, indices = arrays["boards"], arrays["policy_indices"]
    if boards.dtype != np.uint8 or boards.ndim != 2 or boards.shape[1] not in (LEGACY_PACKED_BOARD_BYTES, PACKED_BOARD_BYTES, FULL_BITPACKED_BOARD_BYTES):
        raise ValueError("boards must be a supported uint8 packed-position matrix")
    count = len(boards)
    if not count or indices.ndim != 2 or indices.shape[0] != count or not indices.shape[1] or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("policy_indices must be a nonempty integer [positions, candidates] matrix")
    if np.any(indices < -1) or np.any(indices >= POLICY_SIZE):
        raise ValueError("policy indices outside [-1, POLICY_SIZE)")
    valid = indices >= 0
    if not np.all(valid.any(axis=1)):
        raise ValueError("policy row has no valid target")
    sorted_indices = np.sort(np.where(valid, indices, POLICY_SIZE), axis=1)
    if np.any((np.diff(sorted_indices, axis=1) == 0) & (sorted_indices[:, 1:] < POLICY_SIZE)):
        raise ValueError("duplicate policy target indices in a row")
    if "score" in arrays and "policy_scores" in arrays:
        scores, policy_scores = arrays["score"], arrays["policy_scores"]
        if scores.shape != (count,) or policy_scores.shape != indices.shape:
            raise ValueError("score or policy_scores shape mismatch")
        if not np.isfinite(scores).all() or not np.isfinite(policy_scores).all() or np.any(np.abs(scores.astype(np.float64)) > MAX_SCORE) or np.any(np.abs(policy_scores.astype(np.float64)) > MAX_SCORE):
            raise ValueError("nonfinite or out-of-range raw scores")
        if np.any(policy_scores[~valid] != 0):
            raise ValueError("padded policy scores must be zero")
        for name in ("depth", "policy_depth"):
            depth = arrays[name]
            if depth.shape != (count,) or not np.issubdtype(depth.dtype, np.integer) or np.any(depth <= 0):
                raise ValueError(f"invalid {name} array")
        return boards, scores, indices, policy_scores, "evals-v2"
    wdl, probs = arrays["wdl"], arrays["policy_probs"]
    if wdl.shape != (count, 3) or probs.shape != indices.shape:
        raise ValueError("legacy WDL or policy probability shape mismatch")
    for values in (wdl, probs):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("legacy probabilities must be finite, nonnegative, and normalized")
    if np.any(probs[~valid] != 0):
        raise ValueError("legacy padding has nonzero probability")
    scores, policy_scores = reconstruct_legacy_scores(wdl, indices, probs, legacy_config)
    return boards, scores, indices, policy_scores, "evals-v1"


def _add_histogram(histogram: dict[str, int], values: np.ndarray) -> None:
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique, counts, strict=True):
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + int(count)


def _packed_board_errors(boards: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pieces = boards[:, :96].reshape(-1, 12, 8)
    counts = _BYTE_COUNTS[pieces].sum(axis=2)
    occupied = np.zeros((len(boards), 8), dtype=np.uint8)
    overlap = np.zeros(len(boards), dtype=bool)
    for piece_index in range(12):
        overlap |= np.any(occupied & pieces[:, piece_index], axis=1)
        occupied |= pieces[:, piece_index]
    flags = boards[:, 96:136].reshape(-1, 5, 8)
    ep = boards[:, 136:144]
    errors = {
        "piece_overlap": overlap,
        "piece_count": (counts[:, :6].sum(axis=1) > 16) | (counts[:, 6:].sum(axis=1) > 16),
        "king_count": (counts[:, 5] != 1) | (counts[:, 11] != 1),
        "pawn_count": (counts[:, 0] > 8) | (counts[:, 6] > 8),
        "backrank_pawns": np.any(boards[:, [0, 7, 48, 55]] != 0, axis=1),
        "side_castling_planes": np.any(flags != flags[:, :, :1], axis=(1, 2)) | np.any((flags[:, :, 0] != 0) & (flags[:, :, 0] != 255), axis=1),
        "en_passant_plane": np.any(ep != ep[:, :1], axis=1) | (_BYTE_COUNTS[ep[:, 0]] > 1),
    }
    return counts.sum(axis=1), errors


def audit_eval_dataset(
    data_dir: Path, *, max_shards: int | None = None, legal_samples: int = 128, verify_checksums: bool = True
) -> dict[str, Any]:
    """Audit structure everywhere scanned, legality at deterministic sampled rows.

    ``legal_samples=0`` checks every position. A sampled scan is never reported
    as a complete dataset audit. Sorted deduplicated manifests permit exact
    duplicate detection in constant memory; legacy flat shards do not.
    """
    from superchess.targets import DEFAULT_CP_SCALE

    if max_shards is not None and max_shards < 1 or legal_samples < 0:
        raise ValueError("max_shards must be positive and legal_samples nonnegative")
    report: dict[str, Any] = {
        "data": str(data_dir.resolve()), "dataset_id": None, "valid": False,
        "shards_total": 0, "shards_checked": 0, "positions_checked": 0,
        "legal_positions_checked": 0, "invalid_positions": 0, "board_errors": {}, "checksums_verified": 0,
        "complete_dataset_scan": False, "complete_legal_scan": False,
        "split_assignment_verified": False, "duplicate_order_verified": False,
        "split_positions": {}, "formats": {}, "policy_candidates": {}, "piece_counts": {},
        "depth": {}, "policy_depth_gap": {}, "turns": {"white": 0, "black": 0},
        "score_bin_edges": [-12501, -600, -200, -50, 50, 200, 600, 12501],
        "score_bin_counts": [0] * 7, "clock_bytes_min": None, "clock_bytes_max": None,
        "value_policy_expected_gap_mean": None, "value_policy_expected_gap_max": None,
        "error_count": 0, "errors": [], "limitations": [],
    }

    def error(message: str) -> None:
        report["error_count"] += 1
        if len(report["errors"]) < 32:
            report["errors"].append(message)

    try:
        metadata = load_manifest(data_dir)
        if metadata is not None:
            paths = [path for split in SPLITS for path in manifest_shards(data_dir, metadata, split)]
            entries = {entry["path"]: entry for entry in metadata["shards"]}
            report["dataset_id"] = metadata["dataset_id"]
            config = metadata["config"]
        else:
            paths = sorted(data_dir.glob("shard-*.npz"))
            entries = {}
            metadata_path = data_dir / "metadata.json"
            config = json.loads(metadata_path.read_text()).get("config", {}) if metadata_path.exists() else {}
            report["limitations"].append("No frozen manifest: source provenance, checksums, split isolation, and global deduplication are unverified.")
        if not paths:
            raise ValueError("no shards found")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error(str(exc))
        return report

    report["shards_total"] = len(paths)
    if max_shards is not None and len(paths) > max_shards:
        selected = np.linspace(0, len(paths) - 1, max_shards, dtype=np.int64)
        paths = [paths[int(index)] for index in selected]
        report["limitations"].append("Only a deterministic shard sample was scanned; counts describe that sample.")
    previous: dict[str, tuple[bytes, bytes]] = {}
    gap_sum = 0.0
    gap_count = 0
    for path in paths:
        report["shards_checked"] += 1
        split = path.parent.name if metadata is not None else "unspecified"
        try:
            entry = entries.get(str(path.relative_to(data_dir)))
            if entry is not None and verify_checksums:
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("SHA256 checksum mismatch")
                report["checksums_verified"] += 1
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            boards, scores, indices, policy_scores, data_format = _audit_arrays(arrays, config)
            count = len(boards)
            valid = indices >= 0
            piece_counts, board_errors = _packed_board_errors(boards)
            invalid = np.zeros(count, dtype=bool)
            for name, mask in board_errors.items():
                invalid |= mask
                failures = int(mask.sum())
                if failures:
                    report["board_errors"][name] = report["board_errors"].get(name, 0) + failures
                    error(f"{path.relative_to(data_dir)}: {name} in {failures} positions")
            report["invalid_positions"] += int(invalid.sum())
            if entry is not None:
                if count != entry["rows"]:
                    raise ValueError("manifest row-count mismatch")
                hashes = arrays["position_hash"]
                if hashes.shape != (count, 16) or hashes.dtype != np.uint8:
                    raise ValueError("position_hash must be uint8 [positions, 16]")
                for row, (packed, stored) in enumerate(zip(boards, hashes, strict=True)):
                    key = position_key(packed)
                    if stored.tobytes() != key:
                        raise ValueError(f"position hash mismatch at row {row}")
                    if split_for_key(key, config["validation_fraction"], config["test_fraction"], config["split_seed"]) != split:
                        raise ValueError(f"incorrect frozen split at row {row}")
                    if config["deduplicate"]:
                        order = (key, packed.tobytes())
                        if split in previous and order <= previous[split]:
                            raise ValueError(f"duplicate or unordered position at row {row}")
                        previous[split] = order
                if np.any(arrays["depth"] < config["min_depth"]) or np.any(arrays["policy_depth"] < arrays["depth"].astype(np.int64) - config["policy_max_depth_gap"]):
                    raise ValueError("evaluation depth violates dataset quality policy")
            sample_count = count if legal_samples == 0 else min(count, legal_samples)
            for row in np.linspace(0, count - 1, sample_count, dtype=np.int64):
                report["legal_positions_checked"] += 1
                if invalid[row]:
                    continue
                try:
                    board = board_from_packed(boards[row])
                    legal = set(legal_policy_indices(board).values())
                    targets = set(indices[row][valid[row]].tolist())
                    if not targets <= legal:
                        raise ValueError(f"illegal policy targets: {sorted(targets - legal)}")
                except ValueError as exc:
                    report["invalid_positions"] += 1
                    error(f"{path.relative_to(data_dir)}: row {row}: {exc}")
            report["positions_checked"] += count
            report["formats"][data_format] = report["formats"].get(data_format, 0) + count
            report["split_positions"][split] = report["split_positions"].get(split, 0) + count
            _add_histogram(report["policy_candidates"], valid.sum(axis=1))
            _add_histogram(report["piece_counts"], piece_counts)
            white = int(np.count_nonzero(boards[:, 96] == 255))
            report["turns"]["white"] += white
            report["turns"]["black"] += count - white
            histogram, _ = np.histogram(scores, report["score_bin_edges"])
            report["score_bin_counts"] = [old + int(new) for old, new in zip(report["score_bin_counts"], histogram, strict=True)]
            if data_format == "evals-v2":
                _add_histogram(report["depth"], arrays["depth"])
                _add_histogram(report["policy_depth_gap"], arrays["depth"].astype(np.int64) - arrays["policy_depth"].astype(np.int64))
            if boards.shape[1] == PACKED_BOARD_BYTES:
                low, high = boards[:, -2:].min(axis=0), boards[:, -2:].max(axis=0)
                report["clock_bytes_min"] = np.minimum(low, report["clock_bytes_min"]).tolist() if report["clock_bytes_min"] is not None else low.tolist()
                report["clock_bytes_max"] = np.maximum(high, report["clock_bytes_max"]).tolist() if report["clock_bytes_max"] is not None else high.tolist()
            if data_format == "evals-v2":
                best_scores = np.where(valid, policy_scores, -np.inf).max(axis=1)
                gap = np.abs(1 / (1 + np.exp(-scores.astype(np.float64) / DEFAULT_CP_SCALE)) - 1 / (1 + np.exp(-best_scores.astype(np.float64) / DEFAULT_CP_SCALE)))
                gap_sum += float(gap.sum())
                gap_count += count
                report["value_policy_expected_gap_max"] = max(report["value_policy_expected_gap_max"] or 0.0, float(gap.max()))
                if metadata is not None and config.get("max_policy_value_gap") is not None and np.any(gap > config["max_policy_value_gap"] + 1e-7):
                    raise ValueError("value/policy score disagreement violates dataset quality policy")
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            error(f"{path.relative_to(data_dir)}: {exc}")
    report["value_policy_expected_gap_mean"] = gap_sum / gap_count if gap_count else None
    report["valid"] = report["error_count"] == 0
    report["complete_dataset_scan"] = report["valid"] and report["shards_checked"] == report["shards_total"]
    report["complete_legal_scan"] = report["complete_dataset_scan"] and report["legal_positions_checked"] == report["positions_checked"]
    report["split_assignment_verified"] = report["complete_dataset_scan"] and metadata is not None
    report["duplicate_order_verified"] = report["complete_dataset_scan"] and metadata is not None and config["deduplicate"]
    if not report["complete_legal_scan"]:
        report["limitations"].append("Policy legality was not exhaustively checked across the full dataset.")
    if "evals-v1" in report["formats"]:
        report["limitations"].append("Legacy scores are reconstructed approximately: mate detail and cross-depth policy anchors cannot be recovered.")
    report["limitations"].append("Position grouping does not prove game-family or temporal independence; the eval dump lacks that provenance.")
    return report