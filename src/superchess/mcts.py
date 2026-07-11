from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import chess
import numpy as np
import torch

from superchess.ccrl import result_value
from superchess.encoding import LEGACY_BOARD_CHANNELS, encode_board, legal_policy_indices

Evaluation = tuple[list[chess.Move], np.ndarray, float]
"""Network output for one position: legal moves, prior array, and scalar value."""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    temperature: float = 0.0
    evaluation_batch_size: int = 8
    fpu_reduction: float = 0.3
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0


class MCTSNode:
    """Tree node holding per-edge statistics in numpy arrays (AlphaZero layout).

    Edge ``i`` corresponds to ``moves[i]``. ``value_sums[i] / visit_counts[i]`` is
    the Q-value of that move from the perspective of the player to move at this
    node, so PUCT selection is a single vectorized expression with no per-child
    negation or attribute chasing.
    """

    __slots__ = ("moves", "priors", "visit_counts", "value_sums", "children", "terminal_value")

    def __init__(
        self,
        moves: list[chess.Move],
        priors: np.ndarray,
        terminal_value: float | None = None,
    ) -> None:
        count = len(moves)
        self.moves = moves
        self.priors = np.asarray(priors, dtype=np.float32)
        self.visit_counts = np.zeros(count, dtype=np.float32)
        self.value_sums = np.zeros(count, dtype=np.float32)
        self.children: list[MCTSNode | None] = [None] * count
        self.terminal_value = terminal_value

    @classmethod
    def terminal(cls, value: float) -> "MCTSNode":
        return cls([], np.empty(0, dtype=np.float32), terminal_value=value)

    @property
    def visit_count(self) -> int:
        return int(self.visit_counts.sum())

    def move_index(self, move: chess.Move) -> int:
        try:
            return self.moves.index(move)
        except ValueError:
            return -1

    def child_visits(self, index: int) -> int:
        return int(self.visit_counts[index])

    def child_q(self, index: int) -> float:
        """Q-value of edge ``index`` from this node's side-to-move perspective."""
        visits = self.visit_counts[index]
        return 0.0 if visits <= 0 else float(self.value_sums[index] / visits)


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_move: chess.Move
    visits: dict[chess.Move, int]
    policy: dict[chess.Move, float]
    root: "MCTSNode | None" = None


class NeuralMCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        config: SearchConfig = SearchConfig(),
        device: str | torch.device | None = None,
        *,
        eval_cache: dict[Hashable, Evaluation] | None = None,
        eval_cache_size: int = 50_000,
    ) -> None:
        self.device = torch.device(device or next(model.parameters()).device)
        self.model = model.to(self.device).eval()
        self.config = config
        self.input_channels = getattr(getattr(model, "config", None), "input_channels", 18)
        self._amp_enabled = self.device.type == "cuda"
        self._amp_dtype = (
            torch.bfloat16 if self._amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        )
        # Transposition-aware evaluation cache. Pass a shared dict to persist
        # hits across searcher instances (e.g. GUI requests).
        self._eval_cache: dict[Hashable, Evaluation] = {} if eval_cache is None else eval_cache
        self._eval_cache_size = max(0, eval_cache_size)
        # Clocks only influence the network input when the model consumes the
        # extended planes, so only key on them when they matter.
        self._clock_sensitive = self.input_channels > LEGACY_BOARD_CHANNELS
        self._rng = np.random.default_rng()

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        return self.evaluate_batch([board])[0]

    @torch.inference_mode()
    def evaluate_batch(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], float]]:
        return [
            (dict(zip(moves, priors.tolist(), strict=True)), value)
            for moves, priors, value in self._evaluate_positions(boards)
        ]

    def _position_key(self, board: chess.Board) -> Hashable:
        key = board._transposition_key()
        if self._clock_sensitive:
            return (key, min(board.halfmove_clock, 100), min(board.fullmove_number, 200))
        return key

    def _cache_store(self, key: Hashable, evaluation: Evaluation) -> None:
        if self._eval_cache_size <= 0:
            return
        cache = self._eval_cache
        try:
            if key not in cache and len(cache) >= self._eval_cache_size:
                cache.pop(next(iter(cache)), None)  # FIFO eviction
        except StopIteration:  # pragma: no cover - concurrent eviction race
            pass
        cache[key] = evaluation

    def _evaluate_positions(self, boards: list[chess.Board]) -> list[Evaluation]:
        """Evaluate positions with cache hits and in-batch transposition dedup."""
        if not boards:
            return []
        results: list[Evaluation | None] = [None] * len(boards)
        fresh_indices: list[int] = []
        fresh_keys: list[Hashable] = []
        key_slots: dict[Hashable, list[int]] = {}
        for index, board in enumerate(boards):
            key = self._position_key(board)
            cached = self._eval_cache.get(key)
            if cached is not None:
                results[index] = cached
                continue
            slots = key_slots.setdefault(key, [])
            if not slots:
                fresh_indices.append(index)
                fresh_keys.append(key)
            slots.append(index)
        if fresh_indices:
            evaluations = self._run_model([boards[index] for index in fresh_indices])
            for key, evaluation in zip(fresh_keys, evaluations, strict=True):
                self._cache_store(key, evaluation)
                for slot in key_slots[key]:
                    results[slot] = evaluation
        return results  # type: ignore[return-value]

    @torch.inference_mode()
    def _run_model(self, boards: list[chess.Board]) -> list[Evaluation]:
        tensor = torch.from_numpy(np.stack([self._planes_for_board(board) for board in boards])).to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled):
            outputs = self.model(tensor)
        policy_logits = outputs["policy"].detach().float().cpu().numpy()
        values = outputs["value"].detach().float().cpu().numpy()
        evaluations: list[Evaluation] = []
        for board, logits, value in zip(boards, policy_logits, values, strict=True):
            legal = legal_policy_indices(board)
            if not legal:
                evaluations.append(([], np.empty(0, dtype=np.float32), float(value)))
                continue
            moves = list(legal.keys())
            indices = np.fromiter(legal.values(), dtype=np.int64, count=len(moves))
            masked = logits[indices].astype(np.float32, copy=False)
            masked -= masked.max()
            np.exp(masked, out=masked)
            masked /= masked.sum()
            evaluations.append((moves, masked, float(value)))
        return evaluations

    def _planes_for_board(self, board: chess.Board) -> np.ndarray:
        planes = encode_board(board)[: self.input_channels]
        if planes.shape[0] < self.input_channels:
            planes = np.pad(planes, ((0, self.input_channels - planes.shape[0]), (0, 0), (0, 0)))
        return planes

    def search(self, board: chess.Board, *, tree: MCTSNode | None = None) -> SearchResult:
        """Run PUCT search from ``board``.

        ``tree`` may be a subtree returned by a previous search **for this exact
        position** (see :func:`find_subtree`); its statistics are reused so
        earlier work carries over.
        """
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search a finished game")

        root: MCTSNode | None = tree if tree is not None and tree.terminal_value is None and tree.moves else None
        if root is None:
            ((moves, priors, _),) = self._evaluate_positions([board])
            if not moves:
                raise ValueError("no legal moves available")
            root = MCTSNode(moves, priors.copy())
        if self.config.dirichlet_epsilon > 0.0:
            self._apply_root_noise(root)

        history_counts = _history_key_counts(board)
        simulations = max(0, self.config.simulations)
        batch_size = max(1, self.config.evaluation_batch_size)
        completed = 0
        while completed < simulations:
            batch_limit = min(batch_size, simulations - completed)
            pending: list[tuple[MCTSNode, int, list[tuple[MCTSNode, int]], chess.Board]] = []

            for _ in range(batch_limit):
                parent, index, path, leaf_board, terminal = self._descend(root, board, history_counts)
                if terminal is not None:
                    _backup(path, terminal)
                else:
                    pending.append((parent, index, path, leaf_board))

            if pending:
                evaluations = self._evaluate_positions([leaf_board for *_, leaf_board in pending])
                for (parent, index, path, leaf_board), (moves, priors, value) in zip(
                    pending, evaluations, strict=True
                ):
                    if not moves:  # checkmate/stalemate (mate outranks the 50-move clock)
                        child = MCTSNode.terminal(-1.0 if leaf_board.is_check() else 0.0)
                    elif leaf_board.halfmove_clock >= 100:  # in-check fifty-move edge case
                        child = MCTSNode.terminal(0.0)
                    else:
                        child = MCTSNode(moves, priors.copy())
                    parent.children[index] = child
                    _backup(path, value if child.terminal_value is None else child.terminal_value)

            completed += batch_limit

        visits = {
            move: int(count)
            for move, count in zip(root.moves, root.visit_counts.tolist(), strict=True)
        }
        policy = visit_policy(visits, self.config.temperature)
        best_move = max(policy, key=policy.get)
        return SearchResult(best_move=best_move, visits=visits, policy=policy, root=root)

    def _descend(
        self,
        root: MCTSNode,
        board: chess.Board,
        history_counts: dict[Hashable, int],
    ) -> tuple[MCTSNode, int, list[tuple[MCTSNode, int]], chess.Board, float | None]:
        """Walk one simulation to a leaf, applying virtual loss along the way.

        Returns ``(parent, edge_index, path, leaf_board, terminal_value)``;
        ``terminal_value`` is ``None`` when the leaf still needs a network
        evaluation. Draws by repetition (twofold vs. game history or the search
        path, Lc0-style), the fifty-move rule, and insufficient material are
        adjudicated here without spending network batch slots.
        """
        c_puct = self.config.c_puct
        fpu_reduction = self.config.fpu_reduction
        node = root
        search_board = board.copy(stack=False)
        path: list[tuple[MCTSNode, int]] = []
        path_keys: set[Hashable] = set()

        while True:
            index = _select_index(node, c_puct, fpu_reduction)
            node.visit_counts[index] += 1.0
            node.value_sums[index] -= 1.0  # virtual loss: looks lost until backed up
            path.append((node, index))
            search_board.push(node.moves[index])
            child = node.children[index]

            if child is not None:
                if child.terminal_value is not None:
                    return node, index, path, search_board, child.terminal_value
                # Expanded interior nodes were repetition-checked at creation and
                # tree paths are unique, so only remember the key for cycles.
                path_keys.add(search_board._transposition_key())
                node = child
                continue

            key = search_board._transposition_key()
            if history_counts.get(key, 0) > 0 or key in path_keys:
                node.children[index] = MCTSNode.terminal(0.0)
                return node, index, path, search_board, 0.0
            if search_board.halfmove_clock >= 100 and not search_board.is_check():
                node.children[index] = MCTSNode.terminal(0.0)
                return node, index, path, search_board, 0.0
            if search_board.is_insufficient_material():
                node.children[index] = MCTSNode.terminal(0.0)
                return node, index, path, search_board, 0.0
            return node, index, path, search_board, None

    def principal_variation(self, node: MCTSNode, board: chess.Board, max_len: int = 12) -> list[chess.Move]:
        """Walk the most-visited path from ``node`` (moves are legal by construction)."""
        del board  # kept for API compatibility; tree moves need no re-validation
        line: list[chess.Move] = []
        current: MCTSNode | None = node
        while (
            current is not None
            and current.terminal_value is None
            and current.moves
            and len(line) < max_len
        ):
            counts = current.visit_counts
            index = int(np.argmax(counts + current.priors))  # priors break visit ties
            if counts[index] <= 0:
                break
            line.append(current.moves[index])
            current = current.children[index]
        return line

    def _apply_root_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into root priors (AlphaZero-style exploration)."""
        if not root.moves:
            return
        epsilon = self.config.dirichlet_epsilon
        noise = self._rng.dirichlet(np.full(len(root.moves), self.config.dirichlet_alpha))
        root.priors = (1.0 - epsilon) * root.priors + epsilon * noise.astype(np.float32)


def _select_index(node: MCTSNode, c_puct: float, fpu_reduction: float) -> int:
    """Vectorized PUCT with first-play urgency for unvisited edges."""
    counts = node.visit_counts
    total = float(counts.sum())
    if total > 0.0:
        fpu = float(node.value_sums.sum()) / total - fpu_reduction
        q_values = np.full_like(counts, fpu)
        np.divide(node.value_sums, counts, out=q_values, where=counts > 0.0)
    else:
        q_values = np.full_like(counts, -fpu_reduction)
    scores = q_values + (c_puct * math.sqrt(total + 1.0)) * node.priors / (1.0 + counts)
    return int(np.argmax(scores))


def _backup(path: list[tuple[MCTSNode, int]], value: float) -> None:
    """Propagate a leaf value (leaf side-to-move perspective) and undo virtual loss."""
    for node, index in reversed(path):
        value = -value  # flip into the edge owner's perspective
        node.value_sums[index] += value + 1.0  # +1 reverts the virtual loss


def _history_key_counts(board: chess.Board) -> dict[Hashable, int]:
    """Occurrence counts of every position in the game history (root included)."""
    counts: dict[Hashable, int] = {}
    probe = board.copy()
    while True:
        key = probe._transposition_key()
        counts[key] = counts.get(key, 0) + 1
        if not probe.move_stack:
            return counts
        probe.pop()


def find_subtree(
    root: MCTSNode | None,
    root_board: chess.Board,
    target_board: chess.Board,
    max_plies: int = 4,
) -> MCTSNode | None:
    """Locate the node for ``target_board`` inside a previous search tree.

    Explores visited edges up to ``max_plies`` from ``root_board`` and matches on
    the full FEN, enabling tree reuse across consecutive searches.
    """
    if root is None:
        return None
    target = target_board.fen()
    stack: list[tuple[MCTSNode, chess.Board, int]] = [(root, root_board.copy(stack=False), 0)]
    while stack:
        node, probe, depth = stack.pop()
        if probe.fen() == target:
            return node
        if depth >= max_plies or node.terminal_value is not None:
            continue
        for index, child in enumerate(node.children):
            if child is None or node.visit_counts[index] <= 0:
                continue
            advanced = probe.copy(stack=False)
            advanced.push(node.moves[index])
            stack.append((child, advanced, depth + 1))
    return None


def terminal_value(board: chess.Board) -> float:
    result = board.result(claim_draw=True)
    return 0.0 if result == "*" else result_value(result, board.turn)


def visit_policy(visits: dict[chess.Move, int], temperature: float) -> dict[chess.Move, float]:
    if temperature <= 0:
        best_move = max(visits, key=visits.get)
        return {move: 1.0 if move == best_move else 0.0 for move in visits}
    moves = list(visits)
    counts = np.asarray([visits[move] for move in moves], dtype=np.float64) ** (1.0 / temperature)
    total = counts.sum()
    if total <= 0:
        return {move: 1.0 / len(moves) for move in moves}
    probs = counts / total
    return dict(zip(moves, probs.tolist(), strict=True))