from __future__ import annotations

from dataclasses import dataclass, field
import math

import chess
import numpy as np
import torch

from superchess.ccrl import result_value
from superchess.encoding import encode_board, legal_policy_indices


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    temperature: float = 0.0


@dataclass(slots=True)
class MCTSNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expand(self, priors: dict[chess.Move, float]) -> None:
        for move, prior in priors.items():
            self.children.setdefault(move, MCTSNode(float(prior)))


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_move: chess.Move
    visits: dict[chess.Move, int]
    policy: dict[chess.Move, float]
    root: "MCTSNode | None" = None


class NeuralMCTS:
    def __init__(self, model: torch.nn.Module, config: SearchConfig = SearchConfig(), device: str | torch.device | None = None) -> None:
        self.device = torch.device(device or next(model.parameters()).device)
        self.model = model.to(self.device).eval()
        self.config = config
        self.input_channels = getattr(getattr(model, "config", None), "input_channels", 18)

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        planes = encode_board(board)[: self.input_channels]
        if planes.shape[0] < self.input_channels:
            planes = np.pad(planes, ((0, self.input_channels - planes.shape[0]), (0, 0), (0, 0)))
        tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        outputs = self.model(tensor)
        policy_logits = outputs["policy"][0].detach().float().cpu().numpy()
        value = float(outputs["value"][0].detach().float().cpu())

        legal = legal_policy_indices(board)
        if not legal:
            return {}, value
        moves = list(legal.keys())
        indices = np.fromiter((legal[move] for move in moves), dtype=np.int64)
        logits = policy_logits[indices]
        logits -= logits.max()
        priors = np.exp(logits)
        priors /= priors.sum()
        return dict(zip(moves, priors.tolist(), strict=True)), value

    def search(self, board: chess.Board) -> SearchResult:
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search a finished game")

        root = MCTSNode(1.0)
        priors, _ = self.evaluate(board)
        root.expand(priors)
        if not root.children:
            raise ValueError("no legal moves available")

        for _ in range(self.config.simulations):
            node = root
            search_board = board.copy(stack=False)
            path = [node]

            while node.children:
                move, node = self.select_child(node)
                search_board.push(move)
                path.append(node)

            if search_board.is_game_over(claim_draw=True):
                value = terminal_value(search_board)
            else:
                priors, value = self.evaluate(search_board)
                node.expand(priors)
            self.backup(path, value)

        visits = {move: child.visit_count for move, child in root.children.items()}
        policy = visit_policy(visits, self.config.temperature)
        best_move = max(policy, key=policy.get)
        return SearchResult(best_move=best_move, visits=visits, policy=policy, root=root)

    def principal_variation(self, node: "MCTSNode", board: chess.Board, max_len: int = 12) -> list[chess.Move]:
        """Walk the most-visited path from ``node`` to build a principal variation."""
        line: list[chess.Move] = []
        probe = board.copy(stack=False)
        current = node
        for _ in range(max_len):
            if not current.children:
                break
            move, child = max(
                current.children.items(),
                key=lambda item: (item[1].visit_count, item[1].prior),
            )
            if child.visit_count == 0:
                break
            if move not in probe.legal_moves:
                break
            line.append(move)
            probe.push(move)
            current = child
            if probe.is_game_over(claim_draw=True):
                break
        return line

    def select_child(self, node: MCTSNode) -> tuple[chess.Move, MCTSNode]:
        parent_visits = max(1, node.visit_count)
        exploration = math.sqrt(parent_visits)

        def score(child: MCTSNode) -> float:
            q_value = -child.value
            u_value = self.config.c_puct * child.prior * exploration / (1 + child.visit_count)
            return q_value + u_value

        return max(node.children.items(), key=lambda item: score(item[1]))

    @staticmethod
    def backup(path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


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