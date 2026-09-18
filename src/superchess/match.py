"""Engine-vs-engine matches: the only metric that measures playing strength.

Offline metrics (policy accuracy, value loss) are proxies that can improve while
play gets worse (for example when training-time inputs differ from play-time
inputs). ``superchess match`` plays real games between two engines — two
checkpoints, or a checkpoint and a Lichess Stockfish level — from varied
openings with colours alternated, and reports the score with an Elo difference,
a 95% confidence interval and the likelihood of superiority (cutechess-style
LOS approximation). Confidence bounds group games sharing a paired opening.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Protocol

import chess
import chess.pgn as chess_pgn
import numpy as np

from superchess.mcts import NeuralMCTS, SearchConfig, find_subtree
from superchess.stockfish import StockfishHandle

MAX_ELO = 1000.0


class Player(Protocol):
    name: str

    def new_game(self) -> None: ...

    def move(self, board: chess.Board) -> chess.Move: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MatchConfig:
    games: int = 20
    simulations: int = 256
    c_puct: float = 1.5
    fpu_reduction: float = 0.3
    evaluation_batch_size: int = 32
    policy_softmax_temperature: float = 1.0
    opening_plies: int = 6
    opening_temperature: float = 1.0
    max_plies: int = 400
    seed: int = 0
    device: str | None = None
    allow_legacy_checkpoint: bool = False
    stockfish_command: str = "stockfish"
    root_selection_a: str = "puct"
    root_selection_b: str = "puct"
    root_candidates: int = 16
    root_value_scale: float = 0.1
    root_rescue_margin: float = 0.15
    root_rescue_margin_b: float | None = None

    def search_config(self, root_selection: str = "puct", *, root_rescue_margin: float | None = None) -> SearchConfig:
        return SearchConfig(
            simulations=self.simulations,
            c_puct=self.c_puct,
            evaluation_batch_size=self.evaluation_batch_size,
            fpu_reduction=self.fpu_reduction,
            policy_softmax_temperature=self.policy_softmax_temperature,
            root_selection=root_selection,
            root_candidates=self.root_candidates,
            root_value_scale=self.root_value_scale,
            root_rescue_margin=self.root_rescue_margin if root_rescue_margin is None else root_rescue_margin,
        )


@dataclass(slots=True)
class GameRecord:
    round: int
    white: str
    black: str
    result: str
    termination: str
    plies: int
    opening_plies: int
    start_fen: str
    moves: list[str] = field(default_factory=list)
    white_seconds: float = 0.0
    black_seconds: float = 0.0

    def to_pgn(self, event: str = "Superchess match") -> str:
        game = chess_pgn.Game()
        game.headers["Event"] = event
        game.headers["Round"] = str(self.round)
        game.headers["White"] = self.white
        game.headers["Black"] = self.black
        game.headers["Result"] = self.result
        game.headers["Termination"] = self.termination
        if self.start_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.start_fen
            game.setup(chess.Board(self.start_fen))
        node: chess_pgn.GameNode = game
        for uci in self.moves:
            node = node.add_variation(chess.Move.from_uci(uci))
        return str(game)


@dataclass(slots=True)
class MatchSummary:
    engine_a: str
    engine_b: str
    wins: int
    draws: int
    losses: int
    score: float
    elo: float
    elo_low: float
    elo_high: float
    los: float
    average_plies: float
    games: list[GameRecord]
    pairs: int = 0
    unpaired_games: int = 0
    confidence_method: str = "hoeffding_opening_groups"
    los_method: str = "normal_approximation_ignores_pairing"
    engine_a_seconds: float = 0.0
    engine_b_seconds: float = 0.0
    config: dict[str, Any] | None = None
    opening_groups: int = 0
    engine_a_search: dict[str, int] | None = None
    engine_b_search: dict[str, int] | None = None
    engine_a_spec: str | None = None
    engine_b_spec: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["games"] = [asdict(game) for game in self.games]
        return payload


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class NeuralPlayer:
    """MCTS player backed by a checkpoint, with per-game eval cache and tree reuse."""

    def __init__(self, checkpoint: Path, config: SearchConfig, *, device: str | None, allow_legacy: bool, name: str | None = None) -> None:
        from superchess.training import load_checkpoint_bundle

        bundle = load_checkpoint_bundle(checkpoint, device_name=device, allow_legacy_policy=allow_legacy)
        self.name = name or checkpoint.stem
        self.config = config
        self._cache: dict = {}
        self.searcher = NeuralMCTS(bundle.model, config, eval_cache=self._cache)
        self._tree: tuple[chess.Board, Any] | None = None
        self.search_totals: dict[str, int] = {}

    def new_game(self) -> None:
        self._cache.clear()
        self._tree = None

    def move(self, board: chess.Board) -> chess.Move:
        tree = None
        if self._tree is not None:
            previous_board, previous_root = self._tree
            tree = find_subtree(previous_root, previous_board, board)
        result = self.searcher.search(board, tree=tree)
        self._tree = (board.copy(stack=False), result.root)
        if result.stats is not None:
            for name in ("simulations", "network_evaluations", "network_batches", "cache_hits", "batch_collisions", "root_rescues", "root_rescue_selected"):
                self.search_totals[name] = self.search_totals.get(name, 0) + getattr(result.stats, name)
        return result.best_move

    def sample_move(self, board: chess.Board, temperature: float, rng: np.random.Generator) -> chess.Move:
        """Sample from the raw network policy (used to diversify openings)."""
        policy, _ = self.searcher.evaluate(board)
        moves = list(policy)
        priors = np.asarray([policy[move] for move in moves], dtype=np.float64)
        if temperature <= 0:
            return moves[int(priors.argmax())]
        weights = np.power(np.clip(priors, 1e-12, None), 1.0 / temperature)
        weights /= weights.sum()
        return moves[int(rng.choice(len(moves), p=weights))]

    def close(self) -> None:
        self._cache.clear()


class StockfishPlayer:
    """Stockfish at a Lichess fishnet level (skill, movetime, depth presets)."""

    def __init__(self, level: int, command: str = "stockfish") -> None:
        self.level = int(level)
        self.name = f"stockfish-L{self.level}"
        self.handle = StockfishHandle(command)
        self._game: object = object()

    def new_game(self) -> None:
        self._game = object()

    def move(self, board: chess.Board) -> chess.Move:
        return self.handle.play(board, self.level, game_id=self._game).move

    def close(self) -> None:
        self.handle.close()


def parse_engine_spec(spec: str) -> tuple[str, Any]:
    """``stockfish:<level>`` or a checkpoint path."""
    lowered = spec.lower()
    if lowered.startswith("stockfish:") or lowered.startswith("sf:"):
        level = int(spec.split(":", 1)[1])
        if not 1 <= level <= 8:
            raise ValueError("Stockfish level must be between 1 and 8")
        return "stockfish", level
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"engine spec {spec!r} is neither 'stockfish:<level>' nor an existing checkpoint")
    return "checkpoint", path


def make_player(
    spec: str, config: MatchConfig, *, root_selection: str = "puct", root_rescue_margin: float | None = None
) -> Player:
    kind, value = parse_engine_spec(spec)
    if kind == "stockfish":
        return StockfishPlayer(value, config.stockfish_command)
    return NeuralPlayer(
        value,
        config.search_config(root_selection, root_rescue_margin=root_rescue_margin),
        device=config.device,
        allow_legacy=config.allow_legacy_checkpoint,
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


def play_game(white: Player, black: Player, board: chess.Board, *, max_plies: int, round_index: int) -> GameRecord:
    """Play from ``board`` (whose move stack holds the opening) until the game ends."""
    white.new_game()
    black.new_game()
    opening_plies = len(board.move_stack)
    start_fen = board.root().fen()
    plies = 0
    move_seconds = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = outcome.result()
            termination = outcome.termination.name.lower()
            break
        if plies >= max_plies:
            result = "1/2-1/2"
            termination = "max_plies"
            break
        player = white if board.turn == chess.WHITE else black
        started = time.perf_counter()
        move = player.move(board)
        move_seconds[board.turn] += time.perf_counter() - started
        if move not in board.legal_moves:
            raise RuntimeError(f"{player.name} played illegal move {move.uci()} in {board.fen()}")
        board.push(move)
        plies += 1
    return GameRecord(
        round=round_index,
        white=white.name,
        black=black.name,
        result=result,
        termination=termination,
        plies=plies,
        opening_plies=opening_plies,
        start_fen=start_fen,
        moves=[move.uci() for move in board.move_stack],
        white_seconds=move_seconds[chess.WHITE],
        black_seconds=move_seconds[chess.BLACK],
    )


def load_openings(path: Path) -> list[chess.Board]:
    boards: list[chess.Board] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit():
            board = chess.Board(" ".join(fields[:6]))
        else:
            board, _ = chess.Board.from_epd(text)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            continue
        boards.append(board)
    if not boards:
        raise ValueError(f"no usable positions found in {path}")
    return boards


def random_opening(sampler: NeuralPlayer | None, plies: int, temperature: float, rng: np.random.Generator) -> chess.Board:
    """Play ``plies`` diverse moves from the start (network policy sample, or uniform random)."""
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        if sampler is not None:
            move = sampler.sample_move(board, temperature, rng)
        else:
            legal = list(board.legal_moves)
            move = legal[int(rng.integers(len(legal)))]
        board.push(move)
    if board.is_game_over(claim_draw=True):
        return chess.Board()
    return board


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def elo_from_score(score: float) -> float:
    score = min(1.0 - 1e-9, max(1e-9, score))
    return max(-MAX_ELO, min(MAX_ELO, -400.0 * math.log10(1.0 / score - 1.0)))


def elo_interval(wins: int, draws: int, losses: int, confidence_z: float = 1.959964) -> tuple[float, float, float, float]:
    """Hoeffding bounds for independent bounded game scores, including draws."""
    games = wins + draws + losses
    if games == 0:
        return 0.5, 0.0, -MAX_ELO, MAX_ELO
    score = (wins + 0.5 * draws) / games
    low, high = _bounded_elo_interval(score, 1.0 / games, confidence_z)
    return score, elo_from_score(score), low, high


def _bounded_elo_interval(score: float, squared_weights: float, confidence_z: float = 1.959964) -> tuple[float, float]:
    alpha = math.erfc(confidence_z / math.sqrt(2.0))
    radius = math.sqrt(0.5 * squared_weights * math.log(2.0 / alpha))
    return elo_from_score(score - radius), elo_from_score(score + radius)


def _opening_pairs(games: Sequence[GameRecord]) -> int:
    pairs = 0
    for first, second in zip(games[::2], games[1::2]):
        if (
            first.white == second.black
            and first.black == second.white
            and first.start_fen == second.start_fen
            and first.opening_plies == second.opening_plies
            and first.moves[: first.opening_plies] == second.moves[: second.opening_plies]
        ):
            pairs += 1
    return pairs


def likelihood_of_superiority(wins: int, losses: int) -> float:
    if wins + losses == 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((wins - losses) / math.sqrt(2.0 * (wins + losses))))


def summarize(engine_a: str, engine_b: str, games: Sequence[GameRecord]) -> MatchSummary:
    wins = draws = losses = 0
    for game in games:
        a_is_white = game.white == engine_a
        if game.result == "1/2-1/2":
            draws += 1
        elif (game.result == "1-0") == a_is_white:
            wins += 1
        else:
            losses += 1
    score, elo, low, high = elo_interval(wins, draws, losses)
    pairs = _opening_pairs(games)
    unpaired = len(games) - 2 * pairs
    opening_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    for game in games:
        opening = (game.start_fen, tuple(game.moves[: game.opening_plies]))
        opening_counts[opening] = opening_counts.get(opening, 0) + 1
    if games:
        squared_weights = sum(count**2 for count in opening_counts.values()) / len(games) ** 2
        low, high = _bounded_elo_interval(score, squared_weights)
    return MatchSummary(
        engine_a=engine_a,
        engine_b=engine_b,
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        elo=elo,
        elo_low=low,
        elo_high=high,
        los=likelihood_of_superiority(wins, losses),
        average_plies=float(np.mean([game.plies for game in games])) if games else 0.0,
        games=list(games),
        pairs=pairs,
        unpaired_games=unpaired,
        opening_groups=len(opening_counts),
        engine_a_seconds=sum(game.white_seconds if game.white == engine_a else game.black_seconds for game in games),
        engine_b_seconds=sum(game.black_seconds if game.white == engine_a else game.white_seconds for game in games),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_match(
    spec_a: str,
    spec_b: str,
    config: MatchConfig = MatchConfig(),
    *,
    openings: Sequence[chess.Board] | None = None,
    on_game: Callable[[GameRecord, MatchSummary], None] | None = None,
) -> MatchSummary:
    """Play ``config.games`` games (paired: each opening with both colours) and summarise."""
    player_a = make_player(spec_a, config, root_selection=config.root_selection_a)
    try:
        player_b = make_player(spec_b, config, root_selection=config.root_selection_b, root_rescue_margin=config.root_rescue_margin_b)
    except BaseException:
        player_a.close()
        raise
    if player_a.name == player_b.name:
        player_a.name = f"{player_a.name}#A"
        player_b.name = f"{player_b.name}#B"
    sampler = player_a if isinstance(player_a, NeuralPlayer) else (player_b if isinstance(player_b, NeuralPlayer) else None)
    rng = np.random.default_rng(config.seed)
    records: list[GameRecord] = []
    opening = chess.Board()
    try:
        for index in range(config.games):
            pair = index // 2
            if index % 2 == 0:
                if openings:
                    opening = openings[pair % len(openings)].copy()
                else:
                    opening = random_opening(sampler, config.opening_plies, config.opening_temperature, rng)
            white, black = (player_a, player_b) if index % 2 == 0 else (player_b, player_a)
            record = play_game(white, black, opening.copy(), max_plies=config.max_plies, round_index=index + 1)
            records.append(record)
            if on_game is not None:
                on_game(record, summarize(player_a.name, player_b.name, records))
    finally:
        player_a.close()
        player_b.close()
    summary = summarize(player_a.name, player_b.name, records)
    summary.config = asdict(config)
    summary.engine_a_search = getattr(player_a, "search_totals", None)
    summary.engine_b_search = getattr(player_b, "search_totals", None)
    summary.engine_a_spec = spec_a
    summary.engine_b_spec = spec_b
    return summary


def write_pgn(path: Path, games: Sequence[GameRecord], event: str = "Superchess match") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(game.to_pgn(event) for game in games) + "\n", encoding="utf-8")
