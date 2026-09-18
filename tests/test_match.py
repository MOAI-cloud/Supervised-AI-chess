from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

torch = pytest.importorskip("torch")

from superchess import cli
from superchess.match import (
    GameRecord,
    MatchConfig,
    elo_from_score,
    elo_interval,
    likelihood_of_superiority,
    load_openings,
    parse_engine_spec,
    play_game,
    run_match,
    summarize,
    write_pgn,
)
from superchess.model import ChessCNNTransformer, ModelConfig
from superchess.training import POLICY_SQUARE_ORDER


def test_elo_statistics_match_reference_formulas() -> None:
    assert elo_from_score(0.5) == 0.0
    assert elo_from_score(0.75) == pytest.approx(190.85, abs=0.01)
    assert elo_from_score(1.0) == 1000.0 and elo_from_score(0.0) == -1000.0
    score, elo, low, high = elo_interval(wins=30, draws=40, losses=30)
    assert score == 0.5 and elo == 0.0 and low < 0.0 < high
    assert likelihood_of_superiority(10, 10) == pytest.approx(0.5)
    assert likelihood_of_superiority(20, 5) > 0.99
    assert likelihood_of_superiority(0, 0) == 0.5
    assert elo_interval(0, 0, 0) == (0.5, 0.0, -1000.0, 1000.0)


def test_summarize_counts_results_relative_to_engine_a() -> None:
    games = [
        GameRecord(1, "A", "B", "1-0", "checkmate", 40, 4, chess.STARTING_FEN),
        GameRecord(2, "B", "A", "1-0", "checkmate", 40, 4, chess.STARTING_FEN),
        GameRecord(3, "A", "B", "1/2-1/2", "stalemate", 80, 4, chess.STARTING_FEN),
        GameRecord(4, "B", "A", "0-1", "checkmate", 50, 4, chess.STARTING_FEN),
    ]
    summary = summarize("A", "B", games)
    assert (summary.wins, summary.draws, summary.losses) == (2, 1, 1)
    assert summary.score == pytest.approx(0.625)
    assert summary.average_plies == pytest.approx(52.5)
    payload = summary.to_dict()
    assert len(payload["games"]) == 4 and payload["games"][0]["result"] == "1-0"


@pytest.mark.parametrize("results", [(0, 10, 0), (10, 0, 0), (0, 0, 10)])
def test_elo_bounds_do_not_collapse_for_identical_results(results) -> None:
    _, _, low, high = elo_interval(*results)
    assert high > low


def test_summary_accounts_for_correlated_opening_pairs() -> None:
    games = [
        GameRecord(index + 1, "A" if index % 2 == 0 else "B", "B" if index % 2 == 0 else "A",
                   "1/2-1/2", "stalemate", 40, 0, chess.STARTING_FEN)
        for index in range(20)
    ]
    summary = summarize("A", "B", games)
    _, _, independent_low, independent_high = elo_interval(0, 20, 0)
    assert summary.pairs == 10 and summary.unpaired_games == 0
    assert summary.elo_low < independent_low < 0 < independent_high < summary.elo_high
    assert summary.confidence_method == "hoeffding_opening_groups"
    assert summarize("A", "B", games[:-1]).unpaired_games == 1
    assert summary.opening_groups == 1
    repeated = summarize("A", "B", games * 5)
    assert repeated.elo_low == summary.elo_low and repeated.elo_high == summary.elo_high


def test_parse_engine_spec(tmp_path: Path) -> None:
    assert parse_engine_spec("stockfish:5") == ("stockfish", 5)
    assert parse_engine_spec("SF:8") == ("stockfish", 8)
    with pytest.raises(ValueError):
        parse_engine_spec("stockfish:9")
    with pytest.raises(FileNotFoundError):
        parse_engine_spec(str(tmp_path / "missing.pt"))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"")
    assert parse_engine_spec(str(checkpoint)) == ("checkpoint", checkpoint)


def test_match_supports_gate_disabled_control() -> None:
    config = MatchConfig(root_selection_a="value_rescue", root_selection_b="value_rescue", root_rescue_margin_b=2.0)
    assert config.search_config(config.root_selection_a).root_rescue_margin == 0.15
    assert config.search_config(config.root_selection_b, root_rescue_margin=config.root_rescue_margin_b).root_rescue_margin == 2.0


def test_run_match_between_tiny_checkpoints_and_writes_pgn(tmp_path: Path) -> None:
    from dataclasses import asdict

    def save(path: Path, seed: int) -> Path:
        torch.manual_seed(seed)
        config = ModelConfig(channels=16, cnn_blocks=1, transformer_layers=1, attention_heads=4, smolgen_gen=8, value_bins=16)
        torch.save(
            {
                "model": ChessCNNTransformer(config).state_dict(),
                "model_config": asdict(config),
                "policy_square_order": POLICY_SQUARE_ORDER,
                "data_format": "evals",
            },
            path,
        )
        return path

    a = save(tmp_path / "a.pt", 1)
    b = save(tmp_path / "b.pt", 2)
    seen: list[int] = []
    config = MatchConfig(games=2, simulations=2, evaluation_batch_size=2, opening_plies=4, max_plies=12, device="cpu", seed=3,
                         root_selection_a="sequential_halving")
    summary = run_match(str(a), str(b), config, on_game=lambda record, running: seen.append(record.round))
    assert seen == [1, 2]
    assert summary.wins + summary.draws + summary.losses == 2
    assert {summary.games[0].white, summary.games[0].black} == {"a", "b"}
    assert summary.games[0].white == "a" and summary.games[1].white == "b"
    # Paired games share the same opening.
    opening = summary.games[0].moves[:4]
    assert summary.games[1].moves[:4] == opening and summary.games[0].opening_plies == 4
    assert all(game.termination == "max_plies" for game in summary.games)
    assert -1000.0 <= summary.elo <= 1000.0
    assert summary.pairs == 1
    assert summary.engine_a_seconds > 0 and summary.engine_b_seconds > 0
    assert summary.config["root_selection_a"] == "sequential_halving"
    assert summary.config["root_selection_b"] == "puct"
    assert summary.engine_a_search["simulations"] > 0
    assert summary.engine_b_search["simulations"] > 0
    assert summary.engine_a_spec == str(a) and summary.engine_b_spec == str(b)

    pgn_path = tmp_path / "games.pgn"
    write_pgn(pgn_path, summary.games)
    text = pgn_path.read_text(encoding="utf-8")
    assert text.count('[Event "Superchess match"]') == 2
    assert '[White "a"]' in text and '[Black "a"]' in text

    # The same model against itself gets distinct names.
    same = run_match(str(a), str(a), MatchConfig(games=1, simulations=1, opening_plies=0, max_plies=2, device="cpu"))
    assert same.engine_a != same.engine_b


def test_play_game_detects_terminal_positions() -> None:
    class Scripted:
        def __init__(self, name: str, moves: list[str]) -> None:
            self.name = name
            self.moves = list(moves)

        def new_game(self) -> None:
            pass

        def move(self, board: chess.Board) -> chess.Move:
            return chess.Move.from_uci(self.moves.pop(0))

        def close(self) -> None:
            pass

    white = Scripted("w", ["f2f3", "g2g4"])
    black = Scripted("b", ["e7e5", "d8h4"])
    record = play_game(white, black, chess.Board(), max_plies=100, round_index=1)
    assert record.result == "0-1" and record.termination == "checkmate" and record.plies == 4
    assert record.moves == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert "Qh4#" in record.to_pgn()


def test_load_openings_accepts_epd_and_fen(tmp_path: Path) -> None:
    path = tmp_path / "openings.epd"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - bm e4; id "start";',
                "",
            ]
        ),
        encoding="utf-8",
    )
    boards = load_openings(path)
    assert len(boards) == 3 and boards[0].turn == chess.BLACK
    assert boards[2].fen() == chess.STARTING_FEN


def test_cli_match_reports_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    from superchess import match

    def fake_run_match(spec_a, spec_b, config, *, openings=None, on_game=None):
        assert config.games == 4 and config.simulations == 3 and config.stockfish_command == "/opt/sf"
        assert config.root_selection_a == "sequential_halving" and config.root_selection_b == "puct"
        games = [GameRecord(1, "a", "b", "1-0", "checkmate", 30, 2, chess.STARTING_FEN, ["e2e4", "e7e5"])]
        if on_game is not None:
            on_game(games[0], summarize("a", "b", games))
        return summarize("a", "b", games)

    monkeypatch.setattr(match, "run_match", fake_run_match)
    pgn = tmp_path / "out.pgn"
    report = tmp_path / "nested" / "report.json"
    exit_code = cli.main(["match", "a.pt", "stockfish:3", "--games", "4", "--simulations", "3", "--stockfish", "/opt/sf", "--pgn", str(pgn),
                          "--root-selection-a", "sequential_halving", "--json", str(report)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["wins"] == 1 and payload["games"] == 1 and payload["elo"] == 1000.0
    assert "game 1: a vs b 1-0" in captured.err
    assert pgn.exists()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["games"][0]["moves"] == ["e2e4", "e7e5"]
