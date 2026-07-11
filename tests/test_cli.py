from __future__ import annotations

import json
from types import SimpleNamespace

import chess
import pytest

torch = pytest.importorskip("torch")

from superchess import cli


def test_cli_search_passes_legacy_and_batch_options(monkeypatch, tmp_path, capsys):
    from superchess import mcts, training

    called = {}

    def fake_load_model_checkpoint(checkpoint, device_name=None, *, allow_legacy_policy=False):
        called["checkpoint"] = checkpoint
        called["device"] = device_name
        called["allow_legacy_policy"] = allow_legacy_policy
        return object(), None

    class FakeMCTS:
        def __init__(self, model, config):
            called["config"] = config

        def search(self, board):
            move = chess.Move.from_uci("e2e4")
            return SimpleNamespace(best_move=move, visits={move: 3}, policy={move: 1.0})

    monkeypatch.setattr(training, "load_model_checkpoint", fake_load_model_checkpoint)
    monkeypatch.setattr(mcts, "NeuralMCTS", FakeMCTS)

    exit_code = cli.main(
        [
            "search",
            "--checkpoint",
            str(tmp_path / "legacy.pt"),
            "--allow-legacy-checkpoint",
            "--eval-batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["best_move"] == "e2e4"
    assert called["allow_legacy_policy"] is True
    assert called["device"] == "cpu"
    assert called["config"].evaluation_batch_size == 4


def test_cli_gui_passes_stockfish_path(monkeypatch, tmp_path):
    from superchess import gui

    called = {}

    def fake_serve(checkpoint, **kwargs):
        called["checkpoint"] = checkpoint
        called.update(kwargs)

    monkeypatch.setattr(gui, "serve", fake_serve)
    checkpoint = tmp_path / "engine.pt"

    exit_code = cli.main(
        [
            "gui",
            "--checkpoint",
            str(checkpoint),
            "--stockfish",
            "/opt/stockfish/bin/stockfish",
            "--no-browser",
        ]
    )

    assert exit_code == 0
    assert called["checkpoint"] == checkpoint
    assert called["stockfish_path"] == "/opt/stockfish/bin/stockfish"
    assert called["open_browser"] is False


def test_cli_gif_renders_replay(tmp_path, capsys):
    board = chess.Board()
    replay_path = tmp_path / "game.replay.json"
    output_path = tmp_path / "game.gif"
    replay_path.write_text(
        json.dumps(
            {
                "event": "CLI render",
                "white": "Superchess",
                "black": "Stockfish",
                "frames": [{"ply": 0, "fen": board.fen(), "display_ms": 100}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["gif", "--replay", str(replay_path), "--out", str(output_path), "--board-size", "320"])

    assert exit_code == 0
    assert output_path.read_bytes()[:6] == b"GIF89a"
    assert json.loads(capsys.readouterr().out)["gif"] == str(output_path)