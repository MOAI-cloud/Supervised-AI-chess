from pathlib import Path

import chess
import numpy as np

from superchess.ccrl import EngineRating, high_elo_archives, parse_engine_archives, parse_rating_list, preprocess_pgn_files
from superchess.encoding import move_to_policy


def test_parse_rating_list_extracts_engine_elos():
    html = """
    <table>
      <tr><th>Rank</th><th>Name</th><th>Elo</th><th>Games</th></tr>
      <tr><td>1</td><td>Stockfish 18 64-bit 4CPU</td><td>3651</td><td>1200</td></tr>
      <tr><td>2</td><td>Carp 3.0.0 64-bit</td><td>3503</td><td>800</td></tr>
      <tr><td>3</td><td>LowBot 1.0</td><td>2499</td><td>200</td></tr>
    </table>
    """
    ratings = parse_rating_list(html)
    assert ratings["stockfish 18 64-bit 4cpu"].elo == 3651
    assert ratings["carp 3.0.0 64-bit"].elo == 3503


def test_engine_archive_selection_uses_high_elo_rating_cutoff():
    html = """
    <table>
      <tr><th>Engine</th><th>Number of games</th><th>Commented PGN</th></tr>
      <tr><td>Stockfish 18 64-bit 4CPU</td><td>1'017</td><td><a href="sf.7z">0.54 MB</a></td></tr>
      <tr><td>LowBot 1.0</td><td>12</td><td><a href="low.7z">0.01 MB</a></td></tr>
    </table>
    """
    archives = parse_engine_archives(html, base_url="https://example.test/4040/games.html")
    ratings = {
        "stockfish 18 64-bit 4cpu": EngineRating("Stockfish 18 64-bit 4CPU", 3651),
        "lowbot 1.0": EngineRating("LowBot 1.0", 2499),
    }
    selected = high_elo_archives(ratings, archives, min_elo=3500)
    assert [archive.name for archive in selected] == ["Stockfish 18 64-bit 4CPU"]
    assert selected[0].commented_url == "https://example.test/4040/sf.7z"


def test_engine_archive_selection_normalizes_version_separators():
    html = """
    <a href="games-by-engine-commented/Reckless_0_9_0_64-bit_4CPU.commented.[1200].pgn.7z">0.76 MB</a>
    <a href="games-by-engine-commented/PlentyChess_7_0_0_64-bit_4CPU.commented.[900].pgn.7z">0.54 MB</a>
    """
    archives = parse_engine_archives(html, base_url="https://example.test/4040/games.html")
    ratings = {
        "reckless 0.9.0 64-bit 4cpu": EngineRating("Reckless 0.9.0 64-bit 4CPU", 3648),
        "plentychess 7.0.0 64-bit 4cpu": EngineRating("PlentyChess 7.0.0 64-bit 4CPU", 3645),
    }

    selected = high_elo_archives(ratings, archives, min_elo=3500)

    assert [archive.name for archive in selected] == ["Reckless 0.9.0 64-bit 4CPU", "PlentyChess 7.0.0 64-bit 4CPU"]
    assert [archive.games for archive in selected] == [1200, 900]


def test_parse_engine_archives_extracts_current_directory_links():
    html = """
    <a href="games-by-engine-commented/Stockfish_18_64-bit_4CPU.commented.[2240].pgn.7z">1.23 MB</a>
    <a href="games-by-engine-commented/LowBot_1_0.commented.[64].pgn.7z">512 KB</a>
    """

    archives = parse_engine_archives(html, base_url="https://example.test/4040/games.html")

    stockfish = archives["stockfish 18 64-bit 4cpu"]
    assert stockfish.name == "Stockfish 18 64-bit 4CPU"
    assert stockfish.games == 2240
    assert stockfish.commented_url == "https://example.test/4040/games-by-engine-commented/Stockfish_18_64-bit_4CPU.commented.[2240].pgn.7z"
    assert stockfish.size_mb == 1.23
    assert archives["lowbot 1 0"].size_mb == 0.5


def test_preprocess_filters_ccrl_games_and_writes_policy_samples(tmp_path: Path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(
        """
[Event "CCRL"]
[Site "CCRL"]
[Date "2026.05.22"]
[Round "1"]
[White "Stockfish 18 64-bit 4CPU"]
[Black "Reckless 0.9.0 64-bit 4CPU"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "CCRL"]
[Site "CCRL"]
[Date "2026.05.22"]
[Round "2"]
[White "Stockfish 18 64-bit 4CPU"]
[Black "LowBot 1.0"]
[Result "0-1"]

1. e4 e5 0-1
""".lstrip(),
        encoding="utf-8",
    )
    ratings = {
        "stockfish 18 64-bit 4cpu": EngineRating("Stockfish 18 64-bit 4CPU", 3651),
        "reckless 0.9.0 64-bit 4cpu": EngineRating("Reckless 0.9.0 64-bit 4CPU", 3648),
        "lowbot 1.0": EngineRating("LowBot 1.0", 2400),
    }

    stats = preprocess_pgn_files([pgn_path], ratings, tmp_path / "processed", min_elo=3500, shard_size=16)

    assert stats.games_seen == 2
    assert stats.games_kept == 1
    assert stats.games_skipped_rating == 1
    assert stats.samples == 4
    shard = np.load(tmp_path / "processed" / "shard-00000.npz")
    assert shard["boards"].shape == (4, 144)
    assert shard["policies"][0] == move_to_policy(chess.Board(), chess.Move.from_uci("e2e4")).index
    assert shard["values"].tolist() == [1.0, -1.0, 1.0, -1.0]


def test_preprocess_skips_duplicate_games_across_files(tmp_path: Path):
    first_pgn = tmp_path / "stockfish.pgn"
    first_pgn.write_text(
        """
[Event "Stockfish archive"]
[Site "CCRL"]
[Date "2026.05.22"]
[Round "1"]
[White "Stockfish 18 64-bit 4CPU"]
[Black "Torch v4d 64-bit 4CPU"]
[Result "1-0"]

1. e4 e5 1-0
""".lstrip(),
        encoding="utf-8",
    )
    second_pgn = tmp_path / "torch.pgn"
    second_pgn.write_text(
        """
[Event "Torch archive"]
[Site "CCRL"]
[Date "2026.05.22"]
[Round "99"]
[White "Stockfish 18 64-bit 4CPU"]
[Black "Torch v4d 64-bit 4CPU"]
[Result "1-0"]

1. e4 e5 1-0
""".lstrip(),
        encoding="utf-8",
    )
    ratings = {
        "stockfish 18 64-bit 4cpu": EngineRating("Stockfish 18 64-bit 4CPU", 3651),
        "torch v4d 64-bit 4cpu": EngineRating("Torch v4d 64-bit 4CPU", 3633),
    }

    stats = preprocess_pgn_files([first_pgn, second_pgn], ratings, tmp_path / "processed", min_elo=3500, shard_size=16)

    assert stats.games_seen == 2
    assert stats.games_kept == 1
    assert stats.games_skipped_duplicate == 1
    assert stats.samples == 2


def test_preprocess_verbose_output_reports_progress(tmp_path: Path, capsys):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(
        """
[Event "CCRL"]
[Site "CCRL"]
[Date "2026.05.22"]
[Round "1"]
[White "Stockfish 18 64-bit 4CPU"]
[Black "Reckless 0.9.0 64-bit 4CPU"]
[Result "1-0"]

1. e4 e5 1-0
""".lstrip(),
        encoding="utf-8",
    )
    ratings = {
        "stockfish 18 64-bit 4cpu": EngineRating("Stockfish 18 64-bit 4CPU", 3651),
        "reckless 0.9.0 64-bit 4cpu": EngineRating("Reckless 0.9.0 64-bit 4CPU", 3648),
    }

    preprocess_pgn_files([pgn_path], ratings, tmp_path / "processed", min_elo=3500, shard_size=16, verbose=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[preprocess] Processing" in captured.err
    assert "Wrote" in captured.err
    assert "Finished: seen=1 kept=1 skipped=0 (rating=0 result=0 duplicate=0 error=0) samples=2 shards=1" in captured.err