from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from superchess.ccrl import download_ccrl, preprocess_raw_directory
from superchess.evals import (
    EvalConfig,
    download_eval_dump,
    preprocess_eval_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="superchess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ccrl_parser = subparsers.add_parser("ccrl", help="Download and preprocess CCRL PGNs")
    ccrl_subparsers = ccrl_parser.add_subparsers(dest="ccrl_command", required=True)

    download_parser = ccrl_subparsers.add_parser("download", help="Download CCRL commented archives")
    download_parser.add_argument("--out", type=Path, default=Path("data/raw"))
    download_parser.add_argument("--min-elo", type=int, default=3500)
    download_parser.add_argument("--all-games", action="store_true", help="Download the full commented archive")
    download_parser.add_argument("--max-archives", type=int, default=None, help="Limit engine archives for smoke runs")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument("--polite-delay", type=float, default=0.5)

    preprocess_parser = ccrl_subparsers.add_parser("preprocess", help="Filter PGNs and write supervised NPZ shards")
    preprocess_parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    preprocess_parser.add_argument("--out", type=Path, default=Path("data/processed"))
    preprocess_parser.add_argument("--ratings", type=Path, default=None)
    preprocess_parser.add_argument("--min-elo", type=int, default=3500)
    preprocess_parser.add_argument("--shard-size", type=int, default=65_536)
    preprocess_parser.add_argument("--compressed", action="store_true")
    preprocess_parser.add_argument("--no-extract", action="store_true")
    preprocess_parser.add_argument("--max-games", type=int, default=None)
    preprocess_parser.add_argument("--max-positions", type=int, default=None)
    preprocess_parser.add_argument("--verbose", action="store_true", help="Print preprocessing progress to stderr")

    evals_parser = subparsers.add_parser("evals", help="Download and preprocess the Lichess Stockfish eval dump")
    evals_subparsers = evals_parser.add_subparsers(dest="evals_command", required=True)

    evals_download = evals_subparsers.add_parser("download", help="Download lichess_db_eval.jsonl.zst")
    evals_download.add_argument("--out", type=Path, default=Path("data/raw"))
    evals_download.add_argument("--overwrite", action="store_true")

    evals_preprocess = evals_subparsers.add_parser("preprocess", help="Write distillation NPZ shards from the eval dump")
    evals_preprocess.add_argument("--raw", type=Path, default=Path("data/raw"))
    evals_preprocess.add_argument("--out", type=Path, default=Path("data/processed"))
    evals_preprocess.add_argument("--min-depth", type=int, default=12)
    evals_preprocess.add_argument("--max-policy-targets", type=int, default=8)
    evals_preprocess.add_argument("--value-scale", type=float, default=400.0)
    evals_preprocess.add_argument("--wdl-scale", type=float, default=380.0)
    evals_preprocess.add_argument("--wdl-draw-margin", type=float, default=100.0)
    evals_preprocess.add_argument("--policy-temperature", type=float, default=1.0)
    evals_preprocess.add_argument("--shard-size", type=int, default=65_536)
    evals_preprocess.add_argument("--compressed", action="store_true")
    evals_preprocess.add_argument("--max-positions", type=int, default=None)
    evals_preprocess.add_argument("--verbose", action="store_true", help="Print preprocessing progress to stderr")

    train_parser = subparsers.add_parser("train", help="Train the supervised CNN+Transformer")
    train_parser.add_argument("--data", type=Path, default=Path("data/processed"))
    train_parser.add_argument("--out", type=Path, default=Path("checkpoints/superchess.pt"))
    train_parser.add_argument("--data-format", choices=["evals", "games"], default="evals")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--value-weight", type=float, default=1.0, help="Weight of the value loss (eval distillation)")
    train_parser.add_argument("--workers", type=int, default=16)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--channels", type=int, default=256)
    train_parser.add_argument("--cnn-blocks", type=int, default=6)
    train_parser.add_argument("--transformer-layers", type=int, default=10)
    train_parser.add_argument("--heads", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--no-attention-bias", action="store_true", help="Disable the learned per-head square bias")
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--max-steps", type=int, default=None)
    train_parser.add_argument("--validation-fraction", type=float, default=0.05)
    train_parser.add_argument("--validation-seed", type=int, default=0)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a supervised checkpoint on NPZ shards")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--data-format", choices=["evals", "games"], default="evals")
    evaluate_parser.add_argument("--batch-size", type=int, default=512)
    evaluate_parser.add_argument("--value-weight", type=float, default=1.0)
    evaluate_parser.add_argument("--workers", type=int, default=16)
    evaluate_parser.add_argument("--device", default=None)
    evaluate_parser.add_argument("--no-amp", action="store_true")
    evaluate_parser.add_argument("--max-steps", type=int, default=None)
    evaluate_parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Load checkpoints missing current compatibility metadata",
    )

    search_parser = subparsers.add_parser("search", help="Run neural MCTS from a checkpoint")
    search_parser.add_argument("--checkpoint", type=Path, required=True)
    search_parser.add_argument("--fen", default="startpos")
    search_parser.add_argument("--simulations", type=int, default=128)
    search_parser.add_argument("--c-puct", type=float, default=1.5)
    search_parser.add_argument("--temperature", type=float, default=0.0)
    search_parser.add_argument("--eval-batch-size", type=int, default=8)
    search_parser.add_argument("--device", default=None)
    search_parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Load checkpoints missing current compatibility metadata",
    )

    gui_parser = subparsers.add_parser("gui", help="Launch the web GUI to play against the engine")
    gui_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/superchess.pt"))
    gui_parser.add_argument("--host", default="127.0.0.1")
    gui_parser.add_argument("--port", type=int, default=8000)
    gui_parser.add_argument("--device", default=None)
    gui_parser.add_argument(
        "--stockfish",
        default="stockfish",
        help="Stockfish executable name or path (default: search PATH and /usr/games)",
    )
    gui_parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab")
    gui_parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Load checkpoints missing current compatibility metadata",
    )

    gif_parser = subparsers.add_parser("gif", help="Render a Superchess replay JSON as an animated GIF")
    gif_parser.add_argument("--replay", type=Path, required=True)
    gif_parser.add_argument("--out", type=Path, required=True)
    gif_parser.add_argument("--board-size", type=int, default=560)
    gif_parser.add_argument("--orientation", choices=["white", "black"], default=None)

    args = parser.parse_args(argv)

    if args.command == "ccrl" and args.ccrl_command == "download":
        manifest = download_ccrl(
            args.out,
            min_elo=args.min_elo,
            prefer_engine_archives=not args.all_games,
            include_all_games=args.all_games,
            max_archives=args.max_archives,
            overwrite=args.overwrite,
            polite_delay_seconds=args.polite_delay,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "ccrl" and args.ccrl_command == "preprocess":
        stats = preprocess_raw_directory(
            args.raw,
            args.out,
            ratings_path=args.ratings,
            min_elo=args.min_elo,
            shard_size=args.shard_size,
            compressed=args.compressed,
            extract_archives=not args.no_extract,
            max_games=args.max_games,
            max_positions=args.max_positions,
            verbose=args.verbose,
        )
        print(json.dumps(asdict(stats), indent=2))
        return 0

    if args.command == "evals" and args.evals_command == "download":
        path = download_eval_dump(args.out, overwrite=args.overwrite)
        print(json.dumps({"download": str(path)}, indent=2))
        return 0

    if args.command == "evals" and args.evals_command == "preprocess":
        stats = preprocess_eval_directory(
            args.raw,
            args.out,
            EvalConfig(
                min_depth=args.min_depth,
                max_policy_targets=args.max_policy_targets,
                value_scale=args.value_scale,
                wdl_scale=args.wdl_scale,
                wdl_draw_margin=args.wdl_draw_margin,
                policy_temperature=args.policy_temperature,
                shard_size=args.shard_size,
                compressed=args.compressed,
            ),
            max_positions=args.max_positions,
            verbose=args.verbose,
        )
        print(json.dumps(asdict(stats), indent=2))
        return 0

    if args.command == "train":
        from superchess.model import ModelConfig

        model_config = ModelConfig(
            channels=args.channels,
            cnn_blocks=args.cnn_blocks,
            transformer_layers=args.transformer_layers,
            attention_heads=args.heads,
            dropout=args.dropout,
            attention_bias=not args.no_attention_bias,
        )

        if args.data_format == "evals":
            from superchess.training import train_distillation

            history = train_distillation(
                args.data,
                args.out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                value_weight=args.value_weight,
                num_workers=args.workers,
                device_name=args.device,
                model_config=model_config,
                compile_model=args.compile,
                amp=not args.no_amp,
                max_steps=args.max_steps,
                validation_fraction=args.validation_fraction,
                validation_seed=args.validation_seed,
            )
        else:
            from superchess.training import train_supervised

            history = train_supervised(
                args.data,
                args.out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.workers,
                device_name=args.device,
                model_config=model_config,
                compile_model=args.compile,
                amp=not args.no_amp,
                max_steps=args.max_steps,
                validation_fraction=args.validation_fraction,
                validation_seed=args.validation_seed,
            )
        print(json.dumps({"history": history, "checkpoint": str(args.out)}, indent=2))
        return 0

    if args.command == "evaluate":
        if args.data_format == "evals":
            from superchess.training import evaluate_distillation

            metrics = evaluate_distillation(
                args.checkpoint,
                args.data,
                batch_size=args.batch_size,
                value_weight=args.value_weight,
                num_workers=args.workers,
                device_name=args.device,
                amp=not args.no_amp,
                max_steps=args.max_steps,
                allow_legacy_policy=args.allow_legacy_checkpoint,
            )
        else:
            from superchess.training import evaluate_supervised

            metrics = evaluate_supervised(
                args.checkpoint,
                args.data,
                batch_size=args.batch_size,
                num_workers=args.workers,
                device_name=args.device,
                amp=not args.no_amp,
                max_steps=args.max_steps,
                allow_legacy_policy=args.allow_legacy_checkpoint,
            )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "search":
        import chess

        from superchess.mcts import NeuralMCTS, SearchConfig
        from superchess.training import load_model_checkpoint

        board = chess.Board() if args.fen == "startpos" else chess.Board(args.fen)
        model, _ = load_model_checkpoint(
            args.checkpoint,
            device_name=args.device,
            allow_legacy_policy=args.allow_legacy_checkpoint,
        )
        result = NeuralMCTS(
            model,
            SearchConfig(
                simulations=args.simulations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                evaluation_batch_size=args.eval_batch_size,
            ),
        ).search(board)
        payload = {
            "best_move": result.best_move.uci(),
            "visits": {move.uci(): count for move, count in result.visits.items()},
            "policy": {move.uci(): probability for move, probability in result.policy.items()},
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "gui":
        from superchess.gui import serve

        serve(
            args.checkpoint,
            host=args.host,
            port=args.port,
            device=args.device,
            open_browser=not args.no_browser,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
            stockfish_path=args.stockfish,
        )
        return 0

    if args.command == "gif":
        from superchess.gif import render_replay_gif

        payload = json.loads(args.replay.read_text(encoding="utf-8"))
        payload["board_size"] = args.board_size
        if args.orientation is not None:
            payload["orientation"] = args.orientation
        data = render_replay_gif(payload)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(data)
        print(json.dumps({"gif": str(args.out), "bytes": len(data)}, indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())