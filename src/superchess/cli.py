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
    evals_preprocess.add_argument("--min-knodes", type=int, default=0)
    evals_preprocess.add_argument("--policy-max-depth-gap", type=int, default=4)
    evals_preprocess.add_argument("--max-policy-value-gap", type=float, default=0.1, help="Maximum expected-score gap between value and policy snapshots")
    evals_preprocess.add_argument("--validate-pv-plies", type=int, default=8, help="Check this many PV plies (0 checks the complete line)")
    evals_preprocess.add_argument("--validation-fraction", type=float, default=0.02)
    evals_preprocess.add_argument("--test-fraction", type=float, default=0.02)
    evals_preprocess.add_argument("--split-seed", type=int, default=0)
    evals_preprocess.add_argument("--no-deduplicate", action="store_true", help="Stream without the disk-backed strongest-evidence deduplication pass")
    evals_preprocess.add_argument("--max-policy-targets", type=int, default=8)
    evals_preprocess.add_argument("--shard-size", type=int, default=65_536)
    evals_preprocess.add_argument("--compressed", action="store_true")
    evals_preprocess.add_argument("--max-positions", type=int, default=None)
    evals_preprocess.add_argument("--max-records", type=int, default=None, help="Read at most this many raw records for a reproducible preview")
    evals_preprocess.add_argument("--workers", type=int, default=1, help="Parallel JSON/move-generation workers")
    evals_preprocess.add_argument("--verbose", action="store_true", help="Print preprocessing progress to stderr")

    evals_audit = evals_subparsers.add_parser("audit", help="Audit shard integrity, split isolation, targets, and sampled legality")
    evals_audit.add_argument("--data", type=Path, required=True)
    evals_audit.add_argument("--max-shards", type=int, default=None)
    evals_audit.add_argument("--legal-samples", type=int, default=128, help="Legal positions checked per shard (0 checks all)")
    evals_audit.add_argument("--skip-checksums", action="store_true")
    evals_audit.add_argument("--out", type=Path, default=None, help="Write the audit report to JSON")

    train_parser = subparsers.add_parser("train", help="Train the supervised CNN+Transformer")
    train_parser.add_argument("--data", type=Path, default=Path("data/processed"))
    train_parser.add_argument("--out", type=Path, default=Path("checkpoints/superchess.pt"))
    train_parser.add_argument("--data-format", choices=["evals", "games"], default="evals")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=2048)
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--warmup-steps", type=int, default=None, help="Default: min(2000, 5%% of total steps)")
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--ema-decay", type=float, default=0.9999, help="Weight EMA decay (0 disables)")
    train_parser.add_argument("--value-weight", type=float, default=1.0, help="Weight of the value loss")
    train_parser.add_argument("--workers", type=int, default=16)
    train_parser.add_argument("--mix-shards", type=int, default=4, help="Shards mixed per worker before shuffling")
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--resume", type=Path, default=None, help="Continue from a checkpoint with training state")
    train_parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Board planes fed to the model (default: 18; 20 only when the data has real move clocks)",
    )
    train_parser.add_argument("--channels", type=int, default=256)
    train_parser.add_argument("--cnn-blocks", type=int, default=2)
    train_parser.add_argument("--transformer-layers", type=int, default=10)
    train_parser.add_argument("--heads", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--no-attention-bias", action="store_true", help="Disable the learned per-head square bias")
    train_parser.add_argument("--smolgen-hidden", type=int, default=32, help="Smolgen per-square compression (0 disables)")
    train_parser.add_argument("--smolgen-gen", type=int, default=256)
    train_parser.add_argument("--policy-head", choices=["attention", "planes"], default="attention")
    train_parser.add_argument("--value-bins", type=int, default=64, help="Categorical value head resolution")
    train_parser.add_argument("--no-structural-mask", action="store_true", help="Do not mask geometrically impossible moves")
    train_parser.add_argument("--cp-scale", type=float, default=None, help="Centipawns per logit of expected score")
    train_parser.add_argument("--policy-temperature", type=float, default=None, help="Softmax temperature over expected score")
    train_parser.add_argument("--policy-cp-tiebreak", type=float, default=None)
    train_parser.add_argument("--hl-gauss-sigma", type=float, default=None, help="HL-Gauss sigma in bin widths")
    train_parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (about 1.8x slower)")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--max-steps", type=int, default=None)
    train_parser.add_argument("--validation-fraction", type=float, default=0.02)
    train_parser.add_argument("--validation-seed", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=0, help="Model initialization and epoch/worker data-order seed")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a supervised checkpoint on NPZ shards")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--data-format", choices=["evals", "games"], default="evals")
    evaluate_parser.add_argument("--batch-size", type=int, default=2048)
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
    search_parser.add_argument("--policy-softmax-temperature", type=float, default=1.0)
    search_parser.add_argument("--eval-batch-size", type=int, default=8)
    search_parser.add_argument("--root-selection", choices=["puct", "sequential_halving", "value_rescue"], default="puct")
    search_parser.add_argument("--root-candidates", type=int, default=16)
    search_parser.add_argument("--root-value-scale", type=float, default=0.1)
    search_parser.add_argument("--root-rescue-margin", type=float, default=0.15)
    search_parser.add_argument("--device", default=None)
    search_parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Load checkpoints missing current compatibility metadata",
    )

    match_parser = subparsers.add_parser("match", help="Play engine-vs-engine games and estimate the Elo difference")
    match_parser.add_argument("engine_a", help="Checkpoint path or stockfish:<level 1-8>")
    match_parser.add_argument("engine_b", help="Checkpoint path or stockfish:<level 1-8>")
    match_parser.add_argument("--games", type=int, default=20, help="Total games (paired: each opening with both colours)")
    match_parser.add_argument("--simulations", type=int, default=256)
    match_parser.add_argument("--c-puct", type=float, default=1.5)
    match_parser.add_argument("--fpu-reduction", type=float, default=0.3)
    match_parser.add_argument("--eval-batch-size", type=int, default=32)
    match_parser.add_argument("--policy-softmax-temperature", type=float, default=1.0)
    match_parser.add_argument("--root-selection-a", choices=["puct", "sequential_halving", "value_rescue"], default="puct")
    match_parser.add_argument("--root-selection-b", choices=["puct", "sequential_halving", "value_rescue"], default="puct")
    match_parser.add_argument("--root-candidates", type=int, default=16)
    match_parser.add_argument("--root-value-scale", type=float, default=0.1)
    match_parser.add_argument("--root-rescue-margin", type=float, default=0.15)
    match_parser.add_argument("--root-rescue-margin-b", type=float, default=None, help="Override B's rescue margin (2 disables the gate for an ablation)")
    match_parser.add_argument("--opening-plies", type=int, default=6, help="Random opening moves sampled from the network policy")
    match_parser.add_argument("--opening-temperature", type=float, default=1.0)
    match_parser.add_argument("--openings", type=Path, default=None, help="File with one FEN/EPD per line")
    match_parser.add_argument("--max-plies", type=int, default=400, help="Adjudicate a draw after this many plies")
    match_parser.add_argument("--seed", type=int, default=0)
    match_parser.add_argument("--device", default=None)
    match_parser.add_argument("--stockfish", default="stockfish", help="Stockfish executable for stockfish:<level> engines")
    match_parser.add_argument("--pgn", type=Path, default=None, help="Write all games to this PGN file")
    match_parser.add_argument("--json", type=Path, default=None, help="Write the full report, configuration, and games to JSON")
    match_parser.add_argument("--quiet", action="store_true", help="Only print the final summary")
    match_parser.add_argument(
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
                min_knodes=args.min_knodes,
                policy_max_depth_gap=args.policy_max_depth_gap,
                max_policy_value_gap=args.max_policy_value_gap,
                validate_pv_plies=args.validate_pv_plies,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                split_seed=args.split_seed,
                deduplicate=not args.no_deduplicate,
                max_policy_targets=args.max_policy_targets,
                shard_size=args.shard_size,
                compressed=args.compressed,
            ),
            max_positions=args.max_positions,
            verbose=args.verbose,
            workers=args.workers,
            max_records=args.max_records,
        )
        print(json.dumps(asdict(stats), indent=2))
        return 0

    if args.command == "evals" and args.evals_command == "audit":
        from superchess.data_quality import audit_eval_dataset

        report = audit_eval_dataset(args.data, max_shards=args.max_shards, legal_samples=args.legal_samples,
                                    verify_checksums=not args.skip_checksums)
        text = json.dumps(report, indent=2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if report["valid"] else 1

    if args.command == "train":
        from superchess.encoding import BOARD_CHANNELS, LEGACY_BOARD_CHANNELS
        from superchess.model import ModelConfig
        from superchess.targets import TargetConfig
        from superchess.training import clock_planes_informative, train_distillation, train_supervised

        input_channels = args.input_channels
        if input_channels is None:
            shards = sorted(args.data.glob("shard-*.npz"))
            informative = args.data_format == "games" and bool(shards) and clock_planes_informative(shards)
            input_channels = BOARD_CHANNELS if informative else LEGACY_BOARD_CHANNELS

        model_config = ModelConfig(
            input_channels=input_channels,
            channels=args.channels,
            cnn_blocks=args.cnn_blocks,
            transformer_layers=args.transformer_layers,
            attention_heads=args.heads,
            dropout=args.dropout,
            attention_bias=not args.no_attention_bias,
            smolgen_hidden=args.smolgen_hidden,
            smolgen_gen=args.smolgen_gen,
            policy_head=args.policy_head,
            value_bins=args.value_bins,
            structural_mask=not args.no_structural_mask,
        )
        target_overrides = {
            name: value
            for name, value in (
                ("cp_scale", args.cp_scale),
                ("policy_temperature", args.policy_temperature),
                ("policy_cp_tiebreak", args.policy_cp_tiebreak),
                ("hl_gauss_sigma", args.hl_gauss_sigma),
            )
            if value is not None
        }
        target_config = TargetConfig(**target_overrides)
        trainer = train_distillation if args.data_format == "evals" else train_supervised
        history = trainer(
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
            target_config=target_config,
            compile_model=not args.no_compile,
            amp=not args.no_amp,
            max_steps=args.max_steps,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
            seed=args.seed,
            ema_decay=args.ema_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            mix_shards=args.mix_shards,
            resume=args.resume,
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
                policy_softmax_temperature=args.policy_softmax_temperature,
                root_selection=args.root_selection,
                root_candidates=args.root_candidates,
                root_value_scale=args.root_value_scale,
                root_rescue_margin=args.root_rescue_margin,
            ),
        ).search(board)
        payload = {
            "best_move": result.best_move.uci(),
            "visits": {move.uci(): count for move, count in result.visits.items()},
            "policy": {move.uci(): probability for move, probability in result.policy.items()},
            "stats": asdict(result.stats) if result.stats is not None else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "match":
        import sys

        from superchess.match import MatchConfig, load_openings, run_match, write_pgn

        config = MatchConfig(
            games=args.games,
            simulations=args.simulations,
            c_puct=args.c_puct,
            fpu_reduction=args.fpu_reduction,
            evaluation_batch_size=args.eval_batch_size,
            policy_softmax_temperature=args.policy_softmax_temperature,
            root_selection_a=args.root_selection_a,
            root_selection_b=args.root_selection_b,
            root_candidates=args.root_candidates,
            root_value_scale=args.root_value_scale,
            root_rescue_margin=args.root_rescue_margin,
            root_rescue_margin_b=args.root_rescue_margin_b,
            opening_plies=args.opening_plies,
            opening_temperature=args.opening_temperature,
            max_plies=args.max_plies,
            seed=args.seed,
            device=args.device,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
            stockfish_command=args.stockfish,
        )
        openings = load_openings(args.openings) if args.openings is not None else None

        def report(record, summary):
            if args.quiet:
                return
            print(
                f"game {record.round}: {record.white} vs {record.black} {record.result} "
                f"({record.termination}, {record.plies} plies) | {summary.engine_a}: "
                f"+{summary.wins} ={summary.draws} -{summary.losses} elo {summary.elo:+.0f} "
                f"[{summary.elo_low:+.0f}, {summary.elo_high:+.0f}] LOS {summary.los:.1%}",
                file=sys.stderr,
                flush=True,
            )

        summary = run_match(args.engine_a, args.engine_b, config, openings=openings, on_game=report)
        if args.pgn is not None:
            write_pgn(args.pgn, summary.games)
        payload = summary.to_dict()
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["games"] = len(summary.games)
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