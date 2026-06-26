# Superchess

A supervised CNN+Transformer chess-engine project for CCRL preprocessing, policy/value training, and neural MCTS.

The data path is optimized around CCRL 40/15 commented PGNs and rating metadata. The default cutoff is 3500 Elo, using the current CCRL rating list to decide which engine games to keep.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,extract]"
```

Install PyTorch with the CUDA wheel that matches your driver, for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then run:

```bash
superchess ccrl download --out data/raw --min-elo 3500
superchess ccrl preprocess --raw data/raw --out data/processed --min-elo 3500
superchess train --data data/processed --out checkpoints/superchess.pt --epochs 1
superchess search --checkpoint checkpoints/superchess.pt --fen "startpos" --simulations 128
```

Training holds out 5% of NPZ shards for validation by default and reports `val_*`
metrics in the checkpoint JSON. Use `--validation-fraction 0` to disable the
split, or adjust `--validation-seed` for a different deterministic shard holdout.
Checkpoints trained before the policy square-order fix should be retrained.

## Play in the GUI

Launch the graphical arena to play against the trained engine in your browser:

```bash
superchess gui --checkpoint checkpoints/superchess.pt
```

A polished single-page board opens automatically at `http://127.0.0.1:8000/`. It
features drag-and-drop moves with legal-move hints, a live evaluation bar driven
by the value head, engine analysis (top moves by visit count), move history,
captured-material tracking, adjustable engine strength (simulations) and
exploration (temperature), board flipping, undo, and hints. Choose to play
White, Black, or watch the engine play both sides. The backend uses only the
standard library plus python-chess, so no extra dependencies are required.

The downloader first tries the smaller per-engine commented archives for engines above the Elo cutoff and falls back to the full commented archive if needed. For smoke tests, limit work before touching the full database:

```bash
superchess ccrl download --out data/raw --min-elo 3500 --max-archives 2
superchess ccrl preprocess --raw data/raw --out data/processed --min-elo 3500 --max-games 100 --verbose
```

## Design

- Perspective-normalized board planes keep the model invariant to side to move.
- AlphaZero-style 4672-action policy targets cover legal queen-like moves, knights, and underpromotions.
- Preprocessing writes NPZ shards with packed bit planes for fast sequential reads; add `--compressed` when disk space matters more than load speed.
- Training unpacks batches vectorized in the collate function and uses AMP, channels-last tensors, and optional `torch.compile`.
- MCTS masks policy logits to legal moves and evaluates positions on GPU when PyTorch CUDA is available.
