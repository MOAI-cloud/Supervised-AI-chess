# Superchess

<p align="center">
	<img src="superchess.png" alt="Superchess logo" width="420" />
</p>

A supervised CNN+Transformer chess engine: Stockfish-evaluation distillation (or
CCRL game training), neural MCTS, engine-vs-engine strength measurement, and a
Lichess-style GUI.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,extract,evals]"
```

Install a PyTorch build that supports your GPU using the
[official installation selector](https://pytorch.org/get-started/locally/).
The RTX 5090 checks here used PyTorch 2.12.0+cu132; older CUDA wheels may not
support Blackwell. Pin the selected build for reproducible runs.

Distil the Lichess Stockfish evaluation database (the recommended path):

```bash
superchess evals download --out data/raw
superchess evals preprocess --raw data/raw --out data/curated-v1 --workers 8 --verbose
superchess evals audit --data data/curated-v1 --legal-samples 0 --out benchmarks/curated-v1-audit.json
superchess train --data data/curated-v1 --out checkpoints/superchess.pt --epochs 2 --seed 0
superchess match checkpoints/superchess.pt stockfish:5 --games 40 --simulations 400
superchess gui --checkpoint checkpoints/superchess.pt
```

Or train from CCRL games:

```bash
superchess ccrl download --out data/raw --min-elo 3500
superchess ccrl preprocess --raw data/raw --out data/processed --min-elo 3500
superchess train --data data/processed --data-format games --out checkpoints/superchess.pt --epochs 1
```

## Why offline metrics can lie (and what was fixed)

A training run whose validation accuracy improves can still play *worse*. Two
concrete causes were found in this codebase and are now guarded against:

1. **Constant clock planes.** The Lichess eval dump stores four-field FENs, so
   every training position gets `halfmove=0, fullmove=1`. A model configured
   with the 20-plane input therefore never sees a non-zero clock during
   training but receives real clocks at play time — an out-of-distribution input
   that corrupts the stem activations more and more as the game progresses,
   while every offline metric (computed on the constant planes) keeps improving.
   `ModelConfig.input_channels` now defaults to the 18 clock-free planes, the
   trainer scans the shards and **refuses** to train a clock-aware model on data
   whose clock planes are constant, and search/GUI feed the model only the planes
   it was trained on.
2. **Near-uniform policy targets.** The old pipeline turned multi-PV scores into
   `softmax(cp / 400)`: the best move of a 5-PV position received ~0.36
   probability on average (median 0.34), almost indistinguishable from the 4th
   or 5th choice. Longer training fit that flat target *better* (lower KL, higher
   top-1 vs the argmax) while producing a flatter prior for MCTS. Targets are now
   shaped at training time from raw scores (see below); with the default
   temperature the best move receives ~0.52 on average and a 100 cp inferior
   move ~0.05.

The real metric is playing strength, so `superchess match` (below) is part of
the standard workflow. Validation `policy_kl`, `value_mae` and top-1/top-5
accuracy versus Stockfish's best move are still logged as diagnostics.

## Data format

`superchess evals preprocess` writes `evals-v2` shards holding **raw**
side-to-move scores instead of pre-shaped probabilities. New builds have a frozen
manifest and separate `train/`, `validation/`, and `test/` directories:

| array | dtype | meaning |
|---|---|---|
| `boards` | `uint8[N,146]` | bit-packed 18 planes + 2 (uninformative) clock bytes |
| `score`, `depth` | `int16`, `uint16` | deepest valid eval's best-line score and depth |
| `policy_indices` | `int16[N,K]` | 64×73 policy index of each multi-PV move (−1 padding) |
| `policy_scores` | `int16[N,K]` | score of each multi-PV move (same eval) |
| `policy_depth` | `uint16` | depth of the selected coherent multi-PV eval |
| `value_knodes`, `policy_knodes` | `uint64[N]` | teacher work recorded for the selected snapshots |
| `position_hash` | `uint8[N,16]` | split group, ignoring clock/en-passant variants |

Scores are centipawns clamped to ±11 999, or `±(12 000 + 500 − n)` for mate in
`n` so that faster mates order first. Preprocessing runs in a process pool
(`--workers`), writes deterministic output, and needs no target hyper-parameters.
Legacy shards (`wdl`, `policy_probs`) remain readable, but reconstructed scores
are approximate: saturation loses mate detail, and policy score differences do
not recover an unknown absolute anchor from a different-depth evaluation.

### Data quality and reproducibility

Preprocessing rejects invalid standard-chess boards, malformed/bound-only scores,
and illegal PV moves. The default validates eight plies of each PV; use
`--validate-pv-plies 0` for the full line. Duplicate moves no longer consume
target slots. Wider policy supervision must agree with the deepest valid
snapshot's best move, be within `--policy-max-depth-gap 4`, and satisfy
`--max-policy-value-gap 0.1` in mapped expected-score units. Otherwise the deepest
snapshot supplies both heads. These thresholds are ablation settings, not
universally optimal quality scores.

New builds deduplicate exact packed inputs on disk, retain the strongest whole
sample, hash-group clock/en-passant variants into immutable splits, and publish
atomically. They refuse to overwrite a nonempty directory. Budget disk for
staging plus final shards; `--no-deduplicate` gives up exact duplicate removal
and hash-order mixing. Input planes and existing weights are unchanged.
`--max-records` supports bounded previews without rebuilding the full dump.

Manifests record configuration, source fingerprints, library versions, rejection
reasons, checksums, and dataset identity. Training honors the frozen splits,
verifies input checksums, and records provenance, seeds, runtime, and RNG state.
Audit reports state their coverage, including what old shards cannot prove.

**Measured:** 505,522 sampled existing rows contained 222 invalid piece/pawn-count
positions. A new 20,000-record preview retained 19,812 positions; all passed
exhaustive policy-legality and split/checksum checks. Serial and four-worker
builds had identical identities. No full-size retraining or Elo gain is claimed.

See [docs/training-methodology.md](docs/training-methodology.md) for the complete
protocol, measured reports, controlled experiments, and position-only holdout
limitations. A checkpoint trained on the old dump is not made uncontaminated
by assigning that dump new holdout hashes.

## Targets, model, and training recipe

**Targets** (`superchess/targets.py`, all configurable from `superchess train`):

- Expected score `W(cp) = 1 / (1 + exp(−cp / cp_scale))` with the Lichess win
  model `cp_scale ≈ 271.6` (the mapping used by DeepMind's searchless-chess
  work). The GUI's centipawn display inverts exactly this mapping.
- **Value**: a categorical head over 64 bins of `W`, trained with HL-Gauss
  targets (σ = 0.75 bin widths; Farebrother et al. 2024). The search value is
  the bin expectation rescaled to `[−1, 1]`.
- **Policy**: `softmax(W(cp_move) / policy_temperature)` over the multi-PV moves
  (default temperature 0.04 in expected-score units, plus a tiny raw-cp tie-break
  so mates still order above large material edges). Moves outside the PV list get
  zero mass; `policy_kl` reports the loss above the target-entropy floor.

**Model** (`superchess/model.py`):

- CNN stem (2 residual blocks by default) → 10 pre-norm transformer layers over
  the 64 squares (RMSNorm, SwiGLU) with a learned static square-pair bias **and
  smolgen** (Lc0's position-dependent attention logits, shared generator).
- **Attention policy head** (Lc0): logits are `query_from · key_to` for every
  square pair plus learned promotion offsets, scattered into the 64×73 layout.
- **Structural move mask**: moves that are geometrically impossible for the
  piece on the from-square (or leave the board) are masked to `−1e4` inside the
  model, so capacity is spent ranking candidate moves rather than learning
  legality. This is a strict superset of legality; search still masks to legal
  moves.
- Categorical value head: per-square projection → flatten → MLP → 64 bins.
- Default size 25.3M parameters; `--channels/--transformer-layers/--heads`
  scale it. Checkpoints from the previous architecture still load (the GUI,
  `search` and `match` accept them) — they are recognised by their metadata.

**Training** (`superchess/training.py`): AdamW (β₂ = 0.95, decoupled weight decay
on matrices only), linear warm-up then cosine decay to 5 %, bf16 autocast,
gradient clipping, `torch.compile` on by default (≈1.8× faster; `--no-compile`
to disable), shards mixed across workers before shuffling (`--mix-shards`), an
**EMA of the weights** (decay 0.9999) that is what gets saved and validated, and
`--resume` from the optimizer state stored in the last checkpoint. Every epoch
writes `<out>.pt` (+ `.json` history) and `<stem>-best.pt` on validation
improvement.

On an RTX 5090 the default model trains at ≈15 k positions/s with
`--batch-size 2048` (≈17 GB, ≈7 h per pass over the 393 M-position dump). A
reasonable strong run:

```bash
superchess train --data data/curated-v1 --out checkpoints/superchess.pt \
  --epochs 3 --batch-size 2048 --lr 5e-4 --workers 16 --seed 0
# later: continue where it stopped
superchess train --data data/curated-v1 --out checkpoints/superchess.pt --epochs 4 \
  --resume checkpoints/superchess.pt --seed 0
```

Use `--batch-size 1024` on GPUs with less memory. Larger models
(`--channels 384 --transformer-layers 12 --heads 12`) need batch 1024 on 32 GB.

## Measuring strength

```bash
superchess match checkpoints/new.pt checkpoints/old.pt --games 100 --simulations 400
superchess match checkpoints/new.pt stockfish:6 --games 60 --simulations 800 --pgn games.pgn
```

Games are paired (each opening is played with both colours), openings are
diversified by sampling the network policy for `--opening-plies` moves (or from
`--openings file.epd`), and the report gives W/D/L, the Elo difference with a
conservative 95% Hoeffding confidence bound and an approximate likelihood of
superiority (LOS). Games sharing the same opening, including repeated uses of
an opening file entry, are grouped for the bound. This avoids zero-width
intervals after all draws or all wins and avoids treating the two colours as
independent observations. Independence is still assumed between opening groups;
these are fixed-sample bounds, not an SPRT or an anytime-valid stopping rule.
The legacy LOS approximation ignores pairing and must not be used as an
acceptance gate. Elo output is clipped at +/-1000; endpoints at that limit
mean the bound is not usefully finite, not a precise 1000-Elo confidence limit.
`stockfish:<1-8>`
uses the exact Lichess fishnet presets described below. Always compare
checkpoints this way before trusting an offline-metric improvement.

Use `--json report.json` to retain the full configuration, checkpoint paths,
game records, per-engine move time, and neural-work counters, and `--pgn` for
replayable games. These measure fixed-simulation play: equal simulations do not
guarantee equal wall time or equal numbers of uncached neural evaluations.

For frozen datasets, preprocessing reserves 2% validation and 2% test groups by
default; training never consumes test shards. Choose the fractions and split
seed at build time. Evaluate a split explicitly with
`superchess evaluate --checkpoint checkpoints/superchess.pt --data data/curated-v1/validation`.
For legacy flat shards only, the old `--validation-fraction`/`--validation-seed`
training options still apply, with a warning that shard separation does not
prove position-disjoint validation.

## Search experiments

PUCT remains the default. The model, targets, training settings, and saved
checkpoint weights are unchanged by these experiments.

**Batch correctness.** Repeated selection of one pending, unexpanded edge no
longer consumes several simulations or overwrites its child repeatedly. The
duplicate reservation is cancelled, the pending batch is evaluated, and search
continues with real feedback. In the forced-move regression, four simulations
previously stopped at the same first-ply child; they now give that child three
reply visits. Failed inference releases outstanding virtual loss before
propagating the exception. A fresh zero-simulation search uses network priors
instead of move-generation order. `search` JSON reports completed simulations,
neural positions/batches, cache hits, collisions, depth, and elapsed seconds.
Flushing on collision can reduce batch occupancy; node accounting is a
correctness property, not a claim of improved wall-clock strength.

**`sequential_halving` (established-method baseline).** The root considers up to
`--root-candidates` highest-prior legal moves, bounded by the simulation budget.
Candidates receive balanced new visits within each round; elimination happens
only after that round's evaluations are backed up. Ranking uses
`log(prior) + root_value_scale * (50 + max_new_visits) * Q`, with scale 0.1
and Q in the root player's perspective. Interior nodes still use PUCT.
Finalists are selected by this score, not by visit-count ties. This is a
deterministic hybrid, **not Full Gumbel AlphaZero/MuZero**: it has no Gumbel
sampling, mixed-value completion, per-node min-max scaling, or Gumbel interior
selection, and inherits none of that algorithm's policy-improvement claims.

**`value_rescue` (unvalidated research candidate).** The dataset's deepest value
evaluation can disagree with its shallower multi-PV policy supervision. The
testable hypothesis is that, if every explored candidate scores substantially
below the root's neural value, a useful move may lie outside the policy shortlist.
This variant initially probes each shortlisted move once. At round boundaries,
when `max(candidate_Q) < root_network_value - root_rescue_margin`, it keeps the
best half and fills available slots with previously unconsidered moves in prior
order. Challengers consume the existing budget and are evaluated before another
elimination. The default margin is 0.15 in `[-1, 1]` value units. This is not an
uncertainty estimate: root-value error can trigger the same condition. Search
reports both candidates reopened and whether a reopened move was selected.

Disable just the rescue gate with margin 2 while retaining its initial probe
schedule. This is the appropriate ablation; comparing rescue directly with
ordinary halving also changes that schedule:

```bash
superchess search --checkpoint checkpoints/superchess_v4-best.pt \
  --root-selection sequential_halving --simulations 64 --root-candidates 8

superchess match checkpoints/superchess_v4-best.pt checkpoints/superchess_v4-best.pt \
  --root-selection-a value_rescue --root-selection-b value_rescue \
  --root-rescue-margin-b 2 --root-candidates 8 --games 12 \
  --simulations 64 --eval-batch-size 16 --opening-plies 8 \
  --max-plies 240 --seed 29 --device cuda \
  --json benchmarks/v4-rescue-vs-control.json --pgn benchmarks/v4-rescue-vs-control.pgn
```

### Pilot evidence (2026-09-12)

Identical v4-best weights, RTX 5090, PyTorch 2.12.0+cu132, 64 simulations/move,
batch cap 16, 8 root candidates, 8 opening plies, and 240 maximum played plies.
The first two pilots use seed 17 (four opening pairs); the isolated gate control
uses seed 29 (six pairs). Commands were run with `OMP_NUM_THREADS=4` and
`OPENBLAS_NUM_THREADS=1`. Time ratios below are aggregate measured move times,
not results from an equal-time match.

| A versus B | A W/D/L | A score | A/B time | Full report |
|---|---:|---:|---:|---|
| Halving vs PUCT | 4/4/0 | 75.0% | 1.35x | [JSON](benchmarks/v4-halving-vs-puct.json) |
| Rescue vs halving | 4/2/2 | 62.5% | 1.02x | [JSON](benchmarks/v4-rescue-vs-halving.json) |
| Rescue vs gate-disabled control | 2/8/2 | 50.0% | 0.99x | [JSON](benchmarks/v4-rescue-vs-control.json) |

No game ended at the artificial ply limit. **None of these small pilots
establishes an Elo improvement.** Halving's positive fixed-simulation result
comes with a material time cost. In the isolated rescue test, 111 candidates
were reopened, none was selected, and all six paired games followed identical
move sequences with engine identities swapped. The gate therefore showed no
playing-strength benefit on this pilot. Both selectors remain opt-in.

### Prior art and limits

- [Karnin et al., 2013](https://proceedings.mlr.press/v28/karnin13.html): sequential halving for best-arm identification.
- [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO) and [DeepMind Mctx](https://github.com/google-deepmind/mctx): Gumbel planning and policy/value root scoring.
- [Cazenave, 2021](https://arxiv.org/abs/2104.04278): batched MCTS, virtual mean, separate inference caching, and budget-allocation heuristics.
- [Lanctot et al., 2014](https://arxiv.org/abs/1406.0486): implicit minimax backups, which are not a new architecture proposal here.
- [Ruoss et al., 2024](https://arxiv.org/abs/2402.04494): large-scale action-value transformer distillation for chess; adding an action-value head alone would not establish novelty.

This was a targeted literature check, not an exhaustive novelty search.
Value-gated candidate reopening is a repository-specific research hypothesis,
not a demonstrated new state of the art. No equal-wall-clock tournament,
comparison against current full-strength Stockfish/Lc0, or new architecture
training run has been performed. Before promoting an experiment, predeclare a
larger held-out opening suite and compute limits, run the gate-disabled ablation,
and require a positive strength bound as well as acceptable elapsed time.

## Play in the GUI

The GUI auto-detects an installed `stockfish` executable from `PATH` (including
the usual `/usr/games` location). Launch it normally:

```bash
superchess gui --checkpoint checkpoints/superchess.pt
```

If Stockfish is not on `PATH`, pass it explicitly:

```bash
superchess gui --checkpoint checkpoints/superchess.pt --stockfish /path/to/stockfish
```

A Lichess-style analysis board opens automatically at `http://127.0.0.1:8000/`.
It features crisp SVG pieces, drag-and-drop moves with legal-move hints, a live
evaluation gauge, MultiPV engine lines, an evaluation graph, move and material
tracking, editable FEN/PGN fields, responsive layouts, board themes, and an
in-board promotion chooser. Select **Play** to reveal a prominent White, Black,
or Both side picker; choosing a side keeps the current position and lets the bot
move whenever it is its turn. Engine strength, PV length, exploration, arrows,
flipping, undo, and hints remain configurable. The backend uses only the standard
library plus python-chess, so no extra dependencies are required.

<p align="center">
	<img src="chess.gif" alt="Superchess arena with Superchess as White" width="48%" />
	<img src="chess2.gif" alt="Superchess arena with Superchess as Black" width="48%" />
</p>

Select **Arena** to run an automatic **Superchess vs Stockfish** match. Choose
which color Superchess plays and a Lichess Stockfish level from 1 through 8,
then start, pause, or resume the game. These are the pinned
`lichess-org/fishnet` move presets (the time and depth limits are both sent, so
the first reached limit stops the search):

| Lichess level | Skill Level | Move time | Depth |
|---:|---:|---:|---:|
| 1 | -9 | 50 ms | 5 |
| 2 | -5 | 100 ms | 5 |
| 3 | -1 | 150 ms | 5 |
| 4 | 3 | 200 ms | 5 |
| 5 | 7 | 300 ms | 5 |
| 6 | 11 | 400 ms | 8 |
| 7 | 16 | 500 ms | 13 |
| 8 | 20 | 1000 ms | 22 |

Fishnet uses one thread, 16 MB hash, one PV, and disables `UCI_LimitStrength`.
The Arena panel shows these exact requested values. Stockfish 16 advertises a
skill range of 0–20, so its effective skill for levels 1–3 is shown explicitly
as 0 while the exact Lichess time/depth limits are retained.

The vertical evaluation gauge and evaluation graph are always produced by
Superchess, including immediately after Stockfish moves. Stockfish's score is
used only for its own search-line display. Opening names come from a pinned copy
of Lichess's 3,790-position CC0 ECO database; classification walks backward
through the real game history, so the last valid opening remains stable in the
middlegame and common transpositions are recognized.

Arena games can be downloaded as tagged PGN, `superchess-replay-v1` JSON, or a
ready-to-share animated GIF. GIF frames use the same Lichess-style board theme,
Cburnett SVG pieces, orientation, coordinates, check marker, and last-move
highlighting as the browser GUI. They also include player names, the stable
opening name, and the Superchess evaluation gauge. Rendering never reruns either
engine. A saved replay can also be rendered later from the command line:

```bash
superchess gif --replay game.replay.json --out game.gif
```

The downloader first tries the smaller per-engine commented archives for engines above the Elo cutoff and falls back to the full commented archive if needed. For smoke tests, limit work before touching the full database:

```bash
superchess ccrl download --out data/raw --min-elo 3500 --max-archives 2
superchess ccrl preprocess --raw data/raw --out data/processed --min-elo 3500 --max-games 100 --verbose
```

## Design

- Perspective-normalized board planes keep the model invariant to side to move; only the 18 clock-free planes are used unless the data carries real move clocks.
- AlphaZero-style 4672-action policy layout (queen-like moves, knights, underpromotions), produced by a from/to attention head and structurally masked inside the model.
- Preprocessing writes NPZ shards with packed bit planes and raw Stockfish scores for fast sequential reads; add `--compressed` when disk space matters more than load speed.
- Training shapes targets on the GPU, unpacks batches in the loader workers, and uses bf16 AMP, channels-last tensors, `torch.compile`, and weight EMA.
- MCTS masks policy logits to legal moves (with an optional softmax temperature), caches evaluations by transposition key, reuses subtrees between moves, and evaluates positions on GPU when PyTorch CUDA is available.
- Playing strength is measured with paired engine-vs-engine matches (`superchess match`) rather than inferred from offline accuracy.
