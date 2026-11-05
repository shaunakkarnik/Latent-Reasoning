# SpatialLadder training and SPBench-SI evaluation

Run these commands **on the server**, from the repository root. Install the
appropriate CUDA PyTorch build first, then the project extras:

```bash
pip install -e '.[data,teacher]'
# Optional: put the Hugging Face model cache on your server's scratch disk.
export HF_HOME=/path/to/scratch/huggingface

latent-geometry prepare-spatialladder --output-dir data/spatialladder \
  --val-fraction 0.1 --seed 42
```

No local download or SCP is needed. Preparation downloads annotations and image
archives directly from Hugging Face, filters `data_type == "single_image"`, and
extracts only the images needed for those questions and SPBench-SI. SpatialLadder's
full archive is roughly 2.6 GB; allow space for both archives and extracted images.
Archives remain under `data/spatialladder/downloads`. Completed downloads and
extracted images are reused after interruption. Once `spatial.jsonl` exists, use a
new output directory to change the split.

The output contains:

- `spatial.jsonl`: one manifest containing train, val, and test examples.
- `preparation.json`: dataset commit IDs, split seed, validation rooms, and counts.
- `images/`: selected images, with paths relative to the manifest.

The current annotations contain 5,928 single-image training questions. With seed
42 and a 10% room holdout, the split is 5,252 train / 676 val / 1,009 test. The
holdout fraction applies to **rooms**, so the question fraction can differ.
All rescans (`sceneXXXX_00`, `sceneXXXX_01`, etc.) stay together. Preparation
rejects room overlap with SPBench-SI and identical image content across splits.
Use `--training-revision COMMIT` and `--benchmark-revision COMMIT` to reproduce
downloads; resolved revisions are recorded in `preparation.json`.

Questions include options and SPBench's default direct-answer instruction.
All three models receive the same prompts, images, and split. The original model
receives no extra latent tokens or embedding resizing. The answer-only SFT baseline
uses answer-token cross entropy with the same frozen vision encoder/merger and
trainable language model as the latent runs.

## Extract and train

Extract features once, on the server GPU:

```bash
latent-geometry extract --config configs/default.json \
  --manifest data/spatialladder/spatial.jsonl \
  --cache-dir feature_cache/vggt \
  --output-manifest data/spatialladder/spatial-with-features.jsonl --device cuda
```

This also extracts held-out features for optional oracle diagnostics; only
training features fit normalization or optimize weights. The original and SFT
baselines do not read teacher features and can use `spatial.jsonl` directly.

Run SFT, all three latent stages, and validation of the three models:

```bash
STAGE_STEPS=100 ACCUMULATION_STEPS=8 bash scripts/train_spatialladder.sh
```

This is an initial pipeline check, **not a tuned schedule**. Each latent stage
defaults to 100 optimizer steps; SFT defaults to 300, matching the total training
examples consumed across the three latent stages. This matches data exposure,
not compute. One pass over the default training split takes `ceil(5252 / 8) = 657`
optimizer steps. Adjust `STAGE_STEPS`, `SFT_STEPS`, `LEARNING_RATE`,
`ACCUMULATION_STEPS`, `SEED`, `CONFIG`, `MANIFEST`, `RUN_DIR`, or `DEVICE` through
environment variables. Use a new `RUN_DIR` for each fresh experiment. The script
runs sequentially. Both the script and ICE job files enable compact checkpoints:
every 25 steps, save a full recovery checkpoint and remove the preceding save
only after its replacement succeeds. At stage completion, keep only final weights
and metadata. Override the interval with `CHECKPOINT_EVERY`. After interruption,
use `train --resume` on the latest intermediate checkpoint with
`--compact-checkpoints`, as documented in [USAGE.md](USAGE.md). Completed stages
support evaluation and `--init-from`, but their weights-only checkpoints cannot
resume optimizer state. During saving, allow temporary space for two checkpoints.

To train only the answer-only baseline, without VGGT installation or extraction:

```bash
pip install -e '.[data]'
latent-geometry train --stage sft --config configs/default.json \
  --manifest data/spatialladder/spatial.jsonl --output-dir runs/sft \
  --steps 300 --accumulation-steps 8 --learning-rate 1e-5 \
  --checkpoint-every 25 --compact-checkpoints --device cuda
```

Training uses one unpadded example per forward and gradient accumulation. It
remains single-process with FP32 parameters/optimizer and CUDA BF16 autocast;
it does not distribute the model across GPUs. Check GPU memory with a one-step
run before a full schedule, especially during recurrent refinement.

## Evaluate

Use `--split val` to choose schedules and checkpoints. Run SPBench-SI test
evaluation explicitly after those choices are fixed (adjust checkpoint paths
if you changed the default step counts):

```bash
latent-geometry evaluate --original-model --config configs/default.json \
  --manifest data/spatialladder/spatial.jsonl --split test \
  --output runs/spatialladder/eval/original-test.jsonl --device cuda

latent-geometry evaluate --checkpoint runs/spatialladder/sft/checkpoint-000300 \
  --manifest data/spatialladder/spatial.jsonl --split test \
  --output runs/spatialladder/eval/sft-test.jsonl --device cuda

latent-geometry evaluate --checkpoint runs/spatialladder/refinement/checkpoint-000100 \
  --manifest data/spatialladder/spatial.jsonl --split test \
  --output runs/spatialladder/eval/latent-test.jsonl --device cuda
```

Each evaluation writes predictions and a `.jsonl.summary.json` sidecar. On these
manifests, `overall_score` is the unweighted mean of the four task scores;
`mcq_accuracy`, `numerical_mra`, and `by_question_type` give the breakdown. All
scores are fractions in [0, 1]. The evaluator follows the official direct-answer
parsing and MRA threshold construction, including their parsing quirks. Normalized
exact match is retained as a diagnostic, not the primary benchmark metric.

The scoring references are the official
[VSI utilities](https://github.com/ZJU-REAL/SpatialLadder/blob/main/eval_spld/data_utils/vsi_utils.py)
and [task aggregation](https://github.com/ZJU-REAL/SpatialLadder/blob/main/eval_spld/data_utils/vsibench.py).
Our workflow uses direct answers rather than the paper's thinking prompt and
does not reproduce its complete training protocol. Evaluation loads BF16 weights
on CUDA by default; use `--no-bf16` for FP32.
