# Usage

For the goal, method, and experiment sequence, see [RESEARCH_GOAL.md](RESEARCH_GOAL.md).

For server-side downloads, scene splitting, the answer-only baseline, and
SPBench-SI scoring, start with [SPATIALLADDER.md](SPATIALLADDER.md).

## Install and verify without model downloads

Use Python 3.10+ and a suitable PyTorch installation for your machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

The tests use a tiny random Qwen model and synthetic features without downloading pretrained weights. They check gradients, frozen parameters, shifted supervision, cached decoding, and checkpoint resumption. Pretrained execution and GPU memory/throughput still need validation on the cluster.

## Dataset-independent manifest

Prepare one JSON object per line. Paths are relative to the manifest file or
absolute. Only one source image per example is supported initially.

```json
{"id":"scene1-q1","scene_id":"scene1","image":"images/scene1.jpg","question":"Which object is closer?","answer":"The chair.","split":"train"}
{"id":"scene2-q1","scene_id":"scene2","image":"images/scene2.jpg","question":"What is left of the table?","answer":"The lamp.","split":"val"}
```

Every question/view from a scene must share a split. The loader rejects duplicate
IDs, scene leakage, identical image content across splits, and missing files.
The `scene_id` must come from actual dataset metadata: image-level IDs cannot
prevent leakage between different views of the same scene. Do not include
helper views, annotations, or answer-bearing images as source images.

The encoder applies Qwen's chat template to the source image and question, then
tokenizes the answer separately with an assistant terminator. It rejects long
examples rather than truncating image placeholders or answers. The current
runner uses **one unpadded example per forward** and gradient accumulation.
Latent-end is a fixed input boundary: answer loss supervises only answer tokens
and the generated assistant terminator, not the latent-end token.

## Extract teacher features offline

Install the optional VGGT dependency in the extraction environment:

```bash
pip install -e '.[teacher]'
latent-geometry extract \
  --config configs/default.json \
  --manifest data/spatial.jsonl \
  --cache-dir feature_cache/vggt \
  --output-manifest data/spatial-with-features.jsonl \
  --device cuda
```

VGGT source is pinned in `pyproject.toml`. Its model checkpoint revision is
resolved to a commit and recorded in every cache entry. You can specify
`--teacher-revision COMMIT` explicitly. Student revisions are likewise resolved
and recorded in checkpoints. The default feature layer is the final aggregator
layer (`--layer -1`); an unavailable intermediate layer raises an error.

Extraction uses VGGT's **pad** preprocessing to preserve the whole source image,
removes camera/register tokens using the returned patch start index, and applies
2D adaptive average pooling to a 4x4 grid. Tokens have deterministic row-major
order. Non-divisible patch grids are supported without dropping trailing patches.
Both models see the same source content, with their respective preprocessing.

Only pooled FP32 teacher features are cached, not trainable projected targets.
Cache metadata includes image hashes, teacher revision, source-code hashes,
preprocessing, layer, patch grid, pooling, extraction/storage dtypes, and shapes.
The enriched manifest retains scene and example identifiers and adds
`teacher_feature`. Its resolved file paths are absolute; adjust them if moving
the data between machines. Training rejects mixed teacher extraction policies.

Warmup fits feature-channel mean/std from **unique feature files in the selected training subset only**.
Normalization and the linear adapter are stored in the student checkpoint.
The initial adapter/head output scale is matched approximately to input-embedding
RMS; monitor `latent_rms` during training. An adapter trained on answer loss is
not guaranteed to retain useful geometry; that requires subsequent experiments.

## Train on the cluster

These are pilot commands, not tuned schedules. Start with 32–128 examples for a
learning check, then a small end-to-end run before scaling the dataset.

```bash
# Stage 1: establish the VGGT-to-student interface.
latent-geometry train \
  --config configs/default.json \
  --manifest data/spatial-with-features.jsonl \
  --stage warmup --output-dir runs/warmup \
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 \
  --max-train-examples 800 --device cuda

# Stage 2: freeze the oracle adapter and learn to predict its targets.
latent-geometry train \
  --init-from runs/warmup/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage alignment --output-dir runs/alignment \
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 \
  --max-train-examples 800 --device cuda

# Stage 3: answer with self-generated latents and backpropagate through them.
latent-geometry train \
  --init-from runs/alignment/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage refinement --output-dir runs/refinement \
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 \
  --max-train-examples 800 --device cuda
```

`--max-train-examples 800` selects a fixed, seeded subset shared across stages
and the SFT baseline, regardless of teacher-feature fields. The default CLI value
`0` uses all training examples. Validation/test remain unchanged. The examples
above make one pass over the 800-question subset per latent stage; compare with
300 SFT steps on the same subset. The runner records selected IDs in `run.json`
and checkpoint settings.

VGGT is not run during any student-training stage. Refinement also accepts a
manifest with no `teacher_feature` fields. Set feature dimensions, pool size, and
alignment weight in the initial JSON config; subsequent stages inherit the
checkpoint configuration. Stage lengths and learning rates remain experimental.

Shared images are hashed once and shared feature files are validated once during setup. Each example still loads its own pooled features; the dataset is not retained in GPU memory.

The runner uses FP32 model/optimizer storage with CUDA BF16 autocast by default,
gradient clipping, gradient accumulation, and non-reentrant gradient
checkpointing. CPU verification uses FP32. This version is **single-process**;
multi-GPU sharding is not implemented, and `WORLD_SIZE > 1` is rejected. Profile
one complete optimizer step, especially refinement, on the available GPU before
assuming the 3B configuration fits. There is no measured GPU memory guarantee.

Checkpoints contain the complete student (including frozen vision), adapter,
prediction head, statistics, processor, optimizer, RNG, example cursor, and run
settings. They can be large. New stages use `--init-from` and a fresh optimizer.
Add `--compact-checkpoints` to keep only the latest full recovery checkpoint
while training and save a weights-only final checkpoint. A replacement is saved
successfully before the previous checkpoint is deleted. The final checkpoint
retains the student, heads, normalization, processor, and run metadata, but omits
optimizer state. Use it for evaluation or `--init-from`; `--resume` requires an
intermediate full checkpoint. When resuming into a different output directory,
the original checkpoint is preserved. Checkpoint publication temporarily needs
space for both the old and new files. Without this flag, all saves remain full
checkpoints and are retained, as before.

To resume the same stage:

```bash
latent-geometry train \
  --resume runs/alignment/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage alignment --output-dir runs/alignment \
  --steps 200 --learning-rate 1e-5 --accumulation-steps 8 \
  --max-train-examples 800 --device cuda
```

`--steps` is the total optimizer-step target. Resume restores the optimizer,
RNG, and deterministic example order and rejects changes to data or recorded
training settings, including the training subset and answer-loss targets.
Checkpoints from the previous latent-end-supervised objective cannot be resumed
exactly with this version; use a fresh run for the revised experiment. Checkpoints
are saved only after complete optimizer steps.
If logs extend beyond the resumed checkpoint, choose a new output directory.

## Generate and evaluate

```bash
latent-geometry generate \
  --checkpoint runs/refinement/checkpoint-000100 \
  --image /path/to/scene.jpg --question 'Which object is closer?' --device cuda

latent-geometry evaluate \
  --checkpoint runs/refinement/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl --split val \
  --output runs/evaluation/generated.jsonl --device cuda

# Run separately to measure the oracle/generated gap.
latent-geometry evaluate \
  --checkpoint runs/refinement/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl --split val --oracle \
  --output runs/evaluation/oracle.jsonl --device cuda
```

Generation needs only the student checkpoint, image, and question. Evaluation
writes predictions, latent norms, latency, and per-scene normalized exact match.
SpatialLadder manifests additionally enable SPBench scoring as documented in
[SPATIALLADDER.md](SPATIALLADDER.md). Custom manifests without `scoring` use
normalized exact match. Geometry probes, controlled interventions, confidence
intervals, and large research runs remain later work.
