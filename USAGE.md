# Geometry-supervised latent reasoning

A standalone prototype for the method in [RESEARCH_GOAL.md](RESEARCH_GOAL.md):
train Qwen2.5-VL to predict continuous geometry latents supervised by frozen VGGT
features, then answer using its own recurrent latents without VGGT.

The default configuration uses **Qwen2.5-VL-3B-Instruct and 16 latent vectors**.
The language model, input embeddings, and output head are trainable. The student
vision encoder and its existing merger remain frozen in every stage.

## Install and verify without model downloads

Use Python 3.10+ and a suitable PyTorch installation for your machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
latent-geometry smoke --output-dir runs/smoke
```

The smoke command creates a tiny, randomly initialized **real Qwen2.5-VL
architecture**, uses synthetic image/teacher tensors, trains all three stages,
checks loss reduction and frozen vision weights, compares cached and uncached
decoding, and saves/reloads/resumes a checkpoint. It generates exactly 16
continuous latents. No pretrained weights or research dataset are downloaded.
Use a fresh output directory for another smoke run.

This verifies model plumbing, not spatial-reasoning quality or the usefulness of
VGGT supervision. Pretrained Qwen/VGGT execution and GPU memory/throughput still
need validation on the cluster.

## Three training stages

Let `x` be the image/question prompt (ending at the assistant generation prefix),
`t_i` a projected VGGT target, `z_i` a generated vector, and `y` the answer.

| Stage | Model inputs | Objective | Trainable new modules |
| --- | --- | --- | --- |
| `warmup` | `x, start, t_1 ... t_K, end, y` | Answer cross-entropy | Oracle adapter |
| `alignment` | Same oracle-conditioned sequence | Answer CE + weighted latent MSE | Latent prediction head |
| `refinement` | `x, start, z_1 ... z_K, end, y` | Answer CE only | Latent prediction head |

The language model is trained in all three stages. The oracle adapter is frozen
after warmup. In alignment, the state after `start` predicts `t_1`, the state
after consuming `t_1` predicts `t_2`, and so on. Each prediction precedes its
target input, preventing same-position target leakage. Latent MSE is averaged
over slots and channels. Vocabulary CE supervises `end` and the answer tokens,
including the assistant terminator; the prompt, fixed `start`, and latent slots
are masked from vocabulary CE. The `end` boundary is a fixed control at inference.

Refinement uses all K generated vectors with **no detach and no alignment
loss**. Answer gradients propagate through the entire recurrent computation.
The initial implementation recomputes the prefix at each latent step during
training and uses non-reentrant gradient checkpointing. This is a correctness
reference with substantial compute cost; profile it before a long run.

Inference uses a KV cache and explicit multimodal position IDs. The final latent
is consumed before the end boundary and answer. An uncached reference is
available with `--no-cache`. No teacher module is part of the student checkpoint
or imported by generation.

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

Warmup fits feature-channel mean/std from **unique training feature files only**.
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
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 --device cuda

# Stage 2: freeze the oracle adapter and learn to predict its targets.
latent-geometry train \
  --init-from runs/warmup/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage alignment --output-dir runs/alignment \
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 --device cuda

# Stage 3: answer with self-generated latents and backpropagate through them.
latent-geometry train \
  --init-from runs/alignment/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage refinement --output-dir runs/refinement \
  --steps 100 --learning-rate 1e-5 --accumulation-steps 8 --device cuda
```

VGGT is not run during any student-training stage. Refinement also accepts a
manifest with no `teacher_feature` fields. Set feature dimensions, pool size, and
alignment weight in the initial JSON config; subsequent stages inherit the
checkpoint configuration. Stage lengths and learning rates remain experimental.

The runner uses FP32 model/optimizer storage with CUDA BF16 autocast by default,
gradient clipping, gradient accumulation, and non-reentrant gradient
checkpointing. CPU verification uses FP32. This version is **single-process**;
multi-GPU sharding is not implemented, and `WORLD_SIZE > 1` is rejected. Profile
one complete optimizer step, especially refinement, on the available GPU before
assuming the 3B configuration fits. There is no measured GPU memory guarantee.

Checkpoints contain the complete student (including frozen vision), adapter,
prediction head, statistics, processor, optimizer, RNG, example cursor, and run
settings. They can be large. New stages use `--init-from` and a fresh optimizer.
To resume the same stage:

```bash
latent-geometry train \
  --resume runs/alignment/checkpoint-000100 \
  --manifest data/spatial-with-features.jsonl \
  --stage alignment --output-dir runs/alignment \
  --steps 200 --learning-rate 1e-5 --accumulation-steps 8 --device cuda
```

`--steps` is the total optimizer-step target. Resume restores the optimizer,
RNG, and deterministic example order and rejects changes to data or recorded
training settings. Checkpoints are saved only after complete optimizer steps.
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
Exact match is a plumbing metric; add the selected dataset's scoring rules before
making performance claims. Geometry probes, crop/answer-only baselines, controlled
interventions, confidence intervals, and large research runs remain later work.

## References and implementation scope

The code is original and does not vendor or monkey-patch other training frameworks.
Relevant references inspected for supervision and model-interface conventions:

- [LVR paper](https://arxiv.org/abs/2509.24251) and [forward implementation](https://github.com/VincentLeebang/lvr/blob/main/src/train/monkey_patch_forward.py).
- [LanteRn](https://github.com/GuilhermeViveiros/LantErn), including its [compression helper](https://github.com/GuilhermeViveiros/LantErn/blob/main/src/models/utils.py). Our pooling preserves the 2D grid rather than grouping consecutive tokens and dropping remainders.
- [Mirage](https://vlm-mirage.github.io/) for grounding followed by text-supervised latent refinement. The adapter warmup is specific to this prototype.
- [Qwen2.5-VL implementation, transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
- [VGGT](https://github.com/facebookresearch/vggt), used as an optional external dependency under its upstream license.

The dataset, cluster scheduler, distributed strategy, and actual pretrained GPU
training run remain to be selected/validated. No accuracy improvement is claimed.
