#!/usr/bin/env bash
# Run from the repository root after prepare-spatialladder and feature extraction.
set -euo pipefail

MANIFEST=${MANIFEST:-data/spatialladder/spatial-with-features.jsonl}
RUN_DIR=${RUN_DIR:-runs/spatialladder}
CONFIG=${CONFIG:-configs/default.json}
STAGE_STEPS=${STAGE_STEPS:-100}
SFT_STEPS=${SFT_STEPS:-$((3 * STAGE_STEPS))}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-25}

common=(--manifest "$MANIFEST" --learning-rate "$LEARNING_RATE"
        --accumulation-steps "$ACCUMULATION_STEPS" --device "$DEVICE" --seed "$SEED"
        --checkpoint-every "$CHECKPOINT_EVERY" --compact-checkpoints)
printf -v stage_checkpoint 'checkpoint-%06d' "$STAGE_STEPS"
printf -v sft_checkpoint 'checkpoint-%06d' "$SFT_STEPS"

# Match the total number of training examples seen across all three latent stages.
latent-geometry train "${common[@]}" --config "$CONFIG" --stage sft \
  --steps "$SFT_STEPS" --output-dir "$RUN_DIR/sft"
latent-geometry train "${common[@]}" --config "$CONFIG" --stage warmup \
  --steps "$STAGE_STEPS" --output-dir "$RUN_DIR/warmup"
latent-geometry train "${common[@]}" --stage alignment \
  --init-from "$RUN_DIR/warmup/$stage_checkpoint" \
  --steps "$STAGE_STEPS" --output-dir "$RUN_DIR/alignment"
latent-geometry train "${common[@]}" --stage refinement \
  --init-from "$RUN_DIR/alignment/$stage_checkpoint" \
  --steps "$STAGE_STEPS" --output-dir "$RUN_DIR/refinement"

# Development evaluation only; keep the benchmark test split for final reporting.
latent-geometry evaluate --original-model --config "$CONFIG" --manifest "$MANIFEST" \
  --split val --output "$RUN_DIR/eval/original-val.jsonl" --device "$DEVICE"
latent-geometry evaluate --checkpoint "$RUN_DIR/sft/$sft_checkpoint" --manifest "$MANIFEST" \
  --split val --output "$RUN_DIR/eval/sft-val.jsonl" --device "$DEVICE"
latent-geometry evaluate --checkpoint "$RUN_DIR/refinement/$stage_checkpoint" --manifest "$MANIFEST" \
  --split val --output "$RUN_DIR/eval/latent-val.jsonl" --device "$DEVICE"
