from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import torch

from .checkpoint import restore_checkpoint, save_checkpoint
from .config import Stage
from .data import manifest_fingerprint, validate_feature_policy
from .geometry import fit_statistics


def train(model, processor, encoder, records, *, output_dir, stage, steps=100,
          learning_rate=1e-5, accumulation_steps=1, seed=42, checkpoint_every=100,
          resume=None, gradient_checkpointing=True, bf16=True, max_grad_norm=1.0):
    if min(steps, accumulation_steps, checkpoint_every) < 1 or learning_rate <= 0 or max_grad_norm <= 0:
        raise ValueError("Steps, accumulation, checkpoint interval, LR and max_grad_norm must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This initial runner is single-process; distributed sharding is not implemented")
    stage = Stage(stage)
    training = [r for r in records if r["split"] == "train"]
    if not training:
        raise ValueError("Manifest has no training split")
    fingerprint = manifest_fingerprint(records)
    policy = validate_feature_policy(records, model.config) if stage != Stage.REFINEMENT else None
    model.set_stage(stage)
    if stage == Stage.WARMUP and resume is None:
        mean, std = fit_statistics([r["teacher_feature"] for r in training])
        model.oracle.set_statistics(mean, std)
    if stage == Stage.ALIGNMENT and not bool(model.oracle.stats_ready):
        raise ValueError("Initialize alignment from a completed warmup checkpoint")
    if gradient_checkpointing:
        # Non-reentrant checkpointing supports gradients through recurrent inputs.
        model.backbone.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    settings = {"stage": stage.value, "manifest_fingerprint": fingerprint, "seed": seed,
                "accumulation_steps": accumulation_steps, "learning_rate": learning_rate,
                "gradient_checkpointing": gradient_checkpointing, "bf16": bf16,
                "max_grad_norm": max_grad_norm, "teacher_policy": policy,
                "max_prompt_tokens": encoder.max_prompt_tokens, "max_answer_tokens": encoder.max_answer_tokens}
    progress = {"step": 0, "examples_seen": 0, "settings": settings}
    if resume:
        progress = restore_checkpoint(resume, model, optimizer=optimizer, restore_random=True)
        if progress["settings"] != settings:
            raise ValueError("Resume data or optimizer settings changed; use --init-from for a new run")
    else:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if progress["step"] >= steps:
        raise ValueError("--steps is the total optimizer-step target; checkpoint already reached it")
    output_dir = Path(output_dir)
    if (output_dir / "run.json").exists() and resume is None:
        raise FileExistsError("Output directory already contains a run; choose a new directory or --resume")
    metrics_path = output_dir / "metrics.jsonl"
    if resume and metrics_path.exists():
        logged_steps = [json.loads(line)["step"] for line in metrics_path.read_text().splitlines() if line.strip()]
        if logged_steps and max(logged_steps) > progress["step"]:
            raise ValueError("Output has steps beyond this checkpoint; resume into a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    use_amp = bf16 and model.backbone.device.type == "cuda"
    model.train()
    order, order_epoch = [], -1
    start_time = time.monotonic()
    for step in range(progress["step"], steps):
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "answer_loss": 0.0, "alignment_loss": 0.0, "latent_rms": 0.0}
        for _ in range(accumulation_steps):
            epoch, index = divmod(progress["examples_seen"], len(training))
            if epoch != order_epoch:
                order = list(range(len(training)))
                random.Random(seed + epoch).shuffle(order)
                order_epoch = epoch
            example = encoder(training[order[index]], require_features=stage != Stage.REFINEMENT)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with context:
                result = model(**example)
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step + 1}")
            (result.loss / accumulation_steps).backward()
            for name in ("loss", "answer_loss", "alignment_loss"):
                totals[name] += float(getattr(result, name).detach()) / accumulation_steps
            totals["latent_rms"] += float(result.latents.detach().float().square().mean().sqrt()) / accumulation_steps
            progress["examples_seen"] += 1
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        progress["step"] = step + 1
        metrics = {"stage": stage.value, "step": step + 1, **totals, "gradient_norm": float(norm),
                   "examples_seen": progress["examples_seen"], "elapsed_seconds": time.monotonic() - start_time}
        if model.backbone.device.type == "cuda":
            metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(model.backbone.device)
        print(json.dumps(metrics), flush=True)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            save_checkpoint(output_dir / f"checkpoint-{step + 1:06d}", model,
                            processor=processor, optimizer=optimizer, progress=progress)
    return progress
