import json
from pathlib import Path
import time
from contextlib import nullcontext

import torch

from .geometry import load_feature
from .scoring import score_spbench, summarize_spbench


def normalize_answer(text):
    return " ".join(text.strip().casefold().split())


def evaluate(model, processor, encoder, records, *, output_path, split="val", oracle=False,
             max_new_tokens=64, use_cache=True, bf16=True):
    records = [r for r in records if r["split"] == split]
    if not records:
        raise ValueError(f"No examples in split {split}")
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    correct, by_scene, scored = 0, {}, []
    scoring = {r.get("scoring", "exact_match") for r in records}
    if len(scoring) != 1:
        raise ValueError("Cannot mix scoring protocols in one evaluation")
    protocol = scoring.pop()
    if protocol not in ("exact_match", "spbench-si"):
        raise ValueError(f"Unknown scoring protocol: {protocol}")
    # Use exclusive creation so research outputs are not silently overwritten.
    with path.open("x") as stream:
        for record in records:
            inputs = encoder.prompt(record["image"], record["question"])
            features = None
            if oracle:
                features, _ = load_feature(record["teacher_feature"],
                    expected_image_hash=record["image_sha256"], num_latents=model.config.num_latents,
                    feature_dim=model.config.teacher_dim)
                features = features.unsqueeze(0)
            start = time.monotonic()
            context = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if bf16 and model.backbone.device.type == "cuda" else nullcontext())
            with context:
                result = model.generate(inputs, max_new_tokens=max_new_tokens, eos_token_ids=[eos],
                                        use_cache=use_cache, oracle_features=features)
            text = processor.tokenizer.decode(result.token_ids, skip_special_tokens=True)
            match = normalize_answer(text) == normalize_answer(record["answer"])
            correct += int(match)
            scene = by_scene.setdefault(record["scene_id"], {"correct": 0, "count": 0})
            scene["correct"] += int(match)
            scene["count"] += 1
            score = score_spbench(text, record) if protocol == "spbench-si" else float(match)
            scored.append({"question_type": record.get("question_type"), "score": score})
            stream.write(json.dumps({"id": record["id"], "scene_id": record["scene_id"],
                "question": record["question"], "reference": record["answer"], "prediction": text,
                "question_type": record.get("question_type"), "score": score,
                "exact_match": match, "oracle": oracle, "latent_count": result.latents.shape[1],
                "latent_rms": float(result.latents.float().square().mean().sqrt()) if result.latents.numel() else 0.0,
                "latency_seconds": time.monotonic() - start}) + "\n")
    summary = {"count": len(records), "exact_match": correct / len(records), "by_scene": by_scene,
               "oracle": oracle, "split": split,
               "scoring": protocol,
               "model_kind": "answer_only" if model.stage.value == "sft" else "latent",
               "student": model.config.student, "student_revision": model.config.student_revision,
               "max_new_tokens": max_new_tokens, "bf16": bf16}
    if protocol == "spbench-si":
        summary.update(summarize_spbench(scored))
    path.with_suffix(path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
