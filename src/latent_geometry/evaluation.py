import json
from pathlib import Path
import time

from .geometry import load_feature


def normalize_answer(text):
    return " ".join(text.strip().casefold().split())


def evaluate(model, processor, encoder, records, *, output_path, split="val", oracle=False,
             max_new_tokens=64, use_cache=True):
    records = [r for r in records if r["split"] == split]
    if not records:
        raise ValueError(f"No examples in split {split}")
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    correct, by_scene = 0, {}
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
            result = model.generate(inputs, max_new_tokens=max_new_tokens, eos_token_ids=[eos],
                                    use_cache=use_cache, oracle_features=features)
            text = processor.tokenizer.decode(result.token_ids, skip_special_tokens=True)
            match = normalize_answer(text) == normalize_answer(record["answer"])
            correct += int(match)
            scene = by_scene.setdefault(record["scene_id"], {"correct": 0, "count": 0})
            scene["correct"] += int(match)
            scene["count"] += 1
            stream.write(json.dumps({"id": record["id"], "scene_id": record["scene_id"],
                "question": record["question"], "reference": record["answer"], "prediction": text,
                "exact_match": match, "oracle": oracle, "latent_count": result.latents.shape[1],
                "latent_rms": float(result.latents.float().square().mean().sqrt()),
                "latency_seconds": time.monotonic() - start}) + "\n")
    summary = {"count": len(records), "exact_match": correct / len(records), "by_scene": by_scene,
               "oracle": oracle, "split": split,
               "metric_note": "Whitespace/case normalized exact match; dataset-specific scoring still required"}
    path.with_suffix(path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
