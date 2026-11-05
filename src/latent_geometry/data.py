"""Dataset-independent JSONL interface; one original image per example."""
import json
from pathlib import Path

from PIL import Image
import torch

from .geometry import cache_signature, image_sha256, load_feature


def read_manifest(path):
    path = Path(path).resolve()
    records, ids, scene_splits, image_splits = [], set(), {}, {}
    image_hashes, checked_paths = {}, set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        for field in ("id", "scene_id", "image", "question", "answer", "split"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{path}:{line_number}: nonempty {field} is required")
        if record["split"] not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if record["id"] in ids:
            raise ValueError(f"Duplicate example id: {record['id']}")
        ids.add(record["id"])
        split = record["split"]
        scene = record["scene_id"]
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError(f"Scene leakage across splits: {scene}")
        scene_splits[scene] = split
        for field in ("image", "teacher_feature"):
            if field in record:
                record[field] = str((path.parent / record[field]).resolve())
                if record[field] not in checked_paths:
                    if not Path(record[field]).is_file():
                        raise ValueError(f"Missing {field}: {record[field]}")
                    checked_paths.add(record[field])
        image_path = record["image"]
        if image_path not in image_hashes:
            image_hashes[image_path] = image_sha256(image_path)
        digest = image_hashes[image_path]
        if digest in image_splits and image_splits[digest] != split:
            raise ValueError("Identical image content occurs across splits")
        image_splits[digest] = split
        record["image_sha256"] = digest
        records.append(record)
    if not records:
        raise ValueError("Manifest is empty")
    return records


def manifest_fingerprint(records):
    """Content-based provenance, independent of absolute workspace paths."""
    rows, feature_hashes = [], {}
    for record in records:
        row = {k: v for k, v in record.items() if k not in ("image", "teacher_feature")}
        if "teacher_feature" in record:
            feature_path = record["teacher_feature"]
            if feature_path not in feature_hashes:
                feature_hashes[feature_path] = image_sha256(feature_path)
            row["teacher_feature_sha256"] = feature_hashes[feature_path]
        rows.append(row)
    return cache_signature({"records": rows})


def validate_feature_policy(records, config):
    signature = None
    checked_features = {}
    for record in records:
        if "teacher_feature" not in record:
            raise ValueError("Extract teacher features before warmup/alignment training")
        feature_path = record["teacher_feature"]
        if feature_path in checked_features:
            if checked_features[feature_path] != record["image_sha256"]:
                raise ValueError(f"Teacher cache belongs to a different source image: {feature_path}")
            continue
        _, metadata = load_feature(feature_path, expected_image_hash=record["image_sha256"],
                                   num_latents=config.num_latents, feature_dim=config.teacher_dim)
        if metadata.get("pool_shape") != [config.pool_height, config.pool_width]:
            raise ValueError("Cache spatial layout differs from model configuration")
        # Per-image fields differ; teacher/code/preprocessing policy must not.
        policy = {key: metadata[key] for key in (
            "teacher", "teacher_revision", "aggregator_source_sha256", "preprocess_source_sha256",
            "layer", "pool_shape", "preprocessing", "extraction_dtype", "storage_dtype", "view_count")}
        current = cache_signature(policy)
        if signature is not None and current != signature:
            raise ValueError("Manifest mixes incompatible teacher extraction policies")
        signature = current
        checked_features[feature_path] = record["image_sha256"]
    return signature


class ExampleEncoder:
    def __init__(self, processor, config, *, max_prompt_tokens=4096, max_answer_tokens=256):
        self.processor, self.config = processor, config
        self.max_prompt_tokens, self.max_answer_tokens = max_prompt_tokens, max_answer_tokens

    def prompt(self, image_path, question):
        if any(token in question for token in (self.config.start_token, self.config.end_token)):
            raise ValueError("Question contains reserved latent boundary tokens")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=False)
        if inputs["input_ids"].shape[1] > self.max_prompt_tokens:
            raise ValueError("Prompt exceeds max_prompt_tokens; do not truncate image placeholders")
        return dict(inputs)

    def __call__(self, record, *, require_features=True):
        if any(token in record["answer"] for token in (self.config.start_token, self.config.end_token)):
            raise ValueError("Answer contains reserved latent boundary tokens")
        inputs = self.prompt(record["image"], record["question"])
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(record["answer"], add_special_tokens=False)
        ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if len(ids) > self.max_answer_tokens:
            raise ValueError("Answer exceeds max_answer_tokens")
        features = None
        if require_features:
            # Image association and extraction policy were checked once during setup.
            features, _ = load_feature(record["teacher_feature"])
            features = features.unsqueeze(0)
        return {"inputs": inputs, "answer_ids": torch.tensor([ids]), "teacher_features": features}
