"""Offline VGGT extraction and immutable pooled-feature caches."""
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def image_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_pool(tokens, patch_start_idx, grid_height, grid_width, pool_shape=(4, 4)):
    """Remove camera/register tokens; preserve 2D topology and row-major order."""
    if tokens.ndim != 4 or tokens.shape[1] != 1:
        raise ValueError("Expected [batch, 1 view, tokens, channels]; multi-view is deferred")
    patches = tokens[:, 0, patch_start_idx:]
    if patches.shape[1] != grid_height * grid_width:
        raise ValueError("Patch count and spatial grid disagree")
    if min(grid_height, grid_width) < 1 or min(pool_shape) < 1:
        raise ValueError("Grid sizes must be positive")
    if grid_height < pool_shape[0] or grid_width < pool_shape[1]:
        raise ValueError("Pooling grid must not exceed the patch grid")
    grid = patches.reshape(patches.shape[0], grid_height, grid_width, patches.shape[-1])
    pooled = F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2).float(), pool_shape)
    return pooled.flatten(2).transpose(1, 2).contiguous()


def cache_signature(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def load_feature(path, *, expected_image_hash=None, num_latents=None, feature_dim=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features, metadata = payload["features"], payload["metadata"]
    if payload.get("format_version") != 1 or features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError(f"Invalid teacher feature cache: {path}")
    if expected_image_hash is not None and metadata["image_sha256"] != expected_image_hash:
        raise ValueError(f"Teacher cache belongs to a different source image: {path}")
    if num_latents is not None and features.shape[0] != num_latents:
        raise ValueError("Cached pool size differs from configured latent count")
    if feature_dim is not None and features.shape[1] != feature_dim:
        raise ValueError("Cached teacher channels differ from configured feature dimension")
    return features, metadata


def fit_statistics(feature_paths):
    """Streaming FP64 moments over unique training images, never validation/test."""
    count, mean, m2 = 0, None, None
    for path in sorted(set(map(str, feature_paths))):
        features, _ = load_feature(path)
        values = features.double()
        n = values.shape[0]
        batch_mean = values.mean(0)
        batch_m2 = (values - batch_mean).square().sum(0)
        if mean is None:
            count, mean, m2 = n, batch_mean, batch_m2
        else:
            if mean.shape != batch_mean.shape:
                raise ValueError("Mixed teacher feature dimensions")
            delta = batch_mean - mean
            total = count + n
            m2 = m2 + batch_m2 + delta.square() * count * n / total
            mean = mean + delta * n / total
            count = total
    if count == 0:
        raise ValueError("No training teacher features found")
    return mean.float(), (m2 / count).clamp_min(0).sqrt().float().clamp_min(1e-6)


class VGGTExtractor:
    def __init__(self, *, checkpoint="facebook/VGGT-1B", revision="main", layer=-1,
                 pool_shape=(4, 4), device="cuda", dtype=torch.bfloat16):
        # VGGT is an optional, separately installed upstream dependency.
        from huggingface_hub import HfApi
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        self.preprocess = load_and_preprocess_images
        self.revision = HfApi().model_info(checkpoint, revision=revision).sha
        self.model = VGGT.from_pretrained(checkpoint, revision=self.revision).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.layer, self.pool_shape = layer, pool_shape
        self.spec = {
            "teacher": checkpoint, "teacher_revision": self.revision,
            "aggregator_source_sha256": image_sha256(inspect.getfile(type(self.model.aggregator))),
            "preprocess_source_sha256": image_sha256(inspect.getfile(load_and_preprocess_images)),
            "layer": layer, "pool_shape": list(pool_shape), "preprocessing": "VGGT pad (whole image)",
            "extraction_dtype": str(dtype), "storage_dtype": "torch.float32", "view_count": 1,
        }

    @torch.no_grad()
    def extract(self, image_path):
        from PIL import Image

        # pad preserves the whole image; default VGGT crop preprocessing does not.
        images = self.preprocess([str(image_path)], mode="pad").unsqueeze(0).to(self.device)
        enabled = torch.device(self.device).type == "cuda" and self.dtype != torch.float32
        with torch.autocast(device_type=torch.device(self.device).type,
                            dtype=self.dtype, enabled=enabled):
            layers, start = self.model.aggregator(images)
        tokens = layers[self.layer]
        if tokens is None:
            raise ValueError("Selected VGGT layer is not cached by this aggregator revision")
        patch_size = self.model.aggregator.patch_size
        h, w = images.shape[-2] // patch_size, images.shape[-1] // patch_size
        pooled = spatial_pool(tokens, start, h, w, self.pool_shape)[0].cpu()
        with Image.open(image_path) as image:
            source_size = list(image.size)
        metadata = {**self.spec, "image_sha256": image_sha256(image_path),
                    "source_size_wh": source_size, "processed_size_hw": list(images.shape[-2:]),
                    "patch_grid_hw": [h, w], "patch_start_idx": start,
                    "feature_shape": list(pooled.shape)}
        return {"format_version": 1, "features": pooled, "metadata": metadata}

    def cache(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        key = cache_signature({**self.spec, "image_sha256": image_sha256(image_path)})
        path = output_dir / f"{key}.pt"
        if not path.exists():
            payload = self.extract(image_path)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        else:
            load_feature(path, expected_image_hash=image_sha256(image_path))
        return path.resolve()
