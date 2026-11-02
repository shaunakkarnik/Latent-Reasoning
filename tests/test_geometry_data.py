import json

from PIL import Image
import pytest
import torch

from latent_geometry.config import LatentConfig
from latent_geometry.data import read_manifest, validate_feature_policy
from latent_geometry.geometry import fit_statistics, image_sha256, load_feature, spatial_pool


def test_spatial_pool_removes_special_tokens_and_preserves_grid():
    # 5 special tokens, then a non-divisible 3x5 patch grid.
    patches = torch.arange(15).float().reshape(1, 1, 15, 1)
    tokens = torch.cat((torch.full((1, 1, 5, 1), 1e8), patches), dim=2)
    pooled = spatial_pool(tokens, 5, 3, 5, (2, 2))
    expected = torch.nn.functional.adaptive_avg_pool2d(patches.reshape(1, 1, 3, 5), (2, 2))
    torch.testing.assert_close(pooled, expected.flatten(2).transpose(1, 2))
    assert pooled.max() < 15
    with pytest.raises(ValueError, match="Patch count"):
        spatial_pool(tokens, 4, 3, 5)


def feature_file(path, features, **metadata):
    torch.save({"format_version": 1, "features": features, "metadata": metadata}, path)


def test_streaming_statistics_and_deduplication(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    x, y = torch.randn(4, 8), torch.randn(4, 8)
    feature_file(a, x)
    feature_file(b, y)
    mean, std = fit_statistics([a, a, b])
    expected = torch.cat((x, y))
    torch.testing.assert_close(mean, expected.mean(0))
    torch.testing.assert_close(std, expected.std(0, correction=0))


def test_feature_cache_rejects_wrong_image_and_shape(tmp_path):
    path = tmp_path / "cache.pt"
    feature_file(path, torch.ones(16, 8), image_sha256="correct")
    with pytest.raises(ValueError, match="different source"):
        load_feature(path, expected_image_hash="wrong")
    with pytest.raises(ValueError, match="pool size"):
        load_feature(path, num_latents=4)


def test_manifest_rejects_scene_and_image_leakage(tmp_path):
    for name, color in (("a", "red"), ("b", "blue")):
        Image.new("RGB", (8, 8), color).save(tmp_path / f"{name}.png")
    rows = [{"id": "a", "scene_id": "same", "image": "a.png", "question": "Where?", "answer": "Left", "split": "train"},
            {"id": "b", "scene_id": "same", "image": "b.png", "question": "Where?", "answer": "Right", "split": "val"}]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="Scene leakage"):
        read_manifest(path)
    rows[1]["scene_id"] = "different"
    rows[1]["image"] = "a.png"
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="Identical image"):
        read_manifest(path)
    rows[1]["image"] = "b.png"
    path.write_text("\n".join(map(json.dumps, rows)))
    assert len(read_manifest(path)) == 2


def test_reject_mixed_extraction_policies(tmp_path):
    metadata = {"image_sha256": "image", "teacher": "vggt", "teacher_revision": "rev",
                "aggregator_source_sha256": "source", "preprocess_source_sha256": "preprocess",
                "layer": -1, "pool_shape": [1, 4], "preprocessing": "pad", "extraction_dtype": "fp32",
                "storage_dtype": "fp32", "view_count": 1}
    records = []
    for i in range(2):
        path = tmp_path / f"{i}.pt"
        feature_file(path, torch.ones(4, 8), **{**metadata, "teacher_revision": str(i)})
        records.append({"teacher_feature": str(path), "image_sha256": "image"})
    with pytest.raises(ValueError, match="incompatible"):
        validate_feature_policy(records, LatentConfig(teacher_dim=8, pool_height=1, pool_width=4))
