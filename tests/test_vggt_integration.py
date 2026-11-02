"""Optional upstream API check with a tiny random aggregator, no network/weights."""
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from latent_geometry.geometry import VGGTExtractor, load_feature


def test_upstream_aggregator_preprocessing_and_cache(tmp_path, monkeypatch):
    pytest.importorskip("vggt")
    from huggingface_hub import HfApi
    from vggt.models.aggregator import Aggregator
    from vggt.models.vggt import VGGT

    teacher = torch.nn.Module()
    # 518px pad preprocessing -> 7x7 patches; small random upstream transformer.
    teacher.aggregator = Aggregator(img_size=518, patch_size=74, embed_dim=32,
        depth=1, num_heads=4, patch_embed="conv", cached_layer_indices=(0,))
    monkeypatch.setattr(HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="test-revision"))
    monkeypatch.setattr(VGGT, "from_pretrained", lambda *args, **kwargs: teacher)
    image = tmp_path / "wide.png"
    Image.new("RGB", (32, 16), "red").save(image)
    extractor = VGGTExtractor(device="cpu", dtype=torch.float32)
    assert not any(p.requires_grad for p in extractor.model.parameters())
    path = extractor.cache(image, tmp_path / "cache")
    features, metadata = load_feature(path, num_latents=16, feature_dim=64)
    assert features.shape == (16, 64)
    assert metadata["source_size_wh"] == [32, 16]
    assert metadata["processed_size_hw"] == [518, 518]
    assert metadata["patch_grid_hw"] == [7, 7]
    assert metadata["patch_start_idx"] == 5
    assert metadata["teacher_revision"] == "test-revision"
    # Existing immutable cache is reused without rerunning the teacher.
    monkeypatch.setattr(extractor, "extract", lambda *args: pytest.fail("Cache not reused"))
    assert extractor.cache(image, tmp_path / "cache") == path
