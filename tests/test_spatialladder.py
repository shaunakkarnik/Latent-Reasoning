from zipfile import ZipFile
import json
from types import SimpleNamespace

import pytest

from latent_geometry.spatialladder import build_records, extract_images
from latent_geometry.scoring import score_spbench, summarize_spbench


def training_row(index, scan, data_type="single_image"):
    return {"question_id": index, "question": "Which is closer?", "answer": "A",
            "image": [f"{scan}/10.jpg"], "options": ["A. chair", "B. table"],
            "question_type": "relative distance", "data_type": data_type}


def test_room_split_and_single_image_filter():
    rows = [training_row(1, "scene0000_00"), training_row(2, "scene0000_01"),
            training_row(3, "scene0001_00"), training_row(4, "scene0002_00", "multi_view")]
    test = [{"id": 1, "question": "How big?", "ground_truth": "12.5", "options": None,
             "scene_name": "scene0003_00", "images": ["20.jpg"], "question_type": "object_size_estimation"}]
    records, rooms = build_records(rows, test, seed=42)
    assert len(records) == 4
    assert len(rooms) == 1
    assert records[0]["split"] == records[1]["split"]
    assert {r["split"] for r in records} == {"train", "val", "test"}
    assert "A. chair" in records[0]["question"]
    assert "numerical value" in records[-1]["question"]
    assert records == build_records(rows, test, seed=42)[0]
    test[0]["scene_name"] = "scene0000_02"
    with pytest.raises(ValueError, match="overlapping rooms"):
        build_records(rows, test)


def test_selective_image_extraction(tmp_path):
    archive = tmp_path / "images.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("wrapper/scene0000_00/10.jpg", b"image")
        zipped.writestr("wrapper/scene0001_00/20.jpg", b"unused")
        zipped.writestr("../../unrelated.txt", b"ignore")
    extract_images(archive, tmp_path / "out", {"scene0000_00/10.jpg"})
    assert (tmp_path / "out/scene0000_00/10.jpg").read_bytes() == b"image"
    assert len(list((tmp_path / "out").rglob("*.jpg"))) == 1
    with pytest.raises(ValueError, match="missing requested"):
        extract_images(archive, tmp_path / "out", {"scene0002_00/10.jpg"})


def test_spbench_scoring_and_macro_average():
    mc = {"question_type": "object_rel_direction", "answer": "A"}
    numeric = {"question_type": "object_abs_distance", "answer": "100"}
    assert score_spbench("A. left", mc) == 1
    assert score_spbench("B", mc) == 0
    assert score_spbench("100", numeric) == 1
    assert score_spbench("150", numeric) == pytest.approx(0.1)
    assert score_spbench("unknown", numeric) == 0
    assert score_spbench("200", numeric) == 0
    scores = [{"question_type": "object_rel_direction", "score": 1}] * 9
    scores += [{"question_type": "object_abs_distance", "score": 0}]
    summary = summarize_spbench(scores)
    assert summary["overall_score"] == 0.5
    assert summary["mcq_accuracy"] == 1
    assert summary["numerical_mra"] == 0


def test_prepare_download_to_validated_manifest(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    import huggingface_hub
    from PIL import Image
    from latent_geometry.spatialladder import prepare_spatialladder
    from latent_geometry.data import read_manifest

    sources = tmp_path / "sources"
    sources.mkdir()
    training = [training_row(1, "scene0000_00"), training_row(2, "scene0001_00")]
    (sources / "spld_spatial_data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in training))
    test = [{"id": 1, "question": "How big?", "ground_truth": "12.5", "options": None,
             "scene_name": "scene0003_00", "images": ["20.jpg"], "question_type": "object_size_estimation"}]
    pq.write_table(pa.Table.from_pylist(test), sources / "SPBench-SI.parquet")
    for archive, scans in (("images.zip", ["scene0000_00", "scene0001_00"]),
                           ("SPBench-SI-images.zip", ["scene0003_00"])):
        with ZipFile(sources / archive, "w") as zipped:
            for scan in scans:
                image = sources / f"{scan}.jpg"
                Image.new("RGB", (8, 8), (int(scan[5:9]) * 70, 0, 0)).save(image)
                zipped.write(image, f"wrapper/{scan}/{'20' if scan == 'scene0003_00' else '10'}.jpg")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="fixed-commit")))

    def download(repo, filename, **kwargs):
        assert kwargs["revision"] == "fixed-commit"
        return str(sources / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = prepare_spatialladder(tmp_path / "output")
    assert result["counts"] == {"train": 1, "val": 1, "test": 1}
    assert len(read_manifest(result["manifest"])) == 3
    assert json.loads((tmp_path / "output/preparation.json").read_text())["sources"]["spatialladder"]["revision"] == "fixed-commit"


def test_evaluate_writes_spbench_summary(tmp_path):
    import torch
    from latent_geometry.config import LatentConfig, Stage
    from latent_geometry.evaluation import evaluate

    model = SimpleNamespace(backbone=SimpleNamespace(device=torch.device("cpu")),
        config=LatentConfig(), stage=Stage.SFT,
        generate=lambda *args, **kwargs: SimpleNamespace(token_ids=[1], latents=torch.empty(1, 0, 8)))
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 2,
                                decode=lambda *args, **kwargs: "100")
    encoder = SimpleNamespace(prompt=lambda *args: {})
    record = {"id": "q1", "scene_id": "scene", "image": "unused.jpg", "question": "How far?",
              "answer": "100", "split": "val", "question_type": "object_abs_distance", "scoring": "spbench-si"}
    output = tmp_path / "predictions.jsonl"
    summary = evaluate(model, SimpleNamespace(tokenizer=tokenizer), encoder, [record], output_path=output)
    assert summary["overall_score"] == 1
    assert json.loads(output.read_text())["latent_rms"] == 0
    assert output.with_suffix(".jsonl.summary.json").is_file()
