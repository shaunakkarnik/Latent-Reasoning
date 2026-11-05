"""Download SpatialLadder single-image QA and SPBench-SI on the training server."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
from zipfile import ZipFile

from .data import read_manifest


QUESTION_TYPES = {
    "relative direction": "object_rel_direction",
    "relative distance": "object_rel_distance",
    "absolute distance": "object_abs_distance",
    "object size": "object_size_estimation",
}


def room_id(scan_id):
    # ScanNet sceneXXXX_00 and sceneXXXX_01 are rescans of the same room.
    if not re.fullmatch(r"scene\d{4}_\d{2}", scan_id):
        raise ValueError(f"Unexpected ScanNet scan ID: {scan_id}")
    return scan_id.split("_")[0]


def format_question(question, options):
    """Use SPBench's direct-answer prompt for every model and every split."""
    text = question.strip()
    if options:
        text += "\nOptions:\n" + "\n".join(option.strip() for option in options)
        instruction = "Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly."
    else:
        instruction = "Please answer the question using a numerical value (e.g., 42 or 3.1) directly."
    return f"Question: {text}\n\n{instruction}"


def build_records(training_rows, test_rows, *, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rows = [r for r in training_rows if r["data_type"] == "single_image"]
    rooms = sorted({room_id(PurePosixPath(r["image"][0]).parent.name) for r in rows})
    if len(rooms) < 2:
        raise ValueError("Need at least two training rooms for a validation holdout")
    random.Random(seed).shuffle(rooms)
    count = max(1, min(len(rooms) - 1, round(len(rooms) * val_fraction)))
    val_rooms = set(rooms[:count])
    test_rooms = {room_id(r["scene_name"]) for r in test_rows}
    if set(rooms) & test_rooms:
        raise ValueError("Training and SPBench-SI contain overlapping rooms")
    records = []
    for source, examples in (("spatialladder", rows), ("spbench-si", test_rows)):
        for row in examples:
            is_train = source == "spatialladder"
            images = row["image"] if is_train else row["images"]
            if len(images) != 1:
                raise ValueError("Single-image QA must contain exactly one image")
            image = PurePosixPath(images[0])
            scan = image.parent.name if is_train else row["scene_name"]
            room = room_id(scan)
            question_type = QUESTION_TYPES[row["question_type"]] if is_train else row["question_type"]
            if question_type not in QUESTION_TYPES.values():
                raise ValueError(f"Unsupported single-image question type: {question_type}")
            options = row.get("options") or []
            answer = str(row["answer"] if is_train else row["ground_truth"]).strip()
            if bool(options) != (question_type in ("object_rel_direction", "object_rel_distance")):
                raise ValueError("Options disagree with question type")
            if options and answer not in [option.split(".", 1)[0].strip() for option in options]:
                raise ValueError(f"Answer is not an option letter: {answer}")
            records.append({
                "id": f"{source}-{row['question_id'] if is_train else row['id']}",
                "scene_id": room, "scan_id": scan,
                "image": f"images/{source}/{scan}/{image.name}",
                "question": format_question(row["question"], options), "answer": answer,
                "split": ("val" if room in val_rooms else "train") if is_train else "test",
                "question_type": question_type, "options": options, "source": source,
                "scoring": "spbench-si",
            })
    return records, sorted(val_rooms)


def extract_images(archive, destination, image_keys):
    """Extract only requested scene/frame pairs, ignoring archive wrapper folders."""
    destination = Path(destination)
    with ZipFile(archive) as zipped:
        members = {}
        for member in zipped.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            key = "/".join(parts[-2:])
            if key in image_keys:
                if key in members:
                    raise ValueError(f"Ambiguous archive image: {key}")
                members[key] = member
        missing = image_keys - members.keys()
        if missing:
            raise ValueError(f"Archive is missing requested images: {sorted(missing)[:5]}")
        for key, member in members.items():
            # Write to our own validated scene/frame path, never the archive path.
            target = destination / key
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            temporary = target.with_suffix(target.suffix + ".incomplete")
            with zipped.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)


def prepare_spatialladder(output_dir, *, val_fraction=0.1, seed=42,
                         training_revision="main", benchmark_revision="main"):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    output_dir = Path(output_dir).resolve()
    manifest = output_dir / "spatial.jsonl"
    if manifest.exists():
        raise FileExistsError(f"{manifest} exists; use a new directory to change the split")
    api = HfApi()
    sources = {}
    files = {}
    for name, repo, revision, names in (
        ("spatialladder", "hongxingli/SpatialLadder-26k", training_revision,
         ("spld_spatial_data.jsonl", "images.zip")),
        ("spbench-si", "hongxingli/SPBench", benchmark_revision,
         ("SPBench-SI.parquet", "SPBench-SI-images.zip")),
    ):
        commit = api.dataset_info(repo, revision=revision).sha
        sources[name] = {"repo": repo, "revision": commit}
        for filename in names:
            files[filename] = hf_hub_download(repo, filename, repo_type="dataset", revision=commit,
                local_dir=output_dir / "downloads" / name)
    training_rows = [json.loads(line) for line in Path(files["spld_spatial_data.jsonl"]).read_text().splitlines()
                     if line.strip()]
    test_rows = pq.read_table(files["SPBench-SI.parquet"]).to_pylist()
    records, val_rooms = build_records(training_rows, test_rows, val_fraction=val_fraction, seed=seed)
    for name, archive in (("spatialladder", "images.zip"), ("spbench-si", "SPBench-SI-images.zip")):
        keys = {"/".join(PurePosixPath(r["image"]).parts[-2:]) for r in records if r["source"] == name}
        extract_images(files[archive], output_dir / "images" / name, keys)
    # Validate paths, image hashes, and leakage before publishing the final manifest.
    temporary = output_dir / "spatial.incomplete.jsonl"
    temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
    read_manifest(temporary)
    metadata = {"sources": sources, "seed": seed, "val_fraction": val_fraction,
                "split_unit": "ScanNet room (all rescans grouped)", "validation_rooms": val_rooms,
                "counts": dict(Counter(r["split"] for r in records)),
                "question_types": dict(Counter(r["question_type"] for r in records)),
                "prompt": "SPBench default direct-answer; identical across splits/models"}
    (output_dir / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.rename(manifest)
    return {"manifest": str(manifest), **metadata}
