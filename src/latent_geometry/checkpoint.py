"""Student checkpoints, optionally including optimizer state for exact resumption."""
import json
from pathlib import Path
import random
import shutil

import torch


def capture_rng():
    result = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, *, processor=None, optimizer=None, progress=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete checkpoint exists; inspect before removing: {temporary}")
    temporary.mkdir()
    try:
        model.config.save(temporary / "latent_config.json")
        metadata = {"format_version": 1, "stage": model.stage.value,
                    "checkpoint_type": "full" if optimizer is not None else "weights_only",
                    "model_kind": "answer_only" if model.stage.value == "sft" else "latent",
                    "start_id": model.start_id, "end_id": model.end_id,
                    "transformers": "4.57.1", "torch": str(torch.__version__)}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if processor is not None:
            processor.save_pretrained(temporary / "processor")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "progress": progress or {}, "rng": capture_rng()}, temporary / "state.pt")
        temporary.rename(path)
    except Exception:
        shutil.rmtree(temporary)
        raise


def restore_checkpoint(path, model, *, optimizer=None, restore_random=False):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    if (metadata["start_id"], metadata["end_id"]) != (model.start_id, model.end_id):
        raise ValueError("Checkpoint latent boundary token IDs do not match tokenizer")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if optimizer is not None:
        if state["optimizer"] is None:
            raise ValueError("Checkpoint has no optimizer state; use --init-from instead of --resume")
        if metadata["stage"] != model.stage.value:
            raise ValueError("Resume must use the checkpoint stage; use --init-from for a new stage")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_random:
        restore_rng(state["rng"])
    return state["progress"]
