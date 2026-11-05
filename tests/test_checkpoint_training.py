import json

import pytest
import torch

from latent_geometry.checkpoint import restore_checkpoint, save_checkpoint
from latent_geometry.cli import load_run
from latent_geometry.config import Stage
from helpers import tiny_model
from latent_geometry.training import train


def test_checkpoint_exact_resume_and_stage_transition(tmp_path, model, example):
    model.set_stage(Stage.ALIGNMENT)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(**example).loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, optimizer=optimizer, progress={"step": 1})
    expected_random = torch.rand(5)
    optimizer.zero_grad(set_to_none=True)
    model(**example).loss.backward()
    optimizer.step()
    reloaded = tiny_model()
    reloaded.set_stage(Stage.ALIGNMENT)
    restored_optimizer = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=1e-3)
    progress = restore_checkpoint(path, reloaded, optimizer=restored_optimizer, restore_random=True)
    assert progress["step"] == 1
    torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
    reloaded(**example).loss.backward()
    restored_optimizer.step()
    for a, b in zip(model.parameters(), reloaded.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model)
    reloaded.set_stage(Stage.REFINEMENT)
    assert not reloaded.oracle.projection.weight.requires_grad
    assert reloaded.latent_head.weight.requires_grad
    with pytest.raises(ValueError, match="Resume must use"):
        restore_checkpoint(path, reloaded, optimizer=restored_optimizer)


class FixtureEncoder:
    max_prompt_tokens = 4096
    max_answer_tokens = 256

    def __init__(self, example):
        self.example = example

    def __call__(self, record, *, require_features):
        return {**self.example, "teacher_features": None}


def test_shared_subset_matches_sft_and_latent_exposure(tmp_path, example):
    from collections import Counter
    from latent_geometry.baseline import AnswerOnlyReasoner
    from latent_geometry.training import select_training_records

    records = [{"id": str(i), "split": "train"} for i in range(10)]
    records.append({"id": "held-out", "split": "val"})
    enriched = [{**r, "teacher_feature": "unused"} for r in reversed(records)]
    selected = select_training_records(records, 4, seed=42)
    assert [r["id"] for r in selected] == [
        r["id"] for r in select_training_records(enriched, 4, seed=42)]

    class RecordingEncoder(FixtureEncoder):
        def __init__(self):
            super().__init__(example)
            self.seen = []

        def __call__(self, record, **kwargs):
            self.seen.append(record["id"])
            return super().__call__(record, **kwargs)

    latent = tiny_model()
    baseline_backbone = tiny_model()
    baseline = AnswerOnlyReasoner(baseline_backbone.backbone, baseline_backbone.config)
    counts = []
    for model, stage, steps in ((latent, Stage.REFINEMENT, 2), (baseline, Stage.SFT, 6)):
        encoder = RecordingEncoder()
        progress = train(model, None, encoder, records, output_dir=tmp_path / stage.value,
                         stage=stage, steps=steps, accumulation_steps=2, max_train_examples=4,
                         gradient_checkpointing=False, bf16=False)
        assert progress["settings"]["training_example_ids"] == [r["id"] for r in selected]
        counts.append(Counter(encoder.seen))
    assert len(counts[0]) == 4
    assert counts[1] == Counter({key: value * 3 for key, value in counts[0].items()})


def test_training_runner_resume_matches_uninterrupted(tmp_path, example, monkeypatch):
    # Exercise real training-loop bookkeeping without downloading a tokenizer.
    records = [{"id": "fixture", "scene_id": "scene", "split": "train", "question": "Q", "answer": "A"}]
    encoder = FixtureEncoder(example)
    torch.manual_seed(42)
    continuous = tiny_model()
    continuous.oracle.set_statistics(torch.zeros(8), torch.ones(8))
    initial = tmp_path / "initial"
    continuous.set_stage(Stage.ALIGNMENT)
    save_checkpoint(initial, continuous)
    kwargs = dict(processor=None, encoder=encoder, records=records, stage=Stage.REFINEMENT,
                  accumulation_steps=2, learning_rate=1e-3, checkpoint_every=10,
                  gradient_checkpointing=True, bf16=False)
    train(continuous, output_dir=tmp_path / "continuous", steps=3, **kwargs)
    interrupted = tiny_model()
    restore_checkpoint(initial, interrupted)
    train(interrupted, output_dir=tmp_path / "interrupted", steps=1, **kwargs)
    checkpoint = tmp_path / "interrupted" / "checkpoint-000001"
    # CLI construction must leave weight loading to train() when resuming.
    monkeypatch.setattr("latent_geometry.cli.load_student",
                        lambda *args, **kwargs: (tiny_model().backbone, None, 60, 61))
    original_load = torch.load
    checkpoint_reads = []

    def count_checkpoint_reads(path, **kwargs):
        checkpoint_reads.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", count_checkpoint_reads)
    resumed, _, _ = load_run(checkpoint, load_weights=False)
    train(resumed, output_dir=tmp_path / "resumed", steps=3, resume=checkpoint, **kwargs)
    assert checkpoint_reads == [checkpoint / "state.pt"]
    for a, b in zip(continuous.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    metrics = [json.loads(line) for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [2, 3]
    assert metrics[-1]["examples_seen"] == 6


def test_compact_checkpoints_recover_then_finish_weights_only(tmp_path, example):
    class InterruptedEncoder(FixtureEncoder):
        calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("Simulated interruption")
            return super().__call__(*args, **kwargs)

    records = [{"id": "fixture", "scene_id": "scene", "split": "train", "question": "Q", "answer": "A"}]
    kwargs = dict(processor=None, records=records, stage=Stage.REFINEMENT, steps=3,
                  checkpoint_every=1, learning_rate=1e-3, gradient_checkpointing=False,
                  bf16=False, compact_checkpoints=True)
    torch.manual_seed(42)
    continuous = tiny_model()
    initial = {key: value.clone() for key, value in continuous.state_dict().items()}
    train(continuous, encoder=FixtureEncoder(example), output_dir=tmp_path / "continuous", **kwargs)
    interrupted = tiny_model()
    interrupted.load_state_dict(initial)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        train(interrupted, encoder=InterruptedEncoder(example), output_dir=output, **kwargs)
    recovery = output / "checkpoint-000002"
    assert list(output.glob("checkpoint-*")) == [recovery]
    assert torch.load(recovery / "state.pt", weights_only=True)["optimizer"] is not None
    full_size = (recovery / "state.pt").stat().st_size

    # Branching a resume must not remove the source directory's recovery checkpoint.
    branched = tiny_model()
    train(branched, encoder=FixtureEncoder(example), output_dir=tmp_path / "branched", resume=recovery, **kwargs)
    assert recovery.exists()
    train(interrupted, encoder=FixtureEncoder(example), output_dir=output, resume=recovery, **kwargs)
    final = output / "checkpoint-000003"
    assert list(output.glob("checkpoint-*")) == [final]
    assert json.loads((final / "metadata.json").read_text())["checkpoint_type"] == "weights_only"
    assert torch.load(final / "state.pt", weights_only=True)["optimizer"] is None
    assert (final / "state.pt").stat().st_size < full_size
    for a, b in zip(continuous.parameters(), interrupted.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    reloaded = tiny_model()
    restore_checkpoint(final, reloaded)
    for key, value in interrupted.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
    optimizer = torch.optim.AdamW(reloaded.parameters())
    with pytest.raises(ValueError, match="--init-from instead of --resume"):
        restore_checkpoint(final, reloaded, optimizer=optimizer)


def test_compact_checkpoint_save_failure_keeps_recovery(tmp_path, example, monkeypatch):
    def fail_second_save(path, *args, **kwargs):
        if path.name == "checkpoint-000002":
            raise OSError("Simulated disk error")
        save_checkpoint(path, *args, **kwargs)

    monkeypatch.setattr("latent_geometry.training.save_checkpoint", fail_second_save)
    records = [{"id": "fixture", "scene_id": "scene", "split": "train", "question": "Q", "answer": "A"}]
    with pytest.raises(OSError, match="Simulated disk error"):
        train(tiny_model(), None, FixtureEncoder(example), records, output_dir=tmp_path,
              stage=Stage.REFINEMENT, steps=3, checkpoint_every=1,
              gradient_checkpointing=False, bf16=False, compact_checkpoints=True)
    assert (tmp_path / "checkpoint-000001/state.pt").exists()
