import json

import pytest
import torch

from latent_geometry.checkpoint import restore_checkpoint, save_checkpoint
from latent_geometry.config import Stage
from latent_geometry.smoke import tiny_model
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


def test_training_runner_resume_matches_uninterrupted(tmp_path, example):
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
    resumed = tiny_model()
    resumed.set_stage(Stage.REFINEMENT)
    checkpoint = tmp_path / "interrupted" / "checkpoint-000001"
    restore_checkpoint(checkpoint, resumed)
    train(resumed, output_dir=tmp_path / "resumed", steps=3, resume=checkpoint, **kwargs)
    for a, b in zip(continuous.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    metrics = [json.loads(line) for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [2, 3]
    assert metrics[-1]["examples_seen"] == 6
