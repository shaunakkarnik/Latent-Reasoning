import torch

from latent_geometry.baseline import AnswerOnlyReasoner
from latent_geometry.config import Stage
from latent_geometry.training import train
from helpers import tiny_model, tiny_example


def test_baseline_matches_native_qwen_and_cached_generation():
    latent = tiny_model()
    model = AnswerOnlyReasoner(latent.backbone, latent.config)
    example = tiny_example()
    inputs, answer = example["inputs"], example["answer_ids"]
    model.eval()
    result = model(inputs, answer)
    ids = torch.cat((inputs["input_ids"], answer), dim=1)
    labels = ids.clone()
    labels[:, :inputs["input_ids"].shape[1]] = -100
    expected = model.backbone.model(input_ids=ids, attention_mask=torch.ones_like(ids),
        labels=labels, pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
        use_cache=False)
    torch.testing.assert_close(result.loss, expected.loss)
    result.loss.backward()
    assert model.backbone.model.lm_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.model.model.visual.parameters())
    a = model.generate(inputs, max_new_tokens=5, use_cache=True)
    b = model.generate(inputs, max_new_tokens=5, use_cache=False)
    assert a.token_ids == b.token_ids
    assert a.latents.shape[1] == 0


def test_baseline_training_checkpoint_resume(tmp_path, monkeypatch):
    from test_checkpoint_training import FixtureEncoder

    latent = tiny_model()
    model = AnswerOnlyReasoner(latent.backbone, latent.config)
    records = [{"id": "fixture", "scene_id": "scene", "split": "train", "question": "Q", "answer": "A"}]
    kwargs = dict(processor=None, encoder=FixtureEncoder(tiny_example()), records=records,
                  stage=Stage.SFT, learning_rate=1e-3, gradient_checkpointing=True, bf16=False)
    train(model, output_dir=tmp_path / "first", steps=1, **kwargs)
    checkpoint = tmp_path / "first/checkpoint-000001"
    from latent_geometry.cli import load_run

    def load_baseline(*args, **kwargs):
        assert kwargs["add_latent_tokens"] is False
        return tiny_model().backbone, None, None, None

    monkeypatch.setattr("latent_geometry.cli.load_student", load_baseline)
    restored, _, _ = load_run(checkpoint)
    assert isinstance(restored, AnswerOnlyReasoner)
    assert restored.evaluation_source == {
        "model_kind": "sft", "checkpoint_path": str(checkpoint.resolve()),
        "stage": "sft", "training_step": 1,
    }
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    progress = train(restored, output_dir=tmp_path / "resumed", steps=2, resume=checkpoint, **kwargs)
    assert progress["step"] == 2
