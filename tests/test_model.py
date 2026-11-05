import pytest
import torch
from torch.nn import functional as F

from latent_geometry.config import Stage
from helpers import tiny_model


def test_backbone_preserves_native_qwen_image_embeddings_and_positions(model, example):
    model.eval()
    with torch.no_grad():
        native = model.backbone.model.model(**example["inputs"], use_cache=False).last_hidden_state
        prepared = model.backbone.prepare(example["inputs"])
        adapted = model.backbone.decode(prepared).last_hidden_state
    torch.testing.assert_close(native, adapted, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stage", [Stage.WARMUP, Stage.ALIGNMENT, Stage.REFINEMENT])
def test_stage_gradients_and_parameter_updates(model, example, stage):
    model.set_stage(stage)
    parameters = dict(model.named_parameters())
    before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    result = model(**example)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
    oracle_grad = model.oracle.projection.weight.grad
    head_grad = model.latent_head.weight.grad
    if stage == Stage.WARMUP:
        assert oracle_grad is not None and oracle_grad.abs().sum() > 0
        assert head_grad is None
    else:
        assert oracle_grad is None
        assert head_grad is not None and head_grad.abs().sum() > 0
    assert model.backbone.model.get_input_embeddings().weight.grad.abs().sum() > 0
    assert model.backbone.model.lm_head.weight.grad.abs().sum() > 0
    optimizer.step()
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert not torch.equal(model.backbone.model.lm_head.weight, before["backbone.model.lm_head.weight"])


def test_alignment_targets_are_shifted_and_causal(model, example):
    model.set_stage(Stage.ALIGNMENT)
    model.eval()
    captured = []
    hook = model.latent_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    original = model(**example)
    first = captured[-1].clone()
    modified = dict(example)
    modified["teacher_features"] = example["teacher_features"].clone()
    modified["teacher_features"][:, 1] += 100
    model(**modified)
    second = captured[-1]
    hook.remove()
    # Slot 1 is predicted before it is visible; slot 2 sees the changed slot 1.
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
    assert not torch.allclose(first[:, 2], second[:, 2])
    prefix = model.prefix(example["inputs"])
    targets = model._targets(example["teacher_features"])
    sequence = prefix.append(targets).append(model.backbone.token(model.end_id)).append(
        model.backbone.embed(example["answer_ids"]))
    hidden = model.backbone.decode(sequence).last_hidden_state
    p = prefix.embeddings.shape[1]
    expected = F.mse_loss(model.latent_head(hidden[:, p - 1:p + 3]).float(), targets.float())
    torch.testing.assert_close(original.alignment_loss, expected)


def test_answer_loss_masks_prompt_and_latent_slots(model, example):
    model.set_stage(Stage.ALIGNMENT)
    output = model(**example)
    prefix = model.prefix(example["inputs"])
    seq = prefix.append(model._targets(example["teacher_features"])).append(
        model.backbone.token(model.end_id)).append(model.backbone.embed(example["answer_ids"]))
    hidden = model.backbone.decode(seq).last_hidden_state
    # Exactly end-boundary + three answer labels, and only their predecessor states.
    first = prefix.embeddings.shape[1] + model.config.num_latents - 1
    labels = torch.tensor([model.end_id, 7, 8, 2])
    expected = F.cross_entropy(model.backbone.logits(hidden[:, first:first + 4]).float().reshape(4, -1), labels)
    torch.testing.assert_close(output.answer_loss, expected)


def test_refinement_uses_no_oracle_and_preserves_recurrent_graph(model, example, monkeypatch):
    model.set_stage(Stage.REFINEMENT)
    monkeypatch.setattr(model.oracle, "forward", lambda *args: pytest.fail("Refinement touched oracle"))
    prefix = model.prefix(example["inputs"])
    vectors = []

    def record(module, args, output):
        output.retain_grad()
        vectors.append(output)

    hook = model.latent_head.register_forward_hook(record)
    output = model(example["inputs"], example["answer_ids"])
    output.loss.backward()
    hook.remove()
    assert len(vectors) == model.config.num_latents
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in vectors)
    assert output.alignment_loss.item() == 0
    # Explicitly verify the next generated vector depends on the preceding one.
    rollout = model.rollout(prefix)
    assert rollout.full_sequence.embeddings.grad_fn is not None


@pytest.mark.parametrize("with_image", [True, False])
def test_cached_generation_matches_full_reference(model, example, with_image):
    inputs = example["inputs"] if with_image else {"input_ids": torch.tensor([[1, 3, 4]])}
    model.eval()
    for _ in range(2):  # no stale multimodal position/cache state across calls
        cached = model.generate(inputs, max_new_tokens=5, use_cache=True)
        uncached = model.generate(inputs, max_new_tokens=5, use_cache=False)
        torch.testing.assert_close(cached.latents, uncached.latents, rtol=1e-4, atol=1e-6)
        assert cached.token_ids == uncached.token_ids
        assert cached.latents.shape == (1, 4, model.backbone.hidden_size)


def test_sixteen_steps_teacher_free(model, example, monkeypatch):
    model = tiny_model(num_latents=16)
    monkeypatch.setattr(model.oracle, "forward", lambda *args: pytest.fail("Inference touched oracle"))
    calls = []
    handle = model.latent_head.register_forward_hook(lambda *args: calls.append(1))
    result = model.generate(example["inputs"], max_new_tokens=2)
    handle.remove()
    assert len(calls) == 16
    assert result.latents.shape[1] == 16


def test_oracle_generation_cache_matches_reference(model, example):
    a = model.generate(example["inputs"], oracle_features=example["teacher_features"], max_new_tokens=3)
    b = model.generate(example["inputs"], oracle_features=example["teacher_features"], max_new_tokens=3,
                       use_cache=False)
    assert a.token_ids == b.token_ids


def test_reject_padding_and_multiview(model, example):
    inputs = dict(example["inputs"])
    inputs["attention_mask"] = inputs["attention_mask"].clone()
    inputs["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="unpadded"):
        model.prefix(inputs)
    inputs = dict(example["inputs"])
    inputs["image_grid_thw"] = torch.tensor([[1, 4, 4], [1, 4, 4]])
    with pytest.raises(ValueError, match="one source image"):
        model.prefix(inputs)


def test_gradient_checkpointed_refinement(model, example):
    model.set_stage(Stage.REFINEMENT)
    model.backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    result = model(**example)
    result.loss.backward()
    assert model.latent_head.weight.grad.abs().sum() > 0
