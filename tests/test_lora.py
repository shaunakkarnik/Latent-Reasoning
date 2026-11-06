"""Exercise real PEFT adapters on tiny Qwen models, without pretrained downloads."""
import pytest
import torch

pytest.importorskip("peft")

from helpers import tiny_model
from latent_geometry.backbone import QwenBackbone
from latent_geometry.baseline import AnswerOnlyReasoner
from latent_geometry.checkpoint import restore_checkpoint, save_checkpoint
from latent_geometry.cli import load_run
from latent_geometry.config import LatentConfig, Stage


@pytest.mark.parametrize("stage", list(Stage))
def test_lora_updates_only_adapters_and_stage_heads(stage, example):
    if stage == Stage.SFT:
        original = tiny_model()
        model = AnswerOnlyReasoner(QwenBackbone(original.backbone.model),
                                  LatentConfig(finetuning="lora"))
    else:
        model = tiny_model(finetuning="lora")
        model.oracle.set_statistics(torch.zeros(8), torch.ones(8))
    model.set_stage(stage)
    model.backbone.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    vectors = []
    if stage == Stage.REFINEMENT:
        def retain_vector(module, args, output):
            output.retain_grad()
            vectors.append(output)
        model.latent_head.register_forward_hook(retain_vector)
    model(**example).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_B" in name)
    if stage != Stage.SFT:
        assert model.backbone.boundary_embeddings.grad.abs().sum() > 0
        assert model.oracle.projection.weight.requires_grad == (stage == Stage.WARMUP)
        assert model.latent_head.weight.requires_grad == (stage != Stage.WARMUP)
    if stage == Stage.REFINEMENT:
        assert len(vectors) == model.config.num_latents
        assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in vectors)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.model.") and "lora_" not in name:
            assert not parameter.requires_grad and parameter.grad is None, name
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
               if "lora_B" in name)
    # A subsequent stage must not accidentally unfreeze the base model.
    model.set_stage(Stage.SFT if stage == Stage.SFT else Stage.REFINEMENT)
    assert all(not p.requires_grad for name, p in model.backbone.model.named_parameters()
               if "lora_" not in name)


def test_lora_checkpoint_resume_and_configuration(tmp_path, example, monkeypatch):
    model = tiny_model(finetuning="lora")
    model.set_stage(Stage.REFINEMENT)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(**example).loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer=optimizer, progress={"step": 1})
    assert LatentConfig.load(checkpoint / "latent_config.json").finetuning == "lora"

    # Let load_run reconstruct the adapters using the checkpoint config.
    def load_tiny(config, **kwargs):
        return QwenBackbone(tiny_model().backbone.model), None, 60, 61
    monkeypatch.setattr("latent_geometry.cli.load_student", load_tiny)
    resumed, _, _ = load_run(checkpoint, load_weights=False)
    restored_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-3)
    restore_checkpoint(checkpoint, resumed, optimizer=restored_optimizer)
    for candidate, opt in ((model, optimizer), (resumed, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        candidate(**example).loss.backward()
        opt.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    resumed.config.lora_alpha *= 2
    with pytest.raises(ValueError, match="LoRA settings"):
        restore_checkpoint(checkpoint, resumed)
    with pytest.raises(ValueError, match="fine-tuning mode"):
        restore_checkpoint(checkpoint, tiny_model())


def test_lora_bf16_base_with_fp32_trainables_and_generation(example):
    model = tiny_model(finetuning="lora", dtype=torch.bfloat16)
    model.set_stage(Stage.REFINEMENT)
    assert model.backbone.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    model(**example).loss.backward()
    assert model.latent_head.weight.grad.abs().sum() > 0
    cached = model.generate(example["inputs"], max_new_tokens=2)
    uncached = model.generate(example["inputs"], max_new_tokens=2, use_cache=False)
    torch.testing.assert_close(cached.latents, uncached.latents, rtol=0.03, atol=0.001)
    assert cached.token_ids == uncached.token_ids


def test_original_baseline_ignores_lora_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    LatentConfig(finetuning="lora").save(path)
    monkeypatch.setattr("latent_geometry.cli.load_student",
                        lambda *args, **kwargs: (QwenBackbone(tiny_model().backbone.model), None, None, None))
    model, _, _ = load_run(config_path=path, answer_only=True, original_model=True)
    assert model.config.finetuning == "full"
    assert not any("lora_" in name for name, _ in model.named_parameters())
