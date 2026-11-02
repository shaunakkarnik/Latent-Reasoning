"""Synthetic wiring check using the actual Qwen architecture, without downloads.

Random image tensors and fake teacher features cannot validate VGGT quality or
spatial reasoning. They exercise the same model/loss/rollout/checkpoint code.
"""
from pathlib import Path

import torch

from .backbone import QwenBackbone
from .checkpoint import restore_checkpoint, save_checkpoint
from .config import LatentConfig, Stage
from .model import LatentReasoner


def tiny_model(*, num_latents=4):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 64, "intermediate_size": 96,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "max_position_embeddings": 1024, "rope_scaling": {"type": "default", "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                       "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": 64, "fullatt_block_indexes": [0], "window_size": 8},
        image_token_id=58, video_token_id=56, vision_start_token_id=57, vision_end_token_id=59,
    )
    config._attn_implementation = "sdpa"
    backbone = QwenBackbone(Qwen2_5_VLForConditionalGeneration(config))
    return LatentReasoner(backbone, LatentConfig(teacher_dim=8, pool_height=1, pool_width=num_latents), 60, 61)


def tiny_example():
    # 4x4 patches -> 2x2 merged image embeddings; patches include 2 temporal frames.
    return {"inputs": {"input_ids": torch.tensor([[1, 57, 58, 58, 58, 58, 59, 3, 4]]),
                       "attention_mask": torch.ones(1, 9, dtype=torch.long),
                       "image_grid_thw": torch.tensor([[1, 4, 4]]),
                       "pixel_values": torch.linspace(0, 1, 16 * 24).reshape(16, 24)},
            "answer_ids": torch.tensor([[7, 8, 2]]),
            "teacher_features": torch.linspace(-1, 1, 32).reshape(1, 4, 8)}


def run_smoke(output_dir, *, steps=12):
    if steps < 2:
        raise ValueError("At least two steps are needed for the learning check")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, example = tiny_model(num_latents=16), tiny_example()
    example["teacher_features"] = example["teacher_features"].repeat(1, 4, 1)
    model.oracle.set_statistics(torch.zeros(8), torch.ones(8))
    frozen = {n: p.detach().clone() for n, p in model.backbone.model.model.visual.named_parameters()}
    results = {}
    for stage in Stage:
        model.set_stage(stage)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = model(**example)
            if not torch.isfinite(output.loss):
                raise AssertionError("Nonfinite tiny-model loss")
            output.loss.backward()
            if stage == Stage.REFINEMENT:
                grad = model.latent_head.weight.grad
                assert grad is not None and grad.abs().sum() > 0, "Answer loss did not reach latent generator"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output.loss.detach()))
        if losses[-1] >= losses[0]:
            raise AssertionError(f"Tiny-set loss did not decrease in {stage.value}")
        results[stage.value] = {"first_loss": losses[0], "last_loss": losses[-1]}
    for name, parameter in model.backbone.model.model.visual.named_parameters():
        torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    model.eval()
    cached = model.generate(example["inputs"], max_new_tokens=5, eos_token_ids=[2], use_cache=True)
    reference = model.generate(example["inputs"], max_new_tokens=5, eos_token_ids=[2], use_cache=False)
    torch.testing.assert_close(cached.latents, reference.latents, rtol=1e-4, atol=1e-5)
    assert cached.token_ids == reference.token_ids
    path = Path(output_dir) / "checkpoint"
    save_checkpoint(path, model, optimizer=optimizer, progress={"step": steps})
    reloaded = tiny_model(num_latents=16)
    reloaded.set_stage(Stage.REFINEMENT)
    restored_optimizer = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=3e-3)
    restore_checkpoint(path, reloaded, optimizer=restored_optimizer, restore_random=True)
    restored = reloaded.generate(example["inputs"], max_new_tokens=5, eos_token_ids=[2])
    assert cached.token_ids == restored.token_ids
    torch.testing.assert_close(cached.latents, restored.latents, rtol=0, atol=0)
    restored_optimizer.zero_grad(set_to_none=True)
    reloaded(**example).loss.backward()
    restored_optimizer.step()
    results.update({"cached_matches_reference": True, "frozen_vision_unchanged": True,
                    "checkpoint_reload_and_resume": True, "generated_latent_count": cached.latents.shape[1],
                    "generated_token_ids": cached.token_ids,
                    "note": "Tiny random Qwen and synthetic teacher features; no pretrained-model quality claim"})
    return results
