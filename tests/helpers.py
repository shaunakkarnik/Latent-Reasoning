"""Tiny random Qwen fixtures for offline model tests."""
import torch

from latent_geometry.backbone import QwenBackbone
from latent_geometry.config import LatentConfig
from latent_geometry.model import LatentReasoner


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
