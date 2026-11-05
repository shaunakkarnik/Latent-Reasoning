"""Small Qwen adapter; no monkey-patches or copied upstream model code.

The API/layout here is pinned to transformers 4.57.1. We encode images once,
obtain multimodal RoPE positions from Qwen, and call its text decoder directly.
"""
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class Sequence:
    embeddings: torch.Tensor  # [1, length, hidden]
    positions: torch.Tensor  # [3, 1, length], Qwen multimodal positions

    def append(self, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("This prototype uses one unpadded example per forward")
        count = embeddings.shape[1]
        start = self.positions.max() + 1
        positions = (torch.arange(count, device=self.positions.device) + start).view(1, 1, -1)
        return Sequence(torch.cat((self.embeddings, embeddings), dim=1),
                        torch.cat((self.positions, positions.expand(3, 1, -1)), dim=-1))


class QwenBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if not hasattr(model.model, "language_model"):
            raise ValueError("Expected Qwen2.5-VL layout from transformers==4.57.1")
        self.model.model.visual.requires_grad_(False)
        self.model.model.visual.eval()

    @property
    def hidden_size(self):
        return self.model.get_input_embeddings().embedding_dim

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.model.get_input_embeddings().weight.dtype

    def train(self, mode=True):
        super().train(mode)
        self.model.model.visual.eval()
        return self

    def embed(self, ids):
        return self.model.get_input_embeddings()(ids.to(self.device))

    def token(self, token_id):
        return self.embed(torch.tensor([[token_id]], device=self.device))

    def prepare(self, inputs: dict[str, Any]):
        ids = inputs["input_ids"].to(self.device)
        mask = inputs.get("attention_mask", torch.ones_like(ids)).to(self.device)
        if ids.shape[0] != 1 or not bool(mask.bool().all()):
            raise ValueError("Use a single unpadded prompt; accumulate gradients across examples")
        if "pixel_values_videos" in inputs or "video_grid_thw" in inputs:
            raise ValueError("Video is not supported in the single-image prototype")
        embeddings = self.embed(ids)
        grid = inputs.get("image_grid_thw")
        has_image = bool((ids == self.model.config.image_token_id).any())
        if has_image:
            if grid is None or len(grid) != 1 or "pixel_values" not in inputs:
                raise ValueError("Exactly one source image and image grid are required")
            grid = grid.to(self.device)
            with torch.no_grad():
                features = torch.cat(self.model.get_image_features(
                    inputs["pixel_values"].to(self.device), grid), dim=0)
            image_mask = (ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
            if int(image_mask.sum()) != features.numel():
                raise ValueError("Image placeholder count does not match vision features")
            embeddings = embeddings.masked_scatter(image_mask, features.to(embeddings.dtype))
        elif grid is not None or "pixel_values" in inputs:
            raise ValueError("Pixels provided without image placeholders")
        positions, _ = self.model.model.get_rope_index(
            input_ids=ids, image_grid_thw=grid, attention_mask=mask)
        return Sequence(embeddings, positions)

    def decode(self, sequence, *, past_key_values=None, use_cache=False, cached_token_count=0):
        length = sequence.embeddings.shape[1]
        return self.model.model.language_model(
            inputs_embeds=sequence.embeddings,
            position_ids=sequence.positions,
            attention_mask=torch.ones((1, cached_token_count + length), dtype=torch.long, device=self.device),
            cache_position=torch.arange(cached_token_count, cached_token_count + length, device=self.device),
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

    def logits(self, hidden):
        return self.model.lm_head(hidden)


def load_student(config, *, dtype=torch.bfloat16, device="cpu", processor_path=None, add_latent_tokens=True):
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install the validated transformers==4.57.1 API before loading Qwen")
    processor = AutoProcessor.from_pretrained(
        processor_path or config.student,
        **({} if processor_path else {"revision": config.student_revision}),
        max_pixels=config.max_pixels, use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.student, revision=config.student_revision,
        dtype=dtype, attn_implementation="sdpa",
    )
    # Record the resolved upstream commit when loading a named revision.
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved:
        config.student_revision = resolved
    if add_latent_tokens:
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.start_token, config.end_token]},
            replace_additional_special_tokens=False,
        )
        model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)
    start_id = processor.tokenizer.convert_tokens_to_ids(config.start_token) if add_latent_tokens else None
    end_id = processor.tokenizer.convert_tokens_to_ids(config.end_token) if add_latent_tokens else None
    return QwenBackbone(model), processor, start_id, end_id
