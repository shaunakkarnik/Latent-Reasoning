from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Sequence
from .config import LatentConfig, Stage


class OracleAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_size, embedding_rms):
        super().__init__()
        # Keep statistics in FP32; cast only the normalized features for projection.
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))
        self.register_buffer("stats_ready", torch.tensor(False))
        self.projection = nn.Linear(feature_dim, hidden_size)
        nn.init.normal_(self.projection.weight, std=embedding_rms / feature_dim**0.5)
        nn.init.zeros_(self.projection.bias)

    @torch.no_grad()
    def set_statistics(self, mean, std):
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError("Training statistics do not match teacher feature dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Invalid feature statistics")
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))
        self.stats_ready.fill_(True)

    def forward(self, features):
        if not bool(self.stats_ready):
            raise RuntimeError("Fit normalization on training-only teacher features first")
        normalized = (features.float() - self.mean.float()) / self.std.float()
        return self.projection(normalized.to(self.projection.weight.dtype))


@dataclass
class LossOutput:
    loss: torch.Tensor
    answer_loss: torch.Tensor
    alignment_loss: torch.Tensor
    latents: torch.Tensor


@dataclass
class GenerationOutput:
    token_ids: list[int]
    latents: torch.Tensor


class LatentReasoner(nn.Module):
    def __init__(self, backbone, config: LatentConfig, start_id: int, end_id: int):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.start_id, self.end_id = start_id, end_id
        rms = backbone.model.get_input_embeddings().weight.detach().float().square().mean().sqrt().item()
        self.oracle = OracleAdapter(config.teacher_dim, backbone.hidden_size, rms).to(backbone.device)
        self.oracle.projection.to(dtype=backbone.dtype)
        self.latent_head = nn.Linear(backbone.hidden_size, backbone.hidden_size).to(
            device=backbone.device, dtype=backbone.dtype)
        nn.init.normal_(self.latent_head.weight, std=rms / backbone.hidden_size**0.5)
        nn.init.zeros_(self.latent_head.bias)
        self.set_stage(Stage.WARMUP)

    def set_stage(self, stage):
        self.stage = Stage(stage)
        self.backbone.requires_grad_(True)
        self.backbone.model.model.visual.requires_grad_(False)
        self.oracle.requires_grad_(self.stage == Stage.WARMUP)
        self.latent_head.requires_grad_(self.stage != Stage.WARMUP)
        # Clear stale gradients when changing stages before building a new optimizer.
        self.zero_grad(set_to_none=True)

    def prefix(self, inputs):
        return self.backbone.prepare(inputs).append(self.backbone.token(self.start_id))

    def _targets(self, features):
        expected = (1, self.config.num_latents, self.config.teacher_dim)
        if tuple(features.shape) != expected or not torch.isfinite(features).all():
            raise ValueError(f"Expected finite teacher features with shape {expected}")
        return self.oracle(features.to(self.backbone.device))

    def rollout(self, prefix: Sequence, *, use_cache=False):
        """Generate exactly K vectors. Uncached mode preserves the full training graph."""
        if use_cache and self.training:
            raise ValueError("Cached rollout is inference-only; use uncached differentiable training")
        sequence, current = prefix, prefix
        cache, offset = None, 0
        vectors = []
        for _ in range(self.config.num_latents):
            output = self.backbone.decode(current, past_key_values=cache, use_cache=use_cache, offset=offset)
            vector = self.latent_head(output.last_hidden_state[:, -1:])
            vectors.append(vector)
            sequence = sequence.append(vector)
            if use_cache:
                cache = output.past_key_values
                offset += current.embeddings.shape[1]
                current = Sequence(vector, sequence.positions[..., -1:])
            else:
                current = sequence
        # The final generated vector has not been consumed by the cached decoder yet.
        return sequence, torch.cat(vectors, dim=1), cache, offset

    def _answer_loss(self, hidden, prefix_length, answer_ids):
        # Fixed start control is masked. Predict latent_end from the final latent,
        # then every answer token (including the assistant terminator). Oracle
        # slots and all original prompt tokens have no vocabulary loss.
        labels = torch.cat((torch.tensor([[self.end_id]], device=hidden.device), answer_ids), dim=1)
        first = prefix_length + self.config.num_latents - 1
        logits = self.backbone.logits(hidden[:, first:first + labels.shape[1]])
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))

    def forward(self, inputs, answer_ids, teacher_features=None):
        answer_ids = answer_ids.to(self.backbone.device)
        if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] == 0:
            raise ValueError("answer_ids must be a nonempty unpadded [1, length] tensor")
        prefix = self.prefix(inputs)
        prefix_length = prefix.embeddings.shape[1]
        if self.stage == Stage.REFINEMENT:
            sequence, latents, _, _ = self.rollout(prefix)
        else:
            if teacher_features is None:
                raise ValueError("Warmup and alignment require pooled teacher features")
            latents = self._targets(teacher_features)
            if self.stage == Stage.ALIGNMENT:
                latents = latents.detach()
            sequence = prefix.append(latents)
        sequence = sequence.append(self.backbone.token(self.end_id)).append(self.backbone.embed(answer_ids))
        hidden = self.backbone.decode(sequence).last_hidden_state
        answer_loss = self._answer_loss(hidden, prefix_length, answer_ids)
        alignment_loss = answer_loss.new_zeros(())
        if self.stage == Stage.ALIGNMENT:
            # State immediately BEFORE each target; never regress a state that
            # already consumed the target at the same position.
            predictions = self.latent_head(hidden[:, prefix_length - 1:prefix_length - 1 + self.config.num_latents])
            alignment_loss = F.mse_loss(predictions.float(), latents.float())
        loss = answer_loss + self.config.alignment_weight * alignment_loss
        return LossOutput(loss, answer_loss, alignment_loss, latents)

    @torch.no_grad()
    def generate(self, inputs, *, max_new_tokens=64, eos_token_ids=(), use_cache=True,
                 oracle_features=None):
        """Greedy decoding. Default path never touches VGGT, cache files, or adapter."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        was_training = self.training
        self.eval()
        try:
            prefix = self.prefix(inputs)
            if oracle_features is None:
                sequence, latents, cache, offset = self.rollout(prefix, use_cache=use_cache)
                sequence = sequence.append(self.backbone.token(self.end_id))
                current = (Sequence(sequence.embeddings[:, -2:], sequence.positions[..., -2:])
                           if use_cache else sequence)
            else:
                latents = self._targets(oracle_features)
                sequence = prefix.append(latents).append(self.backbone.token(self.end_id))
                current, cache, offset = sequence, None, 0
            tokens = []
            for _ in range(max_new_tokens):
                output = self.backbone.decode(current, past_key_values=cache,
                                              use_cache=use_cache, offset=offset)
                token_id = int(self.backbone.logits(output.last_hidden_state[:, -1]).argmax(-1).item())
                tokens.append(token_id)
                if token_id in eos_token_ids:
                    break
                sequence = sequence.append(self.backbone.token(token_id))
                if use_cache:
                    cache = output.past_key_values
                    offset += current.embeddings.shape[1]
                    current = Sequence(sequence.embeddings[:, -1:], sequence.positions[..., -1:])
                else:
                    current = sequence
            return GenerationOutput(tokens, latents)
        finally:
            self.train(was_training)
