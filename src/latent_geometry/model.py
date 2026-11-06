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


@dataclass
class RolloutOutput:
    full_sequence: Sequence
    latents: torch.Tensor
    past_key_values: object
    cached_token_count: int


class LatentReasoner(nn.Module):
    def __init__(self, backbone, config: LatentConfig, start_id: int, end_id: int):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.start_id, self.end_id = start_id, end_id
        backbone.configure_finetuning(config, (start_id, end_id))
        head_dtype = torch.float32 if config.finetuning == "lora" else backbone.dtype
        rms = backbone.model.get_input_embeddings().weight.detach().float().square().mean().sqrt().item()
        self.oracle = OracleAdapter(config.teacher_dim, backbone.hidden_size, rms).to(backbone.device)
        self.oracle.projection.to(dtype=head_dtype)
        self.latent_head = nn.Linear(backbone.hidden_size, backbone.hidden_size).to(
            device=backbone.device, dtype=head_dtype)
        nn.init.normal_(self.latent_head.weight, std=rms / backbone.hidden_size**0.5)
        nn.init.zeros_(self.latent_head.bias)
        self.set_stage(Stage.WARMUP)

    def set_stage(self, stage):
        if Stage(stage) == Stage.SFT:
            raise ValueError("Use AnswerOnlyReasoner for sft")
        self.stage = Stage(stage)
        self.backbone.set_trainable_parameters()
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
        return self.oracle(features.to(self.backbone.device)).to(self.backbone.dtype)

    def predict_latent(self, hidden):
        return self.latent_head(hidden.to(self.latent_head.weight.dtype)).to(self.backbone.dtype)

    def rollout(self, prefix: Sequence, *, use_cache=False):
        """Generate exactly K vectors. Uncached mode preserves the full training graph."""
        if use_cache and self.training:
            raise ValueError("Cached rollout is inference-only; use uncached differentiable training")
        full_sequence = decoder_input = prefix
        past_key_values, cached_token_count = None, 0
        vectors = []
        for _ in range(self.config.num_latents):
            output = self.backbone.decode(
                decoder_input, past_key_values=past_key_values,
                use_cache=use_cache, cached_token_count=cached_token_count)
            vector = self.predict_latent(output.last_hidden_state[:, -1:])
            vectors.append(vector)
            full_sequence = full_sequence.append(vector)
            if use_cache:
                past_key_values = output.past_key_values
                cached_token_count += decoder_input.embeddings.shape[1]
                decoder_input = Sequence(vector, full_sequence.positions[..., -1:])
            else:
                decoder_input = full_sequence
        # The final generated vector has not been consumed by the cached decoder yet.
        return RolloutOutput(full_sequence, torch.cat(vectors, dim=1),
                             past_key_values, cached_token_count)

    def _answer_loss(self, hidden, prefix_length, answer_ids):
        # The fixed latent_end boundary is an input, not a prediction target.
        # Its state predicts the first answer token; supervise the answer and
        # assistant terminator only, just like the answer-only baseline.
        labels = answer_ids
        first = prefix_length + self.config.num_latents
        logits = self.backbone.logits(hidden[:, first:first + labels.shape[1]])
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))

    def forward(self, inputs, answer_ids, teacher_features=None):
        answer_ids = answer_ids.to(self.backbone.device)
        if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] == 0:
            raise ValueError("answer_ids must be a nonempty unpadded [1, length] tensor")
        prefix = self.prefix(inputs)
        if self.stage == Stage.REFINEMENT:
            return self._refinement_loss(prefix, answer_ids)
        return self._oracle_conditioned_loss(prefix, answer_ids, teacher_features)

    def _decode_answer(self, sequence, answer_ids):
        sequence = sequence.append(self.backbone.token(self.end_id)).append(self.backbone.embed(answer_ids))
        return self.backbone.decode(sequence).last_hidden_state

    def _oracle_conditioned_loss(self, prefix, answer_ids, teacher_features):
        if teacher_features is None:
            raise ValueError("Warmup and alignment require pooled teacher features")
        latents = self._targets(teacher_features)
        if self.stage == Stage.ALIGNMENT:
            latents = latents.detach()
        prefix_length = prefix.embeddings.shape[1]
        hidden = self._decode_answer(prefix.append(latents), answer_ids)
        answer_loss = self._answer_loss(hidden, prefix_length, answer_ids)
        alignment_loss = answer_loss.new_zeros(())
        if self.stage == Stage.ALIGNMENT:
            # State immediately BEFORE each target; never regress a state that
            # already consumed the target at the same position.
            predictions = self.predict_latent(hidden[:, prefix_length - 1:prefix_length - 1 + self.config.num_latents])
            alignment_loss = F.mse_loss(predictions.float(), latents.float())
        loss = answer_loss + self.config.alignment_weight * alignment_loss
        return LossOutput(loss, answer_loss, alignment_loss, latents)

    def _refinement_loss(self, prefix, answer_ids):
        rollout = self.rollout(prefix)
        hidden = self._decode_answer(rollout.full_sequence, answer_ids)
        answer_loss = self._answer_loss(hidden, prefix.embeddings.shape[1], answer_ids)
        return LossOutput(answer_loss, answer_loss, answer_loss.new_zeros(()), rollout.latents)

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
                rollout = self.rollout(prefix, use_cache=use_cache)
                latents = rollout.latents
                past_key_values = rollout.past_key_values
                cached_token_count = rollout.cached_token_count
                full_sequence = rollout.full_sequence.append(self.backbone.token(self.end_id))
                # The cache ends before the final latent; consume it and the end boundary.
                decoder_input = (Sequence(full_sequence.embeddings[:, -2:], full_sequence.positions[..., -2:])
                                 if use_cache else full_sequence)
            else:
                latents = self._targets(oracle_features)
                full_sequence = prefix.append(latents).append(self.backbone.token(self.end_id))
                decoder_input = full_sequence
                past_key_values, cached_token_count = None, 0
            tokens = []
            for _ in range(max_new_tokens):
                output = self.backbone.decode(
                    decoder_input, past_key_values=past_key_values,
                    use_cache=use_cache, cached_token_count=cached_token_count)
                token_id = int(self.backbone.logits(output.last_hidden_state[:, -1]).argmax(-1).item())
                tokens.append(token_id)
                if token_id in eos_token_ids:
                    break
                full_sequence = full_sequence.append(self.backbone.token(token_id))
                if use_cache:
                    past_key_values = output.past_key_values
                    cached_token_count += decoder_input.embeddings.shape[1]
                    decoder_input = Sequence(full_sequence.embeddings[:, -1:], full_sequence.positions[..., -1:])
                else:
                    decoder_input = full_sequence
            return GenerationOutput(tokens, latents)
        finally:
            self.train(was_training)
