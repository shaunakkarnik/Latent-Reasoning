"""Answer-only Qwen baseline using the same image encoder and training loop."""
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Sequence
from .config import Stage
from .model import GenerationOutput, LossOutput


class AnswerOnlyReasoner(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone, self.config = backbone, config
        self.start_id = self.end_id = None
        backbone.configure_finetuning(config)
        self.set_stage(Stage.SFT)

    def set_stage(self, stage):
        if Stage(stage) != Stage.SFT:
            raise ValueError("Answer-only models support only the sft stage")
        self.stage = Stage.SFT
        self.backbone.set_trainable_parameters()
        self.zero_grad(set_to_none=True)

    def forward(self, inputs, answer_ids, teacher_features=None):
        prefix = self.backbone.prepare(inputs)
        answer_ids = answer_ids.to(self.backbone.device)
        sequence = prefix.append(self.backbone.embed(answer_ids))
        hidden = self.backbone.decode(sequence).last_hidden_state
        first = prefix.embeddings.shape[1] - 1
        logits = self.backbone.logits(hidden[:, first:first + answer_ids.shape[1]])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), answer_ids.reshape(-1))
        return LossOutput(loss, loss, loss.new_zeros(()), hidden.new_empty(1, 0, self.backbone.hidden_size))

    @torch.no_grad()
    def generate(self, inputs, *, max_new_tokens=64, eos_token_ids=(), use_cache=True,
                 oracle_features=None):
        if oracle_features is not None:
            raise ValueError("The answer-only baseline does not accept oracle features")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        was_training = self.training
        self.eval()
        try:
            full_sequence = decoder_input = self.backbone.prepare(inputs)
            cache, cached_count, tokens = None, 0, []
            for _ in range(max_new_tokens):
                output = self.backbone.decode(decoder_input, past_key_values=cache,
                    use_cache=use_cache, cached_token_count=cached_count)
                token = int(self.backbone.logits(output.last_hidden_state[:, -1]).argmax(-1).item())
                tokens.append(token)
                if token in eos_token_ids:
                    break
                full_sequence = full_sequence.append(self.backbone.token(token))
                if use_cache:
                    cache = output.past_key_values
                    cached_count += decoder_input.embeddings.shape[1]
                    decoder_input = Sequence(full_sequence.embeddings[:, -1:], full_sequence.positions[..., -1:])
                else:
                    decoder_input = full_sequence
            return GenerationOutput(tokens, full_sequence.embeddings.new_empty(1, 0, self.backbone.hidden_size))
        finally:
            self.train(was_training)
