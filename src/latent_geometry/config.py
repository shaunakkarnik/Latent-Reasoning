from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path


class Stage(str, Enum):
    SFT = "sft"
    WARMUP = "warmup"
    ALIGNMENT = "alignment"
    REFINEMENT = "refinement"


@dataclass
class LatentConfig:
    student: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    student_revision: str = "main"
    teacher_dim: int = 2048
    pool_height: int = 4
    pool_width: int = 4
    alignment_weight: float = 1.0
    start_token: str = "<|latent_start|>"
    end_token: str = "<|latent_end|>"
    max_pixels: int = 512 * 28 * 28
    finetuning: str = "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    def __post_init__(self):
        if self.finetuning not in ("full", "lora"):
            raise ValueError("finetuning must be 'full' or 'lora'")
        if self.lora_rank < 1 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
        if min(self.teacher_dim, self.pool_height, self.pool_width, self.max_pixels) <= 0:
            raise ValueError("Feature dimensions, pool sizes, and max_pixels must be positive")
        if self.alignment_weight < 0:
            raise ValueError("alignment_weight cannot be negative")
        if not self.start_token or not self.end_token or self.start_token == self.end_token:
            raise ValueError("Latent boundary tokens must be distinct and nonempty")

    @property
    def num_latents(self):
        return self.pool_height * self.pool_width

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
