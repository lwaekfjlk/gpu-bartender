from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LoraArgs:
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_target: Optional[str] = field(default=None)

@dataclass
class QLoraArgs:
    qlora_alpha: Optional[int] = field(default=None)
    qlora_dropout: Optional[float] = field(default=None)

@dataclass
class FinetuningArgs(LoraArgs, QLoraArgs):
    training_precision: str = field(default='mixed')
    is_fsdp: bool = field(default=True)
