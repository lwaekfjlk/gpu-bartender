from dataclasses import dataclass, field


@dataclass
class LoraArgs:
    lora_alpha: int | None = field(default=None)
    lora_dropout: float | None = field(default=None)
    lora_rank: int | None = field(default=8)
    lora_target: str | None = field(default=None)

@dataclass
class QLoraArgs:
    qlora_alpha: int | None = field(default=None)
    qlora_dropout: float | None = field(default=None)

@dataclass
class FinetuningArgs(LoraArgs, QLoraArgs):
    training_precision: str = field(default='mixed')
    is_fsdp: bool = field(default=True)
