from dataclasses import dataclass, field


@dataclass
class OptimizerArgs:
    optimizer: str = field(default="adam")
    optimizer_sgd_momentum: float | None = field(default=None, metadata={"help": "Momentum for SGD optimizer, if used."})
