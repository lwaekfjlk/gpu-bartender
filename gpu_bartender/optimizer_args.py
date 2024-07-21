from dataclasses import dataclass, field
from typing import Optional

@dataclass
class OptimizerArgs:
    optimizer: str = field(default="adam")
    optimizer_sgd_momentum: Optional[float] = field(default=None, metadata={"help": "Momentum for SGD optimizer, if used."})
