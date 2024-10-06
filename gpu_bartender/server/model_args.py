from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelArgs:
    num_params: int = field(default=1)
    vocab_size: int = field(default=1)
    hidden_size: int = field(default=1)
    num_attention_heads: int = field(default=1)
    num_key_value_heads: int = field(default=1)
    intermediate_size: int = field(default=1)
    num_layers: int = field(default=1)
    qquantization: Literal['float32', 'float16', 'bfloat16', 'int8'] = field(default='float32')