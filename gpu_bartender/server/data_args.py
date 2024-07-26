from dataclasses import dataclass, field


@dataclass
class DataArgs:
    batch_size: int = field(default=4)
    sequence_length: int = field(default=512)
