from dataclasses import dataclass, field


@dataclass
class DeviceArgs:
    gpu_num: int = field(default=1)
    node_num: int = field(default=1)
    gpu_memory_limit: int = field(default=0)
    gpu_type: str = field(default='A100')

