import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_bartender.server import (
    DataArgs,
    DeviceArgs,
    FinetuningArgs,
    ModelArgs,
    OptimizerArgs,
    VRAMCalculator,
)

# Example usage
model_args = ModelArgs(
    num_params=123456789,
    vocab_size=30522,
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=12,
    intermediate_size=3072,
    num_layers=12
)

data_args = DataArgs(
    batch_size=32,
    sequence_length=128
)

optimizer_args = OptimizerArgs(
    optimizer='Adam',
    optimizer_sgd_momentum=0.9
)

finetuning_args = FinetuningArgs(
    training_precision='mixed',
    is_fsdp=True
)

device_args = DeviceArgs(
    gpu_num=4,
)

calculator = VRAMCalculator(
    model_args=model_args,
    finetuning_args=finetuning_args,
    optimizer_args=optimizer_args,
    data_args=data_args,
    device_args=device_args,
    unit="MiB"
)

result_estimation = calculator.estimate_result()
total_usage_per_gpu = calculator.get_total_usage_per_gpu(result_estimation, is_first=True)

print("Result Estimation:", result_estimation)
print("Total VRAM Usage per GPU:", total_usage_per_gpu, "MiB")
