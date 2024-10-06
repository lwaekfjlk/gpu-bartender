from typing import Dict

from data_args import DataArgs
from device_args import DeviceArgs
from finetuning_args import FinetuningArgs
from model_args import ModelArgs
from optimizer_args import OptimizerArgs


class VRAMCalculator:
    def __init__(
        self,
        model_args: ModelArgs,
        finetuning_args: FinetuningArgs,
        optimizer_args: OptimizerArgs,
        data_args: DataArgs,
        device_args: DeviceArgs,
        num_gpus: int = 1,
        unit: str = "MiB"
    ):
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.optimizer_args = optimizer_args
        self.data_args = data_args
        self.device_args = device_args
        self.num_gpus = num_gpus
        self.unit = unit
        self.divisor = 2 ** 20 if unit == "MiB" else 2 ** 30
        self.precision = 0 if unit == "MiB" else 3

        self.bytes_per_param = self.compute_bytes_per_param()
        self.gpu_divisor = self.compute_gpu_divisor()

    def compute_bytes_per_param(self) -> int:
        training_precision = self.finetuning_args.training_precision
        if training_precision == 'mixed':
            return 6
        else:
            return 4

    def calculate_bytes_per_param(self) -> int:
        return self.bytes_per_param

    def compute_gpu_divisor(self) -> int:
        is_fsdp = self.finetuning_args.is_fsdp
        is_parallel_mode = is_fsdp
        return self.num_gpus if self.num_gpus > 1 and is_parallel_mode else 1

    def calculate_gpu_divisor(self) -> int:
        return self.gpu_divisor

    def compute_cuda_kernels(self) -> float:
        cuda_kernels = 1000 * 2 ** 20
        return cuda_kernels

    def calculate_cuda_kernels(self) -> float:
        cuda_kernels = self.compute_cuda_kernels()
        return self.round_num(cuda_kernels / self.divisor)

    def compute_parameters(self) -> float:
        num_params = self.model_args.num_params
        parameters = (self.bytes_per_param * num_params * 10 ** 9) / self.gpu_divisor
        return float(parameters)

    def calculate_parameters(self) -> float:
        parameters = self.compute_parameters()
        return self.round_num(parameters / self.divisor)

    def compute_activations(self) -> float:
        hidden_size = self.model_args.hidden_size
        num_attention_heads = self.model_args.num_attention_heads
        num_key_value_heads = self.model_args.num_key_value_heads
        intermediate_size = self.model_args.intermediate_size
        num_layers = self.model_args.num_layers

        batch_size = self.data_args.batch_size
        sequence_length = self.data_args.sequence_length

        bytes_per_param = self.bytes_per_param
        head_dim = hidden_size // num_attention_heads

        attention_input = bytes_per_param * batch_size * sequence_length * hidden_size
        q = bytes_per_param * batch_size * sequence_length * head_dim * num_attention_heads
        k = bytes_per_param * batch_size * sequence_length * head_dim * num_key_value_heads
        softmax_output = bytes_per_param * batch_size * num_attention_heads * sequence_length ** 2
        softmax_dropout_mask = batch_size * num_attention_heads * sequence_length ** 2
        dropout_output = bytes_per_param * batch_size * num_attention_heads * sequence_length ** 2
        v = bytes_per_param * batch_size * sequence_length * head_dim * num_key_value_heads
        out_proj_input = bytes_per_param * batch_size * sequence_length * num_attention_heads * head_dim
        attention_dropout = batch_size * sequence_length * hidden_size

        attention_block = (
            attention_input + q + k + softmax_output + v + out_proj_input + softmax_dropout_mask + dropout_output + attention_dropout
        )

        mlp_input = bytes_per_param * batch_size * sequence_length * hidden_size
        activation_input = bytes_per_param * batch_size * sequence_length * intermediate_size
        down_proj_input = bytes_per_param * batch_size * sequence_length * intermediate_size
        dropout_mask = batch_size * sequence_length * hidden_size

        mlp_block = mlp_input + activation_input + down_proj_input + dropout_mask

        layer_norms = bytes_per_param * batch_size * sequence_length * hidden_size * 2

        layer = attention_block + mlp_block + layer_norms

        activations = layer * num_layers

        return int(activations)

    def calculate_activations(self) -> float:
        activations = self.compute_activations()
        return self.round_num(activations / self.divisor)

    def compute_outputs(self) -> float:
        batch_size = self.data_args.batch_size
        sequence_length = self.data_args.sequence_length
        vocab_size = self.model_args.vocab_size
        outputs = 4 * batch_size * sequence_length * vocab_size * 2
        return float(outputs)

    def calculate_outputs(self) -> float:
        outputs = self.compute_outputs()
        return self.round_num(outputs / self.divisor)

    def compute_gradients(self) -> float:
        num_params = self.model_args.num_params
        gradients = (4 * num_params * 10 ** 9) / self.gpu_divisor
        return float(gradients)

    def calculate_gradients(self) -> float:
        gradients = self.compute_gradients()
        return self.round_num(gradients / self.divisor)

    def compute_first_moments(self) -> float:
        optimizer = self.optimizer_args.optimizer
        optimizer_sgd_momentum = self.optimizer_args.optimizer_sgd_momentum
        if not ((optimizer == 'SGD' and optimizer_sgd_momentum) or optimizer == 'Adam'):
            raise ValueError("First moments are only calculated for Adam or SGD optimizer with momentum")
        num_params = self.model_args.num_params
        first_moments = (4 * num_params * 10 ** 9) / self.gpu_divisor
        return float(first_moments)

    def calculate_first_moments(self) -> float:
        first_moments = self.compute_first_moments()
        if first_moments is None:
            raise ValueError("First moments are only calculated for Adam or SGD optimizer with momentum")
        return self.round_num(first_moments / self.divisor)

    def compute_second_moments(self) -> float:
        optimizer = self.optimizer_args.optimizer
        if optimizer != 'Adam':
            raise ValueError("Second moments are only calculated for Adam optimizer")
        num_params = self.model_args.num_params
        second_moments = (4 * num_params * 10 ** 9) / self.gpu_divisor
        return float(second_moments)

    def calculate_second_moments(self) -> float:
        second_moments = self.compute_second_moments()
        if second_moments is None:
            raise ValueError("Second moments are only calculated for Adam optimizer")
        return self.round_num(second_moments / self.divisor)

    def estimate_result(self) -> Dict[str, float]:
        result_estimation = {
            'cudaKernels': self.calculate_cuda_kernels(),
            'parameters': self.calculate_parameters(),
            'outputs': self.calculate_outputs(),
            'activations': self.calculate_activations(),
            'gradients': self.calculate_gradients(),
            'firstMoments': self.calculate_first_moments(),
            'secondMoments': self.calculate_second_moments()
        }

        return result_estimation

    def get_total_usage_per_gpu(self, result_estimation: Dict[str, float], is_first: bool) -> float:
        total_usage = (
            result_estimation['cudaKernels'] +
            result_estimation['parameters'] +
            (result_estimation['outputs'] if result_estimation['outputs'] is not None else 0) * int(is_first) +
            (result_estimation['activations'] if result_estimation['activations'] is not None else 0) +
            (result_estimation['gradients'] if result_estimation['gradients'] is not None else 0) +
            (result_estimation['firstMoments'] if result_estimation['firstMoments'] is not None else 0) +
            (result_estimation['secondMoments'] if result_estimation['secondMoments'] is not None else 0)
        )

        return self.round_num(total_usage)

    @staticmethod
    def round_num(num: float, fraction_digits: int = 3) -> float:
        return round(num, fraction_digits)
