import { DataArgs } from './dataArgs';
import { FinetuningArgs } from './finetuningArgs';
import { ModelArgs } from './modelArgs';
import { OptimizerArgs } from './optimizerArgs';

export class VRAMCalculator {
    private bytesPerParam: number;
    private gpuDivisor: number;

    constructor(
        private modelArgs: ModelArgs,
        private finetuningArgs: FinetuningArgs,
        private optimizerArgs: OptimizerArgs,
        private dataArgs: DataArgs,
        private numGpus: number = 1,
        private unit: string = "MiB"
    ) {
        this.bytesPerParam = this.computeBytesPerParam();
        this.gpuDivisor = this.computeGpuDivisor();
    }

    private computeBytesPerParam(): number {
        return this.finetuningArgs.trainingPrecision === 'mixed' ? 6 : 4;
    }

    private computeGpuDivisor(): number {
        return this.finetuningArgs.isFsdp && this.numGpus > 1 ? this.numGpus : 1;
    }

    private computeParameters(): number {
        return (this.bytesPerParam * this.modelArgs.numParams * 1e9) / this.gpuDivisor;
    }

    private computeActivations(): number {
        const { hiddenSize, numAttentionHeads, numKeyValueHeads, intermediateSize, numLayers } = this.modelArgs;
        const { batchSize, sequenceLength } = this.dataArgs;
        const headDim = hiddenSize / numAttentionHeads;
        const attentionBlock = 2 * this.bytesPerParam * batchSize * sequenceLength * hiddenSize;
        const mlpBlock = this.bytesPerParam * batchSize * sequenceLength * intermediateSize;
        const layerNorms = 2 * this.bytesPerParam * batchSize * sequenceLength * hiddenSize;
        const layer = attentionBlock + mlpBlock + layerNorms;
        return layer * numLayers;
    }

    private computeOutputs(): number {
        return 4 * this.dataArgs.batchSize * this.dataArgs.sequenceLength * this.modelArgs.vocabSize * 2;
    }

    private computeGradients(): number {
        return (4 * this.modelArgs.numParams * 1e9) / this.gpuDivisor;
    }

    private computeFirstMoments(): number | null {
        const { optimizer, optimizerSgdMomentum } = this.optimizerArgs;
        if (!((optimizer === 'SGD' && optimizerSgdMomentum) || optimizer === 'Adam')) {
            return null;
        }
        return (4 * this.modelArgs.numParams * 1e9) / this.gpuDivisor;
    }

    private computeSecondMoments(): number | null {
        if (this.optimizerArgs.optimizer !== 'Adam') {
            return null;
        }
        return (4 * this.modelArgs.numParams * 1e9) / this.gpuDivisor;
    }

    private roundNum(num: number): number {
        const divisor = this.unit === "MiB" ? 2 ** 20 : 2 ** 30;
        return Math.round((num / divisor) * 1e3) / 1e3;
    }

    public estimate_result(): Record<string, number | null> {
        return {
            cudaKernels: this.roundNum(1e6 * 2 ** 20),
            parameters: this.roundNum(this.computeParameters()),
            activations: this.roundNum(this.computeActivations()),
            outputs: this.roundNum(this.computeOutputs()),
            gradients: this.roundNum(this.computeGradients()),
            firstMoments: this.roundNum(this.computeFirstMoments() || 0),
            secondMoments: this.roundNum(this.computeSecondMoments() || 0)
        };
    }
}
