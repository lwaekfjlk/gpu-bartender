export class ModelArgs {
    constructor(
        public numParams: number = 1,
        public vocabSize: number = 1,
        public hiddenSize: number = 1,
        public numAttentionHeads: number = 1,
        public numKeyValueHeads: number = 1,
        public intermediateSize: number = 1,
        public numLayers: number = 1
    ) {}
}
