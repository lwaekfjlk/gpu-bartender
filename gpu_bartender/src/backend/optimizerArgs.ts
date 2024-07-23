export class OptimizerArgs {
    constructor(
        public optimizer: string = "adam",
        public optimizerSgdMomentum?: number
    ) {}
}
