import { LoraArgs, QLoraArgs } from './loraArgs';

export class FinetuningArgs {
    public loraArgs: LoraArgs;
    public qloraArgs: QLoraArgs;

    constructor(
        public trainingPrecision: string = 'mixed',
        public isFsdp: boolean = true,
        loraAlpha?: number,
        loraDropout?: number,
        loraRank?: number,
        loraTarget?: string,
        qloraAlpha?: number,
        qloraDropout?: number
    ) {
        this.loraArgs = new LoraArgs(loraAlpha, loraDropout, loraRank, loraTarget);
        this.qloraArgs = new QLoraArgs(qloraAlpha, qloraDropout);
    }
}
