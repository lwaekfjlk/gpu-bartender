export class LoraArgs {
    constructor(
        public loraAlpha?: number,
        public loraDropout?: number,
        public loraRank: number = 8,
        public loraTarget?: string
    ) {}
}

export class QLoraArgs {
    constructor(
        public qloraAlpha?: number,
        public qloraDropout?: number
    ) {}
}
