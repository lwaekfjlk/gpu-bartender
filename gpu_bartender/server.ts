import express, { Request, Response } from 'express';
import bodyParser from 'body-parser';
import path from 'path';
import { VRAMCalculator } from './src/backend/calculator';
import { DataArgs } from './src/backend/dataArgs';
import { FinetuningArgs } from './src/backend/finetuningArgs';
import { ModelArgs } from './src/backend/modelArgs';
import { OptimizerArgs } from './src/backend/optimizerArgs';

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'src', 'frontend')));
app.use('/dist', express.static(path.join(__dirname, 'dist')));
app.post('/calculate', (req: Request, res: Response) => {
    const data = req.body;

    const modelArgs = new ModelArgs(
        data.modelSize,
        data.vocabSize,
        data.hiddenSize,
        data.numAttentionHeads,
        data.numKeyValueHeads,
        data.intermediateSize,
        data.numLayers
    );
    const finetuningArgs = new FinetuningArgs(
        data.trainingPrecision,
        data.isFsdp,
        data.loraAlpha,
        data.loraDropout,
        data.loraRank,
        data.loraTarget,
        data.qloraAlpha,
        data.qloraDropout
    );
    const optimizerArgs = new OptimizerArgs(
        data.optimizer,
        data.optimizerSgdMomentum
    );
    const dataArgs = new DataArgs(
        data.batchSize,
        data.sequenceLength
    );
    const calculator = new VRAMCalculator(
        modelArgs,
        finetuningArgs,
        optimizerArgs,
        dataArgs,
        data.numGpus,
        data.unit
    );

    const result = calculator.estimate_result();
    res.json(result);
});
app.get('*', (req: Request, res: Response) => {
    res.sendFile(path.join(__dirname, 'src', 'frontend', 'index.html'));
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
