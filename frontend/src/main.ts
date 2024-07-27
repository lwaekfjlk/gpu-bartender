import Chart from 'chart.js/auto';
import { ChartConfiguration } from 'chart.js';
document.addEventListener('DOMContentLoaded', () => {
    const mixedPrecisionBtn = document.getElementById('mixedPrecision') as HTMLButtonElement;
    const fullPrecisionBtn = document.getElementById('fullPrecision') as HTMLButtonElement;
    const adamOptimizerBtn = document.getElementById('adamOptimizer') as HTMLButtonElement;
    const sgdOptimizerBtn = document.getElementById('sgdOptimizer') as HTMLButtonElement;
    const mibBtn = document.getElementById('mibBtn') as HTMLButtonElement;
    const gibBtn = document.getElementById('gibBtn') as HTMLButtonElement;

    //User input elements
    const momentumCheckbox = document.getElementById('momentum') as HTMLInputElement;
    const sequenceLengthInput = document.getElementById('sequenceLength') as HTMLInputElement;
    const batchSizeInput = document.getElementById('batchSize') as HTMLInputElement;
    const numGPUsInput = document.getElementById('numGPUs') as HTMLInputElement;
    const parametersPresetSelect = document.getElementById('parametersPreset') as HTMLSelectElement;
    const numParametersInput = document.getElementById('numParameters') as HTMLInputElement;
    const numLayersInput = document.getElementById('numLayers') as HTMLInputElement;
    const vocabSizeInput = document.getElementById('vocabSize') as HTMLInputElement;
    const hiddenSizeInput = document.getElementById('hiddenSize') as HTMLInputElement;
    const numAttentionHeadsInput = document.getElementById('numAttentionHeads') as HTMLInputElement;
    const intermediateSizeInput = document.getElementById('intermediateSize') as HTMLInputElement;
    const numKeyValueHeadsInput = document.getElementById('numKeyValueHeads') as HTMLInputElement;

    // Output elements
    const totalVRAMElement = document.getElementById('totalVRAM') as HTMLParagraphElement;
    const vramChartElement = document.getElementById('vramChart') as HTMLDivElement;

    let vramUsageChart: Chart | undefined;

    [sequenceLengthInput, batchSizeInput, numGPUsInput, numParametersInput, numLayersInput, 
        vocabSizeInput, hiddenSizeInput, numAttentionHeadsInput, intermediateSizeInput, 
        numKeyValueHeadsInput].forEach(input => {
           input.addEventListener('change', calculateGPUUsage);
    });

    [mixedPrecisionBtn, fullPrecisionBtn, 
        adamOptimizerBtn, sgdOptimizerBtn, mibBtn, gibBtn].forEach(btn => {
           btn.addEventListener('click', () => {
               btn.parentElement?.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
               btn.classList.add('active');
               calculateGPUUsage();
           });
    });

    momentumCheckbox.addEventListener('change', calculateGPUUsage);
    parametersPresetSelect.addEventListener('change', updateModelParameters);

    function updateModelParameters() {
        const numParameters = numParametersInput.value;
        const numLayers = numLayersInput.value;
        const vocabSize = vocabSizeInput.value;
        const hiddenSize = hiddenSizeInput.value;
        const numAttentionHeads = numAttentionHeadsInput.value;
        const intermediateSize = intermediateSizeInput.value;
        const numKeyValueHeads = numKeyValueHeadsInput.value;

        console.log('Current model parameters:', {
            numParameters,
            numLayers,
            vocabSize,
            hiddenSize,
            numAttentionHeads,
            intermediateSize,
            numKeyValueHeads
        });
    
        calculateGPUUsage();
    }

    async function calculateGPUUsage() {
        const precision = document.querySelector('#mixedPrecision.active') ? 'mixed' : 'full';
        const optimizer = document.querySelector('#adamOptimizer.active') ? 'Adam' : 'SGD';
        const momentum = (document.getElementById('momentum') as HTMLInputElement).checked;
        const sequenceLength = parseInt(sequenceLengthInput.value);
        const batchSize = parseInt(batchSizeInput.value);
        const numGPUs = parseInt(numGPUsInput.value);
        const numParams = parseFloat(numParametersInput.value) * 1e9;
        const numLayers = parseInt(numLayersInput.value);
        const vocabSize = parseInt(vocabSizeInput.value);
        const hiddenSize = parseInt(hiddenSizeInput.value);
        const numAttentionHeads = parseInt(numAttentionHeadsInput.value);
        const intermediateSize = parseInt(intermediateSizeInput.value);
        const numKeyValueHeads = parseInt(numKeyValueHeadsInput.value);


        try {
            const response = await fetch('/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                precision,
                optimizer,
                momentum,
                sequenceLength,
                batchSize,
                numGPUs,
                numParams,
                numLayers,
                vocabSize,
                hiddenSize,
                numAttentionHeads,
                intermediateSize,
                numKeyValueHeads
            })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        updateUI(result);
    } catch (error) {
        totalVRAMElement.textContent = 'Error calculating VRAM';
    }}

    function updateUI(result: any) {
        console.log('Updating UI with:', result);
        
        const unit = mibBtn.classList.contains('active') ? 'MiB' : 'GiB';
        const divisor = unit === 'MiB' ? 1 : 1024;

        totalVRAMElement.textContent = `Total VRAM usage is ${(result.totalVRAM / divisor).toFixed(2)} ${unit}`;

        updateChart(result, unit);
        updateVRAMDetails(result, unit);
    }

    function updateChart(result: any, unit: string) {
        const vramChartCanvas = document.getElementById('vramChartCanvas') as HTMLCanvasElement;
        const ctx = vramChartCanvas.getContext('2d');

        if (!ctx) {
            console.error('Failed to get 2D context');
            return;
        }

        if (vramUsageChart) {
            vramUsageChart.destroy();
        }

        const divisor = unit === 'MiB' ? 1 : 1024;
        const data = [
            { name: 'CUDA Kernels', value: result.cudaKernels / divisor },
            { name: 'Parameters', value: result.parameters / divisor },
            { name: 'Activations', value: result.activations / divisor },
            { name: 'Gradients', value: result.gradients / divisor },
            { name: 'First Moments', value: result.firstMoments / divisor },
            { name: 'Outputs', value: result.outputs / divisor }
        ];

        const config: ChartConfiguration = {
            type: 'line',
            data: {
                labels: data.map(item => item.name),
                datasets: [{
                    label: 'GPU Usage (%)',
                    data: data,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }as any]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        }
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'GPU Usage (%)'
                        },
                        min: 0,
                        max: 100
                    }
                } as any
            }
        };

        vramUsageChart = new Chart(ctx, config);
    }

    function handleToggleClick(event: Event) {
        const clickedButton = event.target as HTMLButtonElement;
        if (!clickedButton.classList.contains('toggle-btn')) return;

        const toggleGroup = clickedButton.closest('.toggle-buttons');
        if (!toggleGroup) return;

        toggleGroup.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        clickedButton.classList.add('active');

        calculateGPUUsage();
    }

    document.querySelectorAll('.toggle-buttons').forEach(container => {
        container.addEventListener('click', handleToggleClick);
    });

    [mixedPrecisionBtn, fullPrecisionBtn, adamOptimizerBtn, sgdOptimizerBtn, mibBtn, gibBtn].forEach(btn => {
        btn.removeEventListener('click', handleToggleClick);
    });
    
    function updateVRAMDetails(result: any, unit: string) {
        const divisor = unit === 'MiB' ? 1 : 1024;
        const vramDetails = document.querySelector('.vram-details') as HTMLDivElement;
        vramDetails.innerHTML = `
            <p><strong>CUDA Kernels</strong> use ${(result.cudaKernels / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>When PyTorch uses CUDA for the first time, it allocates between 300 MiB and 2 GiB of VRAM</p>
            <p><strong>Parameters</strong> use ${(result.parameters / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>Number of Parameters (${(result.numParams / 1e9).toFixed(3)} billion) × number of bytes per parameter (${result.bytesPerParam}; parameters are stored in both full precision and half precision)</p>
            <p><strong>Activations</strong> use ${(result.activations / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>Sum of sizes of all intermediate tensors during forward pass across all ${result.numLayers} layers. Activations size have quadratic dependence on Sequence Length.</p>
            <p><strong>Gradients</strong> use ${(result.gradients / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>Gradient is stored for each parameter in full precision, so it is Number of Parameters (${(result.numParams / 1e9).toFixed(3)} billion) × number of bytes per parameter (4)</p>
            <p><strong>First Moments</strong> use ${(result.firstMoments / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>Optimizer stores moving average of gradients for each parameter in full precision, so it is Number of Parameters (${(result.numParams / 1e9).toFixed(3)} billion) × number of bytes per parameter (4)</p>
            <p><strong>Output tensor</strong> uses ${(result.outputs / divisor).toFixed(2)} ${unit} of VRAM</p>
            <p>Batch Size (${result.batchSize}) × Sequence Length (${result.sequenceLength}) × Vocabulary Size (${result.vocabSize}) × number of bytes per parameter (4) × 2 (storing probabilities after softmax output which are the same size as output)</p>
        `;
    }
    calculateGPUUsage();
});
