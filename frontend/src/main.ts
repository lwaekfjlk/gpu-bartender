import Chart from 'chart.js/auto';
import { ChartConfiguration } from 'chart.js';
import './styles/styles.css';

document.addEventListener('DOMContentLoaded', () => {
    const mixedPrecisionBtn = document.getElementById('mixedPrecision') as HTMLButtonElement;
    const fullPrecisionBtn = document.getElementById('fullPrecision') as HTMLButtonElement;
    const adamOptimizerBtn = document.getElementById('adamOptimizer') as HTMLButtonElement;
    const sgdOptimizerBtn = document.getElementById('sgdOptimizer') as HTMLButtonElement;
    const mibBtn = document.getElementById('mibBtn') as HTMLButtonElement;
    const gibBtn = document.getElementById('gibBtn') as HTMLButtonElement;
    const calculateBtn = document.getElementById('calculateBtn') as HTMLButtonElement;


    //User input elements
    const momentumCheckbox = document.getElementById('momentum') as HTMLInputElement;
    const sequenceLengthInput = document.getElementById('sequenceLength') as HTMLInputElement;
    const numGPUsInput = document.getElementById('numGPUs') as HTMLInputElement;
    const parametersPresetSelect = document.getElementById('parametersPreset') as HTMLSelectElement;
    const numParametersInput = document.getElementById('numParameters') as HTMLInputElement;
    const numLayersInput = document.getElementById('numLayers') as HTMLInputElement;
    const vocabSizeInput = document.getElementById('vocabSize') as HTMLInputElement;
    const hiddenSizeInput = document.getElementById('hiddenSize') as HTMLInputElement;
    const numAttentionHeadsInput = document.getElementById('numAttentionHeads') as HTMLInputElement;
    const intermediateSizeInput = document.getElementById('intermediateSize') as HTMLInputElement;
    const numKeyValueHeadsInput = document.getElementById('numKeyValueHeads') as HTMLInputElement;
    const batchSizeInput = document.getElementById('batchSize') as HTMLInputElement;

    // Output elements
    const totalVRAMElement = document.getElementById('totalVRAM') as HTMLParagraphElement;
    const vramChartElement = document.getElementById('vramChart') as HTMLDivElement;

    let vramUsageChart: Chart | undefined;
    calculateBtn.addEventListener('click', calculateGPUUsage);

    const inputElements = [
        sequenceLengthInput, numGPUsInput, numParametersInput, numLayersInput,
        vocabSizeInput, hiddenSizeInput, numAttentionHeadsInput, intermediateSizeInput,
        numKeyValueHeadsInput, batchSizeInput
    ];

    inputElements.forEach(input => {
        if (input) {
            input.addEventListener('change', calculateGPUUsage);
        } else {
            console.error(input, 'Input element not found');
        }
    });

    function handleToggleClick(event: Event) {
        const clickedButton = event.target as HTMLButtonElement;
        console.log('Clicked button:', clickedButton);
        if (!clickedButton.classList.contains('toggle-btn')) {
            console.log('Clicked element is not a toggle button');
            return;
        }

        const toggleGroup = clickedButton.closest('.toggle-buttons');
        console.log('Toggle group:', toggleGroup);
        if (!toggleGroup) {
            console.log('No parent .toggle-buttons found');
            return;
        }

        toggleGroup.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.classList.remove('active');
            console.log('Removed active class from:', btn);
        });
        clickedButton.classList.add('active');
        console.log('Added active class to:', clickedButton);

        calculateGPUUsage();
    }

    document.querySelectorAll('.toggle-buttons').forEach(container => {
        container.addEventListener('click', handleToggleClick);
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
        const batchSize = batchSizeInput.value;

        console.log('Current model parameters:', {
            numParameters,
            numLayers,
            vocabSize,
            hiddenSize,
            numAttentionHeads,
            intermediateSize,
            numKeyValueHeads,
            batchSize
        });
    }

    async function calculateGPUUsage() {
        const precision = document.querySelector('#mixedPrecision.active') ? 'mixed' : 'full';
        const optimizer = document.querySelector('#adamOptimizer.active') ? 'Adam' : 'SGD';
        const momentum = (document.getElementById('momentum') as HTMLInputElement).checked;
        const sequenceLength = parseInt(sequenceLengthInput.value);
        const numGPUs = parseInt(numGPUsInput.value);
        const numParams = parseFloat(numParametersInput.value);
        const numLayers = parseInt(numLayersInput.value);
        const vocabSize = parseInt(vocabSizeInput.value);
        const hiddenSize = parseInt(hiddenSizeInput.value);
        const numAttentionHeads = parseInt(numAttentionHeadsInput.value);
        const intermediateSize = parseInt(intermediateSizeInput.value);
        const numKeyValueHeads = parseInt(numKeyValueHeadsInput.value);
        const unit = mibBtn.classList.contains('active') ? 'MiB' : 'GiB';
        const batchSize = parseInt(batchSizeInput.value);

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
                numGPUs,
                numParams,
                numLayers,
                vocabSize,
                hiddenSize,
                numAttentionHeads,
                intermediateSize,
                numKeyValueHeads,
                unit,
                batchSize
            })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        updateUI(result);
    } catch (error) {
        totalVRAMElement.textContent = 'Error calculating VRAM: First moments are only calculated for Adam or SGD optimizer with momentum and Second moments are only calculated for Adam optimizer';
    }}

    function updateUI(response: any) {
        console.log('Updating UI with:', response);
        const result = response.result;
        const totalVRAMUsage = response.totalVRAMUsage;
        const unit = mibBtn.classList.contains('active') ? 'MiB' : 'GiB';

        totalVRAMElement.innerHTML = `<strong>Total VRAM usage</strong> is ${(totalVRAMUsage).toFixed(2)} ${unit}`;

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

        const data = [
            { name: 'CUDA Kernels', value: result.cudaKernels},
            { name: 'Parameters', value: result.parameters},
            { name: 'Activations', value: result.activations},
            { name: 'Gradients', value: result.gradients},
            { name: 'First Moments', value: result.firstMoments},
            { name: 'Second Moments', value: result.secondMoments},
            { name: 'Outputs', value: result.outputs}
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

    function updateVRAMDetails(result: any, unit: string) {
        const vramDetails = document.querySelector('.vram-details') as HTMLDivElement;
        vramDetails.innerHTML = `
        <p><strong>CUDA Kernels</strong> use ${(result.cudaKernels).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>Parameters</strong> use ${(result.parameters).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>Activations</strong> use ${(result.activations).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>Gradients</strong> use ${(result.gradients).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>First Moments</strong> use ${(result.firstMoments).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>Second Moments</strong> use ${(result.secondMoments).toFixed(2)} ${unit} of VRAM</p>
        <p><strong>Output tensor</strong> uses ${(result.outputs).toFixed(2)} ${unit} of VRAM</p>
        `;
    }
    calculateGPUUsage();
});
