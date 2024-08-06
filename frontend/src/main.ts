import Chart from 'chart.js/auto';
import { ChartConfiguration, ChartTypeRegistry, TooltipItem} from 'chart.js';
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
        const pluginTextElement = document.getElementById('pluginText') as HTMLDivElement;
        pluginTextElement.innerHTML = `
            <p>Due to some data being too large or too small, they may be hidden in the chart. Users can toggle the visibility of specific data segments by interacting with the corresponding legend elements.</p>
        `;
        updateChart(result, unit);
        updateVRAMDetails(result, totalVRAMUsage, unit);
    }

    function updateChart(result: any, unit: string) {
        const vramChartCanvas = document.getElementById('vramChartCanvas') as HTMLCanvasElement;
        const ctx = vramChartCanvas.getContext('2d');
        vramChartCanvas.style.width = '120%';
        if (!ctx) {
            console.error('Failed to get 2D context');
            return;
        }
    
        if (vramUsageChart) {
            vramUsageChart.destroy();
        }
        const numGPUs = parseInt(numGPUsInput.value);
        const components = [
            { name: 'CUDA Kernels', key: 'cudaKernels' },
            { name: 'Parameters', key: 'parameters' },
            { name: 'Activations', key: 'activations' },
            { name: 'Gradients', key: 'gradients' },
            { name: 'First Moments', key: 'firstMoments' },
            { name: 'Second Moments', key: 'secondMoments' },
            { name: 'Outputs', key: 'outputs' }
        ];
        
        const totalVRAM = components.reduce((sum, item) => sum + result[item.key], 0);
    
        const datasets = components.map(item => ({
            label: item.name,
            data: Array(numGPUs).fill((result[item.key] / totalVRAM) * 100),
            backgroundColor: getColor(item.name),
        }));
    
        const config: ChartConfiguration<'bar'> = {
            type: 'bar',
            data: {
                labels: Array.from({length: numGPUs}, (_, i) => `GPU ${i}`),
                datasets: datasets
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    title: {
                        display: true,
                        text: `GPU Usage %`
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context: TooltipItem<'bar'>) {
                                const value = context.raw as number;
                                return `${context.dataset.label}: ${value.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        title: {
                            display: true,
                            text: `VRAM Usage %`
                        },
                        ticks: {
                            callback: function(value) {
                                return `${Number(value)}%`;
                            }
                        },
                    },
                    y: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'GPU Index'
                        }
                    }
                }
            },
        };
    
        vramUsageChart = new Chart(ctx, config);
    }

    function getColor(name: string): string {
        const colorMap: {[key: string]: string} = {
            'CUDA Kernels': 'rgba(153, 102, 255, 0.8)',
            'Parameters': 'rgba(75, 192, 192, 0.8)',
            'Activations': 'rgba(255, 159, 64, 0.8)',
            'Gradients': 'rgba(255, 99, 132, 0.8)',
            'First Moments': 'rgba(54, 162, 235, 0.8)',
            'Second Moments': 'rgba(255, 206, 86, 0.8)',
            'Outputs': 'rgba(201, 203, 207, 0.8)'
        };
        return colorMap[name] || 'rgba(0, 0, 0, 0.8)';
    }

    function updateVRAMDetails(result: any, totalVRAMUsage: number, unit: string) {
        const vramDetails = document.querySelector('.vram-details') as HTMLDivElement;
        vramDetails.innerHTML = `
        <p><strong>Total VRAM usage</strong> is ${totalVRAMUsage.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit}</p>
        <p><strong>CUDA Kernels</strong> use ${result.cudaKernels.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>Parameters</strong> use ${result.parameters.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>Activations</strong> use ${result.activations.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>Gradients</strong> use ${result.gradients.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>First Moments</strong> use ${result.firstMoments.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>Second Moments</strong> use ${result.secondMoments.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        <p><strong>Output tensor</strong> uses ${result.outputs.toLocaleString(undefined, {maximumFractionDigits: 2})} ${unit} of VRAM</p>
        `;
    }
    calculateGPUUsage();
});
