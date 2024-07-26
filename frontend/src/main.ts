import Chart from 'chart.js/auto';
import { ChartConfiguration } from 'chart.js';
document.addEventListener('DOMContentLoaded', () => {
    const calculateBtn = document.getElementById('calculateBtn') as HTMLButtonElement;
    const trainingTime = document.getElementById('trainingTime') as HTMLParagraphElement;
    const memoryUtilization = document.getElementById('memoryUtilization') as HTMLParagraphElement;
    const powerConsumption = document.getElementById('powerConsumption') as HTMLParagraphElement;
    const downloadResults = document.getElementById('downloadResults') as HTMLAnchorElement;
    let gpuUsageChart: Chart | undefined;

    calculateBtn.addEventListener('click', calculateGPUUsage);
    downloadResults.addEventListener('click', downloadResultsFile);

    async function calculateGPUUsage() {
        const modelSize = parseFloat((document.getElementById('modelSize') as HTMLInputElement).value);
        const batchSize = parseInt((document.getElementById('batchSize') as HTMLInputElement).value);
        const epochs = parseInt((document.getElementById('epochs') as HTMLInputElement).value);
        const gpuModel = (document.getElementById('gpuModel') as HTMLSelectElement).value;
        const learningRate = parseFloat((document.getElementById('learningRate') as HTMLInputElement).value);
        const optimizer = (document.getElementById('optimizer') as HTMLSelectElement).value;
        const dataSize = parseFloat((document.getElementById('dataSize') as HTMLInputElement).value);

        try {
            const response = await fetch('/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                modelSize,
                batchSize,
                epochs,
                gpuModel,
                learningRate,
                optimizer,
                dataSize,
                numGpus: 1,
                unit: "MiB",
                vocabSize: 30522,
                hiddenSize: 768,
                numAttentionHeads: 12,
                numKeyValueHeads: 12,
                intermediateSize: 3072,
                numLayers: 12,
                trainingPrecision: 'mixed',
                isFsdp: true,
                optimizerSgdMomentum: null,
                sequenceLength: 512
            })
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        console.log('Backend response:', result);
        const estimatedTime = (modelSize * batchSize * epochs) / 1000;
        const estimatedMemory = (modelSize * 4 + dataSize * 1024) / 1024;
        const estimatedPower = modelSize * 0.5;
        const combinedResult = {
            ...result,
            estimatedTime,
            estimatedMemory,
            estimatedPower
        };
        updateUI(combinedResult);
    } catch (error) {
        console.error('Error fetching data:', error);
        trainingTime.textContent = 'Error fetching data';
        memoryUtilization.textContent = 'Error fetching data';

        powerConsumption.textContent = 'Error fetching data';
    }}

    function updateUI(result: any) {
        console.log('Updating UI with:', result);
        if (trainingTime) trainingTime.textContent = `${result.estimatedTime.toFixed(2)} hours`;
        if (memoryUtilization) memoryUtilization.textContent = `${result.estimatedMemory.toFixed(2)} GB`;
        if (powerConsumption) powerConsumption.textContent = `${result.estimatedPower.toFixed(2)} W`;

        updateChart(result.estimatedTime || 0);
    }

    function updateChart(estimatedTime: number) {
        const canvas = document.getElementById('gpuUsageChart') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            console.error('Failed to get 2D context');
            return;
        }
        if (gpuUsageChart) {
            gpuUsageChart.destroy();
        }

        const labels = Array.from({ length: 10 }, (_, i) => i * estimatedTime / 10);
        const data = labels.map(x => Math.sin(x) * 50 + 50);

        const config: ChartConfiguration = {
            type: 'line',
            data: {
                labels: labels,
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

        gpuUsageChart = new Chart(ctx, config);
    }

    function downloadResultsFile(e: Event) {
        e.preventDefault();
        const results = `
            Estimated Training Time: ${trainingTime.textContent}
            Memory Utilization: ${memoryUtilization.textContent}
            Power Consumption: ${powerConsumption.textContent}
        `;
        const blob = new Blob([results], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'gpu_usage_results.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});
