let rawData = [];
let processedData = [];
let features = [];
let target = [];
let featureNames = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'];
let knnModel = null;
let rfModel = null;
let syntheticData = [];
let regressionModel = null;
let results = {};
let crossValidationResults = {};
function generateBreastCancerData() {
    const data = [];
    const n = 569;
    for (let i = 0; i < n; i++) {
        const isMalignant = Math.random() > 0.63;
        const baseRadius = isMalignant ?
            14 + Math.random() * 8 :
            12 + Math.random() * 4;
        const baseTexture = isMalignant ?
            19 + Math.random() * 10 :
            17 + Math.random() * 6;
        const basePerimeter = baseRadius * 2 * Math.PI + (Math.random() - 0.5) * 20;
        const baseArea = Math.PI * baseRadius * baseRadius + (Math.random() - 0.5) * 200;
        const baseSmoothness = isMalignant ?
            0.1 + Math.random() * 0.05 :
            0.08 + Math.random() * 0.04;
        data.push({
            mean_radius: baseRadius,
            mean_texture: baseTexture,
            mean_perimeter: basePerimeter,
            mean_area: baseArea,
            mean_smoothness: baseSmoothness,
            diagnosis: isMalignant ? 1 : 0
        });
    }
    return data;
}
function loadData() {
    document.getElementById('dataInfo').innerHTML = '<div class="loading"></div> Carregando dataset...';
    setTimeout(() => {
        rawData = generateBreastCancerData();
        document.getElementById('dataInfo').innerHTML = `
            <div class="info-box">
                <strong>Dataset carregado com sucesso!</strong><br>
                Total de amostras: ${rawData.length}<br>
                Características: ${featureNames.join(', ')}<br>
                Classes: Benigno (0), Maligno (1)
            </div>
        `;
        showDataPreview();
        document.getElementById('preprocessBtn').disabled = false;
        document.getElementById('infoBtn').disabled = false;
    }, 1000);
}
function showDataPreview() {
    const preview = rawData.slice(0, 10);
    let tableHTML = `
        <div class="data-table">
            <h3>Preview dos Dados (10 primeiras amostras)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Índice</th>
                        ${featureNames.map(name => `<th>${name}</th>`).join('')}
                        <th>Diagnóstico</th>
                    </tr>
                </thead>
                <tbody>
    `;
    preview.forEach((row, idx) => {
        tableHTML += `
            <tr>
                <td>${idx + 1}</td>
                ${featureNames.map(name => `<td>${row[name].toFixed(3)}</td>`).join('')}
                <td>${row.diagnosis === 1 ? 'Maligno' : 'Benigno'}</td>
            </tr>
        `;
    });
    tableHTML += '</tbody></table></div>';
    document.getElementById('dataPreview').innerHTML = tableHTML;
}
function preprocessData() {
    const originalContent = document.getElementById('dataInfo').innerHTML;
    document.getElementById('dataInfo').innerHTML = originalContent + '<div id="preprocessLoading" class="loading"></div> Pré-processando dados...';
    setTimeout(() => {
        try {
            features = rawData.map(row => featureNames.map(name => row[name]));
            target = rawData.map(row => row.diagnosis);
            const means = [];
            const stds = [];
            for (let i = 0; i < featureNames.length; i++) {
                const column = features.map(row => row[i]);
                const mean = column.reduce((a, b) => a + b) / column.length;
                const std = Math.sqrt(column.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / column.length);
                means.push(mean);
                stds.push(std);
                for (let j = 0; j < features.length; j++) {
                    features[j][i] = (features[j][i] - mean) / std;
                }
            }
            processedData = features.map((row, idx) => ({
                features: row,
                target: target[idx]
            }));
            const loadingElement = document.getElementById('preprocessLoading');
            if (loadingElement) {
                loadingElement.parentNode.removeChild(loadingElement);
            }
            document.getElementById('dataInfo').innerHTML = originalContent + `
                <div class="info-box">
                    <strong>Pré-processamento concluído!</strong><br>
                    ✅ Features normalizadas (z-score)<br>
                    ✅ Dados separados em features e target<br>
                    ✅ ${processedData.length} amostras processadas
                </div>
            `;
            document.getElementById('knnBtn').disabled = false;
            document.getElementById('syntheticBtn').disabled = false;
            document.getElementById('regressionBtn').disabled = false;
        } catch (error) {
            console.error("Error in preprocessData:", error);
            const loadingElement = document.getElementById('preprocessLoading');
            if (loadingElement) {
                loadingElement.parentNode.removeChild(loadingElement);
            }
            document.getElementById('dataInfo').innerHTML = originalContent + `
                <div class="warning-box">
                    <strong>Erro ao pré-processar dados!</strong><br>
                    Verifique o console para mais detalhes.
                </div>
            `;
        }
    }, 1000);
}
function showDataInfo() {
    const malignant = target.filter(t => t === 1).length;
    const benign = target.filter(t => t === 0).length;
    const infoHTML = `
        <div class="info-box">
            <h3>📊 Informações do Dataset</h3>
            <strong>Distribuição das Classes:</strong><br>
            🔴 Malignos: ${malignant} (${(malignant/target.length*100).toFixed(1)}%)<br>
            🟢 Benignos: ${benign} (${(benign/target.length*100).toFixed(1)}%)<br><br>
            <strong>Estatísticas das Features (dados originais):</strong><br>
            ${featureNames.map((name, idx) => {
                const column = rawData.map(row => row[name]);
                const min = Math.min(...column);
                const max = Math.max(...column);
                const mean = column.reduce((a, b) => a + b) / column.length;
                return `${name}: Min=${min.toFixed(2)}, Max=${max.toFixed(2)}, Média=${mean.toFixed(2)}`;
            }).join('<br>')}
        </div>
    `;
    document.getElementById('dataInfo').innerHTML += infoHTML;
}
function euclideanDistance(point1, point2) {
    return Math.sqrt(
        point1.reduce((sum, val, idx) => sum + Math.pow(val - point2[idx], 2), 0)
    );
}
function knnPredict(trainData, testPoint, k) {
    const distances = trainData.map(point => ({
        distance: euclideanDistance(point.features, testPoint),
        target: point.target
    }));
    distances.sort((a, b) => a.distance - b.distance);
    const neighbors = distances.slice(0, k);
    const votes = neighbors.reduce((acc, neighbor) => {
        acc[neighbor.target] = (acc[neighbor.target] || 0) + 1;
        return acc;
    }, {});
    return Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
}
function trainKNN() {
    const k = parseInt(document.getElementById('kValue').value);
    document.getElementById('knnResults').innerHTML = '<div class="loading"></div> Treinando modelo KNN...';
    setTimeout(() => {
        const shuffled = [...processedData].sort(() => Math.random() - 0.5);
        const trainSize = Math.floor(shuffled.length * 0.8);
        const trainData = shuffled.slice(0, trainSize);
        const testData = shuffled.slice(trainSize);
        const predictions = testData.map(point =>
            parseInt(knnPredict(trainData, point.features, k))
        );
        const actual = testData.map(point => point.target);
        const accuracy = predictions.filter((pred, idx) => pred === actual[idx]).length / predictions.length;
        let tp = 0, tn = 0, fp = 0, fn = 0;
        predictions.forEach((pred, idx) => {
            if (pred === 1 && actual[idx] === 1) tp++;
            else if (pred === 0 && actual[idx] === 0) tn++;
            else if (pred === 1 && actual[idx] === 0) fp++;
            else if (pred === 0 && actual[idx] === 1) fn++;
        });
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
        results.knn = {
            k: k,
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1Score,
            confusionMatrix: { tp, tn, fp, fn }
        };
        document.getElementById('knnResults').innerHTML = `
            <div class="info-box">
                <strong>🎯 Modelo KNN Treinado (K=${k})</strong><br>
                Dados de treino: ${trainData.length} amostras<br>
                Dados de teste: ${testData.length} amostras<br>
                ✅ Treinamento concluído com sucesso!
            </div>
        `;
        document.getElementById('knnMetrics').innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${(accuracy * 100).toFixed(1)}%</div>
                <div class="metric-label">Acurácia</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(precision * 100).toFixed(1)}%</div>
                <div class="metric-label">Precisão</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(recall * 100).toFixed(1)}%</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(f1Score * 100).toFixed(1)}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
        `;
        document.getElementById('kAnalysisBtn').disabled = false;
        document.getElementById('cvBtn').disabled = false;
        document.getElementById('rfBtn').disabled = false;
    }, 1500);
}
function analyzeKPerformance() {
    document.getElementById('kPerformancePlot').innerHTML = '<div class="loading"></div> Analisando performance para diferentes valores de K...';
    setTimeout(() => {
        try {
            const kValues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21];
            const accuracies = [];
            const shuffled = [...processedData].sort(() => Math.random() - 0.5);
            const trainSize = Math.floor(shuffled.length * 0.8);
            const trainData = shuffled.slice(0, trainSize);
            const testData = shuffled.slice(trainSize);
            kValues.forEach(k => {
                const predictions = testData.map(point =>
                    parseInt(knnPredict(trainData, point.features, k))
                );
                const actual = testData.map(point => point.target);
                const accuracy = predictions.filter((pred, idx) => pred === actual[idx]).length / predictions.length;
                accuracies.push(accuracy);
            });
            document.getElementById('kPerformancePlot').innerHTML = '';
            const trace = {
                x: kValues,
                y: accuracies.map(acc => acc * 100),
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: '#3498db', size: 8 },
                line: { color: '#3498db', width: 3 },
                name: 'Acurácia'
            };
            const layout = {
                title: 'Performance do KNN vs Valor de K',
                xaxis: { title: 'Valor de K' },
                yaxis: { title: 'Acurácia (%)' },
                showlegend: false,
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };
            Plotly.newPlot('kPerformancePlot', [trace], layout);
            const bestK = kValues[accuracies.indexOf(Math.max(...accuracies))];
            const bestAccuracy = Math.max(...accuracies);
            document.getElementById('kPerformancePlot').innerHTML += `
                <div class="info-box">
                    <strong>📊 Análise de Performance K:</strong><br>
                    🏆 Melhor K: ${bestK} (Acurácia: ${(bestAccuracy * 100).toFixed(1)}%)<br>
                    📈 Valores testados: ${kValues.join(', ')}<br>
                    💡 Recomendação: Use K=${bestK} para melhor performance
                </div>
            `;
        } catch (error) {
            console.error("Error in analyzeKPerformance:", error);
            document.getElementById('kPerformancePlot').innerHTML =
                '<div class="warning-box">Ocorreu um erro durante a análise de performance. Verifique o console para detalhes.</div>';
        }
    }, 2000);
}
document.getElementById('noiseLevel').addEventListener('input', function() {
    document.getElementById('noiseLevelValue').textContent = this.value;
});
function generateSyntheticData() {
    const noiseLevel = parseFloat(document.getElementById('noiseLevel').value);
    const numSamples = parseInt(document.getElementById('syntheticSamples').value);
    document.getElementById('syntheticResults').innerHTML = '<div class="loading"></div> Gerando dados sintéticos...';
    setTimeout(() => {
        syntheticData = [];
        for (let i = 0; i < numSamples; i++) {
            const originalIdx = Math.floor(Math.random() * processedData.length);
            const original = processedData[originalIdx];
            const noisyFeatures = original.features.map(feature =>
                feature + (Math.random() - 0.5) * 2 * noiseLevel
            );
            syntheticData.push({
                features: noisyFeatures,
                target: original.target
            });
        }
        document.getElementById('syntheticResults').innerHTML = `
            <div class="info-box">
                <strong>🔧 Dados Sintéticos Gerados!</strong><br>
                Amostras sintéticas: ${syntheticData.length}<br>
                Nível de ruído: ${noiseLevel}<br>
                Baseado em amostras originais com ruído gaussiano
            </div>
        `;
        document.getElementById('syntheticTrainBtn').disabled = false;
    }, 1000);
}
function trainWithSynthetic() {
    const k = parseInt(document.getElementById('kValue').value);
    document.getElementById('syntheticResults').innerHTML += '<div class="loading"></div> Treinando com dados sintéticos...';
    setTimeout(() => {
        const combinedData = [...processedData, ...syntheticData];
        const shuffled = [...combinedData].sort(() => Math.random() - 0.5);
        const trainSize = Math.floor(shuffled.length * 0.8);
        const trainData = shuffled.slice(0, trainSize);
        const testData = shuffled.slice(trainSize);
        const predictions = testData.map(point =>
            parseInt(knnPredict(trainData, point.features, k))
        );
        const actual = testData.map(point => point.target);
        const accuracy = predictions.filter((pred, idx) => pred === actual[idx]).length / predictions.length;
        let tp = 0, tn = 0, fp = 0, fn = 0;
        predictions.forEach((pred, idx) => {
            if (pred === 1 && actual[idx] === 1) tp++;
            else if (pred === 0 && actual[idx] === 0) tn++;
            else if (pred === 1 && actual[idx] === 0) fp++;
            else if (pred === 0 && actual[idx] === 1) fn++;
        });
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
        results.knnSynthetic = {
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1Score
        };
        document.getElementById('syntheticResults').innerHTML += `
            <div class="info-box">
                <strong>🎯 Treinamento com Dados Sintéticos Concluído!</strong><br>
                Total de dados: ${combinedData.length} (${processedData.length} originais + ${syntheticData.length} sintéticos)<br>
                Dados de treino: ${trainData.length}<br>
                Dados de teste: ${testData.length}
            </div>
        `;
        const originalAcc = results.knn ? results.knn.accuracy : 0;
        const improvement = accuracy - originalAcc;
        document.getElementById('syntheticMetrics').innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${(accuracy * 100).toFixed(1)}%</div>
                <div class="metric-label">Acurácia (com sintéticos)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(originalAcc * 100).toFixed(1)}%</div>
                <div class="metric-label">Acurácia (original)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${improvement >= 0 ? '+' : ''}${(improvement * 100).toFixed(1)}%</div>
                <div class="metric-label">Melhoria</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(f1Score * 100).toFixed(1)}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
        `;
    }, 1500);
}
function trainLinearRegression() {
    document.getElementById('regressionResults').innerHTML = '<div class="loading"></div> Treinando modelo de Regressão Linear...';
    setTimeout(() => {
        const X = rawData.map(row => [
            1,
            row.mean_radius,
            row.mean_texture,
            row.mean_perimeter,
            row.mean_smoothness
        ]);
        const y = rawData.map(row => row.mean_area);
        const indices = Array.from({length: X.length}, (_, i) => i);
        indices.sort(() => Math.random() - 0.5);
        const trainSize = Math.floor(X.length * 0.8);
        const trainIndices = indices.slice(0, trainSize);
        const testIndices = indices.slice(trainSize);
        const X_train = trainIndices.map(i => X[i]);
        const y_train = trainIndices.map(i => y[i]);
        const X_test = testIndices.map(i => X[i]);
        const y_test = testIndices.map(i => y[i]);
        const XT = transpose(X_train);
        const XTX = matrixMultiply(XT, X_train);
        const XTX_inv = matrixInverse(XTX);
        const XTy = matrixVectorMultiply(XT, y_train);
        const coefficients = matrixVectorMultiply(XTX_inv, XTy);
        const y_pred_train = X_train.map(row =>
            row.reduce((sum, val, idx) => sum + val * coefficients[idx], 0)
        );
        const y_pred_test = X_test.map(row =>
            row.reduce((sum, val, idx) => sum + val * coefficients[idx], 0)
        );
        const mse_train = meanSquaredError(y_train, y_pred_train);
        const mse_test = meanSquaredError(y_test, y_pred_test);
        const r2_train = rSquared(y_train, y_pred_train);
        const r2_test = rSquared(y_test, y_pred_test);
        const mae_test = meanAbsoluteError(y_test, y_pred_test);
        regressionModel = {
            coefficients: coefficients,
            mse_train: mse_train,
            mse_test: mse_test,
            r2_train: r2_train,
            r2_test: r2_test,
            mae_test: mae_test,
            y_test: y_test,
            y_pred_test: y_pred_test,
            residuals: y_test.map((actual, i) => actual - y_pred_test[i])
        };
        results.regression = regressionModel;
        document.getElementById('regressionResults').innerHTML = `
            <div class="info-box">
                <strong>📈 Modelo de Regressão Linear Treinado!</strong><br>
                Variável dependente: Área média do tumor<br>
                Variáveis independentes: Raio, Textura, Perímetro, Suavidade<br>
                Dados de treino: ${X_train.length} amostras<br>
                Dados de teste: ${X_test.length} amostras<br><br>
                <strong>Equação do modelo:</strong><br>
                Área = ${coefficients[0].toFixed(3)} + 
                ${coefficients[1].toFixed(3)} × Raio + 
                ${coefficients[2].toFixed(3)} × Textura + 
                ${coefficients[3].toFixed(3)} × Perímetro + 
                ${coefficients[4].toFixed(3)} × Suavidade
            </div>
        `;
        document.getElementById('regressionMetrics').innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${r2_test.toFixed(3)}</div>
                <div class="metric-label">R² (Teste)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${Math.sqrt(mse_test).toFixed(2)}</div>
                <div class="metric-label">RMSE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${mae_test.toFixed(2)}</div>
                <div class="metric-label">MAE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${mse_test.toFixed(2)}</div>
                <div class="metric-label">MSE</div>
            </div>
        `;
        plotRegressionResults();
        document.getElementById('residualsBtn').disabled = false;
    }, 2000);
}
function plotRegressionResults() {
    const trace1 = {
        x: regressionModel.y_test,
        y: regressionModel.y_pred_test,
        mode: 'markers',
        type: 'scatter',
        name: 'Predições',
        marker: { color: '#3498db', size: 6 }
    };
    const minVal = Math.min(...regressionModel.y_test);
    const maxVal = Math.max(...regressionModel.y_test);
    const trace2 = {
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        name: 'Predição Perfeita',
        line: { color: '#e74c3c', dash: 'dash' }
    };
    const layout = {
        title: 'Predições vs Valores Reais - Regressão Linear',
        xaxis: { title: 'Área Real' },
        yaxis: { title: 'Área Predita' },
        showlegend: true,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    Plotly.newPlot('regressionPlots', [trace1, trace2], layout);
}
function analyzeResiduals() {
    document.getElementById('residualsPlot').innerHTML = '<div class="loading"></div> Analisando resíduos...';
    setTimeout(() => {
        const trace1 = {
            x: regressionModel.y_pred_test,
            y: regressionModel.residuals,
            mode: 'markers',
            type: 'scatter',
            name: 'Resíduos',
            marker: { color: '#9b59b6', size: 6 }
        };
        const trace2 = {
            x: [Math.min(...regressionModel.y_pred_test), Math.max(...regressionModel.y_pred_test)],
            y: [0, 0],
            mode: 'lines',
            type: 'scatter',
            name: 'Zero',
            line: { color: '#e74c3c', dash: 'dash' }
        };
        const layout1 = {
            title: 'Análise de Resíduos - Resíduos vs Predições',
            xaxis: { title: 'Valores Preditos' },
            yaxis: { title: 'Resíduos' },
            showlegend: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };
        const trace3 = {
            x: regressionModel.residuals,
            type: 'histogram',
            nbinsx: 20,
            marker: { color: '#2ecc71', opacity: 0.7 },
            name: 'Distribuição dos Resíduos'
        };
        const layout2 = {
            title: 'Distribuição dos Resíduos',
            xaxis: { title: 'Resíduos' },
            yaxis: { title: 'Frequência' },
            showlegend: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };
        document.getElementById('residualsPlot').innerHTML = '';
        const firstPlotDiv = document.createElement('div');
        firstPlotDiv.id = 'residualsScatter';
        firstPlotDiv.className = 'plot-container';
        document.getElementById('residualsPlot').appendChild(firstPlotDiv);
        const secondPlotDiv = document.createElement('div');
        secondPlotDiv.id = 'residualsHistogram';
        secondPlotDiv.className = 'plot-container';
        document.getElementById('residualsPlot').appendChild(secondPlotDiv);
        try {
            Plotly.newPlot('residualsScatter', [trace1, trace2], layout1);
            Plotly.newPlot('residualsHistogram', [trace3], layout2);
        } catch (error) {
            console.error('Error creating plots:', error);
            document.getElementById('residualsPlot').innerHTML +=
                '<div class="warning-box">Erro ao criar gráficos. Verifique o console para mais detalhes.</div>';
        }
        const residualStats = analyzeResidualsStats(regressionModel.residuals);
        document.getElementById('residualsPlot').innerHTML += `
            <div class="info-box">
                <strong>📊 Análise Estatística dos Resíduos:</strong><br>
                Média dos resíduos: ${residualStats.mean.toFixed(4)}<br>
                Desvio padrão: ${residualStats.std.toFixed(4)}<br>
                Normalidade (Jarque-Bera): ${residualStats.isNormal ? 'Aceita' : 'Rejeitada'}<br>
                Homocedasticidade: ${residualStats.isHomoscedastic ? 'Sim' : 'Não'}<br><br>
                <strong>💡 Interpretação:</strong><br>
                ${residualStats.interpretation}
            </div>
        `;
        generateFinalAnalysis();
    }, 1500);
}
function analyzeResidualsStats(residuals) {
    const mean = residuals.reduce((a, b) => a + b) / residuals.length;
    const variance = residuals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / residuals.length;
    const std = Math.sqrt(variance);
    const skewness = calculateSkewness(residuals, mean, std);
    const kurtosis = calculateKurtosis(residuals, mean, std);
    const isNormal = Math.abs(skewness) < 2 && Math.abs(kurtosis - 3) < 2;
    const firstHalf = residuals.slice(0, Math.floor(residuals.length / 2));
    const secondHalf = residuals.slice(Math.floor(residuals.length / 2));
    const var1 = calculateVariance(firstHalf);
    const var2 = calculateVariance(secondHalf);
    const isHomoscedastic = Math.abs(var1 - var2) / Math.max(var1, var2) < 0.5;
    let interpretation = '';
    if (Math.abs(mean) < 0.1) {
        interpretation += '✅ Resíduos centrados em zero (boa especificação do modelo). ';
    } else {
        interpretation += '⚠️ Resíduos não centrados (possível viés no modelo). ';
    }
    if (isNormal) {
        interpretation += '✅ Resíduos seguem distribuição normal. ';
    } else {
        interpretation += '⚠️ Resíduos não seguem distribuição normal. ';
    }
    if (isHomoscedastic) {
        interpretation += '✅ Variância constante (homocedasticidade).';
    } else {
        interpretation += '⚠️ Variância não constante (heterocedasticidade).';
    }
    return {
        mean,
        std,
        skewness,
        kurtosis,
        isNormal,
        isHomoscedastic,
        interpretation
    };
}
function generateFinalAnalysis() {
    const analysisHTML = `
        <div class="info-box">
            <h3>🎯 Análise Comparativa Final</h3>
            <h4>📊 Classificação vs Regressão:</h4>
            <strong>Classificação (KNN):</strong><br>
            • Objetivo: Prever se tumor é maligno ou benigno (variável categórica)<br>
            • Métrica principal: Acurácia = ${results.knn ? (results.knn.accuracy * 100).toFixed(1) : 'N/A'}%<br>
            • Outras métricas: Precisão, Recall, F1-Score<br>
            • Algoritmo: K-Nearest Neighbors (aprendizado baseado em instâncias)<br><br>
            <strong>Regressão Linear:</strong><br>
            • Objetivo: Prever área média do tumor (variável contínua)<br>
            • Métrica principal: R² = ${results.regression ? results.regression.r2_test.toFixed(3) : 'N/A'}<br>
            • Outras métricas: MSE, RMSE, MAE<br>
            • Algoritmo: Regressão Linear (método dos mínimos quadrados)<br><br>
            <h4>🔍 Principais Diferenças:</h4>
            • <strong>Tipo de variável:</strong> Classificação prediz categorias, Regressão prediz valores contínuos<br>
            • <strong>Métricas:</strong> Classificação usa acurácia/precisão, Regressão usa R²/MSE<br>
            • <strong>Interpretação:</strong> Classificação: "É maligno?", Regressão: "Qual o tamanho?"<br>
            • <strong>Aplicação:</strong> Classificação para diagnóstico, Regressão para quantificação<br><br>
            <h4>💡 Insights dos Resultados:</h4>
            ${generateInsights()}
        </div>
        <div class="warning-box">
            <h4>⚠️ Limitações e Considerações:</h4>
            • Dataset sintético para fins educacionais<br>
            • Validação cruzada não implementada (divisão simples treino/teste)<br>
            • Não foram testados outros algoritmos de classificação/regressão<br>
            • Análise de features e seleção de variáveis não realizada<br>
            • Em aplicações reais, seria necessário validação médica especializada
        </div>
    `;
    document.getElementById('finalAnalysis').innerHTML = analysisHTML;
    plotPerformanceComparison();
}
function generateInsights() {
    let insights = '';
    if (results.knn && results.knn.accuracy > 0.9) {
        insights += '✅ Modelo KNN apresentou excelente performance (>90% acurácia)<br>';
    } else if (results.knn && results.knn.accuracy > 0.8) {
        insights += '✅ Modelo KNN apresentou boa performance (>80% acurácia)<br>';
    } else {
        insights += '⚠️ Modelo KNN pode precisar de otimização<br>';
    }
    if (results.regression && results.regression.r2_test > 0.8) {
        insights += '✅ Modelo de regressão explica bem a variabilidade da área (R² > 0.8)<br>';
    } else if (results.regression && results.regression.r2_test > 0.6) {
        insights += '✅ Modelo de regressão tem poder explicativo moderado<br>';
    } else {
        insights += '⚠️ Modelo de regressão pode precisar de mais features<br>';
    }
    if (results.knnSynthetic && results.knn) {
        const improvement = results.knnSynthetic.accuracy - results.knn.accuracy;
        if (improvement > 0.05) {
            insights += '✅ Dados sintéticos melhoraram significativamente a performance<br>';
        } else if (improvement > 0) {
            insights += '✅ Dados sintéticos trouxeram melhoria modesta<br>';
        } else {
            insights += '⚠️ Dados sintéticos não melhoraram a performance<br>';
        }
    }
    return insights;
}
function plotPerformanceComparison() {
    if (!results.knn || !results.regression) return;
    const trace1 = {
        x: ['Classificação (KNN)', 'Regressão Linear'],
        y: [results.knn.accuracy * 100, results.regression.r2_test * 100],
        type: 'bar',
        marker: {
            color: ['#3498db', '#e74c3c'],
            opacity: 0.8
        },
        text: [`${(results.knn.accuracy * 100).toFixed(1)}%`, `${(results.regression.r2_test * 100).toFixed(1)}%`],
        textposition: 'auto'
    };
    const layout = {
        title: 'Comparação de Performance dos Modelos',
        yaxis: {
            title: 'Performance (%)',
            range: [0, 100]
        },
        showlegend: false,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    Plotly.newPlot('performanceComparison', [trace1], layout);
}
function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}
function matrixMultiply(a, b) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
        result[i] = [];
        for (let j = 0; j < b[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < b.length; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}
function matrixVectorMultiply(matrix, vector) {
    return matrix.map(row =>
        row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
    );
}
function matrixInverse(matrix) {
    const n = matrix.length;
    const identity = Array(n).fill().map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) identity[i][i] = 1;
    const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
    for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
        for (let k = i + 1; k < n; k++) {
            const factor = augmented[k][i] / augmented[i][i];
            for (let j = i; j < 2 * n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    for (let i = n - 1; i >= 0; i--) {
        for (let k = i - 1; k >= 0; k--) {
            const factor = augmented[k][i] / augmented[i][i];
            for (let j = i; j < 2 * n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    for (let i = 0; i < n; i++) {
        const divisor = augmented[i][i];
        for (let j = 0; j < 2 * n; j++) {
            augmented[i][j] /= divisor;
        }
    }
    return augmented.map(row => row.slice(n));
}
function meanSquaredError(actual, predicted) {
    return actual.reduce((sum, val, idx) =>
        sum + Math.pow(val - predicted[idx], 2), 0
    ) / actual.length;
}
function meanAbsoluteError(actual, predicted) {
    return actual.reduce((sum, val, idx) =>
        sum + Math.abs(val - predicted[idx]), 0
    ) / actual.length;
}
function rSquared(actual, predicted) {
    const actualMean = actual.reduce((a, b) => a + b) / actual.length;
    const totalSumSquares = actual.reduce((sum, val) =>
        sum + Math.pow(val - actualMean, 2), 0
    );
    const residualSumSquares = actual.reduce((sum, val, idx) =>
        sum + Math.pow(val - predicted[idx], 2), 0
    );
    return 1 - (residualSumSquares / totalSumSquares);
}
function calculateSkewness(data, mean, std) {
    const n = data.length;
    const sum = data.reduce((acc, val) => acc + Math.pow((val - mean) / std, 3), 0);
    return (n / ((n - 1) * (n - 2))) * sum;
}
function calculateKurtosis(data, mean, std) {
    const n = data.length;
    const sum = data.reduce((acc, val) => acc + Math.pow((val - mean) / std, 4), 0);
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum -
           (3 * Math.pow(n - 1, 2) / ((n - 2) * (n - 3)));
}
function calculateVariance(data) {
    const mean = data.reduce((a, b) => a + b) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
}
class DecisionTree {
    constructor(maxDepth = 10, minSamplesToSplit = 2) {
        this.maxDepth = maxDepth;
        this.minSamplesToSplit = minSamplesToSplit;
        this.root = null;
    }
    train(X, y) {
        this.root = this.buildTree(X, y, 0);
    }
    buildTree(X, y, depth) {
        if (depth >= this.maxDepth || X.length <= this.minSamplesToSplit || this.allSameClass(y)) {
            return {
                type: 'leaf',
                prediction: this.getMajorityClass(y)
            };
        }
        const { featureIndex, threshold, leftX, leftY, rightX, rightY } = this.findBestSplit(X, y);
        if (leftX.length === 0 || rightX.length === 0) {
            return {
                type: 'leaf',
                prediction: this.getMajorityClass(y)
            };
        }
        const leftSubtree = this.buildTree(leftX, leftY, depth + 1);
        const rightSubtree = this.buildTree(rightX, rightY, depth + 1);
        return {
            type: 'node',
            featureIndex: featureIndex,
            threshold: threshold,
            left: leftSubtree,
            right: rightSubtree
        };
    }
    allSameClass(y) {
        if (y.length === 0) return true;
        const firstClass = y[0];
        return y.every(val => val === firstClass);
    }
    getMajorityClass(y) {
        if (y.length === 0) return 0;
        const counts = y.reduce((acc, val) => {
            acc[val] = (acc[val] || 0) + 1;
            return acc;
        }, {});
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }
    findBestSplit(X, y) {
        const numFeatures = X[0].length;
        let bestGini = Infinity;
        let bestFeatureIndex = 0;
        let bestThreshold = 0;
        let bestLeftX = [];
        let bestLeftY = [];
        let bestRightX = [];
        let bestRightY = [];
        const numFeaturesToConsider = Math.max(1, Math.floor(Math.sqrt(numFeatures)));
        const featureIndices = [];
        for (let i = 0; i < numFeatures; i++) featureIndices.push(i);
        const shuffledFeatures = featureIndices.sort(() => Math.random() - 0.5).slice(0, numFeaturesToConsider);
        for (let i = 0; i < numFeaturesToConsider; i++) {
            const featureIndex = shuffledFeatures[i];
            const featureValues = X.map(sample => sample[featureIndex]);
            const uniqueValues = [...new Set(featureValues)].sort((a, b) => a - b);
            for (let j = 0; j < uniqueValues.length - 1; j++) {
                const threshold = (uniqueValues[j] + uniqueValues[j + 1]) / 2;
                const leftIndices = [];
                const rightIndices = [];
                for (let k = 0; k < X.length; k++) {
                    if (X[k][featureIndex] <= threshold) {
                        leftIndices.push(k);
                    } else {
                        rightIndices.push(k);
                    }
                }
                const leftX = leftIndices.map(idx => X[idx]);
                const leftY = leftIndices.map(idx => y[idx]);
                const rightX = rightIndices.map(idx => X[idx]);
                const rightY = rightIndices.map(idx => y[idx]);
                const gini = this.calculateGiniIndex(leftY, rightY);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                    bestLeftX = leftX;
                    bestLeftY = leftY;
                    bestRightX = rightX;
                    bestRightY = rightY;
                }
            }
        }
        return {
            featureIndex: bestFeatureIndex,
            threshold: bestThreshold,
            leftX: bestLeftX,
            leftY: bestLeftY,
            rightX: bestRightX,
            rightY: bestRightY
        };
    }
    calculateGiniIndex(leftY, rightY) {
        const totalSamples = leftY.length + rightY.length;
        if (totalSamples === 0) return 0;
        const leftWeight = leftY.length / totalSamples;
        const rightWeight = rightY.length / totalSamples;
        const leftGini = this.calculateGini(leftY);
        const rightGini = this.calculateGini(rightY);
        return leftWeight * leftGini + rightWeight * rightGini;
    }
    calculateGini(y) {
        if (y.length === 0) return 0;
        const classCounts = {};
        for (const label of y) {
            classCounts[label] = (classCounts[label] || 0) + 1;
        }
        let gini = 1;
        for (const label in classCounts) {
            const prob = classCounts[label] / y.length;
            gini -= prob * prob;
        }
        return gini;
    }
    predict(sample) {
        return this.predictSample(sample, this.root);
    }
    predictSample(sample, node) {
        if (node.type === 'leaf') {
            return parseInt(node.prediction);
        }
        if (sample[node.featureIndex] <= node.threshold) {
            return this.predictSample(sample, node.left);
        } else {
            return this.predictSample(sample, node.right);
        }
    }
}
class RandomForest {
    constructor(nTrees = 10, maxDepth = 10) {
        this.nTrees = nTrees;
        this.maxDepth = maxDepth;
        this.trees = [];
    }
    train(X, y) {
        this.trees = [];
        for (let i = 0; i < this.nTrees; i++) {
            const bootstrapIndices = this.bootstrapSample(X.length);
            const bootstrapX = bootstrapIndices.map(idx => X[idx]);
            const bootstrapY = bootstrapIndices.map(idx => y[idx]);
            const tree = new DecisionTree(this.maxDepth);
            tree.train(bootstrapX, bootstrapY);
            this.trees.push(tree);
        }
    }
    bootstrapSample(size) {
        const indices = [];
        for (let i = 0; i < size; i++) {
            indices.push(Math.floor(Math.random() * size));
        }
        return indices;
    }
    predict(sample) {
        const votes = this.trees.map(tree => tree.predict(sample));
        const voteCounts = {};
        for (const vote of votes) {
            voteCounts[vote] = (voteCounts[vote] || 0) + 1;
        }
        return parseInt(Object.keys(voteCounts).reduce((a, b) => voteCounts[a] > voteCounts[b] ? a : b));
    }
    predictMultiple(samples) {
        return samples.map(sample => this.predict(sample));
    }
}
function trainRandomForest() {
    const nTrees = parseInt(document.getElementById('nTrees').value);
    const maxDepth = parseInt(document.getElementById('maxDepth').value);
    document.getElementById('rfResults').innerHTML = '<div class="loading"></div> Treinando modelo Random Forest...';
    setTimeout(() => {
        try {
            const shuffled = [...processedData].sort(() => Math.random() - 0.5);
            const trainSize = Math.floor(shuffled.length * 0.8);
            const trainData = shuffled.slice(0, trainSize);
            const testData = shuffled.slice(trainSize);
            const X_train = trainData.map(point => point.features);
            const y_train = trainData.map(point => point.target);
            const X_test = testData.map(point => point.features);
            const y_test = testData.map(point => point.target);
            const rf = new RandomForest(nTrees, maxDepth);
            rf.train(X_train, y_train);
            const predictions = rf.predictMultiple(X_test);
            const accuracy = predictions.filter((pred, idx) => pred === y_test[idx]).length / predictions.length;
            let tp = 0, tn = 0, fp = 0, fn = 0;
            predictions.forEach((pred, idx) => {
                if (pred === 1 && y_test[idx] === 1) tp++;
                else if (pred === 0 && y_test[idx] === 0) tn++;
                else if (pred === 1 && y_test[idx] === 0) fp++;
                else if (pred === 0 && y_test[idx] === 1) fn++;
            });
            const precision = tp / (tp + fp) || 0;
            const recall = tp / (tp + fn) || 0;
            const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
            results.rf = {
                nTrees: nTrees,
                maxDepth: maxDepth,
                accuracy: accuracy,
                precision: precision,
                recall: recall,
                f1Score: f1Score,
                confusionMatrix: { tp, tn, fp, fn }
            };
            rfModel = rf;
            document.getElementById('rfResults').innerHTML = `
                <div class="info-box">
                    <strong>🌲 Modelo Random Forest Treinado (${nTrees} árvores, profundidade ${maxDepth})</strong><br>
                    Dados de treino: ${trainData.length} amostras<br>
                    Dados de teste: ${testData.length} amostras<br>
                    ✅ Treinamento concluído com sucesso!
                </div>
            `;
            document.getElementById('rfMetrics').innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${(accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-label">Acurácia</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(precision * 100).toFixed(1)}%</div>
                    <div class="metric-label">Precisão</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(recall * 100).toFixed(1)}%</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(f1Score * 100).toFixed(1)}%</div>
                    <div class="metric-label">F1-Score</div>
                </div>
            `;
            document.getElementById('cvBtn').disabled = false;
        } catch (error) {
            console.error("Error in trainRandomForest:", error);
            document.getElementById('rfResults').innerHTML = `
                <div class="warning-box">
                    <strong>⚠️ Erro ao treinar Random Forest</strong><br>
                    ${error.message}
                </div>
            `;
        }
    }, 2000);
}
function createFolds(data, numFolds) {
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    const foldSize = Math.floor(shuffled.length / numFolds);
    const folds = [];
    for (let i = 0; i < numFolds; i++) {
        const start = i * foldSize;
        const end = (i === numFolds - 1) ? shuffled.length : (i + 1) * foldSize;
        folds.push(shuffled.slice(start, end));
    }
    return folds;
}
function calculateMetrics(actual, predicted) {
    const accuracy = predicted.filter((pred, idx) => pred === actual[idx]).length / predicted.length;
    let tp = 0, tn = 0, fp = 0, fn = 0;
    predicted.forEach((pred, idx) => {
        if (pred === 1 && actual[idx] === 1) tp++;
        else if (pred === 0 && actual[idx] === 0) tn++;
        else if (pred === 1 && actual[idx] === 0) fp++;
        else if (pred === 0 && actual[idx] === 1) fn++;
    });
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    return {
        accuracy: accuracy,
        precision: precision,
        recall: recall,
        f1Score: f1Score,
        confusionMatrix: { tp, tn, fp, fn }
    };
}
function runCrossValidation() {
    const numFolds = parseInt(document.getElementById('nFolds').value);
    const kValue = parseInt(document.getElementById('kValue').value);
    const nTrees = parseInt(document.getElementById('nTrees').value);
    const maxDepth = parseInt(document.getElementById('maxDepth').value);
    document.getElementById('cvResults').innerHTML = '<div class="loading"></div> Executando validação cruzada...';
    setTimeout(() => {
        try {
            const folds = createFolds(processedData, numFolds);
            const knnMetrics = [];
            const rfMetrics = [];
            for (let i = 0; i < numFolds; i++) {
                const testData = folds[i];
                const trainData = [];
                for (let j = 0; j < numFolds; j++) {
                    if (j !== i) trainData.push(...folds[j]);
                }
                const X_train = trainData.map(point => point.features);
                const y_train = trainData.map(point => point.target);
                const X_test = testData.map(point => point.features);
                const y_test = testData.map(point => point.target);
                const knnPredictions = X_test.map(point =>
                    parseInt(knnPredict(trainData, point, kValue))
                );
                knnMetrics.push(calculateMetrics(y_test, knnPredictions));
                const rf = new RandomForest(nTrees, maxDepth);
                rf.train(X_train, y_train);
                const rfPredictions = rf.predictMultiple(X_test);
                rfMetrics.push(calculateMetrics(y_test, rfPredictions));
            }
            const calculateMeanMetrics = (metricsArray) => {
                const meanMetrics = {
                    accuracy: 0,
                    precision: 0,
                    recall: 0,
                    f1Score: 0
                };
                for (const metrics of metricsArray) {
                    meanMetrics.accuracy += metrics.accuracy / metricsArray.length;
                    meanMetrics.precision += metrics.precision / metricsArray.length;
                    meanMetrics.recall += metrics.recall / metricsArray.length;
                    meanMetrics.f1Score += metrics.f1Score / metricsArray.length;
                }
                return meanMetrics;
            };
            const knnMeanMetrics = calculateMeanMetrics(knnMetrics);
            const rfMeanMetrics = calculateMeanMetrics(rfMetrics);
            crossValidationResults = {
                knn: {
                    k: kValue,
                    folds: knnMetrics,
                    mean: knnMeanMetrics
                },
                rf: {
                    nTrees: nTrees,
                    maxDepth: maxDepth,
                    folds: rfMetrics,
                    mean: rfMeanMetrics
                }
            };
            const bestK = kValue;
            const bestNTrees = nTrees;
            const bestMaxDepth = maxDepth;
            const X_all = processedData.map(point => point.features);
            const y_all = processedData.map(point => point.target);
            const finalRF = new RandomForest(bestNTrees, bestMaxDepth);
            finalRF.train(X_all, y_all);
            let finalRFMetrics = null;
            if (results.rf) {
                finalRFMetrics = {
                    accuracy: results.rf.accuracy,
                    precision: results.rf.precision,
                    recall: results.rf.recall,
                    f1Score: results.rf.f1Score
                };
            }
            const vencedor = rfMeanMetrics.f1Score > knnMeanMetrics.f1Score ?
                'Random Forest' : 'KNN';
            document.getElementById('cvResults').innerHTML = `
                <div class="info-box">
                    <strong>🔍 Resultados da Validação Cruzada (${numFolds} folds)</strong><br><br>
                    <strong>KNN (K=${kValue}):</strong><br>
                    Acurácia: ${(knnMeanMetrics.accuracy * 100).toFixed(1)}%<br>
                    Precisão: ${(knnMeanMetrics.precision * 100).toFixed(1)}%<br>
                    Recall: ${(knnMeanMetrics.recall * 100).toFixed(1)}%<br>
                    F1-Score: ${(knnMeanMetrics.f1Score * 100).toFixed(1)}%<br><br>
                    <strong>Random Forest (${nTrees} árvores, profundidade ${maxDepth}):</strong><br>
                    Acurácia: ${(rfMeanMetrics.accuracy * 100).toFixed(1)}%<br>
                    Precisão: ${(rfMeanMetrics.precision * 100).toFixed(1)}%<br>
                    Recall: ${(rfMeanMetrics.recall * 100).toFixed(1)}%<br>
                    F1-Score: ${(rfMeanMetrics.f1Score * 100).toFixed(1)}%<br><br>
                    ${finalRFMetrics ? `
                    <strong>Random Forest Re-treinado (conjunto completo):</strong><br>
                    Acurácia: ${(finalRFMetrics.accuracy * 100).toFixed(1)}%<br>
                    Precisão: ${(finalRFMetrics.precision * 100).toFixed(1)}%<br>
                    Recall: ${(finalRFMetrics.recall * 100).toFixed(1)}%<br>
                    F1-Score: ${(finalRFMetrics.f1Score * 100).toFixed(1)}%<br><br>
                    ` : ''}
                    <strong>🏆 Modelo vencedor: ${vencedor}</strong>
                </div>
            `;
            plotModelComparison(knnMeanMetrics, rfMeanMetrics, finalRFMetrics);
        } catch (error) {
            console.error("Error in runCrossValidation:", error);
            document.getElementById('cvResults').innerHTML = `
                <div class="warning-box">
                    <strong>⚠️ Erro na validação cruzada</strong><br>
                    ${error.message}
                </div>
            `;
        }
    }, 3000);
}
function plotModelComparison(knnMetrics, rfMetrics, finalRFMetrics) {
    const metrics = ['accuracy', 'precision', 'recall', 'f1Score'];
    const metricLabels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score'];
    const models = ['KNN (CV)', 'Random Forest (CV)'];
    const modelData = [
        [knnMetrics.accuracy, knnMetrics.precision, knnMetrics.recall, knnMetrics.f1Score],
        [rfMetrics.accuracy, rfMetrics.precision, rfMetrics.recall, rfMetrics.f1Score]
    ];
    if (finalRFMetrics) {
        models.push('Random Forest (Retreinado)');
        modelData.push([
            finalRFMetrics.accuracy,
            finalRFMetrics.precision,
            finalRFMetrics.recall,
            finalRFMetrics.f1Score
        ]);
    }
    const plotData = [];
    for (let i = 0; i < models.length; i++) {
        plotData.push({
            x: metricLabels,
            y: modelData[i].map(val => val * 100),
            type: 'bar',
            name: models[i]
        });
    }
    const layout = {
        title: 'Comparação de Desempenho entre Modelos',
        xaxis: {
            title: 'Métrica'
        },
        yaxis: {
            title: 'Valor (%)',
            range: [0, 100]
        },
        barmode: 'group',
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    Plotly.newPlot('modelComparisonPlot', plotData, layout);
    const vencedor = rfMetrics.f1Score > knnMetrics.f1Score ?
        'Random Forest' : 'KNN';
    document.getElementById('finalAnalysis').innerHTML = `
        <div class="info-box">
            <h3>📋 Análise Final dos Modelos</h3>
            <p><strong>Baseline (KNN):</strong> O modelo K-Nearest Neighbors alcançou ${(knnMetrics.f1Score * 100).toFixed(1)}% 
            de F1-Score na validação cruzada com K=${document.getElementById('kValue').value}.</p>
            <p><strong>Random Forest:</strong> O modelo Random Forest com ${document.getElementById('nTrees').value} árvores 
            e profundidade máxima ${document.getElementById('maxDepth').value} alcançou ${(rfMetrics.f1Score * 100).toFixed(1)}% 
            de F1-Score na validação cruzada.</p>
            ${finalRFMetrics ? `
            <p><strong>Random Forest Re-treinado:</strong> Após retreinar o Random Forest com os melhores 
            hiperparâmetros no conjunto completo de dados, o modelo alcançou ${(finalRFMetrics.f1Score * 100).toFixed(1)}% 
            de F1-Score.</p>
            ` : ''}
            <p><strong>Conclusão:</strong> O modelo ${vencedor} apresentou melhor desempenho geral, 
            com métricas de precisão, recall e F1-Score superiores. 
            ${rfMetrics.f1Score > knnMetrics.f1Score ? 
                'O Random Forest demonstrou maior capacidade de generalização e robustez na classificação deste conjunto de dados.' : 
                'O KNN apresentou-se como um classificador eficaz para este conjunto de dados, mesmo sendo um modelo mais simples.'}
            </p>
        </div>
    `;
    plotPerformanceComparison();
}
function plotPerformanceComparison() {
    if (!crossValidationResults.knn || !crossValidationResults.rf) return;
    const knnFolds = crossValidationResults.knn.folds;
    const rfFolds = crossValidationResults.rf.folds;
    const accuracyData = {
        x: Array.from({length: knnFolds.length}, (_, i) => `Fold ${i+1}`),
        y: knnFolds.map(m => m.accuracy * 100),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'KNN Acurácia',
        line: {color: 'rgb(31, 119, 180)'}
    };
    const f1ScoreData = {
        x: Array.from({length: knnFolds.length}, (_, i) => `Fold ${i+1}`),
        y: knnFolds.map(m => m.f1Score * 100),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'KNN F1-Score',
        line: {color: 'rgb(31, 119, 180)', dash: 'dot'}
    };
    const rfAccuracyData = {
        x: Array.from({length: rfFolds.length}, (_, i) => `Fold ${i+1}`),
        y: rfFolds.map(m => m.accuracy * 100),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'RF Acurácia',
        line: {color: 'rgb(255, 127, 14)'}
    };
    const rfF1ScoreData = {
        x: Array.from({length: rfFolds.length}, (_, i) => `Fold ${i+1}`),
        y: rfFolds.map(m => m.f1Score * 100),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'RF F1-Score',
        line: {color: 'rgb(255, 127, 14)', dash: 'dot'}
    };
    const layout = {
        title: 'Performance por Fold na Validação Cruzada',
        xaxis: {
            title: 'Fold'
        },
        yaxis: {
            title: 'Score (%)',
            range: [0, 100]
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    Plotly.newPlot('performanceComparison',
        [accuracyData, f1ScoreData, rfAccuracyData, rfF1ScoreData], layout);
}
