let analyticsTrendChart = null;
let analyticsAverageChart = null;
let lastAnalyticsData = null;

document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('analyticsSummaryCards')) {
        return;
    }

    const refreshBtn = document.getElementById('refreshAnalyticsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => loadLastRunAnalytics(true));
    }

    const downloadBtn = document.getElementById('downloadAnalyticsBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadAnalyticsDataAsCSV);
    }

    loadLastRunAnalytics();
});

async function loadLastRunAnalytics(forceReload = false) {
    const summaryCards = document.getElementById('analyticsSummaryCards');
    const tableBody = document.getElementById('analyticsTableBody');

    if (summaryCards) {
        summaryCards.innerHTML = '<div class="history-loading">Loading last run analytics...</div>';
    }
    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="10" class="history-empty-state">Loading analytics...</td></tr>';
    }

    try {
        const params = new URLSearchParams(window.location.search);
        const sessionId = params.get('session_id');
        const endpoint = sessionId
            ? `/api/analytics-last-run?session_id=${encodeURIComponent(sessionId)}`
            : '/api/analytics-last-run';

        const response = await fetch(endpoint, {
            headers: forceReload ? { 'Cache-Control': 'no-cache' } : {}
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || `Unable to load analytics (HTTP ${response.status})`);
        }

        renderAnalyticsDashboard(data);
    } catch (error) {
        console.error('Failed to load analytics:', error);
        if (summaryCards) {
            summaryCards.innerHTML = '<div class="history-loading error">Unable to load analytics.</div>';
        }
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="10" class="history-empty-state">Unable to load analytics.</td></tr>';
        }
    }
}

function renderAnalyticsDashboard(data) {
    if (!data.has_data) {
        renderEmptyAnalytics(data.message || 'No stored runs yet.');
        return;
    }

    lastAnalyticsData = data;
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const session = data.session || {};
    const summary = data.summary || {};

    const summaryCards = document.getElementById('analyticsSummaryCards');
    const countLabel = document.getElementById('analyticsCountLabel');

    if (summaryCards) {
        summaryCards.innerHTML = buildSummaryCards(session, rows, summary);
    }

    if (countLabel) {
        countLabel.textContent = `${rows.length} row${rows.length === 1 ? '' : 's'}`;
    }

    renderTrendChart(data.trend_points || []);
    renderAverageChart(summary);
    renderRowsTable(rows, data);
    renderCodeAnalysis(data.code_analysis || {});
    renderMultimodal(data.multimodal || {});
}

function renderEmptyAnalytics(message) {
    const summaryCards = document.getElementById('analyticsSummaryCards');
    const tableBody = document.getElementById('analyticsTableBody');

    if (summaryCards) {
        summaryCards.innerHTML = `<div class="history-loading">${escapeHtml(message)}</div>`;
    }

    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="10" class="history-empty-state">No rows available for analytics.</td></tr>';
    }

    const codeContainer = document.getElementById('codeAnalysisContainer');
    if (codeContainer) {
        codeContainer.innerHTML = '<div class="history-empty-state">No code analysis available yet.</div>';
    }

    const multimodalContainer = document.getElementById('multimodalContainer');
    if (multimodalContainer) {
        multimodalContainer.innerHTML = '<div class="history-empty-state">No multimodal evaluations available yet.</div>';
    }
}

function buildSummaryCards(session, rows, summary) {
    const cards = [
        { label: 'Source', value: session.source_url || 'unknown', hint: 'Source name/url attached to the run' },
        { label: 'Session Time', value: formatTime(session.timestamp), hint: 'Timestamp of latest run' },
        { label: 'Messages', value: String(session.message_count || 0), hint: 'Messages processed in this run' },
        { label: 'Avg Overall Score', value: formatPercent(summary.overall_score), hint: 'Average overall score in this run' },
    ];

    return cards.map((card) => `
        <article class="history-summary-card">
            <span class="history-summary-label">${escapeHtml(card.label)}</span>
            <strong class="history-summary-value" style="font-size: 2rem;">${escapeHtml(card.value)}</strong>
            <p class="history-summary-hint">${escapeHtml(card.hint)}</p>
        </article>
    `).join('');
}

function renderTrendChart(points) {
    const canvas = document.getElementById('analyticsTrendChart');
    if (!canvas || typeof Chart === 'undefined') {
        return;
    }

    const labels = points.map((point) => point.label || `Pair ${point.id || ''}`);
    const values = points.map((point) => toChartValue(point.overall_score));

    if (analyticsTrendChart) {
        analyticsTrendChart.destroy();
    }

    analyticsTrendChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Overall Score',
                data: values,
                borderColor: '#00f0ff',
                backgroundColor: 'rgba(0, 240, 255, 0.18)',
                pointRadius: 4,
                pointBackgroundColor: '#ffffff',
                tension: 0.35,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#cfd8e3' },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#cfd8e3',
                        callback: (value) => formatPercent(value)
                    },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                }
            }
        }
    });
}

function renderAverageChart(summary) {
    const canvas = document.getElementById('analyticsAverageChart');
    if (!canvas || typeof Chart === 'undefined') {
        return;
    }

    const labels = ['Relevance', 'Length', 'Coherence', 'Toxicity', 'Bias', 'Hallucination'];
    const values = [
        toChartValue(summary.relevance),
        toChartValue(summary.length_appropriateness),
        toChartValue(summary.coherence),
        toChartValue(summary.toxicity),
        toChartValue(summary.bias),
        toChartValue(summary.hallucination),
    ];

    if (analyticsAverageChart) {
        analyticsAverageChart.destroy();
    }

    analyticsAverageChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Average Value',
                data: values,
                backgroundColor: [
                    'rgba(66,165,245,0.6)',
                    'rgba(0,240,255,0.6)',
                    'rgba(76,175,80,0.6)',
                    'rgba(255,152,0,0.6)',
                    'rgba(233,30,99,0.6)',
                    'rgba(255,87,34,0.6)'
                ],
                borderColor: [
                    'rgba(66,165,245,1)',
                    'rgba(0,240,255,1)',
                    'rgba(76,175,80,1)',
                    'rgba(255,152,0,1)',
                    'rgba(233,30,99,1)',
                    'rgba(255,87,34,1)'
                ],
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#cfd8e3' },
                    grid: { display: false }
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#cfd8e3',
                        callback: (value) => formatPercent(value)
                    },
                    grid: { color: 'rgba(255,255,255,0.06)' }
                }
            }
        }
    });
}

function renderRowsTable(rows, data) {
    const tableBody = document.getElementById('analyticsTableBody');
    if (!tableBody) {
        return;
    }

    if (!rows.length) {
        tableBody.innerHTML = '<tr><td colspan="10" class="history-empty-state">No rows in last run.</td></tr>';
        return;
    }

    tableBody.innerHTML = rows.map((row, index) => {
        const cleanQuestion = (row.question || '').replace(/^you\s+said\s*/i, '').trim();
        const cleanResponse = (row.response || '').replace(/^copilot\s+said\s*/i, '').replace(/^assistant\s+said\s*/i, '').trim();
        const rowId = index + 1;
        return `
        <tr>
            <td>${rowId}</td>
            <td>${escapeHtml(formatTime(row.timestamp))}</td>
            <td title="${escapeHtml(cleanQuestion)}">${escapeHtml(truncateText(cleanQuestion, 40))}</td>
            <td title="${escapeHtml(cleanResponse)}">${escapeHtml(truncateText(cleanResponse, 40))}</td>
            <td><span class="history-score-badge">${formatPercent(row.overall_score)}</span></td>
            <td>${formatMetricCell(row.relevance)}</td>
            <td>${formatMetricCell(row.coherence)}</td>
            <td>${formatMetricCell(row.toxicity)}</td>
            <td>${formatMetricCell(row.bias)}</td>
            <td>${formatMetricCell(row.hallucination)}</td>
        </tr>
    `;
    }).join('');
}

function renderCodeAnalysis(codeAnalysis) {
    const countLabel = document.getElementById('codeAnalysisCountLabel');
    const container = document.getElementById('codeAnalysisContainer');

    if (countLabel) {
        countLabel.textContent = String(codeAnalysis.count || 0);
    }

    if (!container) {
        return;
    }

    const reports = Array.isArray(codeAnalysis.reports) ? codeAnalysis.reports : [];
    if (!reports.length) {
        container.innerHTML = '<div class="history-empty-state">No code blocks found in the last run responses.</div>';
        return;
    }

    container.innerHTML = reports.map((item) => {
        const report = item.report || {};
        const metrics = report.metrics || {};
        const errors = Array.isArray(report.errors) ? report.errors : [];
        // Use highlight.js for syntax highlighting if available
        const codeHtml = window.hljs
            ? `<pre class="logs-content"><code class="language-${escapeHtml(item.language || 'python')}">${window.hljs.highlight(item.snippet || '', {language: item.language || 'python', ignoreIllegals: true}).value}</code></pre>`
            : `<pre class="logs-content">${escapeHtml(item.snippet || '')}</pre>`;
        return `
            <article class="chart-card" style="min-height: auto; margin-bottom: 1rem;">
                <div class="chart-card-header">
                    <h3>Row ${escapeHtml(String(item.row_id || '-'))} - Code Block ${escapeHtml(String(item.block_index || 1))}</h3>
                    <p>Syntax Valid: <strong>${report.syntax_valid ? 'Yes' : 'No'}</strong></p>
                </div>
                <p style="font-size:1.3rem; color:#b8c4d6; margin-bottom:0.8rem;">${escapeHtml(report.explanation || '')}</p>
                ${codeHtml}
                <p style="font-size:1.3rem; margin-top:0.8rem;">Lines: ${escapeHtml(String(metrics['Total Lines'] || 0))}, Functions: ${escapeHtml(String(metrics['Functions'] || 0))}, Classes: ${escapeHtml(String(metrics['Classes'] || 0))}</p>
                ${errors.length ? `<p style="font-size:1.3rem; color:#ffb3b3;">Errors: ${escapeHtml(errors.join(' | '))}</p>` : ''}
            </article>
        `;
    }).join('');
}

function renderMultimodal(multimodal) {
    const countLabel = document.getElementById('multimodalCountLabel');
    const container = document.getElementById('multimodalContainer');

    if (countLabel) {
        countLabel.textContent = String(multimodal.count || 0);
    }

    if (!container) {
        return;
    }

    const reports = Array.isArray(multimodal.reports) ? multimodal.reports : [];
    if (!reports.length) {
        container.innerHTML = '<div class="history-empty-state">No image URLs found in last run responses.</div>';
        return;
    }

    container.innerHTML = reports.map((item) => {
        if (!item.success) {
            return `
                <article class="chart-card" style="min-height: auto; margin-bottom: 1rem;">
                    <div class="chart-card-header">
                        <h3>Row ${escapeHtml(String(item.row_id || '-'))} - Image Evaluation</h3>
                        <p>Unable to evaluate this image</p>
                    </div>
                    <p style="font-size:1.3rem; color:#ffb3b3;">${escapeHtml(item.error || 'Unknown error')}</p>
                    <a href="${escapeHtml(item.image_url || '#')}" target="_blank" style="font-size:1.3rem; color:#00f0ff;">Open Image URL</a>
                </article>
            `;
        }

        const metrics = item.metrics || {};
        // Show the image directly
        const imageTag = `<img src="${escapeHtml(item.image_url || '')}" alt="Evaluated Image" style="max-width: 320px; max-height: 220px; border-radius: 0.7rem; border: 2px solid #00f0ff; margin-bottom: 0.7rem; display: block;">`;
        return `
            <article class="chart-card" style="min-height: auto; margin-bottom: 1rem;">
                <div class="chart-card-header">
                    <h3>Row ${escapeHtml(String(item.row_id || '-'))} - Image Evaluation</h3>
                    <p>Overall Accuracy: ${formatPercent(metrics['Overall Accuracy'])}</p>
                </div>
                ${imageTag}
                <p style="font-size:1.3rem; color:#b8c4d6; margin-bottom:0.8rem;">${escapeHtml(item.explanation || '')}</p>
                <a href="${escapeHtml(item.image_url || '#')}" target="_blank" style="font-size:1.3rem; color:#00f0ff;">Open Image URL</a>
            </article>
        `;
    }).join('');
}

function formatMetricCell(value) {
    return `<span class="history-metric-chip">${formatPercent(value)}</span>`;
}

function formatPercent(value) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return '0.0%';
    }
    return `${(numeric * 100).toFixed(1)}%`;
}

function toChartValue(value) {
    const numeric = Number(value);
    return Number.isNaN(numeric) ? 0 : Math.max(0, Math.min(1, numeric));
}

function truncateText(value, maxLength) {
    const text = String(value || '');
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
}

function formatTime(timestamp) {
    if (!timestamp) {
        return 'Unknown';
    }

    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
        return String(timestamp);
    }

    return date.toLocaleString();
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function downloadAnalyticsDataAsCSV() {
    if (!lastAnalyticsData || !Array.isArray(lastAnalyticsData.rows) || lastAnalyticsData.rows.length === 0) {
        alert('No data available to download');
        return;
    }

    const rows = lastAnalyticsData.rows;
    const session = lastAnalyticsData.session || {};
    
    // Build CSV header
    const headers = ['ID', 'Timestamp', 'Question', 'Response', 'Overall Score', 'Relevance', 'Coherence', 'Toxicity', 'Bias', 'Hallucination'];
    const csvData = [headers.join(',')];

    // Build CSV rows with ID from 1 upward
    rows.forEach((row, index) => {
        const rowId = index + 1;
        const timestamp = row.timestamp || '';
        const question = (row.question || '').replace(/"/g, '""').replace(/,/g, ';');
        const response = (row.response || '').replace(/"/g, '""').replace(/,/g, ';');
        const overallScore = formatPercentRaw(row.overall_score);
        const relevance = formatPercentRaw(row.relevance);
        const coherence = formatPercentRaw(row.coherence);
        const toxicity = formatPercentRaw(row.toxicity);
        const bias = formatPercentRaw(row.bias);
        const hallucination = formatPercentRaw(row.hallucination);

        const csvRow = [
            rowId,
            `"${timestamp}"`,
            `"${question}"`,
            `"${response}"`,
            overallScore,
            relevance,
            coherence,
            toxicity,
            bias,
            hallucination
        ].join(',');

        csvData.push(csvRow);
    });

    // Add session metadata at end
    csvData.push('');
    csvData.push(`Session ID,${session.session_id || 'N/A'}`);
    csvData.push(`Source,${(session.source_url || 'N/A').replace(/"/g, '""')}`);
    csvData.push(`Timestamp,${session.timestamp || 'N/A'}`);
    csvData.push(`Total Rows,${rows.length}`);

    // Create blob and download
    const csv = csvData.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `analytics-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function formatPercentRaw(value) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return '0';
    }
    return (numeric * 100).toFixed(1);
}
