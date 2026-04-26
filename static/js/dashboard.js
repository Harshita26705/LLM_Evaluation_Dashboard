let historyTrendChart = null;
let historyRadarChart = null;

document.addEventListener('DOMContentLoaded', () => {
    const hasHistoryWidgets = document.getElementById('historySummaryCards') || document.getElementById('historyTableBody');
    if (!hasHistoryWidgets) {
        return;
    }

    setupHistoryControls();
    loadEvaluationHistory();
});

function setupHistoryControls() {
    const refreshButton = document.getElementById('refreshHistoryBtn');
    const downloadButton = document.getElementById('downloadHistoryBtn');
    const clearButton = document.getElementById('clearHistoryBtn');
    const tableBody = document.getElementById('historyTableBody');

    if (refreshButton) {
        refreshButton.addEventListener('click', () => loadEvaluationHistory(true));
    }

    if (downloadButton) {
        downloadButton.addEventListener('click', downloadHistorySnapshot);
    }

    if (clearButton) {
        clearButton.addEventListener('click', clearAllHistory);
    }

    if (tableBody) {
        tableBody.addEventListener('click', async (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }

            const deleteButton = target.closest('.delete-session-btn');
            if (!deleteButton) {
                return;
            }

            const sessionId = deleteButton.getAttribute('data-session-id');
            if (!sessionId) {
                return;
            }

            await deleteHistorySession(sessionId);
        });
    }
}

async function loadEvaluationHistory(forceReload = false) {
    const summaryCards = document.getElementById('historySummaryCards');
    const tableBody = document.getElementById('historyTableBody');
    const countLabel = document.getElementById('historyCountLabel');

    if (summaryCards) {
        summaryCards.innerHTML = '<div class="history-loading">Loading evaluation history...</div>';
    }

    if (tableBody) {
        tableBody.innerHTML = '<tr><td colspan="12" class="history-empty-state">Loading history...</td></tr>';
    }

    try {
        const response = await fetch('/api/evaluation-history', {
            headers: forceReload ? { 'Cache-Control': 'no-cache' } : {}
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || `Unable to load history (HTTP ${response.status})`);
        }

        renderHistoryDashboard(data);
    } catch (error) {
        console.error('Failed to load evaluation history:', error);
        if (summaryCards) {
            summaryCards.innerHTML = '<div class="history-loading error">Unable to load evaluation history.</div>';
        }
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="12" class="history-empty-state">Unable to load history.</td></tr>';
        }
        if (countLabel) {
            countLabel.textContent = '0 sessions';
        }
    }
}

function renderHistoryDashboard(data) {
    const sessions = Array.isArray(data.sessions) ? data.sessions.slice() : [];
    const summary = data.summary || {};
    const summaryCards = document.getElementById('historySummaryCards');
    const tableBody = document.getElementById('historyTableBody');
    const countLabel = document.getElementById('historyCountLabel');

    if (countLabel) {
        countLabel.textContent = `${sessions.length} session${sessions.length === 1 ? '' : 's'}`;
    }

    if (summaryCards) {
        summaryCards.innerHTML = buildSummaryCards(summary, sessions);
    }

    renderTrendChart(sessions);
    renderRadarChart(summary);
    renderSessionsTable(sessions, tableBody);
}

function buildSummaryCards(summary, sessions) {
    const averageSummary = summary.average_summary || {};
    const latestSession = summary.latest_session || sessions[sessions.length - 1] || null;

    const cards = [
        { label: 'Total Sessions', value: summary.total_sessions || sessions.length || 0, hint: 'Evaluation runs stored in the dashboard' },
        { label: 'Total Rows', value: summary.total_rows || 0, hint: 'CSV-compatible rows synced from the extension' },
        { label: 'Avg Overall Score', value: formatPercent(averageSummary.average_score), hint: 'Mean session score across all stored runs' },
        { label: 'Latest Session Score', value: formatPercent(latestSession?.summary?.average_score), hint: latestSession ? formatTime(latestSession.timestamp) : 'No runs yet' }
    ];

    return cards.map(card => `
        <article class="history-summary-card">
            <span class="history-summary-label">${card.label}</span>
            <strong class="history-summary-value">${card.value}</strong>
            <p class="history-summary-hint">${card.hint}</p>
        </article>
    `).join('');
}

function renderTrendChart(sessions) {
    const canvas = document.getElementById('historyTrendChart');
    if (!canvas || typeof Chart === 'undefined') {
        return;
    }

    const labels = sessions.map(session => formatShortDate(session.timestamp));
    const scores = sessions.map(session => toChartValue(session?.summary?.average_score));

    if (historyTrendChart) {
        historyTrendChart.destroy();
    }

    historyTrendChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Average Score',
                data: scores,
                borderColor: '#00f0ff',
                backgroundColor: 'rgba(0, 240, 255, 0.16)',
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#00f0ff',
                pointRadius: 4,
                tension: 0.35,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: '#cfd8e3' },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' }
                },
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#cfd8e3',
                        callback: value => formatPercent(value)
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' }
                }
            }
        }
    });
}

function renderRadarChart(summary) {
    const canvas = document.getElementById('historyRadarChart');
    if (!canvas || typeof Chart === 'undefined') {
        return;
    }

    const averageSummary = summary.average_summary || {};

    if (historyRadarChart) {
        historyRadarChart.destroy();
    }

    historyRadarChart = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: ['Relevance', 'Length', 'Coherence', 'Toxicity', 'Bias', 'Hallucination'],
            datasets: [{
                label: 'Average Metric Value',
                data: [
                    toChartValue(averageSummary.relevance),
                    toChartValue(averageSummary.length_appropriateness),
                    toChartValue(averageSummary.coherence),
                    toChartValue(averageSummary.toxicity),
                    toChartValue(averageSummary.bias),
                    toChartValue(averageSummary.hallucination)
                ],
                borderColor: '#42a5f5',
                backgroundColor: 'rgba(66, 165, 245, 0.22)',
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#42a5f5'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    ticks: {
                        backdropColor: 'transparent',
                        color: '#cfd8e3'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' },
                    angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                    pointLabels: { color: '#f5f7fb', font: { size: 12 } }
                }
            }
        }
    });
}

function renderSessionsTable(sessions, tableBody) {
    if (!tableBody) {
        return;
    }

    if (!sessions.length) {
        tableBody.innerHTML = '<tr><td colspan="12" class="history-empty-state">No evaluation history yet. Run the Chrome extension to populate this dashboard.</td></tr>';
        return;
    }

    const rows = sessions.slice().reverse().map(session => `
        <tr>
            <td>${escapeHtml(formatTime(session.timestamp))}</td>
            <td title="${escapeHtml(session.source_url || 'unknown')}">${escapeHtml(truncateText(session.source_url || 'unknown', 42))}</td>
            <td>${escapeHtml(String(session.row_count ?? 0))}</td>
            <td><span class="history-score-badge">${formatPercent(session?.summary?.average_score)}</span></td>
            <td>${formatMetricCell(session?.summary?.relevance)}</td>
            <td>${formatMetricCell(session?.summary?.length_appropriateness)}</td>
            <td>${formatMetricCell(session?.summary?.coherence)}</td>
            <td>${formatMetricCell(session?.summary?.toxicity)}</td>
            <td>${formatMetricCell(session?.summary?.bias)}</td>
            <td>${formatMetricCell(session?.summary?.hallucination)}</td>
            <td>
                <a class="btn btn-icon btn-secondary btn-sm" href="/analytics?session_id=${encodeURIComponent(String(session.session_id || ''))}" title="View Analytics">
                    <i class="fa-solid fa-chart-simple"></i>
                </a>
                <button class="btn btn-icon btn-secondary btn-sm delete-session-btn" data-session-id="${escapeHtml(session.session_id || '')}" type="button" title="Delete">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </td>
        </tr>
    `).join('');

    tableBody.innerHTML = rows;
}

async function downloadHistorySnapshot() {
    try {
        const anchor = document.createElement('a');
        anchor.href = '/api/evaluation-history?format=csv';
        anchor.download = 'evaluation-history.csv';
        anchor.click();
    } catch (error) {
        console.error('Failed to download history snapshot:', error);
    }
}

async function clearAllHistory() {
    const approved = window.confirm('Clear all stored evaluation history? This cannot be undone.');
    if (!approved) {
        return;
    }

    try {
        await mutateHistoryWithFallback('/api/evaluation-history', 'DELETE', '/api/evaluation-history/clear');
        await loadEvaluationHistory(true);
    } catch (error) {
        console.error('Failed to clear history:', error);
        window.alert(`Unable to clear history right now. ${error.message || ''}`.trim());
    }
}

async function deleteHistorySession(sessionId) {
    const approved = window.confirm('Delete this evaluation session?');
    if (!approved) {
        return;
    }

    try {
        await mutateHistoryWithFallback(
            `/api/evaluation-history/${encodeURIComponent(sessionId)}`,
            'DELETE',
            `/api/evaluation-history/delete/${encodeURIComponent(sessionId)}`
        );
        await loadEvaluationHistory(true);
    } catch (error) {
        console.error('Failed to delete session:', error);
        window.alert(`Unable to delete this session right now. ${error.message || ''}`.trim());
    }
}

async function mutateHistoryWithFallback(primaryUrl, primaryMethod, fallbackPostUrl) {
    const headers = { 'Content-Type': 'application/json' };

    const primaryResponse = await fetch(primaryUrl, {
        method: primaryMethod,
        headers,
    });

    if (primaryResponse.ok) {
        const data = await safeJson(primaryResponse);
        if (data.success === false) {
            throw new Error(data.error || `Request failed (${primaryResponse.status})`);
        }
        return;
    }

    if (primaryResponse.status !== 405 && primaryResponse.status !== 404) {
        const primaryData = await safeJson(primaryResponse);
        throw new Error(primaryData.error || `Request failed (${primaryResponse.status})`);
    }

    const fallbackResponse = await fetch(fallbackPostUrl, {
        method: 'POST',
        headers,
    });
    const fallbackData = await safeJson(fallbackResponse);
    if (!fallbackResponse.ok || !fallbackData.success) {
        throw new Error(fallbackData.error || `Fallback failed (${fallbackResponse.status})`);
    }
}

async function safeJson(response) {
    try {
        return await response.json();
    } catch (_error) {
        return { success: false, error: 'Server returned a non-JSON response' };
    }
}

function formatMetricCell(value) {
    return `<span class="history-metric-chip">${formatPercent(value)}</span>`;
}

function formatPercent(value, fromRatio = false) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return '0.0%';
    }

    const ratio = fromRatio ? numeric : numeric;
    return `${(ratio * 100).toFixed(1)}%`;
}

function toChartValue(value) {
    const numeric = Number(value);
    return Number.isNaN(numeric) ? 0 : Math.max(0, Math.min(1, numeric));
}

function formatShortDate(timestamp) {
    if (!timestamp) {
        return 'Unknown';
    }

    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
        return String(timestamp).slice(0, 10);
    }

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
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

function truncateText(value, maxLength) {
    const text = String(value || '');
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function escapeCsvCell(value) {
    const text = String(value ?? '');
    if (/[",\n\r]/.test(text)) {
        return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
// Dashboard functionality is embedded in dashboard.html for tight integration
// This file serves as a reference for additional dashboard utilities

/**
 * Helper function to determine color based on score
 * @param {number} score - Score between 0 and 1
 * @returns {string} - HEX color code
 */
function getScoreColor(score) {
    if (score < 0.3) return '#00ff00';  // Green - Good
    if (score < 0.6) return '#ffff00';  // Yellow - Medium
    return '#ff0000';                   // Red - Bad
}

/**
 * Helper function to determine color based on risk
 * @param {number} risk - Risk score between 0 and 1
 * @returns {string} - HEX color code
 */
function getRiskColor(risk) {
    if (risk < 0.3) return '#00ff00';   // Green - Low risk
    if (risk < 0.6) return '#ffff00';   // Yellow - Medium risk
    return '#ff0000';                   // Red - High risk
}

/**
 * Format evaluation results into HTML
 * @param {object} data - Evaluation results object
 * @returns {string} - Formatted HTML
 */
function formatResults(data) {
    return `
        <div class="results-grid">
            <div class="result-item">
                <h4>Semantic Similarity</h4>
                <div class="score">${(data.semantic_similarity * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <h4>ROUGE-1 F1</h4>
                <div class="score">${(data.rouge1_f1 * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <h4>Coherence</h4>
                <div class="score">${(data.coherence * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <h4>Toxicity</h4>
                <div class="score" style="color: ${getScoreColor(data.toxicity_penalty)}">${(data.toxicity_penalty * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <h4>Bias</h4>
                <div class="score" style="color: ${getScoreColor(data.bias_penalty)}">${(data.bias_penalty * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item">
                <h4>Hallucination Risk</h4>
                <div class="score" style="color: ${getRiskColor(data.hallucination_risk)}">${(data.hallucination_risk * 100).toFixed(1)}%</div>
            </div>
            <div class="result-item highlight">
                <h4>Final Score</h4>
                <div class="score" style="color: #00ff00; font-size: 2rem;">${(data.final_score * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}

/**
 * Show loading spinner
 * @param {string} elementId - ID of element to show spinner in
 */
function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = 
        '<p><i class="fa-solid fa-spinner fa-spin"></i> Analyzing...</p>';
}

/**
 * Show error message
 * @param {string} elementId - ID of element to show error in
 * @param {string} message - Error message
 */
function showError(elementId, message) {
    document.getElementById(elementId).innerHTML = 
        `<div class="error-message">${message}</div>`;
}
