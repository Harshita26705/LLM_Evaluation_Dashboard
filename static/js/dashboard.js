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
