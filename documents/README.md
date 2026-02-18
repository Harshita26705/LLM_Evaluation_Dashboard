# 🤖 LLM Evaluation Dashboard

A comprehensive Gradio-based dashboard for evaluating Large Language Model (LLM) responses across multiple dimensions including semantic similarity, toxicity, bias, hallucinations, and code quality.

## 📋 Features

### Tabs/Modules:

1. **Single Response Evaluation** 📊
   - Evaluate a single LLM response against a reference
   - Metrics: Semantic Similarity, ROUGE-1 F1, Length Fit, Relevance, Coherence, Toxicity, Bias, Hallucination Risk
   - Final composite score

2. **Hallucination Detection** 🔍
   - Detect unsupported claims and hallucinations
   - Named entity extraction and verification
   - Risk scoring (0-1 scale)

3. **Bias Detection** ⚖️
   - Gender and demographic bias detection
   - Named entity analysis
   - Bias penalty scoring

4. **Code Quality Check** 💻
   - Python code syntax validation
   - Code complexity metrics
   - Security issue detection (eval, exec)
   - Code smell detection
   - Suggestions for improvement

5. **Dataset Evaluation** 📂
   - Batch evaluate multiple responses
   - Support for CSV and JSON formats
   - Requires: `question`, `model_name`, `response` columns (optional: `reference`)
   - Export results as CSV

6. **Results Visualization** 📈
   - Compare up to 3 models side-by-side
   - Bar chart visualization
   - Detailed metrics table

## 🚀 Installation

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd LLM_Dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

The app will be available at `http://localhost:7860`

## 🌐 Deployment on Hugging Face Spaces

### Method 1: Direct GitHub Integration

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select "Gradio" as the space type
3. Connect your GitHub repository
4. Spaces will automatically detect and run `app.py`

### Method 2: Manual Upload

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space
3. Upload these files to the Space:
   - `app.py`
   - `requirements.txt`

The Space will automatically install dependencies and launch the app.

## 📊 Evaluation Metrics

### Semantic Metrics
- **Semantic Similarity** (0-1): Cosine similarity of embeddings
- **ROUGE-1 F1** (0-1): Unigram overlap measure
- **Length Fit** (0-1): How well response length matches expected
- **Relevance** (0-1): Keyword overlap with query
- **Coherence** (0-1): Logical consistency and clarity

### Safety Metrics
- **Toxicity Penalty** (0-1): Harmful content detection
- **Bias Penalty** (0-1): Gender and demographic bias
- **Hallucination Risk** (0-1): Unsupported claims detection

### Final Score
Composite score combining all metrics with configurable weights.

## 📦 Input Formats

### For Dataset Evaluation:

**CSV Format:**
```
question,model_name,response,reference
"What is AI?","GPT-4","AI is...",""
"What is AI?","Claude","AI stands for..."
```

**JSON Format:**
```json
[
  {"question": "What is AI?", "model_name": "GPT-4", "response": "AI is...", "reference": ""},
  {"question": "What is AI?", "model_name": "Claude", "response": "AI stands for..."}
]
```

## ⚙️ Configuration

Edit the hyperparameters in `app.py`:

```python
LAMBDA_LEN = 0.5              # Length penalty control
EXPECTED_LEN_RATIO = 2.0      # Expected response/query length ratio
GAMMA_TOX = 5.0               # Toxicity penalty weight
GAMMA_BIAS = 4.0              # Bias penalty weight
ALPHA_COH = 0.6               # Coherence alpha parameter
HALL_MULT = 0.3               # Hallucination multiplier
SAFETY_THRESHOLD = 0.30       # Toxicity threshold
SAFETY_CAP = 0.50             # Maximum toxic response score
```

Adjust the `WEIGHTS` dictionary to change metric importance:

```python
WEIGHTS = {
    "relevance": 0.25,
    "length_fit": 0.15,
    "coherence": 0.20,
    "rouge1_f1": 0.10,
    "toxicity_inv": 0.15,
    "bias_inv": 0.10,
    "hall_inv": 0.05
}
```

## 🔧 Dependencies

- `gradio>=5.45.0` - Web UI
- `sentence-transformers>=2.2.2` - Semantic similarity
- `detoxify>=0.5.0` - Toxicity detection
- `spacy>=3.8.0` - NLP and entity extraction
- `pandas>=2.0.0` - Data handling
- `matplotlib>=3.7.0` - Visualization
- `torch>=2.1.0` - Deep learning
- `transformers>=4.36.0` - LLM utilities

## 📝 Examples

### Single Response Evaluation
```
Reference: "The Earth orbits the Sun."
Response: "The Earth revolves around the Sun in approximately 365 days."
```

### Code Quality Check
```python
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### Dataset Batch Processing
Upload a CSV with multiple Q&A pairs and model responses to get aggregated metrics across your dataset.

## 🐛 Troubleshooting

### Error: "python-3.10 not found"
✅ **Fixed** - Removed invalid `python-3.10` from requirements.txt

### Error: "Module not found"
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- If using Hugging Face Spaces, wait for the build to complete

### Slow Performance
- The first run downloads pre-trained models (~500MB)
- Subsequent runs will be faster
- Consider using GPU for faster inference on Spaces

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review evaluation metrics documentation
3. Create an issue with detailed error information

---

**Built with ❤️ using Gradio and Hugging Face**
