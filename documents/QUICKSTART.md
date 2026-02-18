# 🚀 Quick Start Guide

## Start in 3 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python app.py
```

### Step 3: Open in Browser
Visit: **http://localhost:7860**

---

## Using Each Tab

### 1️⃣ Single Response Evaluation
**What it does**: Score a single response

**Steps**:
1. Enter reference/query text
2. Enter model response
3. Click "Analyze Response"
4. Get 9 metrics + final score

**Example**:
```
Reference: "What is AI?"
Response: "AI is artificial intelligence"
→ Get semantic similarity, toxicity score, hallucination risk, etc.
```

### 2️⃣ Hallucination Detection
**What it does**: Find incorrect claims in responses

**Steps**:
1. Enter source/reference text
2. Enter model response
3. Click "Detect Hallucinations"
4. See hallucination risk & unsupported entities

**Example**:
```
Reference: "Paris is the capital of France"
Response: "London is the capital of France"
→ Detects entities not matching reference
```

### 3️⃣ Bias Detection
**What it does**: Find gender/demographic bias

**Steps**:
1. Enter model response
2. Click "Detect Bias"
3. See bias score & named entities found

**Example**:
```
Response: "The man is a doctor, the woman is a nurse"
→ Highlights gender bias in role assignments
```

### 4️⃣ Code Quality Check
**What it does**: Evaluate Python code

**Steps**:
1. Paste Python code
2. Click "Analyze Code Quality"
3. Get syntax check, metrics, and suggestions

**Detects**:
- ✅ Syntax errors
- 🚨 Security issues (eval, exec)
- ⚠️ Code smells
- 💡 Improvement suggestions

**Example**:
```python
def hello():
    return "world"
→ Valid syntax, clean code!
```

### 5️⃣ Dataset Evaluation
**What it does**: Evaluate multiple responses at once

**Supported Formats**: CSV, JSON

**Required Columns**:
- `question` - The query
- `model_name` - Name of LLM
- `response` - Model's answer
- `reference` (optional) - Ground truth

**CSV Template**:
```csv
question,model_name,response,reference
"What is AI?","GPT-4","AI is...",""
"What is AI?","Claude","AI stands for..."
```

**Steps**:
1. Upload CSV/JSON
2. Click "Evaluate Dataset"
3. Get aggregated metrics for each model
4. Download results

### 6️⃣ Results Visualization
**What it does**: Compare up to 3 models

**Steps**:
1. Enter reference text
2. Enter responses for Model 1, 2, 3
3. Click "Generate Comparison Chart"
4. See side-by-side comparison

**Output**:
- 📊 Bar chart comparison
- 📋 Detailed metrics table

---

## Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| Semantic Similarity | 0-1 | How similar is response to reference? |
| ROUGE-1 F1 | 0-1 | Word overlap between texts |
| Length Fit | 0-1 | Is response length appropriate? |
| Relevance | 0-1 | Does response answer the question? |
| Coherence | 0-1 | Is response logical and clear? |
| Toxicity Penalty | 0-1 | Does response contain harmful content? |
| Bias Penalty | 0-1 | Does response show demographic bias? |
| Hallucination Risk | 0-1 | Does response make up false claims? |
| Final Score | 0-1 | Overall quality score |

**Scores Interpretation**:
- 0.8-1.0 ✅ Excellent
- 0.6-0.8 ✓ Good
- 0.4-0.6 ⚠️ Average
- 0.0-0.4 ❌ Poor

---

## Sample Data

Use `sample_data.csv` to test dataset evaluation:

```bash
# In the app:
1. Go to "Dataset Evaluation" tab
2. Upload: sample_data.csv
3. Click "Evaluate Dataset"
4. See results for GPT-4 vs Claude
```

---

## Common Tasks

### Compare Two Models
1. Go to "Results Visualization"
2. Enter same reference
3. Enter response from Model A & B
4. Get comparison chart

### Batch Evaluate Responses
1. Go to "Dataset Evaluation"
2. Prepare CSV with your data
3. Use columns: question, model_name, response, reference
4. Upload and get aggregated metrics

### Check Code Quality
1. Go to "Code Quality Check"
2. Paste Python code
3. Get syntax validation + suggestions

### Find Hallucinations
1. Go to "Hallucination Detection"
2. Enter source text
3. Enter model response
4. See what's made up

---

## Tips & Tricks

### ⚡ Speed Tips
- First load: Takes 3-5 minutes (model download)
- Subsequent loads: Much faster ✨
- Use sample_data.csv to test quickly

### 💡 Best Practices
- Always provide reference text for best results
- Longer responses = better evaluation accuracy
- Use consistent naming for models (e.g., "GPT-4", "Claude")
- Test with sample_data.csv first

### 🔧 Customization
Edit `app.py` to adjust:
- Metric weights
- Toxicity threshold
- Bias detection sensitivity
- Length expectations

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt --upgrade
```

### App Starts Slowly
Normal! First run downloads ~500MB of models. Be patient! ⏳

### Out of Memory
- Reduce dataset size
- Close other applications
- Use fewer simultaneous uploads

### Port Already in Use
```bash
python app.py --server_port 7861
```

---

## File Structure

```
LLM_Dashboard/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── sample_data.csv          # Sample CSV for testing
├── README.md                # Full documentation
├── HF_DEPLOYMENT.md         # Hugging Face guide
└── QUICKSTART.md            # This file
```

---

## Next Steps

1. ✅ Run `python app.py`
2. ✅ Try "Single Response Evaluation"
3. ✅ Upload sample_data.csv
4. ✅ Compare models
5. 🚀 Deploy to Hugging Face Spaces (see HF_DEPLOYMENT.md)

---

## 🎯 You're Ready!

```bash
python app.py
# → Open http://localhost:7860
# → Start evaluating LLM responses!
```

For more details, see **README.md** or **HF_DEPLOYMENT.md**
