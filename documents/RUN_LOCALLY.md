# 🚀 Running the LLM Evaluation Dashboard Locally

## Quick Start

### Option 1: Automatic Setup (Recommended)

**Windows:**
```batch
cd "c:\Users\HarshitaSuri\OneDrive - CG Infinity\Desktop\LLM_Dashboard"
run_app.bat
```

The app will start at: **http://localhost:7860**

### Option 2: Manual Setup

1. **Activate virtual environment**:
   ```bash
   .\.venv\Scripts\activate
   ```

2. **Download remaining models** (first time only):
   ```bash
   python -m spacy download en_core_web_sm
   python init_models.py
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```

4. **Open in browser**:
   - Visit: http://localhost:7860

---

## ✅ What's Installed

- ✅ All Python dependencies (in `.venv/`)
- ✅ Sentence-transformers embedding model
- ✅ Detoxify toxicity detection
- ✅ Gradio web interface
- ✅ SpacyNLP (downloads on first run)

## 🎯 First Time Setup

1. Models download automatically on first run
2. Sentence-transformers: ~300MB (one-time download)
3. Spacy model: ~50MB (one-time download)
4. Subsequent runs are much faster

## ⚠️ Notes

- **First run**: Takes 5-10 minutes (model downloads)
- **Subsequent runs**: 30-60 seconds to start
- **System requirements**: 4GB+ RAM, 2GB free disk space
- **Network**: Required for first-time model downloads

## 📊 When App Loads

You'll see:
```
⏳ Loading models...
   ✅ Loaded sentence embedder (from HuggingFace)
   ⚠️ Spacy model not found - Run: python -m spacy download en_core_web_sm
   ✅ Loaded tokenizer
   ⚠️ Could not load toxicity model (optional)
✅ Models loaded!

Running on http://localhost:7860
```

If spacy model is missing, run:
```
python -m spacy download en_core_web_sm
```

Then restart the app.

## 🧪 Testing the App

Create a test file `/Desktop/test_response.txt`:
```
Reference: What is artificial intelligence?
Response: AI is the simulation of human intelligence by computer systems.
```

Then in the app:
1. Go to "Single Response Evaluation" tab
2. Paste the reference
3. Paste the response
4. Click "Analyze Response"
5. See all metrics!

## 📝 Try Dataset Evaluation

Sample CSV (`sample_data.csv`) is already included:
1. Go to "Dataset Evaluation" tab
2. Upload `sample_data.csv`
3. Click "Evaluate Dataset"
4. Get aggregated metrics

## 🛑 Stopping the App

- **In terminal**: Press `Ctrl + C`
- **In browser**: Close the tab
- **Keyboard**: `Ctrl + C` in the terminal running the app

## 🔧 Troubleshooting

### App crashes on startup
```bash
# Clear Python cache and restart
cd "c:\Users\HarshitaSuri\OneDrive - CG Infinity\Desktop\LLM_Dashboard"
rmdir /s __pycache__
python app.py
```

### "Module not found" errors
```bash
# Make sure venv is activated
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Port 7860 is already in use
```bash
# Run on different port
python app.py --server_port 7861
```

### Out of memory errors
- Close other applications
- Reduce dataset size (< 20 rows)
- Reduce batch processing

### Spacy model not found
```bash
python -m spacy download en_core_web_sm
```

## 📈 Performance Tips

- **Optimal**: Multiple evaluations per session is faster
- **Fast**: Smaller datasets process quicker than large batches  
- **GPU**: Can use GPU if CUDA available (auto-detects)
- **Cache**: Model embedding cache improves batch processing speed

## 📚 Available Features

1. ✅ Single Response Evaluation - Full metrics for one response
2. ✅ Hallucination Detection - Find false claims
3. ✅ Bias Detection - Detect demographic bias
4. ✅ Code Quality Check - Validate Python code
5. ✅ Dataset Evaluation - Batch process CSV/JSON
6. ✅ Results Visualization - Compare multiple models
7. ✅ About - Full documentation

## 🌐 Next: Deploy to Hugging Face Spaces

Once you're happy with local testing:
1. See `HF_DEPLOYMENT.md` for deployment steps
2. Upload to Hugging Face Spaces
3. Share with the world!

## 📞 Support

- Check the `README.md` for complete documentation
- See `QUICKSTART.md` for feature guide
- Check logs if errors occur

---

**Enjoy your LLMEvaluation Dashboard! 🎉**
