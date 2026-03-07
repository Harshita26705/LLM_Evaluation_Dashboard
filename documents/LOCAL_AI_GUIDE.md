# 🤖 Local AI Models - No System Installation Required!

## What This Does

Instead of installing Ollama system-wide, you can now download AI models **directly into your project folder**. No admin rights needed, no system configuration - just download and run!

## 🚀 Quick Start (2 Steps)

### Step 1: Install llama-cpp-python
```bash
pip install llama-cpp-python
```

### Step 2: Run Setup Script
```bash
python setup_local_ai.py
```

That's it! The script will:
1. Show you available models
2. Download your chosen model to `./models/` folder
3. Test that it works
4. You're ready to use AI features!

## 📦 Available Models

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| TinyLlama | 700MB | **Getting started** | ⚡⚡⚡ Fastest |
| DeepSeek-Coder 1.3B | 800MB | **Code analysis** | ⚡⚡⚡ Fastest |
| Phi-2 | 1.6GB | **Best quality** | ⚡⚡ Fast |

**Recommendation:** Start with **TinyLlama** (option 1) - it's smallest and works great!

## 💡 How It Works

### Before (Required System Installation)
```
❌ Install Ollama on system
❌ Download models system-wide
❌ Run ollama serve
❌ Configure ports
```

### After (Project-Based)
```
✅ pip install llama-cpp-python
✅ python setup_local_ai.py
✅ Choose a model (downloads to ./models/)
✅ Done! Flask auto-detects it
```

## 📂 What Gets Created

```
LLM_Dashboard/
├── models/                           # 👈 Created automatically
│   ├── tinyllama-1.1b.Q4_K_M.gguf   # Downloaded model
│   └── ...other models
├── local_model_manager.py            # 👈 Model manager
├── setup_local_ai.py                 # 👈 Easy setup script  
└── code_analyzer.py                 # Updated to use local models
```

## 🎯 Using Local Models

The Flask app **automatically detects** and uses local models:

1. **First**: Tries local models in `./models/`
2. **Second**: Falls back to Ollama (if running)
3. **Third**: Shows basic analysis (no AI)

You don't need to change anything - it just works!

## 🧪 Test Your Model

### Manual Test
```bash
python local_model_manager.py
```

### Test in Code
```python
from local_model_manager import setup_local_model

# Download and load model
manager = setup_local_model("tinyllama")

# Use it
if manager:
    response = manager.generate("Explain this code: def hello(): print('hi')")
    print(response)
```

## ⚡ Performance Tips

### For Faster Response
- Use TinyLlama (smallest, fastest)
- Close other programs
- On first run, model loads (takes 5-10 seconds)
- Subsequent calls are fast!

### For Better Quality
- Use Phi-2 model (larger, more accurate)
- Or use DeepSeek-Coder for code-specific tasks

### GPU Acceleration (Optional)
If you have NVIDIA GPU:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

Then in `local_model_manager.py`, change:
```python
n_gpu_layers=0  # Change to -1 for GPU
```

## 🔍 Verify Everything Works

1. **Check model downloaded:**
   ```bash
   ls models/
   # Should show: tinyllama-1.1b.Q4_K_M.gguf (or other model)
   ```

2. **Test the model:**
   ```bash
   python local_model_manager.py
   ```

3. **Start Flask:**
   ```bash
   python flask_app.py
   ```
   
   You should see:
   ```
   ✅ Using local model: tinyllama
   ```

4. **Test in browser:**
   - Go to Code Analysis tab
   - Select "Bug Detection"
   - Paste code and click "Analyze"
   - Should show AI-powered results!

## 📊 Comparison

### Ollama (System-Wide)
| Pros | Cons |
|------|------|
| ✅ Easier updates | ❌ Requires system install |
| ✅ Managed service | ❌ Needs background process |
| ✅ More models | ❌ System-wide configuration |

### Local Models (Project-Based)
| Pros | Cons |
|------|------|
| ✅ No system install | ❌ Manual model management |
| ✅ Portable (in project) | ❌ Larger project size |
| ✅ No background service | ❌ Fewer model options |
| ✅ Works offline | - |

## 🆘 Troubleshooting

### "llama-cpp-python not installed"
```bash
pip install llama-cpp-python
```

### "Download failed"
- Check internet connection
- Try again (script auto-cleans partial downloads)
- Try smaller model first (TinyLlama)

### "Model loading failed"
- Check you have enough RAM (min 4GB free)
- Try smaller model
- Close other programs

### "No response from model"
- Check model loaded: Look for "✅ Using local model" message
- Increase timeout in `local_model_manager.py`
- Try simpler prompt first

### Still shows "Ollama not running"
1. Hard refresh browser (Ctrl+Shift+R)
2. Check Flask server restarted after model download
3. Look for "✅ Using local model" in Flask startup logs

## 🔄 Switching Between Models

### Download Additional Models
```bash
python setup_local_ai.py
# Choose a different model
```

### Manually Switch Models
In `code_analyzer.py`, modify:
```python
self.local_model.load_model("tinyllama")  # Change model name here
```

Models available:
- `tinyllama`
- `deepseek-coder-1.3b`
- `phi-2`

## 📝 Model Details

### TinyLlama 1.1B
- **Size**: 700MB
- **Context**: 2048 tokens
- **Speed**: Very fast (2-3 sec response)
- **Quality**: Good for most tasks
- **Best for**: First-time users, quick testing

### DeepSeek-Coder 1.3B
- **Size**: 800MB
- **Context**: 2048 tokens
- **Speed**: Very fast (2-3 sec response)
- **Quality**: Excellent for code
- **Best for**: Code analysis, bug detection

### Phi-2 2.7B
- **Size**: 1.6GB
- **Context**: 2048 tokens
- **Speed**: Fast (4-6 sec response)
- **Quality**: Best overall
- **Best for**: Complex analysis, documentation

## 🎓 Advanced Usage

### Custom Model Download
Add your own model to `local_model_manager.py`:
```python
self.available_models["my-model"] = {
    "url": "https://huggingface.co/.../model.gguf",
    "size": "1.0GB",
    "file": "my-model.gguf",
    "description": "My custom model"
}
```

### Direct API Usage
```python
from local_model_manager import LocalModelManager

manager = LocalModelManager()
manager.load_model("tinyllama")

response = manager.generate(
    prompt="Your question here",
    system_prompt="You are a helpful assistant",
    max_tokens=1024
)
```

## 🌟 Benefits

1. **Portable**: Models in project folder, move anywhere
2. **No Admin**: No system-wide installation
3. **Offline**: Works without internet (after download)
4. **Fast**: No network latency
5. **Free**: No API costs ever
6. **Private**: Code never leaves your machine

## 📚 Related Files

- **local_model_manager.py** - Core model management
- **setup_local_ai.py** - Interactive setup script
- **code_analyzer.py** - Uses local models automatically
- **models/** - Downloaded models stored here

## 🚀 Next Steps

1. ✅ Run `python setup_local_ai.py`
2. ✅ Choose and download a model
3. ✅ Restart Flask: `python flask_app.py`
4. ✅ Try AI features in Code Analysis tab!

---

**Questions?** Check the setup script output for detailed error messages.

**Want Ollama instead?** That still works too! Local models are just an additional option.
