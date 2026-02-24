# Enhanced Code Analysis Setup Guide

## Overview
Your LLM Dashboard now includes AI-powered code analysis using **Ollama** (free, runs locally).

## Features Added
✅ **Bug Detection with AI** - Find critical bugs automatically  
✅ **Security Analysis** - Detect vulnerabilities (eval, SQL injection, etc.)  
✅ **Code Documentation Generator** - Auto-generate comprehensive docs  
✅ **LLM Code Quality Assessment** - Evaluate AI-generated code  
✅ **GitHub Repository Analysis** - Analyze entire codebases  
✅ **Git Diff Analysis** - Generate changelogs from commits  
✅ **Code Improvement Suggestions** - Get AI-improved versions  

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements_flask.txt
```

### 2. Install Ollama (Local LLM)

**Windows:**
1. Download from: https://ollama.com/download/windows
2. Install the .exe file
3. Ollama will run automatically

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Download AI Model
```bash
# Download Llama 3.2 (recommended, ~2GB)
ollama pull llama3.2

# OR use smaller model (faster but less accurate)
ollama pull llama3.2:1b

# OR use larger model (more accurate but slower)
ollama pull codellama:13b
```

### 4. Start Ollama Server
```bash
ollama serve
```

**Note:** On Windows, Ollama auto-starts. Just verify it's running at http://localhost:11434

### 5. Start Your Dashboard
```bash
python flask_app.py
```

## Usage Examples

### 1. Standard Code Analysis (No AI)
```python
POST /api/analyze-code
{
    "code": "def hello(): print('hi')",
    "language": "python"
}
```

### 2. Enhanced AI Analysis
```python
POST /api/analyze-code-enhanced
{
    "code": "your code here",
    "language": "python",
    "analysis_type": "full"  # or "bugs", "security", "improve", "docs"
}
```

### 3. Analyze LLM-Generated Code
```python
POST /api/analyze-llm-code
{
    "prompt": "Create a function to sort a list",
    "code": "def sort_list(lst): return sorted(lst)",
    "language": "python"
}
```

### 4. Analyze GitHub Repository
```python
POST /api/analyze-repository
{
    "repo_url": "https://github.com/username/repo"
}
```

### 5. Analyze Git Changes
```python
POST /api/analyze-git-diff
{
    "repo_path": "/path/to/local/repo"
}
```

## Available Analysis Types

| Type | Description | Endpoint |
|------|-------------|----------|
| `full` | Complete code analysis | `/api/analyze-code-enhanced` |
| `bugs` | Bug detection only | `/api/analyze-code-enhanced` |
| `security` | Security vulnerabilities | `/api/analyze-code-enhanced` |
| `improve` | Get improved code | `/api/analyze-code-enhanced` |
| `docs` | Generate documentation | `/api/analyze-code-enhanced` |
| `llm-eval` | Evaluate LLM code | `/api/analyze-llm-code` |
| `repo` | Analyze full repository | `/api/analyze-repository` |
| `diff` | Git changelog | `/api/analyze-git-diff` |

## Troubleshooting

### Ollama Not Running
**Error:** `Ollama not running. Start it with: ollama serve`

**Fix:**
```bash
# Check if Ollama is running
curl http://localhost:11434

# If not, start it
ollama serve
```

### Dependencies Missing
**Error:** `gitingest not installed`

**Fix:**
```bash
pip install gitingest gitpython
```

### Model Not Found
**Error:** `model 'llama3.2' not found`

**Fix:**
```bash
ollama pull llama3.2
```

### Slow Analysis
**Solutions:**
1. Use smaller model: `ollama pull llama3.2:1b`
2. Reduce code length (analyze in chunks)
3. Upgrade to GPU-enabled Ollama

## Customization

### Change AI Model
Edit `code_analyzer.py`:
```python
# Use different model
analyzer = OllamaCodeAnalyzer(model="codellama:7b")
```

### Adjust Timeout
```python
# In code_analyzer.py, _call_ollama method
response = requests.post(url, json=payload, timeout=300)  # 5 minutes
```

## Performance Tips

1. **First Run is Slow:** Model loads into memory (wait 10-30 seconds)
2. **Subsequent Runs:** Much faster (model cached in RAM)
3. **GPU Acceleration:** Ollama auto-uses GPU if CUDA available
4. **Concurrent Requests:** Ollama handles multiple requests efficiently

## API Rate Limits
✅ **FREE** - No rate limits  
✅ **LOCAL** - No internet required  
✅ **PRIVATE** - Your code never leaves your machine  

## Next Steps
1. Test basic analysis: `curl -X POST http://localhost:5000/api/analyze-code ...`
2. Try AI features in your dashboard UI
3. Customize prompts in `code_analyzer.py` for your use case
4. Explore other Ollama models: https://ollama.com/library

## Support
- Ollama Docs: https://github.com/ollama/ollama
- Issues: Open GitHub issue on your project
- Community: Ollama Discord

---
**Status Check:**
- [ ] Dependencies installed
- [ ] Ollama installed and running
- [ ] Model downloaded (llama3.2)
- [ ] Flask app running
- [ ] Test API endpoint successful
