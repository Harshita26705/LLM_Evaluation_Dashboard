# Deploying LLM Evaluation Dashboard on Hugging Face Spaces

## Quick Start (2 minutes)

### Option 1: Clone & Deploy

1. **Create a Space on Hugging Face**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose:
     - **Space name**: `llm-eval-dashboard`
     - **License**: Choose one (MIT recommended)
     - **Space SDK**: Gradio
     - **Visibility**: Public or Private

2. **Upload Files**
   - Upload these files to your Space repository:
     - `app.py`
     - `requirements.txt`

3. **Space will auto-build**
   - HF Spaces automatically installs dependencies
   - Your app starts automatically
   - View at: `https://huggingface.co/spaces/YOUR_USERNAME/llm-eval-dashboard`

### Option 2: GitHub Integration

1. **Push code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/llm-eval-dashboard.git
git push -u origin main
```

2. **Create Space on HF**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select SDK: Gradio
   - Link your GitHub repository in settings

3. **Automatic Syncing**
   - Push to GitHub → Space auto-updates
   - No manual uploads needed

## ✅ Verification Checklist

Before deploying, ensure:

- [ ] `app.py` is in root directory
- [ ] `requirements.txt` has NO `python-3.10` line
- [ ] Space SDK is set to "Gradio"
- [ ] All dependencies are in `requirements.txt`
- [ ] No file size exceeds 1GB (model files)

## 🔧 Troubleshooting

### Build Fails with Dependency Error
**Solution**: Check that exact version matches work together
```bash
python -m pip install --dry-run -r requirements.txt
```

### App Takes Too Long to Start
**Normal behavior**: First load takes 3-5 minutes (downloads models)
- Subsequent loads are faster
- Pre-trained embeddings and NLP models need to be downloaded

### Out of Memory Errors
On free tier GPU (if applicable):
- Reduce batch size
- Process smaller datasets
- Use CPU (auto-fall back)

### Model Download Fails
**Solution**: Models are cached after first download
- Retry after 5 minutes
- Check internet connection stability

## 💡 Performance Tips

1. **For Free Tier**:
   - Processing will be CPU-based
   - Batch size limited to ~10 samples
   - Response time: 5-30 seconds per evaluation

2. **For Pro Tier (if available)**:
   - GPU acceleration available
   - Faster processing: 1-5 seconds
   - Can handle larger batches

3. **Optimization**:
   - Cache models locally (already configured)
   - Use sample_data.csv for testing
   - Limit dataset size to <1000 rows initially

## 📦 Space Configuration Files

### Optional: `app_config.yaml` (for advanced setup)
```yaml
title: LLM Evaluation Dashboard
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
```

## 🚀 Advanced Deployment

### Custom Startup with setup.sh
Create `setup.sh`:
```bash
#!/bin/bash
# Optional: custom setup commands
python -m nltk.downloader punkt averaged_perceptron_tagger
python -m spacy download en_core_web_sm
```

Add to `requirements.txt`:
```
nltk>=3.8.1
pandas>=2.0.0
# ... rest of requirements
```

## 📊 Expected File Sizes

- Models downloaded: ~500-700 MB
  - Sentence-transformers: ~300 MB
  - Spacy NLP: ~150 MB
  - Others: ~100 MB
- Your Space needs at least 2GB free disk space

## 🔄 Updating Your Space

### To update the app:

1. **If using GitHub integration**:
   ```bash
   git add .
   git commit -m "Update evaluation metrics"
   git push
   # Space auto-updates in ~1 minute
   ```

2. **If using direct upload**:
   - Edit files directly in Space UI
   - Or upload new files
   - Changes take effect after refresh

## 📈 Monitoring

- Check Space logs: Settings → Space logs
- Monitor hardware usage: Dashboard
- View requests: Activity section

## 🎯 Best Practices

1. **Dataset Size**
   - Start with <100 rows for testing
   - Production: <500 rows at once
   - For larger batches: Process in chunks

2. **Input Validation**
   - Always provide reference text
   - Ensure CSV has required columns
   - Use sample_data.csv as template

3. **Error Handling**
   - Check Spaces logs if issues occur
   - Restart Space from Settings if stuck
   - Clear cache if models seem outdated

## 🆘 Emergency Troubleshooting

### Space not loading
1. Check build logs (Settings → Logs)
2. Restart Space (Settings → Advanced → Restart)
3. Clear cache and rebuild

### Slow performance
1. Check hardware (free tier may be throttled)
2. Reduce batch size for dataset evaluation
3. Use smaller models if available

### Memory errors
1. Close all other Tabs using the Space
2. Clear browser cache
3. Try dataset with fewer rows

## 📞 Support

- HF Spaces Forum: https://discuss.huggingface.co/
- GitHub Issues: Create issue in your repo
- Email: contact@huggingface.co

## 🎉 You're Ready!

Once deployed to Spaces, your dashboard is:
- ✅ Publicly accessible
- ✅ Auto-scaling with traffic
- ✅ Continuously integrated (if GitHub linked)
- ✅ Free to use on free tier

Share your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/llm-eval-dashboard`
