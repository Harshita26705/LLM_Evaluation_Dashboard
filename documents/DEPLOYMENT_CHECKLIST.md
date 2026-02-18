# ✅ Deployment Checklist

## Pre-Deployment (Local Testing)

- [ ] All files created successfully
- [ ] `requirements.txt` has no `python-3.10`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] App runs locally: `python app.py`
- [ ] Can access at `http://localhost:7860`
- [ ] All 6 tabs load without errors
- [ ] Sample CSV works with dataset evaluation

## Tab Testing

- [ ] **Single Response Evaluation**
  - [ ] Can enter reference text
  - [ ] Can enter response text
  - [ ] "Analyze Response" button works
  - [ ] All 9 metrics display
  - [ ] Final score appears

- [ ] **Hallucination Detection**
  - [ ] Can enter reference text
  - [ ] Can enter response text
  - [ ] "Detect Hallucinations" works
  - [ ] Risk score displays
  - [ ] Entities list appears

- [ ] **Bias Detection**
  - [ ] Can enter response text
  - [ ] "Detect Bias" works
  - [ ] Bias score displays
  - [ ] Named entities show
  - [ ] Analysis text appears

- [ ] **Code Quality Check**
  - [ ] Can paste Python code
  - [ ] "Analyze Code Quality" works
  - [ ] Syntax status shows
  - [ ] Metrics display
  - [ ] Suggestions appear

- [ ] **Dataset Evaluation**
  - [ ] Can upload CSV file
  - [ ] Can upload JSON file
  - [ ] "Evaluate Dataset" processes
  - [ ] Results JSON displays
  - [ ] Download button works

- [ ] **Results Visualization**
  - [ ] Can enter reference text
  - [ ] Can enter model responses
  - [ ] "Generate Comparison Chart" works
  - [ ] Chart image displays
  - [ ] Results table shows all metrics

## Hugging Face Spaces Deployment

### Create Space
- [ ] Account on huggingface.co
- [ ] Logged in to HF account
- [ ] Navigated to https://huggingface.co/spaces
- [ ] Clicked "Create new Space"
- [ ] Filled in space name (e.g., `llm-eval-dashboard`)
- [ ] Selected SDK: **Gradio**
- [ ] Selected Visibility: Public (or Private)
- [ ] Created space successfully

### Upload Files
- [ ] Uploaded `app.py`
- [ ] Uploaded `requirements.txt`
- [ ] Checked files appear in Space repo
- [ ] Space build started automatically

### Building & Deployment
- [ ] Space shows "Building" status
- [ ] Build completed (5-10 minutes)
- [ ] No build errors in logs
- [ ] Space shows "Running" status
- [ ] Can access Space URL
- [ ] All tabs load in Space
- [ ] Can test evaluation functions

## Post-Deployment

### Verification
- [ ] Space is publicly accessible
- [ ] URL works: `https://huggingface.co/spaces/USERNAME/llm-eval-dashboard`
- [ ] All features functional
- [ ] No console errors
- [ ] Large inputs handled gracefully

### Documentation
- [ ] README.md is clear and complete
- [ ] QUICKSTART.md is easy to follow
- [ ] HF_DEPLOYMENT.md covers deployment
- [ ] Sample data included for testing

### Performance
- [ ] First load takes 3-5 minutes (normal)
- [ ] Single evaluation: < 30 seconds
- [ ] Dataset processing handles ~10-20 samples
- [ ] No timeout errors

## Optional Enhancements

- [ ] Create custom Space config (app_config.yaml)
- [ ] Set up continuous deployment from GitHub
- [ ] Add custom CSS for branding
- [ ] Set Space pinned (featured)
- [ ] Add Space thumbnail/cover image
- [ ] Write Space description
- [ ] Add topics/tags

## Sharing

- [ ] Share Space URL with others
- [ ] Include in portfolio/resume
- [ ] Add to GitHub profile
- [ ] Share direct link to specific features
- [ ] Create demo video (optional)

## Maintenance

- [ ] Monitor Space usage/logs weekly
- [ ] Update models if new versions available
- [ ] Collect feedback from users
- [ ] Document any custom changes
- [ ] Backup important code
- [ ] Keep GitHub repo in sync (if using)

---

## 🚀 Quick Deployment Command

If using GitHub integration:
```bash
# In your repo after uploading files
git add .
git commit -m "Deploy LLM Evaluation Dashboard to HF Spaces"
git push origin main
# Space auto-syncs and deploys!
```

---

## 📞 Common Issues & Solutions

### ❌ Build fails with dependency error
✅ Solution: 
```bash
pip install -r requirements.txt --upgrade
# Check no conflicting versions
```

### ❌ App takes > 10 minutes to start
✅ Solution:
- Normal for first load (downloading models)
- Subsequent starts are faster
- Free tier may be slower

### ❌ Out of memory error
✅ Solution:
- Use smaller datasets (< 100 rows)
- Process in batches
- Restart Space

### ❌ Can't upload CSV
✅ Solution:
- Check column names: `question`, `model_name`, `response`
- Use UTF-8 encoding
- Keep file < 50MB

### ❌ Metrics not calculating
✅ Solution:
- Ensure both reference AND response text provided
- Check text is not empty
- See browser console for errors

## ✨ Success Indicators

Your deployment is successful if:
✅ Space shows "Running" status
✅ All 6 tabs load
✅ Can evaluate responses
✅ Can process datasets
✅ Can generate comparison charts
✅ No console errors
✅ Accessible to others via URL

---

## 📊 Final Verification

```
Files Present:
✓ app.py (main application)
✓ requirements.txt (dependencies)
✓ sample_data.csv (test data)
✓ README.md (documentation)
✓ QUICKSTART.md (quick guide)
✓ HF_DEPLOYMENT.md (deployment guide)
✓ .gitignore (git configuration)

Features Included:
✓ Single response evaluation
✓ Hallucination detection
✓ Bias detection
✓ Code quality analysis
✓ Dataset batch processing
✓ Results visualization
✓ About/help tab

Fixes Applied:
✓ Removed invalid python-3.10 from requirements
✓ Fixed indentation errors in app.py
✓ Added comprehensive error handling
✓ Enhanced documentation
```

---

## 🎉 You're All Set!

Once this checklist is complete, your LLM Evaluation Dashboard is:
- ✅ Fully functional
- ✅ Deployed to HF Spaces
- ✅ Accessible to users worldwide
- ✅ Ready for production use

**Celebrate! 🎊 Your dashboard is live!**

Share the URL: `https://huggingface.co/spaces/YOUR_USERNAME/llm-eval-dashboard`
