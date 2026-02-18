#!/usr/bin/env python3
"""
Setup script to download required models and data for LLM Evaluation Dashboard
"""
import sys
import os

print("=" * 60)
print("LLM Evaluation Dashboard - Setup Script")
print("=" * 60)

# Step 1: Download NLTK data
print("\n[1/3] Downloading NLTK data (punkt tokenizer)...")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK data downloaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not download NLTK data: {e}")
    print("   The app may still work, but tokenization might be affected")

# Step 2: Load spacy model
print("\n[2/3] Loading spacy model (en_core_web_sm)...")
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Spacy model loaded successfully")
    except OSError:
        print("   Model not found, downloading...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ Spacy model downloaded and loaded")
except Exception as e:
    print(f"❌ Error loading spacy model: {e}")
    print("   Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Step 3: Test imports
print("\n[3/3] Testing all required imports...")
required_modules = [
    'matplotlib',
    'pandas',
    'gradio',
    'sentence_transformers',
    'detoxify',
    'torch',
    'transformers',
    'nltk'
]

all_good = True
for module in required_modules:
    try:
        __import__(module)
        print(f"  ✅ {module}")
    except ImportError as e:
        print(f"  ❌ {module}: {e}")
        all_good = False

if not all_good:
    print("\n❌ Some dependencies are missing!")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All setup complete! Ready to run the app.")
print("=" * 60)
print("\nNext step: python app.py")
print("Then open: http://localhost:7860")
