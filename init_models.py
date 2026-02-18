#!/usr/bin/env python3
"""
Initialize models and data needed for LLM Dashboard
"""
import subprocess
import sys
import nltk

print("⏳ Downloading NLTK tokenizers...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print("✅ NLTK downloads complete")

print("\n⏳ Downloading spacy model...")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
print("✅ Spacy model loaded")

print("\n✅ All setup complete! You can now run: python app.py")
