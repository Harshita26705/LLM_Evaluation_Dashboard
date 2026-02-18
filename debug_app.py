#!/usr/bin/env python
import sys
print("Python version:", sys.version)
print("Starting imports...")

try:
    import gradio as gr
    print("✅ Gradio imported")
except Exception as e:
    print("❌ Gradio error:", e)
    sys.exit(1)

try:
    import numpy as np
    print("✅ NumPy imported")
except Exception as e:
    print("❌ NumPy error:", e)

print("\nAttempting to run app.py...")
exec(open('app.py').read())
