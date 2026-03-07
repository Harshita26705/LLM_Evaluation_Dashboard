#!/usr/bin/env python
from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')

try:
    with app.test_request_context():
        result = render_template('learn-more.html')
        print("✅ Template rendered successfully!")
        print(f"Length: {len(result)} characters")
except Exception as e:
    print(f"❌ Error rendering template: {e}")
    import traceback
    traceback.print_exc()
