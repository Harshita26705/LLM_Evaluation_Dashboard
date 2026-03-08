#!/usr/bin/env python
"""
Quick JSON response validation test—
does NOT load models (uses skip flags)
"""

import sys
import os

# Force disable heavy models
os.environ['DISABLE_HEAVY_MODELS'] = '1'

print("=" * 60)
print("Testing Render-safe JSON API responses...")
print("=" * 60)

# Test 1: Import Flask app with heavy models disabled
print("\n[1] Importing Flask app (heavy models disabled)...")
try:
    from flask_app import app
    print("   ✓ Flask app imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test API endpoint JSON responses without triggering model load
print("\n[2] Testing API JSON responses...")
with app.test_client() as client:
    tests = [
        ('/api/evaluate', 'POST', {'reference': 'test', 'response': 'test'}, 'Evaluate endpoint'),
        ('/api/detect-bias', 'POST', {'text': 'hello world'}, 'Bias detection endpoint'),
        ('/api/check-toxicity', 'POST', {'text': 'hello'}, 'Toxicity check endpoint'),
        ('/api/nonexistent', 'POST', {}, '404 API error'),
    ]
    
    for endpoint, method, payload, desc in tests:
        if method == 'POST':
            resp = client.post(endpoint, json=payload)
        
        print(f"   Testing {desc}:")
        print(f"      HTTP {resp.status_code}")
        
        # Check response is JSON
        content_type = resp.content_type or ''
        if 'json' not in content_type:
            # Try to get_json anyway (Flask is lenient)
            try:
                data = resp.get_json()
                if data is not None:
                    print(f"      ✓ Valid JSON despite Content-Type={content_type}")
                else:
                    print(f"      ✗ get_json() returned None")
                    sys.exit(1)
            except Exception as e:
                print(f"      ✗ Not JSON-decodable: {e}")
                sys.exit(1)
        else:
            try:
                data = resp.get_json()
                print(f"      ✓ Valid JSON")
            except Exception as e:
                print(f"      ✗ JSON parse error: {e}")
                sys.exit(1)

print("\n" + "=" * 60)
print("✓ All JSON responses are valid!")
print("=" * 60)
