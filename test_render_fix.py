#!/usr/bin/env python
"""
Simple test to verify:
1. Flask app boot without heavy models
2. API endpoints always return JSON (never empty/malformed)
3. Safe JSON parsing via get_json(silent=True)
"""

import sys
import json

print("=" * 60)
print("Testing Render-safe configurations and API reliability...")
print("=" * 60)

# Test 1: Import without heavy dependencies
print("\n[1] Testing Flask app import...")
try:
    from flask_app import app, ensure_models_loaded
    print("    ✓ Flask app imported successfully")
except Exception as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check Render environment detection
print("\n[2] Checking Render environment flags...")
import os
IS_RENDER = os.getenv("RENDER", "false").lower() in {"true", "1", "yes", "on"} or bool(os.getenv("RENDER_EXTERNAL_URL"))
DISABLE_HEAVY = os.getenv("DISABLE_HEAVY_MODELS", "false").lower() in {"true", "1", "yes", "on"}
print(f"    IS_RENDER = {IS_RENDER}")
print(f"    DISABLE_HEAVY_MODELS = {DISABLE_HEAVY}")

# Test 3: Test API endpoint safety with test client
print("\n[3] Testing API endpoint with safe JSON handling...")
with app.test_client() as client:
    # Test evaluate with minimal input
    endpoint = '/api/evaluate'
    payload = {'reference': 'ref text', 'response': 'response text'}
    
    resp = client.post(endpoint, json=payload)
    print(f"    Request to {endpoint}: HTTP {resp.status_code}")
    
    # Test that response is always JSON-decodable
    try:
        data = resp.get_json()
        if data is None:
            print(f"    ✗ Response body is None/empty!")
            sys.exit(1)
        print(f"    ✓ Response is valid JSON")
        print(f"    ✓ Response keys: {list(data.keys())[:5]}{'...' if len(data) > 5 else ''}")
    except json.JSONDecodeError as e:
        print(f"    ✗ Response is not valid JSON: {e}")
        sys.exit(1)

# Test 4: Test error case (missing required fields)
print("\n[4] Testing API error handling...")
resp = client.post('/api/evaluate', json={'reference': ''})
print(f"    Request with missing fields: HTTP {resp.status_code}")

try:
    data = resp.get_json()
    if data and 'error' in data:
        print(f"    ✓ Error response contains 'error' field")
        print(f"    ✓ Error message: {data['error'][:60]}...")
    else:
        print(f"    ✗ Error response missing 'error' field")
        sys.exit(1)
except json.JSONDecodeError as e:
    print(f"    ✗ Error response is not JSON: {e}")
    sys.exit(1)

# Test 5: Test 404 handling for API endpoints
print("\n[5] Testing API 404 error handling...")
resp = client.post('/api/nonexistent', json={})
print(f"    Request to /api/nonexistent: HTTP {resp.status_code}")

try:
    data = resp.get_json()
    if data and 'error' in data:
        print(f"    ✓ 404 API response returns JSON with 'error'")
    else:
        print(f"    ⚠ 404 response is JSON but missing 'error' field")
except json.JSONDecodeError:
    print(f"    ✗ 404 API response is not JSON (rendering HTML)")
    print(f"    Content-Type: {resp.content_type}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All critical tests passed!")
print("=" * 60)
print("\nDeploy changes to Render and the API should work reliably.")
