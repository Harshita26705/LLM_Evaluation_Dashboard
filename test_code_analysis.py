"""
Test script for enhanced code analysis features
Run this after setting up Ollama to verify everything works
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, data, description):
    """Test an API endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"✅ Success!")
            print(json.dumps(result, indent=2)[:500])  # Print first 500 chars
        else:
            print(f"❌ Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL}. Is Flask running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    print("🚀 Testing Enhanced Code Analysis Features")
    print(f"Server: {BASE_URL}")
    
    # Test 1: Basic code analysis
    test_endpoint(
        "/api/analyze-code",
        {
            "code": "def hello():\n    print('Hello World')",
            "language": "python"
        },
        "Basic Code Analysis (No AI)"
    )
    
    # Test 2: Enhanced AI analysis
    test_endpoint(
        "/api/analyze-code-enhanced",
        {
            "code": "def divide(a, b):\n    return a / b",
            "language": "python",
            "analysis_type": "bugs"
        },
        "AI Bug Detection"
    )
    
    # Test 3: Security analysis
    test_endpoint(
        "/api/analyze-code-enhanced",
        {
            "code": "user_input = input('Enter code: ')\neval(user_input)",
            "language": "python",
            "analysis_type": "security"
        },
        "AI Security Analysis"
    )
    
    # Test 4: Code improvement
    test_endpoint(
        "/api/analyze-code-enhanced",
        {
            "code": "def calc(x, y):\n    return x + y",
            "language": "python",
            "analysis_type": "improve"
        },
        "AI Code Improvement"
    )
    
    # Test 5: LLM-generated code evaluation
    test_endpoint(
        "/api/analyze-llm-code",
        {
            "prompt": "Create a function that calculates factorial",
            "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "language": "python"
        },
        "LLM Code Quality Evaluation"
    )
    
    # Test 6: Repository analysis (skip if no internet)
    print("\n" + "="*60)
    print("⚠️  Repository analysis requires internet and takes time")
    print("Skipping by default. Uncomment to test.")
    print("="*60)
    
    # Uncomment to test:
    # test_endpoint(
    #     "/api/analyze-repository",
    #     {
    #         "repo_url": "https://github.com/pallets/flask"
    #     },
    #     "GitHub Repository Analysis"
    # )
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. If all tests passed: You're ready to use the dashboard!")
    print("2. If Ollama errors: Make sure Ollama is running (ollama serve)")
    print("3. If connection errors: Start Flask (python flask_app.py)")
    print("4. Check documents/CODE_ANALYSIS_SETUP.md for detailed setup")


if __name__ == "__main__":
    main()
