"""
Setup Local AI - One-click setup for local AI models in your project
No system-wide installation needed!
"""

from local_model_manager import LocalModelManager, setup_local_model


def main():
    print("\n" + "="*70)
    print("🚀 LOCAL AI MODEL SETUP")
    print("="*70)
    print("\nThis will download an AI model directly to your project folder.")
    print("No system-wide installation required!")
    print("\n" + "="*70 + "\n")
    
    manager = LocalModelManager()
    
    # Check if llama-cpp-python is installed
    if not manager.is_llama_cpp_available():
        print("❌ Required package not found: llama-cpp-python\n")
        print("📦 Installing llama-cpp-python...\n")
        
        import subprocess
        import sys
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "llama-cpp-python"
            ])
            print("\n✅ llama-cpp-python installed successfully!\n")
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            print("\nPlease install manually:")
            print("  pip install llama-cpp-python\n")
            return
    
    # Show available models
    manager.list_models()
    
    # Recommend the smallest model
    recommended = "tinyllama"
    print(f"\n💡 RECOMMENDATION: Start with '{recommended}' (smallest, fastest)")
    print("\nChoose a model to download:")
    print("  1. tinyllama (700MB) - Recommended for first-time users")
    print("  2. deepseek-coder-1.3b (800MB) - Better for code analysis")
    print("  3. phi-2 (1.6GB) - Most capable, larger size")
    print("  0. Skip download (use Ollama instead)")
    
    choice = input("\nEnter choice (1-3) or 0 to skip [1]: ").strip() or "1"
    
    model_map = {
        "1": "tinyllama",
        "2": "deepseek-coder-1.3b",
        "3": "phi-2",
        "0": None
    }
    
    selected_model = model_map.get(choice)
    
    if not selected_model:
        print("\n✅ Skipping download. You can use Ollama instead.\n")
        return
    
    if manager.is_model_downloaded(selected_model):
        print(f"\n✅ Model '{selected_model}' is already downloaded!")
        
        # Test the model
        test = input("\nTest the model? (y/n) [y]: ").strip().lower() or "y"
        if test == "y":
            test_model(manager, selected_model)
    else:
        print(f"\n📥 Starting download of '{selected_model}'...")
        print("This may take a few minutes depending on your internet speed.\n")
        
        if manager.download_model(selected_model):
            print("\n✅ Download complete!")
            
            # Test the model
            test = input("\nTest the model? (y/n) [y]: ").strip().lower() or "y"
            if test == "y":
                test_model(manager, selected_model)
        else:
            print("\n❌ Download failed. Please try again later.")
    
    print("\n" + "="*70)
    print("📋 NEXT STEPS:")
    print("="*70)
    print("\n1. Your AI model is ready in: ./models/")
    print("2. The Flask app will automatically use it!")
    print("3. Restart your Flask server:")
    print("   python flask_app.py")
    print("\n4. Go to Code Analysis tab and try AI features!")
    print("\n" + "="*70 + "\n")


def test_model(manager: LocalModelManager, model_name: str):
    """Test the downloaded model"""
    print(f"\n🧪 Testing {model_name}...\n")
    
    if manager.load_model(model_name):
        print("✅ Model loaded successfully!\n")
        print("🔬 Running test generation...\n")
        
        response = manager.generate(
            prompt="Write a Python function that adds two numbers and returns the result.",
            system_prompt="You are a helpful coding assistant. Be concise.",
            max_tokens=200
        )
        
        if response:
            print("📝 Model Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print("\n✅ Test successful! Model is working.\n")
        else:
            print("❌ Test failed - no response from model.\n")
    else:
        print("❌ Failed to load model for testing.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
