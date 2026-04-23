import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🏥 Dr. Ava Medical Chatbot - Installation Script")
    print("=" * 50)
    
    # Required packages
    packages = [
        "langchain",
        "langchain-community", 
        "langchain-huggingface",
        "faiss-cpu",
        "pypdf",
        "pyttsx3",
        "fastapi",
        "uvicorn",
        "transformers",
        "torch"
    ]
    
    print("Installing required packages...")
    print()
    
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print()
    print("=" * 50)
    
    if failed_packages:
        print("❌ Installation completed with errors:")
        for package in failed_packages:
            print(f"   - {package}")
        print()
        print("Please install the failed packages manually:")
        print(f"pip install {' '.join(failed_packages)}")
        return False
    else:
        print("✅ All packages installed successfully!")
        print()
        print("🚀 You can now run the medical chatbot:")
        print("   python server.py")
        print()
        print("📋 Make sure to set your Hugging Face token:")
        print("   export HF_TOKEN=your_token_here")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)