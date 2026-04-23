import subprocess
import sys
import os
import shutil

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_hf_token():
    """Check if Hugging Face token is set"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("⚠️  Hugging Face token not found in environment variables")
        print("   Please set HF_TOKEN environment variable:")
        print("   export HF_TOKEN=your_token_here")
        return False
    print("✅ Hugging Face token found")
    return True

def create_env_file():
    """Create .env file template"""
    env_content = """# Dr. Ava Medical Chatbot Environment Variables
# Copy this file to .env and fill in your values

# Required: Your Hugging Face API token
HF_TOKEN=your_huggingface_token_here

# Optional: Hugging Face model repository ID
HF_REPO_ID=HuggingFaceH4/zephyr-7b-beta

# Optional: Server port (default: 8000)
PORT=8000
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("   Please edit .env file and add your Hugging Face token")
    else:
        print("✅ .env file already exists")

def main():
    print("🏥 Dr. Ava Medical Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Check if vectorstore exists
    if not os.path.exists("vectorstore/db_faiss"):
        print("⚠️  Vectorstore not found. You need to run the medical_chatbot.py script first")
        print("   python medical_chatbot.py")
        return False
    
    # Check Hugging Face token
    token_ok = check_hf_token()
    
    print("\n" + "=" * 50)
    if token_ok:
        print("🎉 Setup completed successfully!")
        print("\n🚀 To run the application:")
        print("   python server.py")
        print("\n🌐 Then open: http://localhost:8000")
    else:
        print("⚠️  Setup completed with warnings")
        print("\n📋 Next steps:")
        print("1. Get a Hugging Face token from: https://huggingface.co/settings/tokens")
        print("2. Set the token: export HF_TOKEN=your_token_here")
        print("3. Run: python server.py")
    
    print("\n📚 For more information, see README.md")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)