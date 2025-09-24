"""
QuteVoice - Advanced Kokoro TTS Web Interface
Automated setup with dependency installation and model downloading
"""

import os
import sys
import tempfile
import asyncio
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import time

# Check and install requirements automatically
def check_and_install_requirements():
    """Check if requirements are installed and install them if needed"""
    print("🔍 Checking Python dependencies...")
    
    try:
        # Try importing key packages
        import torch
        import flask
        import soundfile
        import numpy
        print("✅ All required packages are already installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Installing requirements automatically...")
        
        try:
            # Install requirements
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            print("💡 Please run: pip install -r requirements.txt")
            return False

def download_kokoro_model():
    """Download the Kokoro model file if it doesn't exist"""
    model_path = "./models/Kokoro_espeak_Q8.gguf"
    models_dir = Path("./models")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists: {model_path}")
        return True
    
    print("📥 Model not found. Downloading Kokoro TTS model...")
    print("⚠️  This may take several minutes depending on your internet connection...")
    
    # Model download URLs from the correct Hugging Face repository
    model_urls = [
        "https://huggingface.co/mmwillet2/Kokoro_GGUF/resolve/main/Kokoro_espeak_Q8.gguf",
        "https://huggingface.co/mmwillet2/Kokoro_GGUF/resolve/main/Kokoro_espeak_Q5.gguf",
        "https://huggingface.co/mmwillet2/Kokoro_GGUF/resolve/main/Kokoro_espeak_Q4.gguf"
    ]
    
    for i, url in enumerate(model_urls, 1):
        try:
            print(f"🔄 Attempting download from source {i}/{len(model_urls)}...")
            print(f"📡 URL: {url}")
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) / total_size)
                    print(f"\r📥 Downloading: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
            
            urllib.request.urlretrieve(url, model_path, show_progress)
            print(f"\n✅ Model downloaded successfully to: {model_path}")
            
            # Verify file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size > 100:  # Model should be at least 100MB
                print(f"📁 Model size: {file_size:.1f} MB")
                return True
            else:
                print(f"⚠️  Downloaded file seems too small ({file_size:.1f} MB), trying next source...")
                os.remove(model_path)
                
        except Exception as e:
            print(f"\n❌ Download failed from source {i}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            continue
    
    print("❌ Failed to download model from all sources.")
    print("💡 Please manually download the Kokoro model and place it in the models/ folder")
    print("🔗 You can find the models at: https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main")
    print("📋 Available models:")
    print("   - Kokoro_espeak_Q8.gguf (186 MB) - Recommended")
    print("   - Kokoro_espeak_Q5.gguf (180 MB)")
    print("   - Kokoro_espeak_Q4.gguf (178 MB)")
    return False

def setup_environment():
    """Complete environment setup"""
    print("🚀 Setting up QuteVoice TTS Environment...")
    print("=" * 60)
    
    # Step 1: Check and install requirements
    if not check_and_install_requirements():
        return False
    
    # Step 2: Download model if needed
    if not download_kokoro_model():
        print("⚠️  Continuing without model - some features may not work")
    
    print("=" * 60)
    print("✅ Environment setup complete!")
    return True

# Import after setup
try:
    import torch
    from flask import Flask, render_template, request, jsonify, send_file
    from kokoro_tts_service import KokoroTTSService
except ImportError:
    print("❌ Critical imports failed. Please check your Python environment.")
    sys.exit(1)

app = Flask(__name__)

# Global TTS service
tts_service = None
model_path = "./models/Kokoro_espeak_Q8.gguf"

def load_model():
    """Load the Kokoro TTS model"""
    global tts_service
    try:
        print(f"Loading TTS model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found: {model_path}")
            print("💡 Creating demo TTS service...")
            tts_service = DemoTTSService()
            return True
        
        # Try to load the actual Kokoro service
        try:
            tts_service = KokoroTTSService(model_path)
            
            # Test if we can initialize
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tts_service.initialize())
            loop.close()
            
            if result:
                print("✅ TTS model loaded successfully!")
                return True
            else:
                print("❌ Failed to load TTS model")
                print("💡 Falling back to demo mode...")
                tts_service = DemoTTSService()
                return True
                
        except ImportError as e:
            print(f"⚠️  Kokoro library not available: {e}")
            print("💡 Using demo TTS service instead...")
            tts_service = DemoTTSService()
            return True
            
    except Exception as e:
        print(f"❌ Error loading TTS model: {e}")
        print("💡 Using demo TTS service instead...")
        tts_service = DemoTTSService()
        return True

class DemoTTSService:
    """Demo TTS service for when Kokoro is not available"""
    
    def __init__(self):
        self.device = "cpu"
        self.initialized = True
        self.available_voices = ["demo_voice"]
        self.default_voice = "demo_voice"
        self.sample_rate = 24000
        print("🎭 Demo TTS service initialized")
    
    async def initialize(self):
        return True
    
    def is_available(self):
        return True
    
    async def synthesize_speech(self, text):
        """Generate demo audio"""
        import numpy as np
        import soundfile as sf
        import io
        
        # Create a simple demo audio signal
        duration = len(text) * 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create a more interesting demo sound
        frequency = 440 + (hash(text) % 200)  # Vary frequency based on text
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add envelope
        envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
        audio = audio * envelope
        
        # Convert to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, self.sample_rate, format='WAV')
        return audio_bytes.getvalue()
    
    def get_model_info(self):
        return {
            "model_name": "Demo TTS",
            "model_file": "demo_mode",
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "initialized": self.initialized,
            "available_voices": len(self.available_voices)
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    """Generate speech from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if tts_service is None or not tts_service.is_available():
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate speech using the TTS service
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = loop.run_until_complete(tts_service.synthesize_speech(text))
            loop.close()
            
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(audio_data)
            temp_file.close()
            
            return jsonify({
                'success': True,
                'generated_text': f"Audio generated for: {text[:50]}{'...' if len(text) > 50 else ''}",
                'message': f'Speech generation completed! Audio size: {len(audio_data)} bytes',
                'audio_ready': True,
                'text_length': len(text),
                'audio_path': os.path.basename(temp_file.name),
                'audio_size': len(audio_data)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Check if model is loaded"""
    try:
        loaded = tts_service is not None and tts_service.is_available()
        device_info = 'unknown'
        model_name = 'Unknown'
        available_voices = 0
        
        if tts_service:
            try:
                device_info = str(tts_service.device)
                model_info = tts_service.get_model_info()
                model_name = model_info.get('model_name', 'Unknown')
                available_voices = model_info.get('available_voices', 0)
            except:
                device_info = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_name = 'Demo TTS' if isinstance(tts_service, DemoTTSService) else 'Kokoro TTS'
                available_voices = 1 if isinstance(tts_service, DemoTTSService) else 8
        
        return jsonify({
            'loaded': loaded,
            'model_path': model_path,
            'model_exists': os.path.exists(model_path),
            'device': device_info,
            'model_name': model_name,
            'sample_rate': 24000,
            'available_voices': available_voices,
            'is_demo_mode': isinstance(tts_service, DemoTTSService) if tts_service else False
        })
    except Exception as e:
        return jsonify({
            'loaded': False,
            'error': str(e),
            'model_path': model_path,
            'model_exists': os.path.exists(model_path),
            'is_demo_mode': False
        })

@app.route('/download_audio/<filename>')
def download_audio(filename):
    """Download generated audio file"""
    try:
        # Security check - only allow files from temp directory
        if not filename.endswith('.wav'):
            return jsonify({'error': 'Invalid file type'}), 400
            
        audio_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, as_attachment=True)
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎤 QuteVoice - Advanced Kokoro TTS Application")
    print("🚀 Automated Setup & Launch")
    print("=" * 60)
    
    # Run complete environment setup
    if not setup_environment():
        print("❌ Environment setup failed!")
        print("💡 Please check your Python installation and internet connection")
        sys.exit(1)
    
    print("\n🎯 Starting TTS Application...")
    print("=" * 60)
    
    # Check if model exists after setup
    model_path = "./models/Kokoro_espeak_Q8.gguf"
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        print(f"📁 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("💡 The application will run in demo mode")
    
    print("\n🔄 Loading TTS model...")
    
    # Load the model on startup
    if load_model():
        print("\n🌐 Starting web server...")
        print("📍 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Goodbye!")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            print("💡 Please check the logs above for more details")
    else:
        print("\n⚠️  Model failed to load - running in limited mode")
        print("🌐 Starting web server anyway...")
        print("📍 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Goodbye!")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            print("💡 Please check the logs above for more details")
