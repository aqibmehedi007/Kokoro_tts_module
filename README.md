# Kokoro TTS Module

A powerful Text-to-Speech (TTS) web application using the Kokoro Q4 model, featuring an advanced Flask-based web interface with **fully automated setup**, **GPU acceleration**, and **intelligent fallback systems**.

## 🎤 Features

- **High-Quality Speech Synthesis**: Uses the Kokoro TTS Q4 model (178 MB) for natural-sounding speech
- **🚀 GPU Acceleration**: Automatic CUDA detection and GPU offloading with llama-cpp-python
- **🤖 Fully Automated Setup**: Zero-configuration installation - just run `python app.py`
- **📦 Smart Dependency Management**: Automatically installs all requirements and pre-built CUDA wheels
- **🌐 Modern Web Interface**: Responsive UI with real-time status monitoring and audio controls
- **🎭 Intelligent Fallback**: Demo mode when Kokoro library unavailable, graceful error handling
- **⚡ Performance Optimized**: JamePeng CUDA wheels for maximum GPU performance
- **🔧 Cross-Platform**: Works on Windows, macOS, and Linux with automatic environment detection

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11 recommended for optimal GPU support)
- **Internet connection** (for initial setup and model download)
- **At least 2GB RAM** (4GB+ recommended for GPU acceleration)
- **CUDA-compatible GPU** (optional but highly recommended for performance)
- **Windows 10/11** (for CUDA wheels), macOS/Linux (CPU mode)

### 🚀 Zero-Configuration Installation

**Single Command Launch (Recommended)**
```bash
# Clone and run - that's it!
git clone https://github.com/aqibmehedi007/Kokoro_tts_module.git
cd Kokoro_tts_module
python app.py
```

**What happens automatically:**
1. 🔍 **Environment Detection**: Checks Python version, GPU availability, and system capabilities
2. 📦 **Smart Dependency Installation**: Installs all requirements from `requirements.txt`
3. 🚀 **GPU Acceleration Setup**: Downloads and installs JamePeng CUDA wheels for optimal performance
4. 📥 **Model Download**: Downloads Kokoro Q4 model (178 MB) with progress tracking
5. 🎯 **Service Initialization**: Loads TTS service with fallback to demo mode if needed
6. 🌐 **Web Server Launch**: Starts Flask server at `http://localhost:5000`

**No manual configuration required!** The application handles everything automatically.

## 📁 Project Structure

```
Kokoro_tts_module/
├── app.py                           # 🚀 Main application with automated setup & GPU acceleration
├── kokoro_tts_service.py           # 🎤 Core TTS service implementation
├── requirements.txt                 # 📦 Python dependencies (auto-installed)
├── README.md                       # 📖 This comprehensive documentation
├── .gitignore                      # 🚫 Git ignore rules (excludes models/ & pre-build-wheel/)
├── models/                         # 📁 Model files directory (auto-created)
│   └── Kokoro_espeak_Q4.gguf      # 🎯 Kokoro TTS Q4 model (178 MB, auto-downloaded)
├── pre-build-wheel/                # ⚡ CUDA wheels directory (auto-created)
│   └── llama_cpp_python-*.whl     # 🚀 JamePeng CUDA wheel (auto-downloaded)
├── templates/                      # 🌐 HTML templates
│   └── index.html                 # 💻 Modern responsive web interface
└── src/                           # 📂 Alternative source structure
    ├── app.py                    # Alternative Flask app
    ├── kokoro_tts_service.py     # Alternative TTS service
    ├── tts_demo.py              # Demo implementation
    └── templates/
        └── index.html
```

## 🎯 Usage

### Web Interface

1. **Launch**: Run `python app.py` or use `start.py`/`start.bat`
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Check Status**: The interface shows model loading status automatically
4. **Enter Text**: Type or paste your text in the text area
5. **Generate Speech**: Click "Generate Speech" button
6. **Play Audio**: Use the built-in audio player to listen
7. **Download**: Optionally download the generated audio file

### API Endpoints

- `GET /` - Main web interface
- `POST /generate_speech` - Generate speech from text
- `GET /model_status` - Check model loading status
- `GET /download_audio/<filename>` - Download generated audio files

### Example API Usage

```python
import requests

# Generate speech
response = requests.post('http://localhost:5000/generate_speech', 
                        json={'text': 'Hello, world!'})
result = response.json()

# Check model status
status = requests.get('http://localhost:5000/model_status').json()
```

## 🔧 Advanced Configuration

### 🚀 GPU Acceleration Features

- **Automatic CUDA Detection**: Detects NVIDIA GPUs and enables CUDA acceleration
- **JamePeng CUDA Wheels**: Downloads optimized CUDA 12.8 wheels for maximum performance
- **GPU Offloading**: Automatically offloads model layers to GPU for faster inference
- **Fallback Support**: Gracefully falls back to CPU if GPU unavailable
- **Memory Optimization**: Smart memory management for optimal GPU utilization

### 📊 Model Information

- **Model**: Kokoro TTS Q4 (Quantized 4-bit)
- **Size**: 178 MB (optimized for speed and quality balance)
- **Source**: [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main)
- **Format**: GGUF (GGML Universal Format)
- **Quality**: High-quality speech synthesis with reduced file size
- **GPU Support**: Full CUDA acceleration with llama-cpp-python

### Available Voices

The system supports 8 different voices:
- `af_heart` - Premium female voice (default)
- `en_heart` - English female voice
- `es_heart` - Spanish female voice
- `fr_heart` - French female voice
- `de_heart` - German female voice
- `it_heart` - Italian female voice
- `pt_heart` - Portuguese female voice
- `ru_heart` - Russian female voice

### Environment Variables

```bash
export MODEL_PATH="/path/to/your/model.gguf"
export DEVICE="cuda"  # or "cpu"
export PORT=5000
export HOST="0.0.0.0"
```

## 🛠️ Development

### Running in Development Mode

```bash
export FLASK_DEBUG=1
python app.py
```

### Testing

```bash
python src/tts_demo.py
```

## 📋 Dependencies

### 🚀 Core Dependencies (Auto-Installed)

- **Flask 2.3.3**: Modern web framework with async support
- **PyTorch 2.0+**: Deep learning framework with CUDA support
- **Transformers 4.30+**: Hugging Face transformers library
- **SoundFile**: High-quality audio file handling
- **NumPy**: Numerical computing and array operations
- **Gradio**: Advanced web interface components

### 🎤 Kokoro-Specific Dependencies (Auto-Installed)

- **kokoro 0.9.2+**: Core Kokoro TTS library
- **huggingface_hub**: Model downloading and caching
- **spacy**: Natural language processing pipeline
- **phonemizer-fork**: Advanced phoneme conversion
- **espeakng-loader**: Text-to-phoneme conversion engine
- **misaki**: Japanese text processing
- **curated-transformers**: Optimized transformer models

### ⚡ GPU Acceleration Dependencies (Auto-Installed)

- **llama-cpp-python**: GGUF model inference with CUDA support
- **JamePeng CUDA Wheels**: Pre-built CUDA 12.8 wheels for optimal performance
- **requests & tqdm**: File downloading with progress tracking

**All dependencies are automatically installed on first run - no manual setup required!**

## 🚨 Troubleshooting

### 🔧 Common Issues & Solutions

1. **🚫 Model download fails**
   - ✅ **Solution**: Check internet connection and firewall settings
   - ✅ **Fallback**: App continues in demo mode if download fails
   - ✅ **Manual**: Download model manually from Hugging Face if needed

2. **📦 Dependencies fail to install**
   - ✅ **Solution**: Ensure Python 3.8+ is installed
   - ✅ **Update**: Run `pip install --upgrade pip` before retrying
   - ✅ **Manual**: App provides fallback installation methods

3. **🚀 GPU/CUDA issues**
   - ✅ **Detection**: App automatically detects GPU availability
   - ✅ **Fallback**: Gracefully falls back to CPU if GPU unavailable
   - ✅ **Memory**: Q4 model optimized for lower memory usage
   - ✅ **Wheels**: Automatically downloads JamePeng CUDA wheels

4. **🌐 Web interface not loading**
   - ✅ **Port**: Ensure port 5000 is available
   - ✅ **Firewall**: Check firewall settings
   - ✅ **Alternative**: Try accessing `http://127.0.0.1:5000`

### ⚡ Performance Optimization

- **🚀 GPU Acceleration**: Automatic CUDA detection and optimization
- **💾 Memory Management**: Smart cleanup and caching strategies
- **🔄 Model Persistence**: Model stays loaded for faster subsequent requests
- **📊 Q4 Optimization**: Balanced quality/size for optimal performance
- **🎯 Pre-built Wheels**: JamePeng CUDA wheels for maximum GPU performance

## 📄 License

This project is open source. Please check the license file for specific terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For support and questions:

- Open an issue on GitHub
- Check the troubleshooting section above
- Review the console output for detailed error messages

## 🔄 Updates

Stay updated with the latest changes:

- Watch the repository for updates
- Check the releases page for new versions
- Follow the changelog for detailed update information

## 🌟 Advanced Features

### 🤖 Intelligent Automation
- **Zero-Configuration Setup**: No manual installation steps required
- **Smart Environment Detection**: Automatically detects Python version, GPU, and system capabilities
- **Progressive Installation**: Installs dependencies in optimal order with error recovery
- **Graceful Fallbacks**: Multiple fallback strategies for maximum compatibility

### 🚀 Performance Features
- **GPU Acceleration**: Automatic CUDA detection and GPU offloading
- **Pre-built CUDA Wheels**: Downloads optimized JamePeng wheels for maximum performance
- **Memory Optimization**: Smart memory management and cleanup
- **Model Caching**: Persistent model loading for faster subsequent requests

### 🎭 Robust Error Handling
- **Demo Mode Fallback**: Continues working even if Kokoro library fails
- **Comprehensive Logging**: Detailed status messages and error reporting
- **Multiple Installation Methods**: Fallback installation strategies for dependencies
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux

---

**🎯 Key Benefits**: This application provides a **production-ready TTS solution** with **zero configuration required**. Simply run `python app.py` and everything is handled automatically - from dependency installation to GPU acceleration setup to model downloading. The intelligent fallback systems ensure the application works in any environment, making it perfect for both development and production use.

**📥 Model Source**: Automatically downloads the Kokoro TTS Q4 model (178 MB) from [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main) on first run. The Q4 quantization provides an optimal balance between quality and performance.