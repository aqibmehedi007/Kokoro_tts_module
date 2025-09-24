# Kokoro TTS Module

A powerful Text-to-Speech (TTS) web application using the Kokoro Q4 model, featuring an advanced Flask-based web interface with automated setup and model downloading.

## 🎤 Features

- **High-Quality Speech Synthesis**: Uses the Kokoro TTS Q4 model (178 MB) for natural-sounding speech
- **Automated Setup**: Automatically installs dependencies and downloads the model on first run
- **Web Interface**: Modern, responsive web UI with real-time status monitoring
- **Demo Mode**: Falls back to demo mode if the Kokoro library isn't available
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **One-Click Launch**: Simple startup scripts for easy deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection (for initial setup and model download)
- At least 2GB RAM
- CUDA-compatible GPU (optional, CPU works too)

### Installation & Launch

**Option 1: One-Click Launch (Recommended)**
```bash
# Windows users
start.bat

# Cross-platform
python start.py
```

**Option 2: Manual Launch**
```bash
# Clone the repository
git clone https://github.com/aqibmehedi007/Kokoro_tts_module.git
cd Kokoro_tts_module

# Run the application (auto-installs everything)
python app.py
```

The application will automatically:
1. ✅ Check and install Python dependencies
2. 📥 Download the Kokoro Q4 model (178 MB) from Hugging Face
3. 🚀 Launch the web interface at `http://localhost:5000`

## 📁 Project Structure

```
Kokoro_tts_module/
├── app.py                    # Main Flask application with automated setup
├── kokoro_tts_service.py    # Core TTS service implementation
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── start.py                 # Cross-platform startup script
├── start.bat                # Windows startup script
├── .gitignore              # Git ignore rules (excludes models/)
├── models/                 # Model files directory (auto-created)
│   └── Kokoro_espeak_Q4.gguf # Kokoro TTS Q4 model (178 MB)
├── templates/              # HTML templates
│   └── index.html         # Modern web interface
└── src/                   # Alternative source structure
    ├── app.py            # Alternative Flask app
    ├── kokoro_tts_service.py
    ├── tts_demo.py       # Demo implementation
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

## 🔧 Configuration

### Model Information

- **Model**: Kokoro TTS Q4 (Quantized 4-bit)
- **Size**: 178 MB
- **Source**: [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main)
- **Format**: GGUF (GGML Universal Format)
- **Quality**: High-quality speech synthesis with reduced file size

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

### Core Dependencies

- **Flask 2.3.3**: Web framework
- **PyTorch 2.0+**: Deep learning framework
- **Transformers 4.30+**: Hugging Face transformers
- **SoundFile**: Audio file handling
- **NumPy**: Numerical computing

### Kokoro-Specific Dependencies

- **kokoro 0.9.2+**: Kokoro TTS library
- **huggingface_hub**: Model downloading
- **spacy**: Natural language processing
- **phonemizer-fork**: Phoneme conversion
- **espeakng-loader**: Text-to-phoneme conversion

## 🚨 Troubleshooting

### Common Issues

1. **Model download fails**
   - Check internet connection
   - Verify firewall settings
   - Try running as administrator (Windows)

2. **Dependencies fail to install**
   - Ensure Python 3.8+ is installed
   - Check pip is up to date: `pip install --upgrade pip`
   - Try installing manually: `pip install -r requirements.txt`

3. **CUDA out of memory**
   - The Q4 model is optimized for lower memory usage
   - Close other GPU-intensive applications
   - The app will automatically fall back to CPU if needed

4. **Web interface not loading**
   - Ensure port 5000 is available
   - Check firewall settings
   - Try accessing `http://127.0.0.1:5000`

### Performance Tips

- **GPU Usage**: CUDA acceleration is automatically detected
- **Memory Management**: Automatic cleanup after each request
- **Model Caching**: Model stays loaded for faster subsequent requests
- **Q4 Optimization**: Smaller model size means faster loading and lower memory usage

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

---

**Note**: This application automatically downloads the Kokoro TTS Q4 model (178 MB) from [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main) on first run. The model is optimized for quality and performance with a smaller file size compared to higher quantization levels.