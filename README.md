# Kokoro TTS Module

A powerful Text-to-Speech (TTS) web application using the Kokoro model, featuring a modern Flask-based web interface for converting text to high-quality speech.

## 🎤 Features

- **High-Quality Speech Synthesis**: Uses the Kokoro TTS model for natural-sounding speech generation
- **Web Interface**: Modern, responsive web UI built with Flask and HTML/CSS/JavaScript
- **Multiple Voice Support**: Supports 8 different voices including multilingual options
- **Real-time Processing**: Asynchronous speech generation with progress indicators
- **Audio Playback**: Built-in audio player for immediate playback of generated speech
- **Model Status Monitoring**: Real-time model loading status and health checks
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 4GB RAM
- Internet connection for initial setup

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aqibmehedi007/Kokoro_tts_module.git
   cd Kokoro_tts_module
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Kokoro model**
   - The model file `Kokoro_espeak_Q8.gguf` should be placed in the `models/` directory
   - Due to size constraints, the model is not included in the repository
   - Download the model from: [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main)
   - The application will automatically attempt to download the model on first run

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - The interface will automatically check model status and guide you through usage

## 📁 Project Structure

```
Kokoro_tts_module/
├── app.py                          # Main Flask application (root level)
├── kokoro_tts_service.py          # Core TTS service implementation
├── requirements.txt                # Python dependencies
├── README.md                      # This documentation
├── models/                        # Model files directory (not tracked in git)
│   └── Kokoro_espeak_Q8.gguf     # Kokoro TTS model file
├── templates/                     # HTML templates
│   └── index.html                # Main web interface
├── src/                          # Alternative source structure
│   ├── app.py                    # Alternative Flask app
│   ├── kokoro_tts_service.py     # Alternative TTS service
│   ├── tts_demo.py              # Demo implementation
│   └── templates/
│       └── index.html           # Alternative template
└── docs/                        # Additional documentation
    └── README.md               # Extended documentation
```

## 🎯 Usage

### Web Interface

1. **Start the server**: Run `python app.py`
2. **Open browser**: Navigate to `http://localhost:5000`
3. **Check model status**: The interface will show if the model is loaded
4. **Enter text**: Type or paste your text in the text area
5. **Generate speech**: Click "Generate Speech" button
6. **Play audio**: Use the built-in audio player to listen to the result
7. **Download**: Optionally download the generated audio file

### API Endpoints

The application provides several REST API endpoints:

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

### Model Configuration

The TTS service supports various configuration options:

- **Model Path**: Default is `./models/Kokoro_espeak_Q8.gguf`
- **Device**: Automatically detects CUDA or falls back to CPU
- **Sample Rate**: 24kHz for high-quality audio
- **Default Voice**: `af_heart` (high-quality female voice)

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

You can customize the application using environment variables:

```bash
export MODEL_PATH="/path/to/your/model.gguf"
export DEVICE="cuda"  # or "cpu"
export PORT=5000
export HOST="0.0.0.0"
```

## 🛠️ Development

### Running in Development Mode

```bash
# Enable debug mode
export FLASK_DEBUG=1
python app.py
```

### Testing

The project includes a demo script for testing:

```bash
python src/tts_demo.py
```

### Code Structure

- **`app.py`**: Main Flask application with routes and error handling
- **`kokoro_tts_service.py`**: Core TTS service with async support
- **`templates/index.html`**: Modern web interface with real-time updates
- **`requirements.txt`**: All necessary Python dependencies

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

1. **Model not found**
   - Ensure `Kokoro_espeak_Q8.gguf` is in the `models/` directory
   - Check file permissions and path

2. **CUDA out of memory**
   - Reduce batch size or use CPU mode
   - Close other GPU-intensive applications

3. **Audio generation fails**
   - Check if all dependencies are installed
   - Verify model file integrity
   - Check system audio drivers

4. **Web interface not loading**
   - Ensure port 5000 is available
   - Check firewall settings
   - Try accessing `http://127.0.0.1:5000`

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
- **Memory Management**: The service includes automatic memory cleanup
- **Model Caching**: Model stays loaded in memory for faster subsequent requests

## 📄 License

This project is open source. Please check the license file for specific terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For support and questions:

- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the troubleshooting section above

## 🔄 Updates

Stay updated with the latest changes:

- Watch the repository for updates
- Check the releases page for new versions
- Follow the changelog for detailed update information

---

**Note**: This application requires the Kokoro TTS model file to function. The model is not included in the repository due to size constraints and should be downloaded separately from [https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main). The application will automatically attempt to download the model on first run.
