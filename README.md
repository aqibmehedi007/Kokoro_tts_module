# Simple Kokoro TTS Streaming

A minimal Text-to-Speech (TTS) streaming application using the Kokoro model with GPU acceleration and real-time WebSocket streaming.

## 🎤 Features

- **High-Quality Speech Synthesis**: Uses the Kokoro TTS model for natural-sounding speech
- **🚀 GPU Acceleration**: Automatic CUDA detection and GPU offloading
- **📡 Real-Time Streaming**: WebSocket streaming for low-latency audio generation
- **🎵 Progressive Audio Playback**: Audio chunks play as they're generated
- **🌐 Simple Web Interface**: Clean, responsive UI with auto-connection

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11 recommended for optimal GPU support)
- **Internet connection** (for initial setup and model download)
- **At least 2GB RAM** (4GB+ recommended for GPU acceleration)
- **CUDA-compatible GPU** (optional but highly recommended for performance)

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

3. **Start the server**
```bash
python simple_server.py
```

4. **Open your browser**
Navigate to `http://localhost:5000`

The web interface will automatically connect and you can start streaming TTS immediately!

## 📁 Project Structure

```
Kokoro_tts_module/
├── simple_server.py           # 🚀 Main streaming server
├── simple_stream.html         # 🌐 Web interface
├── kokoro_gpu_tts.py         # 🎤 TTS service implementation
├── requirements.txt           # 📦 Python dependencies
├── README.md                 # 📖 Documentation
└── models/                   # 📁 Model files directory
    └── Kokoro_espeak_Q4.gguf # 🎯 Kokoro TTS model
```

## 🎯 Usage

1. **Enter text** in the text area
2. **Select voice** from the dropdown (8 different voices available)
3. **Click "Start Streaming"** to begin real-time audio generation
4. **Audio plays automatically** as chunks are generated

### Available Voices

- `af_heart` - Premium female voice (default)
- `en_heart` - English female voice
- `es_heart` - Spanish female voice
- `fr_heart` - French female voice
- `de_heart` - German female voice
- `it_heart` - Italian female voice
- `pt_heart` - Portuguese female voice
- `ru_heart` - Russian female voice

## ⚙️ Configuration

The server automatically detects and uses the best available device:
- **CUDA GPU** (if available) - for optimal performance
- **CPU** (fallback) - slower but functional

## 📋 Dependencies

- **torch>=2.0.0**: Deep learning framework with CUDA support
- **kokoro>=0.9.4**: Core Kokoro TTS library
- **fastapi>=0.100.0**: Modern web framework
- **uvicorn[standard]>=0.23.0**: ASGI server
- **websockets>=11.0**: WebSocket support
- **soundfile>=0.12.0**: Audio file handling
- **numpy>=1.24.0**: Numerical computing

## 🔧 Development

### Running in Development Mode

```bash
python simple_server.py
```

The server will start on `http://localhost:5000` with auto-reload enabled.

### API Endpoints

- `GET /` - Web interface
- `WebSocket /ws` - Streaming TTS endpoint

### WebSocket Protocol

**Send:**
```json
{
  "type": "stream_audio",
  "text": "Hello world",
  "voice": "af_heart"
}
```

**Receive:**
```json
{
  "type": "stream_chunk",
  "chunk_data": "base64_encoded_audio",
  "chunk_number": 1
}
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**: The system will automatically fall back to CPU mode
2. **Model loading errors**: Ensure you have internet connection for initial model download
3. **Audio not playing**: Check browser audio permissions and WebSocket connection

### Performance Tips

- Use a CUDA-compatible GPU for best performance
- Keep text chunks reasonable in size for optimal streaming
- Close other GPU-intensive applications when running

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.