# QuteVoice - Kokoro TTS Testing Interface

## Overview
QuteVoice is a web-based testing interface for the Kokoro Text-to-Speech (TTS) model. This project provides a simple and intuitive way to test the Kokoro TTS model with different text inputs.

## Model Information
- **Model**: Kokoro TTS (GGUF format)
- **Quantization**: Q8 (High quality, 186MB)
- **Source**: [Hugging Face - mmwillet2/Kokoro_GGUF](https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main)

## Project Structure
```
QuteVoice/
├── models/
│   └── Kokoro_espeak_Q8.gguf    # The downloaded TTS model
├── src/
│   ├── app.py                   # Flask web application
│   └── templates/
│       └── index.html           # Web interface
├── docs/
│   └── README.md               # This documentation
└── requirements.txt            # Python dependencies
```

## Installation

1. **Clone or download the project**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the web server**:
   ```bash
   python src/app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Test the TTS model**:
   - Enter text in the text area
   - Click "Generate Speech"
   - View the results

## Features

- **Model Status Check**: Automatically checks if the model is loaded
- **Real-time Interface**: Modern, responsive web interface
- **Error Handling**: Comprehensive error messages and status indicators
- **High-Quality Model**: Uses Q8 quantization for best quality

## Model Details

The Kokoro TTS model is a high-quality text-to-speech model converted to GGUF format for efficient inference. The Q8 quantization provides an excellent balance between quality and file size.

### Available Quantizations
- **Q8**: 186MB - Best quality (currently used)
- **Q5**: 180MB - Good quality, smaller size
- **Q4**: 178MB - Lower quality, smallest size

## Troubleshooting

### Model Not Loading
- Ensure the model file exists in the `models/` folder
- Check that you have sufficient RAM (model requires ~2GB)
- Verify all dependencies are installed correctly

### Performance Issues
- The model requires significant computational resources
- Consider using GPU acceleration if available
- For better performance, use the Q5 or Q4 quantization

## Future Improvements

- [ ] Implement actual audio generation
- [ ] Add audio playback functionality
- [ ] Support for different voice styles
- [ ] Batch processing capabilities
- [ ] Audio export functionality

## License

This project uses the Kokoro TTS model which is available under the MIT license.
