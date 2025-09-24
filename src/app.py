import os
import sys
import tempfile
import asyncio
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from kokoro_tts_service import kokoro_tts_service

app = Flask(__name__)

# Global variable to store the loaded model
tts_service = None
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'Kokoro_espeak_Q8.gguf')

async def load_model():
    """Load the Kokoro TTS model"""
    global tts_service
    try:
        tts_service = kokoro_tts_service
        tts_service.model_path = Path(model_path)
        # Ensure the path is correct when running from project root
        if not tts_service.model_path.exists():
            # Try relative path from project root
            tts_service.model_path = Path("./models/Kokoro_espeak_Q8.gguf")
        if await tts_service.initialize():
            print("TTS model loaded successfully!")
            return True
        else:
            print("Failed to load TTS model")
            return False
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        return False

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
        
        if tts_service is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate speech using the TTS service
        try:
            # Run async function in sync context
            audio_data = asyncio.run(tts_service.synthesize_speech(text))
            
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
    return jsonify({
        'loaded': tts_service is not None and tts_service.is_available(),
        'model_path': model_path,
        'model_exists': os.path.exists(model_path),
        'device': tts_service.device if tts_service else 'unknown',
        'model_info': tts_service.get_model_info() if tts_service else {}
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

async def main():
    """Main application entry point"""
    print("Starting QuteVoice TTS Application...")
    print(f"Model path: {model_path}")
    
    # Load the model on startup
    if await load_model():
        print("Ready to serve requests!")
    else:
        print("Warning: Model could not be loaded. Some features may not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    asyncio.run(main())
