"""
FastAPI Server for Kokoro TTS Streaming
Production-ready server with WebSocket streaming
"""

import asyncio
import base64
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from kokoro_gpu_tts import tts_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Kokoro TTS Streaming Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML interface
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kokoro TTS Streaming</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        button {
            flex: 1;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-success {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
        }
        .btn-danger {
            background: linear-gradient(135deg, #dc3545, #c82333);
            color: white;
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 500;
        }
        .status.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.streaming {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .audio-player {
            margin: 20px 0;
        }
        .audio-player audio {
            width: 100%;
            margin: 10px 0;
        }
        .progress {
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 10px 0;
        }
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            width: 0%;
            transition: width 0.3s;
        }
        .info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Kokoro TTS Streaming</h1>
        
        <div class="form-group">
            <label for="textInput">Text to synthesize:</label>
            <textarea id="textInput" placeholder="Enter your text here...">Hello! This is a test of the Kokoro TTS streaming system. It should generate high-quality speech in real-time using GPU acceleration.</textarea>
        </div>
        
        <div class="form-group">
            <label for="voiceSelect">Voice:</label>
            <select id="voiceSelect">
                <option value="af_heart">AF Heart (Female)</option>
                <option value="en_heart">EN Heart (Female)</option>
                <option value="es_heart">ES Heart (Spanish)</option>
                <option value="fr_heart">FR Heart (French)</option>
                <option value="de_heart">DE Heart (German)</option>
                <option value="it_heart">IT Heart (Italian)</option>
                <option value="pt_heart">PT Heart (Portuguese)</option>
                <option value="ru_heart">RU Heart (Russian)</option>
            </select>
        </div>
        
        <div class="button-group">
            <button id="connectBtn" class="btn-primary" onclick="connect()">Connect</button>
            <button id="generateBtn" class="btn-success" onclick="generateComplete()" disabled>Generate Complete</button>
            <button id="streamBtn" class="btn-success" onclick="startStreaming()" disabled>Start Streaming</button>
            <button id="stopBtn" class="btn-danger" onclick="stopStreaming()" disabled>Stop</button>
        </div>
        
        <div id="status" class="status">Ready to connect</div>
        
        <div class="progress">
            <div id="progressBar" class="progress-bar"></div>
        </div>
        
        <div id="audioPlayer" class="audio-player"></div>
        
        <div class="info">
            <strong>GPU Acceleration:</strong> This system uses CUDA acceleration for optimal performance.<br>
            <strong>Streaming:</strong> Audio is generated and played in real-time chunks for low latency.<br>
            <strong>Complete Generation:</strong> Generates full audio before playback for longer texts.
        </div>
    </div>

    <script>
        let ws = null;
        let isStreaming = false;
        let audioChunks = [];
        let streamingAudio = null;
        
        function updateStatus(message, type = 'connected') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
        }
        
        function updateProgress(percent) {
            document.getElementById('progressBar').style.width = percent + '%';
        }
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                updateStatus('Connected to TTS server', 'connected');
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('generateBtn').disabled = false;
                document.getElementById('streamBtn').disabled = false;
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = () => {
                updateStatus('Disconnected from server', 'error');
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('generateBtn').disabled = true;
                document.getElementById('streamBtn').disabled = true;
                document.getElementById('stopBtn').disabled = true;
            };
            
            ws.onerror = (error) => {
                updateStatus('Connection error', 'error');
                console.error('WebSocket error:', error);
            };
        }
        
        function handleMessage(data) {
            switch(data.type) {
                case 'complete_audio':
                    playCompleteAudio(data.audio_data);
                    updateStatus('Audio generated and playing', 'connected');
                    break;
                    
                case 'stream_chunk':
                    addAudioChunk(data.chunk_data);
                    updateStatus(`Streaming... chunk ${data.chunk_number}`, 'streaming');
                    break;
                    
                case 'stream_complete':
                    finishStreaming();
                    updateStatus('Streaming completed', 'connected');
                    break;
                    
                case 'error':
                    updateStatus('Error: ' + data.message, 'error');
                    break;
                    
                case 'status':
                    updateStatus(data.message, 'streaming');
                    break;
            }
        }
        
        function generateComplete() {
            const text = document.getElementById('textInput').value.trim();
            const voice = document.getElementById('voiceSelect').value;
            
            if (!text) {
                updateStatus('Please enter some text', 'error');
                return;
            }
            
            updateStatus('Generating complete audio...', 'streaming');
            updateProgress(50);
            
            ws.send(JSON.stringify({
                type: 'generate_complete',
                text: text,
                voice: voice
            }));
        }
        
        function playCompleteAudio(audioData) {
            const audioPlayer = document.getElementById('audioPlayer');
            
            // Convert base64 to blob
            const audioBytes = atob(audioData);
            const audioArray = new Uint8Array(audioBytes.length);
            for (let i = 0; i < audioBytes.length; i++) {
                audioArray[i] = audioBytes.charCodeAt(i);
            }
            
            const blob = new Blob([audioArray], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            
            audioPlayer.innerHTML = `
                <audio controls autoplay>
                    <source src="${audioUrl}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
            `;
            
            updateProgress(100);
            setTimeout(() => updateProgress(0), 2000);
        }
        
        function startStreaming() {
            const text = document.getElementById('textInput').value.trim();
            const voice = document.getElementById('voiceSelect').value;
            
            if (!text) {
                updateStatus('Please enter some text', 'error');
                return;
            }
            
            isStreaming = true;
            audioChunks = [];
            streamingAudio = null;
            
            document.getElementById('streamBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            updateStatus('Starting streaming...', 'streaming');
            
            ws.send(JSON.stringify({
                type: 'stream_audio',
                text: text,
                voice: voice
            }));
        }
        
        function addAudioChunk(chunkData) {
            // Convert base64 to array
            const audioBytes = atob(chunkData);
            const audioArray = new Uint8Array(audioBytes.length);
            for (let i = 0; i < audioBytes.length; i++) {
                audioArray[i] = audioBytes.charCodeAt(i);
            }
            
            audioChunks.push(audioArray);
            
            // Start playing after first few chunks
            if (audioChunks.length === 3 && !streamingAudio) {
                startStreamingPlayback();
            }
        }
        
        function startStreamingPlayback() {
            if (audioChunks.length === 0) return;
            
            // Create WAV file from accumulated chunks
            const combinedData = combineAudioChunks();
            const wavData = createWavFile(combinedData, 24000, 1);
            const blob = new Blob([wavData], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            
            const audioPlayer = document.getElementById('audioPlayer');
            audioPlayer.innerHTML = `
                <audio id="streamingAudio" controls autoplay>
                    <source src="${audioUrl}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
            `;
            
            streamingAudio = document.getElementById('streamingAudio');
        }
        
        function combineAudioChunks() {
            const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combined = new Uint8Array(totalLength);
            let offset = 0;
            
            for (const chunk of audioChunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            
            return combined;
        }
        
        function createWavFile(pcmData, sampleRate, channels) {
            const dataLength = pcmData.length;
            const buffer = new ArrayBuffer(44 + dataLength);
            const view = new DataView(buffer);
            
            // WAV header
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + dataLength, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, channels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * channels * 2, true);
            view.setUint16(32, channels * 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, dataLength, true);
            
            const uint8Array = new Uint8Array(buffer, 44);
            uint8Array.set(pcmData);
            
            return buffer;
        }
        
        function finishStreaming() {
            isStreaming = false;
            document.getElementById('streamBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            // Update audio with all chunks
            if (audioChunks.length > 0) {
                const combinedData = combineAudioChunks();
                const wavData = createWavFile(combinedData, 24000, 1);
                const blob = new Blob([wavData], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(blob);
                
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.innerHTML = `
                    <audio controls>
                        <source src="${audioUrl}" type="audio/wav">
                        Your browser does not support audio playback.
                    </audio>
                `;
            }
        }
        
        function stopStreaming() {
            if (isStreaming) {
                ws.send(JSON.stringify({ type: 'stop_streaming' }));
                finishStreaming();
                updateStatus('Streaming stopped', 'connected');
            }
        }
        
        // Auto-connect on page load
        window.onload = () => {
            connect();
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the HTML interface"""
    return HTMLResponse(content=HTML_INTERFACE)

@app.get("/info")
async def get_info():
    """Get TTS service information"""
    try:
        if not tts_service.initialized:
            await tts_service.initialize()
        return tts_service.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_audio(request: dict):
    """Generate complete audio file"""
    try:
        text = request.get("text", "")
        voice = request.get("voice", "af_heart")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        if not tts_service.initialized:
            await tts_service.initialize()
        
        audio_data = await tts_service.generate_complete_audio(text, voice)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Initialize TTS service if needed
        if not tts_service.initialized:
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": "Initializing TTS service..."
            }))
            
            success = await tts_service.initialize()
            if not success:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to initialize TTS service"
                }))
                return
        
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "TTS service ready"
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "generate_complete":
                await handle_complete_generation(websocket, message)
            
            elif message["type"] == "stream_audio":
                await handle_streaming(websocket, message)
            
            elif message["type"] == "stop_streaming":
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Streaming stopped"
                }))
                break
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass

async def handle_complete_generation(websocket: WebSocket, message: dict):
    """Handle complete audio generation"""
    try:
        text = message.get("text", "")
        voice = message.get("voice", "af_heart")
        
        if not text.strip():
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No text provided"
            }))
            return
        
        # Generate complete audio
        audio_data = await tts_service.generate_complete_audio(text, voice)
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        await websocket.send_text(json.dumps({
            "type": "complete_audio",
            "audio_data": audio_b64
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

async def handle_streaming(websocket: WebSocket, message: dict):
    """Handle streaming audio generation"""
    try:
        text = message.get("text", "")
        voice = message.get("voice", "af_heart")
        
        if not text.strip():
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No text provided"
            }))
            return
        
        # Start streaming
        chunk_number = 0
        async for chunk_data in tts_service.stream_audio(text, voice):
            chunk_number += 1
            
            # Encode chunk as base64
            chunk_b64 = base64.b64encode(chunk_data).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "chunk_data": chunk_b64,
                "chunk_number": chunk_number
            }))
        
        await websocket.send_text(json.dumps({
            "type": "stream_complete",
            "message": f"Streaming completed with {chunk_number} chunks"
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Kokoro TTS Streaming Server")
    logger.info("GPU Support: " + ("✓" if torch.cuda.is_available() else "✗"))
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
