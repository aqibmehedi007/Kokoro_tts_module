"""
Simple Kokoro TTS Streaming Server
Minimal implementation for streaming TTS
"""

import asyncio
import json
import base64
import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from kokoro_gpu_tts import tts_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_page():
    """Serve the simple HTML page"""
    html_file = Path("simple_stream.html")
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
        <body>
        <h1>Error: simple_stream.html not found</h1>
        <p>Make sure simple_stream.html is in the same directory</p>
        </body>
        </html>
        """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for streaming"""
    await websocket.accept()
    logger.info("WebSocket connected")
    
    try:
        # Initialize TTS if needed
        if not tts_service.initialized:
            logger.info("Initializing TTS service...")
            await tts_service.initialize()
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "stream_audio":
                text = message.get("text", "")
                voice = message.get("voice", "af_heart")
                
                if not text.strip():
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No text provided"
                    }))
                    continue
                
                logger.info(f"Streaming: {text[:50]}...")
                
                try:
                    chunk_number = 0
                    async for chunk_data in tts_service.stream_audio(text, voice, chunk_size=2048):
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
                        "message": f"Generated {chunk_number} chunks"
                    }))
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
            
            elif message["type"] == "stop_streaming":
                break
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Simple Kokoro TTS Streaming Server")
    print("GPU Support:", "Yes" if tts_service.device == "cuda" else "No")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
