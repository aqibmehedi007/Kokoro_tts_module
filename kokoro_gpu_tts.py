"""
Simple Kokoro TTS Service
Streaming TTS with GPU acceleration
"""

import asyncio
import logging
import torch
import numpy as np
import soundfile as sf
import io
from typing import Optional, AsyncGenerator
from kokoro import KPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KokoroGPUTTS:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Kokoro TTS with GPU optimization
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA detected: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                device = 'cpu'
                logger.warning("CUDA not available, using CPU")
        
        self.device = device
        self.pipeline = None
        self.sample_rate = 24000
        self.initialized = False
        
        # Voice options
        self.voices = [
            "af_heart", "en_heart", "es_heart", "fr_heart",
            "de_heart", "it_heart", "pt_heart", "ru_heart"
        ]
        self.default_voice = "af_heart"
        
        logger.info(f"Kokoro TTS initialized for device: {device}")
    
    async def initialize(self) -> bool:
        """Initialize the Kokoro pipeline with GPU optimization"""
        try:
            if self.initialized:
                return True
            
            logger.info("Loading Kokoro TTS model...")
            
            # Initialize pipeline with device specification
            self.pipeline = KPipeline(
                lang_code='a',  # Auto-detect language
                device=self.device
            )
            
            # Test generation to ensure everything works
            logger.info("Testing model with sample text...")
            test_gen = self.pipeline("Test", voice=self.default_voice)
            test_result = next(iter(test_gen))
            
            if len(test_result) >= 3 and test_result[2] is not None:
                logger.info("Model test successful")
                self.initialized = True
                return True
            else:
                logger.error("Model test failed - no audio generated")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            return False
    
    async def generate_complete_audio(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Generate complete audio for the given text
        
        Args:
            text: Text to synthesize
            voice: Voice to use (defaults to af_heart)
            
        Returns:
            WAV audio data as bytes
        """
        if not self.initialized:
            await self.initialize()
        
        voice = voice or self.default_voice
        if voice not in self.voices:
            logger.warning(f"Unknown voice '{voice}', using default")
            voice = self.default_voice
        
        try:
            logger.info(f"Generating audio for: '{text[:50]}...' using voice: {voice}")
            
            # Generate audio
            generator = self.pipeline(text, voice=voice)
            
            # Collect all audio segments
            audio_segments = []
            for result in generator:
                if len(result) >= 3 and result[2] is not None:
                    # Convert tensor to numpy
                    audio_tensor = result[2]
                    if hasattr(audio_tensor, 'cpu'):
                        audio_np = audio_tensor.cpu().numpy()
                    else:
                        audio_np = np.array(audio_tensor)
                    audio_segments.append(audio_np)
            
            if not audio_segments:
                raise Exception("No audio generated")
            
            # Concatenate all segments
            full_audio = np.concatenate(audio_segments)
            
            # Convert to WAV bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, full_audio, self.sample_rate, format='WAV')
            wav_data = audio_buffer.getvalue()
            
            logger.info(f"Generated {len(wav_data)} bytes of audio")
            return wav_data
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise
    
    async def stream_audio(self, text: str, voice: Optional[str] = None, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
        """
        Stream audio generation in real-time chunks
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            chunk_size: Size of audio chunks in samples
            
        Yields:
            Raw PCM audio chunks as bytes
        """
        if not self.initialized:
            await self.initialize()
        
        voice = voice or self.default_voice
        if voice not in self.voices:
            voice = self.default_voice
        
        try:
            logger.info(f"Starting streaming for: '{text[:50]}...' using voice: {voice}")
            
            generator = self.pipeline(text, voice=voice)
            audio_buffer = []
            chunk_count = 0
            
            for result in generator:
                if len(result) >= 3 and result[2] is not None:
                    # Convert tensor to numpy
                    audio_tensor = result[2]
                    if hasattr(audio_tensor, 'cpu'):
                        audio_np = audio_tensor.cpu().numpy()
                    else:
                        audio_np = np.array(audio_tensor)
                    
                    # Add to buffer
                    audio_buffer.extend(audio_np.flatten())
                    
                    # Yield chunks when buffer is large enough
                    while len(audio_buffer) >= chunk_size:
                        chunk_data = np.array(audio_buffer[:chunk_size], dtype=np.float32)
                        audio_buffer = audio_buffer[chunk_size:]
                        
                        # Convert to 16-bit PCM
                        pcm_data = (chunk_data * 32767).astype(np.int16).tobytes()
                        
                        chunk_count += 1
                        logger.debug(f"Yielding chunk {chunk_count}: {len(pcm_data)} bytes")
                        yield pcm_data
                        
                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.001)
            
            # Yield remaining audio
            if audio_buffer:
                chunk_data = np.array(audio_buffer, dtype=np.float32)
                pcm_data = (chunk_data * 32767).astype(np.int16).tobytes()
                chunk_count += 1
                logger.debug(f"Yielding final chunk {chunk_count}: {len(pcm_data)} bytes")
                yield pcm_data
            
            logger.info(f"Streaming completed: {chunk_count} chunks generated")
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def get_info(self) -> dict:
        """Get service information"""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "sample_rate": self.sample_rate,
            "voices": self.voices,
            "initialized": self.initialized
        }
    
    def __del__(self):
        """Cleanup GPU memory on destruction"""
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance
tts_service = KokoroGPUTTS()
