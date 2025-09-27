"""
Quick GPU and Kokoro TTS test
"""

import asyncio
import torch
from kokoro_gpu_tts import tts_service

async def test_system():
    print("Kokoro TTS GPU Test")
    print("=" * 40)
    
    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test TTS
    print("\nInitializing TTS service...")
    success = await tts_service.initialize()
    
    if success:
        print("TTS initialized successfully")
        
        # Generate test audio
        print("\nGenerating test audio...")
        audio_data = await tts_service.generate_complete_audio("Hello, this is a GPU test.")
        
        print(f"Generated {len(audio_data)} bytes of audio")
        
        # Save test file
        with open("gpu_test.wav", "wb") as f:
            f.write(audio_data)
        print("Saved as gpu_test.wav")
        
        # Test streaming
        print("\nTesting streaming...")
        chunk_count = 0
        async for chunk in tts_service.stream_audio("Streaming test", chunk_size=2048):
            chunk_count += 1
            if chunk_count >= 3:  # Test first 3 chunks
                break
        
        print(f"Streaming works - got {chunk_count} chunks")
        
    else:
        print("TTS initialization failed")
    
    # Show service info
    info = tts_service.get_info()
    print(f"\nService Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_system())
