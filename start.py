"""
Startup script for Kokoro TTS Streaming Server
Handles GPU detection and optimal configuration
"""

import sys
import torch
import logging
from pathlib import Path

def check_gpu():
    """Check GPU availability and configuration"""
    print("🔍 Checking GPU configuration...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"🎮 GPU Count: {gpu_count}")
        print(f"🚀 Current GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory test passed")
        except Exception as e:
            print(f"⚠️  GPU memory test failed: {e}")
            return False
            
        return True
    else:
        print("❌ CUDA not available - will use CPU (slower)")
        return False

def check_model():
    """Check if Kokoro model is available"""
    print("\n🔍 Checking Kokoro model...")
    
    try:
        from kokoro import KPipeline
        print("Kokoro library found")
        
        # Test model loading
        pipeline = KPipeline(lang_code='a', device='cuda' if torch.cuda.is_available() else 'cpu')
        test_gen = pipeline("Test", voice="af_heart")
        test_result = next(iter(test_gen))
        
        if len(test_result) >= 3 and test_result[2] is not None:
            print("Kokoro model test passed")
            return True
        else:
            print("❌ Kokoro model test failed - no audio generated")
            return False
            
    except ImportError as e:
        print(f"❌ Kokoro library not found: {e}")
        print("💡 Install with: pip install kokoro")
        return False
    except Exception as e:
        print(f"❌ Kokoro model test failed: {e}")
        return False

def check_dependencies():
    """Check all dependencies"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "websockets", "soundfile", "numpy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages: pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("Kokoro TTS Streaming Server Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check model
    if not check_model():
        print("\n❌ Model check failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🚀 All checks passed! Starting server...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("⚡ GPU Acceleration:", "Enabled" if gpu_available else "Disabled")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        from server import app
        
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
