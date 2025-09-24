#!/usr/bin/env python3
"""
QuteVoice TTS - One-Click Startup Script
This script provides the easiest way to run the QuteVoice TTS application
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main startup function"""
    print("🎤 QuteVoice TTS - One-Click Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found!")
        print("💡 Please run this script from the QuteVoice directory")
        print("📁 Make sure you're in the folder containing app.py")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"💡 You're running Python {sys.version}")
        print("🔗 Download Python from: https://python.org")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Run the main application
    try:
        print("🚀 Starting QuteVoice TTS Application...")
        print("=" * 50)
        
        # Import and run the main app
        import app
        
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        return True
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("💡 Please check the error messages above")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)
