#!/usr/bin/env python3
"""
FORGE-Guard - Elderly Monitoring System
Main entry point for the application.

Usage:
    python main.py                  # Start both API and Dashboard
    python main.py --api-only       # Start only the API server
    python main.py --dashboard-only # Start only the Streamlit dashboard
"""

import argparse
import subprocess
import sys
import os
import threading
import time
import signal

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass  # Ignore if reconfigure not available

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_api_server():
    """Run the FastAPI server."""
    import uvicorn
    from forge_guard.api.server import app
    from forge_guard.config import config
    
    print("\n" + "=" * 60)
    print("🔥 FORGE-Guard API Server")
    print("=" * 60)
    print(f"Starting on http://{config.server.api_host}:{config.server.api_port}")
    print("API Docs: http://localhost:8000/docs")
    print("Video Stream: http://localhost:8000/stream")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=config.server.api_host,
        port=config.server.api_port,
        log_level="info"
    )


def run_dashboard():
    """Run the Streamlit dashboard."""
    from forge_guard.config import config
    
    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        "forge_guard", "dashboard", "app.py"
    )
    
    print("\n" + "=" * 60)
    print("🔥 FORGE-Guard Dashboard")
    print("=" * 60)
    print(f"Starting on http://localhost:{config.server.streamlit_port}")
    print("=" * 60 + "\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(config.server.streamlit_port),
        "--server.headless", "true",
        "--theme.base", "dark",
        "--theme.primaryColor", "#ff8000"
    ])


def run_both():
    """Run both API server and dashboard."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            🔥 FORGE-Guard Monitoring System 🔥           ║")
    print("║                  Elderly Safety First                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Starting services...                                    ║")
    print("║                                                          ║")
    print("║  📡 API Server:    http://localhost:8000                 ║")
    print("║  📊 Dashboard:     http://localhost:8501                 ║")
    print("║  📖 API Docs:      http://localhost:8000/docs            ║")
    print("║  📹 Video Stream:  http://localhost:8000/stream          ║")
    print("║                                                          ║")
    print("║  Press Ctrl+C to stop                                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Start API server in a thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    time.sleep(3)
    
    # Start dashboard (this will block)
    run_dashboard()


def check_dependencies():
    """Check if all required dependencies are installed and FUNCTIONAL."""
    missing = []
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
    ]
    
    print("\n🔍 Checking system dependencies...")
    
    for module, package in dependencies:
        try:
            # First fast check (import in current process)
            # Dangerous if module crashes on import (like numpy MINGW issue)
            # So we use a subprocess check for risky modules
            if module in ["numpy", "cv2", "mediapipe"]:
                # Run a subprocess to test import safely
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module}"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode != 0:
                    print(f"   ❌ {package} CRASHED/FAILED during import check.")
                    missing.append(package)
                    continue
            
            # If safe or passed subprocess, try import here to be sure (and for non-risky ones)
            __import__(module)
            print(f"   ✅ {package}")
            
        except ImportError:
            print(f"   ❌ {package} not found.")
            missing.append(package)
        except Exception as e:
            print(f"   ❌ {package} error: {e}")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Missing or Unstable dependencies - System will run in LIMITED MODE:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nNote: Some detection features (Fall, Gesture) will be disabled.")
        return True  # Allow proceeding in Limited Mode
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FORGE-Guard Elderly Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Start both API and Dashboard
  python main.py --api-only       Start only API server
  python main.py --dashboard-only Start only Streamlit dashboard
  python main.py --check          Check dependencies
        """
    )
    
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the FastAPI server"
    )
    
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Start only the Streamlit dashboard"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check:
        if check_dependencies():
            print("✅ All dependencies installed!")
        sys.exit(0)
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down FORGE-Guard...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run requested mode
    if args.api_only:
        run_api_server()
    elif args.dashboard_only:
        run_dashboard()
    else:
        run_both()


if __name__ == "__main__":
    main()
