#!/usr/bin/env python3
"""
FORGE-Guard - Elderly Safety Monitoring System
Production-Ready Main Entry Point

Usage:
    python main.py                  # Start both API and Dashboard
    python main.py --api-only       # Start only the API server
    python main.py --dashboard-only # Start only the Streamlit dashboard
    python main.py --check          # Check system dependencies
"""

import argparse
import subprocess
import sys
import os
import threading
import time
import signal
import logging
from pathlib import Path

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure production logging."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / "forge_guard.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("FORGE-Guard")

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def fix_windows_console():
    """Fix Windows console encoding for emojis."""
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            # Also set console to UTF-8
            os.system('chcp 65001 >nul 2>&1')
        except Exception:
            pass

def add_project_to_path():
    """Add project directory to Python path."""
    project_root = Path(__file__).parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version < (3, 9):
        print(f"[ERROR] Python {version.major}.{version.minor} is not supported.")
        print("[INFO] Please use Python 3.10 or 3.11 for best compatibility.")
        return False
    elif version >= (3, 13):
        print(f"[WARNING] Python {version.major}.{version.minor} may have limited ML support.")
        print("[INFO] Recommended: Python 3.10 or 3.11 for MediaPipe compatibility.")
    return True

def check_dependencies():
    """Check if all required dependencies are installed and functional."""
    dependencies = [
        ("cv2", "opencv-python", True),
        ("mediapipe", "mediapipe", False),  # Optional - graceful degradation
        ("fastapi", "fastapi", True),
        ("uvicorn", "uvicorn", True),
        ("streamlit", "streamlit", True),
        ("numpy", "numpy", True),
        ("pydantic", "pydantic", True),
        ("PIL", "Pillow", True),
        ("requests", "requests", True),
    ]
    
    print("\n🔍 Checking system dependencies...")
    print("=" * 50)
    
    all_ok = True
    warnings = []
    
    for module, package, required in dependencies:
        try:
            # Safe import check using subprocess for risky modules
            if module in ["numpy", "cv2", "mediapipe"]:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module}; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    if required:
                        print(f"   ❌ {package:<20} FAILED")
                        all_ok = False
                    else:
                        print(f"   ⚠️  {package:<20} NOT AVAILABLE (optional)")
                        warnings.append(package)
                    continue
            
            # Direct import check
            __import__(module)
            print(f"   ✅ {package:<20} OK")
            
        except ImportError:
            if required:
                print(f"   ❌ {package:<20} NOT FOUND")
                all_ok = False
            else:
                print(f"   ⚠️  {package:<20} NOT INSTALLED (optional)")
                warnings.append(package)
        except subprocess.TimeoutExpired:
            if required:
                print(f"   ❌ {package:<20} TIMEOUT")
                all_ok = False
            else:
                print(f"   ⚠️  {package:<20} TIMEOUT (optional)")
                warnings.append(package)
        except Exception as e:
            if required:
                print(f"   ❌ {package:<20} ERROR: {e}")
                all_ok = False
            else:
                print(f"   ⚠️  {package:<20} ERROR (optional)")
                warnings.append(package)
    
    print("=" * 50)
    
    if warnings:
        print("\n⚠️  Optional dependencies not available:")
        for pkg in warnings:
            print(f"   - {pkg}")
        print("\n[INFO] System will run in LIMITED MODE.")
        print("[INFO] Some detection features may be disabled.")
    
    if not all_ok:
        print("\n❌ Missing required dependencies!")
        print("[INFO] Run 'pip install -r requirements.txt' to install.")
        return False
    
    return True

# ============================================================================
# SERVER RUNNERS
# ============================================================================

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server."""
    import uvicorn
    from forge_guard.api.server import app
    
    print("\n" + "=" * 60)
    print("🔥 FORGE-Guard API Server")
    print("=" * 60)
    print(f"   Starting on http://{host}:{port}")
    print(f"   API Docs:     http://localhost:{port}/docs")
    print(f"   Video Stream: http://localhost:{port}/stream")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False
    )

def run_dashboard(port=8501):
    """Run the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "forge_guard" / "dashboard" / "app.py"
    
    print("\n" + "=" * 60)
    print("🔥 FORGE-Guard Dashboard")
    print("=" * 60)
    print(f"   Starting on http://localhost:{port}")
    print("=" * 60 + "\n")
    
    # Streamlit configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--theme.base", "dark",
        "--theme.primaryColor", "#ff6b00",
        "--theme.backgroundColor", "#0a0a0f",
        "--theme.secondaryBackgroundColor", "#14141e",
        "--theme.textColor", "#ffffff",
        "--browser.gatherUsageStats", "false",
    ]
    
    subprocess.run(cmd)

def run_both(api_port=8000, dash_port=8501):
    """Run both API server and dashboard."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║              🔥 FORGE-Guard Monitoring System 🔥                 ║")
    print("║                    Elderly Safety First                          ║")
    print("║                                                                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║                                                                  ║")
    print(f"║   📡 API Server:    http://localhost:{api_port:<5}                      ║")
    print(f"║   📊 Dashboard:     http://localhost:{dash_port:<5}                      ║")
    print(f"║   📖 API Docs:      http://localhost:{api_port}/docs                   ║")
    print(f"║   📹 Video Stream:  http://localhost:{api_port}/stream                 ║")
    print("║   ⚙️  Admin Panel:   Dashboard → Settings                        ║")
    print("║                                                                  ║")
    print("║   Press Ctrl+C to stop                                           ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Start API server in background thread
    api_thread = threading.Thread(
        target=run_api_server,
        kwargs={"host": "0.0.0.0", "port": api_port},
        daemon=True
    )
    api_thread.start()
    
    # Wait for API to initialize
    time.sleep(3)
    
    # Start dashboard (blocking)
    run_dashboard(port=dash_port)

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down FORGE-Guard...")
        print("   Cleaning up resources...")
        time.sleep(0.5)
        print("   Goodbye! Stay safe! 🔥\n")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for FORGE-Guard."""
    # Initial setup
    fix_windows_console()
    add_project_to_path()
    
    # Parse arguments
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
        "--api-port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--dash-port",
        type=int,
        default=8501,
        help="Dashboard port (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if args.check:
        if check_dependencies():
            print("\n✅ All required dependencies installed!")
            print("[INFO] Optional dependencies may enhance functionality.")
        sys.exit(0)
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        print("[INFO] Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup logging and signal handlers
    logger = setup_logging()
    setup_signal_handlers()
    
    # Run requested mode
    try:
        if args.api_only:
            run_api_server(port=args.api_port)
        elif args.dashboard_only:
            run_dashboard(port=args.dash_port)
        else:
            run_both(api_port=args.api_port, dash_port=args.dash_port)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
