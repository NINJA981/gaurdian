# FORGE-Guard | Elderly Monitoring System

<div align="center">

![FORGE-Guard](https://img.shields.io/badge/FORGE--Guard-v1.0.0-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Edge_AI-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal?style=for-the-badge&logo=fastapi)

**Real-time, modular elderly monitoring system with web dashboard**

</div>

---

## 🔥 Features

- **Fall Detection** - MediaPipe Pose + geometry analysis (ratio < 0.8 for 5 frames)
- **Medicine Box Monitoring** - ROI-based background subtraction
- **Emergency SOS Gesture** - Open palm detection (3-second hold)
- **Object Detection** - YOLOv8-nano for persons, wheelchairs, walking sticks
- **Real-time Alerts** - Twilio SMS/Call + local logging
- **Web Dashboard** - Streamlit UI with FORGE dark theme
- **Multi-threaded Pipeline** - Producer-consumer pattern at 30 FPS

---

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project
cd "hackathon project redo"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your Twilio credentials (optional)
```

### Running

```bash
# Start both API and Dashboard
python main.py

# Or start separately:
python main.py --api-only       # API only (port 8000)
python main.py --dashboard-only # Dashboard only (port 8501)
```

### Access

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Video Stream**: http://localhost:8000/stream

---

## 📁 Project Structure

```
forge_guard/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── forge_guard/
│   ├── config.py          # Configuration management
│   ├── pipeline/          # Video processing
│   │   ├── video_pipeline.py
│   │   └── frame_buffer.py
│   ├── detectors/         # Detection modules
│   │   ├── base_detector.py
│   │   ├── fall_detector.py
│   │   ├── medicine_monitor.py
│   │   ├── gesture_detector.py
│   │   └── object_detector.py
│   ├── alerts/            # Notification system
│   │   ├── notification_manager.py
│   │   └── event_logger.py
│   ├── api/               # FastAPI backend
│   │   └── server.py
│   └── dashboard/         # Streamlit UI
│       └── app.py
└── tests/                 # Unit tests
```

---

## 🎛️ Detection Modules

| Module | Technology | Trigger Condition |
|--------|------------|-------------------|
| Fall Detection | MediaPipe Pose | Width/Height ratio < 0.8 for 5 frames |
| Medicine Monitor | Background Subtraction | Pixel change > 20% in ROI |
| SOS Gesture | MediaPipe Hands | Open palm held for 3 seconds |
| Object Detection | YOLOv8-nano | Confidence > 50% |

---

## 🔧 Configuration

Edit `.env` or environment variables:

```env
# Twilio (optional)
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1234567890
EMERGENCY_CONTACT_NUMBER=+0987654321

# Detection thresholds
FALL_RATIO_THRESHOLD=0.8
GESTURE_HOLD_SECONDS=3
MEDICINE_CHANGE_THRESHOLD=0.2

# Video settings
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
VIDEO_FPS=30
```

---

## 📹 Setting Up Zones

1. Open dashboard at http://localhost:8501
2. In sidebar, expand "➕ Add New Zone"
3. Enter zone name (e.g., "medicine_tray")
4. Set X, Y, Width, Height coordinates
5. Click "Create Zone"
6. System auto-captures reference on first detection

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_fall_detector.py -v
```

---

## ⚡ Performance

Optimized for edge devices:
- **Laptop/Desktop**: 30 FPS @ 1280x720
- **Raspberry Pi 4**: 15-20 FPS @ 640x480
- **YOLOv8-nano**: 3MB model, ~20ms inference

For Raspberry Pi, update `.env`:
```env
VIDEO_WIDTH=640
VIDEO_HEIGHT=480
VIDEO_FPS=15
```

---

## 🛡️ License

MIT License - See LICENSE file

---

<div align="center">
Built with 🔥 by FORGE-Guard Team
</div>
