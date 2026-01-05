# 🔥 FORGE-Guard

## Elderly Safety Monitoring System

**AI-Powered Real-Time Protection for Your Loved Ones**

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

<p align="center">
  <img src="https://img.shields.io/badge/🚨-Fall%20Detection-red?style=for-the-badge" alt="Fall Detection"/>
  <img src="https://img.shields.io/badge/🖐️-Gesture%20Recognition-orange?style=for-the-badge" alt="Gesture Recognition"/>
  <img src="https://img.shields.io/badge/💊-Medicine%20Tracking-blue?style=for-the-badge" alt="Medicine Tracking"/>
  <img src="https://img.shields.io/badge/👁️-Object%20Detection-purple?style=for-the-badge" alt="Object Detection"/>
</p>

---

## ⚡ One-Click Setup

### Windows
```batch
# Just run the setup script!
setup.bat
```

### Then start the system:
```batch
run.bat
```

That's it! The system will:
1. ✅ Create a virtual environment
2. ✅ Install all dependencies
3. ✅ Configure settings
4. ✅ Launch the dashboard

---

## 🎯 Features

### 🚨 Fall Detection
- **MediaPipe Pose Estimation** for accurate body tracking
- **Real-time monitoring** with instant alerts
- **Confirmation system** to reduce false positives
- **Configurable sensitivity** via Admin Panel

### 🖐️ Gesture Recognition
- **SOS Signal**: Wave both hands above head
- **Help Request**: Raise one hand and hold
- **Thumbs Up**: Acknowledgment gesture
- **Configurable hold time** for confirmation

### 💊 Medicine Monitoring
- **Zone-based tracking** - click to create zones
- **Visual change detection** in medicine areas
- **Schedule tracking** for medication compliance
- **Caregiver notifications** when medicine is taken

### 👁️ Object Detection (YOLO)
- **Person detection** for room occupancy
- **Safety object detection**
- **Real-time inference** with YOLOv8

### 🔔 Multi-Channel Alerts
- **Dashboard notifications** with sound
- **SMS alerts** via Twilio
- **Voice calls** for critical alerts
- **Email notifications** (configurable)

---

## 🖥️ Screenshots

### Dashboard
```
┌──────────────────────────────────────────────────────────┐
│                 🔥 FORGE-Guard Dashboard                 │
├──────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │🚨 FALL  │ │🖐️ GESTURE│ │💊 MEDS  │ │👁️ DETECT │        │
│  │ ACTIVE  │ │ ACTIVE  │ │ ACTIVE  │ │ ACTIVE  │        │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
├──────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐  │
│  │                                                    │  │
│  │              📹 LIVE VIDEO FEED                    │  │
│  │                                                    │  │
│  │    [Click to create monitoring zones]              │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  📋 Event Log               🚨 Alerts                    │
│  ─────────────              ─────────                    │
│  10:23:45 | Fall Detector   ✅ No active alerts         │
│  10:23:44 | System          System online               │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Requirements

### Recommended Setup
- **Python 3.10 or 3.11** (for MediaPipe compatibility)
- **Webcam** (built-in or USB)
- **4GB+ RAM** recommended
- **Windows 10/11** (macOS/Linux also supported)

### Dependencies (auto-installed)
- OpenCV, MediaPipe, NumPy
- FastAPI, Uvicorn, Streamlit
- Ultralytics (YOLOv8)
- Twilio (optional, for SMS/calls)

---

## 📦 Manual Installation

If you prefer manual setup:

```bash
# 1. Clone the repository
git clone https://github.com/NINJA981/gaurdian.git
cd gaurdian

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python main.py
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or edit the existing one:

```env
# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501

# Detection Settings
FALL_DETECTION_ENABLED=true
GESTURE_DETECTION_ENABLED=true
MEDICINE_MONITORING_ENABLED=true
OBJECT_DETECTION_ENABLED=true

# Alert Settings
ALERT_COOLDOWN_SECONDS=30

# Twilio (Optional - for SMS/Call alerts)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
EMERGENCY_CONTACT=+1234567890
```

### Admin Panel

Access all settings through the web interface:
1. Open http://localhost:8501
2. Click **⚙️ Admin Panel** in sidebar
3. Login with password: `forge2024` (change this!)
4. Configure all detection parameters

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status page |
| `/stream` | GET | MJPEG video stream |
| `/api/status` | GET | System status |
| `/api/events` | GET | Recent events |
| `/api/alerts` | GET | Recent alerts |
| `/api/config` | GET/PATCH | Configuration |
| `/api/zones` | GET/POST/DELETE | Zone management |
| `/ws/events` | WebSocket | Real-time events |

Full API documentation: http://localhost:8000/docs

---

## 🗂️ Project Structure

```
forge-guard/
├── 📄 main.py              # Application entry point
├── 📄 setup.bat            # One-click Windows setup
├── 📄 run.bat              # Quick start script
├── 📄 requirements.txt     # Python dependencies
├── 📄 .env.example         # Environment template
│
├── 📁 forge_guard/         # Main package
│   ├── 📄 config.py        # Configuration management
│   │
│   ├── 📁 detectors/       # AI Detection modules
│   │   ├── 📄 fall_detector.py      # Fall detection
│   │   ├── 📄 gesture_detector.py   # Gesture recognition
│   │   ├── 📄 medicine_monitor.py   # Medicine tracking
│   │   └── 📄 object_detector.py    # YOLO detection
│   │
│   ├── 📁 dashboard/       # Streamlit UI
│   │   └── 📄 app.py       # Main dashboard
│   │
│   ├── 📁 api/             # FastAPI backend
│   │   └── 📄 server.py    # REST API
│   │
│   ├── 📁 alerts/          # Notification system
│   │   ├── 📄 notification_manager.py
│   │   └── 📄 event_logger.py
│   │
│   └── 📁 pipeline/        # Video processing
│       ├── 📄 video_pipeline.py
│       └── 📄 frame_buffer.py
│
├── 📁 tests/               # Unit tests
├── 📁 logs/                # Application logs
└── 📁 docs/                # Documentation
```

---

## 🔧 Troubleshooting

### MediaPipe Not Working?

MediaPipe requires Python 3.10 or 3.11. Check your version:
```bash
python --version
```

If using Python 3.12+, install compatible versions:
```bash
pip install mediapipe==0.10.9
```

### Camera Not Detected?

1. Check camera is connected
2. Close other apps using the camera
3. Try a different camera index in settings

### Port Already in Use?

```bash
# Find process using port 8501
netstat -ano | findstr :8501

# Kill the process
taskkill /PID <process_id> /F
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose/hand estimation
- **Ultralytics** for YOLOv8
- **Streamlit** for the dashboard framework
- **FastAPI** for the backend API

---

<p align="center">
  <strong>Built with ❤️ for elderly safety</strong><br>
  <em>FORGE-Guard - Because every second counts</em>
</p>
