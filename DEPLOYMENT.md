# 🛠️ FORGE-Guard: A to Z Production Deployment Guide

This document provides a comprehensive roadmap for deploying the **FORGE-Guard Elderly Monitoring System** in a production environment. It covers infrastructure, networking, security, and external service integrations.

---

## 🏗️ Phase A: Infrastructure & Hardware

### 1. Host Machine Requirements
- **OS**: Ubuntu 22.04 LTS (Recommended) or Windows 10/11 with Docker Desktop.
- **CPU**: 4+ Cores (needed for real-time video processing).
- **RAM**: 8GB Minimum (16GB recommended for stability).
- **GPU**: NVIDIA GPU with CUDA support (Optional, but highly recommended for 60FPS+ YOLOv8 detection).

### 2. Camera Hardware
- **Integrated Webcam**: Best for local testing.
- **IP Cameras / RTSP**: For production, use IP cameras.
    - **Protocol**: RTSP (Real Time Streaming Protocol).
    - **Format**: H.264.
    - **Connection**: Update `CAMERA_INDEX` in `.env` to the RTSP URL: `rtsp://username:password@camera_ip:554/stream`.

---

## 🔑 Phase B: External Services & API Keys

To work fully, FORGE-Guard needs to connect to several external services:

### 1. Twilio (Alerts)
- **Account SID & Auth Token**: Needed for SMS and Voice call alerts.
- **Phone Number**: A Twilio-purchased number to send from.
- **Emergency Contact**: The number of the caregiver/relative.

### 2. Supabase / ForgeCloud (Persistence)
- **Project URL & API Key**: If you plan to sync events to the cloud.
- **Setup**: Create a project at [supabase.com](https://supabase.com) and add the keys to your environment.

### 3. Google/OpenAI (Optional AI Insights)
- If you use the "Module Improvement Intelligence" or advanced gesture logic, you may need an LLM API Key.

---

## ⚙️ Phase C: Environment Configuration (.env)

The `.env` file is the brain of your deployment. **Never commit this file to Git.**

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TWILIO_ACCOUNT_SID` | Your Twilio SID | `""` |
| `TWILIO_AUTH_TOKEN` | Your Twilio Token | `""` |
| `EMERGENCY_CONTACT` | Caregiver's Phone Number | `""` |
| `CAMERA_INDEX` | 0 for webcam or RTSP URL | `0` |
| `VIDEO_WIDTH` | Video stream width | `1280` |
| `VIDEO_HEIGHT` | Video stream height | `720` |
| `ALERT_COOLDOWN` | Wait time between alerts (sec) | `300` |
| `API_HOST` | Host for FastAPI | `0.0.0.0` |
| `API_PORT` | Port for FastAPI | `8000` |

---

## 🐳 Phase D: Container Orchestration (Docker)

Using Docker ensures the app runs the same on any server.

### 1. Setup
Ensure `docker` and `docker-compose` are installed:
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-compose-plugin
```

### 2. Deployment Command
```bash
# Build and start in detached mode
docker compose up -d --build
```

### 3. Container Services
- **api**: The FastAPI backend processing video.
- **dashboard**: The Streamlit frontend.
- **nginx**: The entry point handling traffic.

---

## 🌐 Phase E: Networking & Security

### 1. Reverse Proxy (Nginx)
The provided `nginx.conf` maps services to port 80:
- `http://your-server-ip/` -> Dashboard.
- `http://your-server-ip/api/` -> Backend.

### 2. SSL/HTTPS (CRITICAL)
For production, you **MUST** use HTTPS.
1. Obtain a certificate via **Certbot/Let's Encrypt**.
2. Update `nginx.conf` to listen on port 443.
3. Mount the certificates into the Nginx container volumes.

### 3. Port Forwarding
If hosting from home/local network:
1. Access your router settings.
2. Forward port **80** and **443** to the host machine's local IP.

---

## 📈 Phase F: Scaling for Production

### 1. Uvicorn Workers
For high traffic, increase workers in the `api` Dockerfile:
```bash
CMD ["uvicorn", "forge_guard.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Log Rotation
Logs are stored in `logs/forge_guard.log`. To prevent disk overflow:
1. Configure `logrotate` on the host machine.
2. Or use Docker's built-in logging driver limits:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## 🚦 Phase G: Maintenance & Monitoring

### 1. Health Checks
Monitor the API health via:
`GET http://your-server-ip/api/health`

### 2. Updates
To deploy code changes:
```bash
git pull origin main
docker compose up -d --build
```

### 3. Troubleshooting
- **No Video**: Check `CAMERA_INDEX` and ensure Docker has access to `/dev/video0`.
- **No Alerts**: Check Twilio logs and account balance.
- **Slow Performance**: Lower `VIDEO_FPS` to `15` or reduce `VIDEO_WIDTH`.

---

**FORGE-Guard** is now ready for world-class elderly care. 🚀
