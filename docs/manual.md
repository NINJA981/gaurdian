# 🔥 FORGE-Guard Instruction Manual

Welcome to the **FORGE-Guard** Elderly Monitoring System. This manual provides detailed instructions on how to set up, configure, and use the system to ensure the safety of your loved ones.

## 🚀 Quick Start Guide

1.  **Installation**: Ensure all dependencies are installed using `pip install -r requirements.txt`.
2.  **Start Services**: Run `python main.py` to start both the API server and the Dashboard.
3.  **Access Dashboard**: Open your browser and navigate to `http://localhost:8501`.

## 🎛️ Detection Modules

FORGE-Guard features four primary detection modules that can be toggled via the dashboard:

*   **🚨 Fall Detection**: Uses AI to detect sudden changes in posture and triggers a CRITICAL alert if a potential fall is identified.
*   **💊 Medicine Monitoring**: Monitors specific "medicine zones" and alerts you if the monitored person is seen (or not seen) taking their medication on schedule.
*   **✋ SOS Gesture Detection**: Allows the person to signal for help using a specific hand gesture.
*   **🎯 Object Detection**: Monitors for specific objects in the environment that might pose a risk.

## 📦 Zone Management

Zones allow you to define specific areas of interest in the camera's field of view.

*   **Creating a Zone**: Use the "Add New Zone" section in the sidebar. Specify a name (e.g., `medicine_shelf`), and set its coordinates (X, Y) and dimensions (Width, Height).
*   **Assigning Zones**: Ensure the zone name matches the expected name for specific modules (e.g., `medicine_area` for the Medicine Monitor).

## 📊 Monitoring & Alerts

*   **Live Feed**: The main dashboard shows a real-time video stream with detection overlays.
*   **Live Alerts**: High-priority alerts appear in red/orange on the right sidebar.
*   **Event Log**: A detailed history of all system events is available at the bottom right.

## 🛠️ Performance Tuning

The dashboard is optimized for efficiency:
*   **API Caching**: Some data is cached for a few seconds to reduce server load.
*   **Dynamic Refresh**: The page refreshes more frequently when activity is detected.

---
*For technical support, please refer to the project repository or contact the administrator.*
