"""
FORGE-Guard Streamlit Dashboard
Web-based monitoring interface with live video feed and controls.
"""

import streamlit as st
import requests
import time
import json
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="FORGE-Guard | Elderly Monitoring",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FORGE theme
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS if file not found
        st.markdown("<style>.stApp { background: #0d0d1a; }</style>", unsafe_allow_html=True)

load_css()

# API Base URL
API_BASE = "http://localhost:8000"

@st.cache_data(ttl=2) # Cache for 2 seconds to boost performance
def fetch_status():
    """Fetch system status from API."""
    try:
        response = requests.get(f"{API_BASE}/api/status", timeout=1)
        return response.json()
    except:
        return None

@st.cache_data(ttl=5) # Events can be slightly older
def fetch_events(count=10):
    """Fetch recent events from API."""
    try:
        response = requests.get(f"{API_BASE}/api/events?count={count}", timeout=1)
        return response.json()
    except:
        return []

@st.cache_data(ttl=2)
def fetch_alerts(count=10):
    """Fetch recent alerts from API."""
    try:
        response = requests.get(f"{API_BASE}/api/alerts?count={count}", timeout=1)
        return response.json()
    except:
        return []

@st.cache_data(ttl=10)
def fetch_zones():
    """Fetch defined zones from API."""
    try:
        response = requests.get(f"{API_BASE}/api/zones", timeout=1)
        return response.json()
    except:
        return []

@st.cache_data(ttl=30) # Config changes rarely
def fetch_config():
    """Fetch configuration from API."""
    try:
        response = requests.get(f"{API_BASE}/api/config", timeout=1)
        return response.json()
    except:
        return {}


def create_zone(name, x, y, width, height):
    """Create a new zone via API."""
    try:
        response = requests.post(
            f"{API_BASE}/api/zones",
            json={"name": name, "x": x, "y": y, "width": width, "height": height},
            timeout=2
        )
        st.cache_data.clear() # Clear cache on write
        return response.status_code == 200
    except:
        return False


def delete_zone(name):
    """Delete a zone via API."""
    try:
        response = requests.delete(f"{API_BASE}/api/zones/{name}", timeout=2)
        st.cache_data.clear() # Clear cache on write
        return response.status_code == 200
    except:
        return False


def update_config(updates):
    """Update configuration via API."""
    try:
        response = requests.patch(
            f"{API_BASE}/api/config",
            json=updates,
            timeout=2
        )
        st.cache_data.clear() # Clear cache on write
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit application entry point."""
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🛠️ Navigation")
        page = st.radio("Go to", ["Dashboard", "Journal", "Instruction Manual"], label_visibility="collapsed")
        
        st.markdown("---")
        
        # System Status (Always visible)
        st.subheader("⚙️ System Status")
        status = fetch_status()
        
        if status and status.get("running"):
            st.success("System Online", icon="✅")
            
            if status.get("pipeline"):
                pipeline = status["pipeline"]
                col1, col2 = st.columns(2)
                p_fps = pipeline.get('producer_fps', 0)
                c_fps = pipeline.get('consumer_fps', 0)
                with col1:
                    st.metric("Inbound FPS", f"{p_fps:.1f}")
                with col2:
                    st.metric("Outbound FPS", f"{c_fps:.1f}")
                
                # Performance Visualizer
                st.write("Processing Health")
                st.progress(min(c_fps / 30.0, 1.0), text=f"Efficiency: {int((c_fps/30)*100)}%")
        else:
            st.error("System Offline", icon="❌")
            st.info("Start the API server: `python main.py`")
        
        st.markdown("---")

    if page == "Dashboard":
        render_dashboard(status)
    elif page == "Journal":
        render_journal()
    elif page == "Instruction Manual":
        render_manual()


def render_journal():
    st.header("📜 Activity Journal")
    st.markdown("Review past detection events and system logs.")
    
    # Import here to avoid circular dependencies if any
    try:
        from forge_guard.utils.journal import JournalManager
        journal = JournalManager()
        events = journal.get_events(limit=100)
    except Exception as e:
        st.error(f"Failed to load journal: {e}")
        return

    if not events:
        st.info("No events found in journal.")
        return

    # Convert to dataframe for better display? Or custom list
    # Custom list for Forge theme
    for event in events:
        level = event.get('level', 'INFO')
        color = "#00c8ff" # Info
        if level == "WARNING": color = "#ffc800"
        if level == "CRITICAL": color = "#ff3232"
        
        dt = datetime.fromtimestamp(event.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S")
        
        st.markdown(f"""
        <div style="background: rgba(30,30,50,0.6); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {color};">
            <div style="display: flex; justify_content: space-between;">
                <strong>{event.get('source')}</strong>
                <span style="color: #666;">{dt}</span>
            </div>
            <div style="margin-top: 5px; color: #ddd;">
                {event.get('message')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_manual():
    st.header("📖 Instruction Manual")
    st.markdown("""
    ### Quick Start
    1. **Start System**: Run `python main.py`
    2. **View Dashboard**: Open `http://localhost:8501`
    
    ### Zones
    - Go to **Dashboard** > **Zone Management** in sidebar.
    - Create zones for specific monitoring (e.g., 'medicine_table').
    
    ### Troubleshooting
    - **No Video**: Check if another app is using the webcam.
    - **Slow Performance**: Reduce resolution in `.env`.
    """)

def render_dashboard(status):
    # Sidebar Module Controls (Only show on Dashboard)
    with st.sidebar:
        st.markdown("### 🎛️ Detection Modules")
        config_data = fetch_config() # Avoid shadowing 'config' module if any
        
        fall_enabled = st.toggle(
            "Fall Detection",
            value=config_data.get("fall_detection_enabled", True),
            key="fall_toggle"
        )
        
        medicine_enabled = st.toggle(
            "Medicine Monitoring",
            value=config_data.get("medicine_monitoring_enabled", True),
            key="medicine_toggle"
        )
        
        gesture_enabled = st.toggle(
            "Gesture Detection",
            value=config_data.get("gesture_detection_enabled", True),
            key="gesture_toggle"
        )
        
        object_enabled = st.toggle(
            "Object Detection",
            value=config_data.get("object_detection_enabled", True),
            key="object_toggle"
        )
        
        if st.button("Apply Changes", use_container_width=True):
            updates = {
                "fall_detection_enabled": fall_enabled,
                "medicine_monitoring_enabled": medicine_enabled,
                "gesture_detection_enabled": gesture_enabled,
                "object_detection_enabled": object_enabled
            }
            if update_config(updates):
                st.success("Configuration updated!")
            else:
                st.error("Failed to update")
        
        st.markdown("---")
        
        # Zone Management
        st.subheader("📦 Zone Management")
        
        zones = fetch_zones()
        if zones:
            for zone in zones:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📍 {zone['name']}")
                with col2:
                    if st.button("🗑️", key=f"del_{zone['name']}"):
                        delete_zone(zone['name'])
                        st.rerun()
        else:
            st.info("No zones defined")
        
        with st.expander("➕ Add New Zone"):
            zone_name = st.text_input("Zone Name", "medicine_area")
            col1, col2 = st.columns(2)
            with col1:
                zone_x = st.number_input("X", 0, 1920, 100)
                zone_width = st.number_input("Width", 10, 1000, 200)
            with col2:
                zone_y = st.number_input("Y", 0, 1080, 100)
                zone_height = st.number_input("Height", 10, 1000, 150)
            
            if st.button("Create Zone", use_container_width=True):
                if create_zone(zone_name, zone_x, zone_y, zone_width, zone_height):
                    st.success(f"Zone '{zone_name}' created!")
                    st.rerun()
                else:
                    st.error("Failed to create zone")
    
    # Main content area (Video & Stats)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video Feed
        st.subheader("📹 Live Monitoring Feed")
        
        try:
            # Using HTML embed for lower latency vs Streamlit image component
            st.markdown(f"""
            <div class="video-container">
                <img src="{API_BASE}/stream" width="100%" 
                     style="display: block; border-radius: 10px;"
                     onerror="this.style.display='none'">
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("Video feed not available. Ensure the API server is running.")
        
        # Quick Stats
        st.markdown("---")
        st.subheader("📊 Detection Analytics")
        
        if status and status.get("detectors"):
            detectors = status["detectors"]
            
            d_cols = st.columns(4)
            
            metrics = [
                ("🚨 Fall", "fall_detection", "detection_count"),
                ("💊 Meds", "medicine_monitor", "detection_count"),
                ("✋ SOS", "gesture_detection", "detection_count"),
                ("🎯 Obj", "object_detection", "detection_count")
            ]
            
            for i, (label, key, val_key) in enumerate(metrics):
                m_stats = detectors.get(key, {})
                count = m_stats.get(val_key, 0)
                time_ms = m_stats.get("avg_processing_time_ms", 0)
                
                with d_cols[i]:
                    st.markdown(f"""
                    <div class="status-card">
                        <h3>{label}</h3>
                        <div class="value">{count}</div>
                        <small>Proc: {time_ms:.1f}ms</small>
                    </div>
                    """, unsafe_allow_html=True)
                    # Dynamic Progress bar for processing load
                    st.progress(min(time_ms / 100.0, 1.0))
    
    with col2:
        # Alerts
        st.subheader("🚨 Live Alerts")
        
        alerts = fetch_alerts(5)
        
        if alerts:
            for alert in reversed(alerts):
                priority = alert.get("priority", "MEDIUM")
                css_class = "alert-critical" if priority == "CRITICAL" else \
                           "alert-warning" if priority in ["HIGH", "WARNING"] else "alert-info"
                
                timestamp = datetime.fromtimestamp(alert.get("timestamp", 0))
                time_str = timestamp.strftime("%H:%M:%S")
                
                st.markdown(f"""
                <div class="alert-item {css_class}">
                    <strong>{time_str}</strong> - {alert.get('source', 'unknown')}<br>
                    {alert.get('message', '')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts")
        
        st.markdown("---")
        
        # Event Log
        with st.expander("📋 System Event Log", expanded=True):
            events = fetch_events(20)
            
            if events:
                st.markdown('<div class="event-log">', unsafe_allow_html=True)
                for event in reversed(events):
                    timestamp = datetime.fromtimestamp(event.get("timestamp", 0))
                    time_str = timestamp.strftime("%H:%M:%S")
                    
                    st.markdown(f"""
                    <div class="event-item">
                        <span style="color: #666; min-width: 65px;">{time_str}</span>
                        <span style="color: var(--primary); min-width: 90px; font-weight: bold;">{event.get('source', 'sys')}</span>
                        <span style="color: #ccc;">{event.get('message', '')[:60]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Collecting system events...")
    
    # Auto-refresh with dynamic interval
    # Faster refresh if system is active, slower otherwise
    if status and status.get("running"):
        time.sleep(1.5)
    else:
        time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file directly, not as __main__
    main()

