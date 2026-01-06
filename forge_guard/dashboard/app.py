"""
FORGE-Guard Enhanced Dashboard
Premium UI with Glassmorphism, Admin Panel, and Interactive Zone Creation
Production-ready with proper error handling and graceful degradation.
"""

import streamlit as st
import requests
import time
import json
import os
import logging
from datetime import datetime

# Optional imports with graceful degradation
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="FORGE-Guard | Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# FORGE-Guard\nElderly Safety Monitoring System v2.0"
    }
)

# ============================================================================
# PREMIUM CSS STYLING - Forge Theme with Glassmorphism
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    /* ===== ROOT VARIABLES ===== */
    :root {
        --forge-primary: #ff6b00;
        --forge-secondary: #ff8c00;
        --forge-accent: #ffa500;
        --forge-dark: #0a0a0f;
        --forge-darker: #050508;
        --forge-card: rgba(20, 20, 30, 0.8);
        --forge-glass: rgba(255, 107, 0, 0.1);
        --forge-border: rgba(255, 107, 0, 0.3);
        --forge-success: #00ff88;
        --forge-warning: #ffcc00;
        --forge-danger: #ff3366;
        --forge-info: #00ccff;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
    }
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        background: linear-gradient(135deg, var(--forge-darker) 0%, #0d0d1a 50%, #1a0a0a 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--forge-darker);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--forge-primary);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--forge-secondary);
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .glass-card {
        background: var(--forge-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--forge-border);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(255, 107, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* ===== HEADER STYLING ===== */
    .forge-header {
        background: linear-gradient(135deg, var(--forge-card) 0%, rgba(255, 107, 0, 0.05) 100%);
        border: 1px solid var(--forge-border);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .forge-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 107, 0, 0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    
    .forge-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--forge-primary) 0%, var(--forge-accent) 50%, #fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 40px rgba(255, 107, 0, 0.5);
    }
    
    .forge-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-top: 8px;
        position: relative;
        z-index: 1;
    }
    
    /* ===== STATUS INDICATORS ===== */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }
    
    .status-card {
        background: var(--forge-card);
        border: 1px solid var(--forge-border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: scale(1.02);
        border-color: var(--forge-primary);
    }
    
    .status-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    
    .status-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .status-active { color: var(--forge-success); }
    .status-warning { color: var(--forge-warning); }
    .status-danger { color: var(--forge-danger); }
    .status-info { color: var(--forge-info); }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--forge-primary) 0%, var(--forge-secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 0, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ===== VIDEO FEED CONTAINER ===== */
    .video-container {
        background: var(--forge-card);
        border: 2px solid var(--forge-border);
        border-radius: 16px;
        padding: 8px;
        position: relative;
        overflow: hidden;
    }
    
    .video-container::before {
        content: 'LIVE';
        position: absolute;
        top: 16px;
        right: 16px;
        background: var(--forge-danger);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        animation: blink 1.5s infinite;
        z-index: 10;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ===== EVENT LOG ===== */
    .event-log {
        background: var(--forge-card);
        border: 1px solid var(--forge-border);
        border-radius: 12px;
        padding: 16px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .event-item {
        display: flex;
        align-items: center;
        padding: 12px;
        margin: 8px 0;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        border-left: 3px solid var(--forge-primary);
        transition: background 0.2s ease;
    }
    
    .event-item:hover {
        background: rgba(255, 107, 0, 0.1);
    }
    
    .event-time {
        color: var(--text-secondary);
        font-size: 0.8rem;
        min-width: 70px;
    }
    
    .event-source {
        background: var(--forge-primary);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin: 0 12px;
        min-width: 60px;
        text-align: center;
    }
    
    .event-message {
        color: var(--text-primary);
        flex: 1;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--forge-card) 0%, rgba(10, 10, 15, 0.95) 100%);
        border-right: 1px solid var(--forge-border);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: var(--forge-card);
        border: 1px solid var(--forge-border);
        border-radius: 12px;
        padding: 16px;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--forge-primary) !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--forge-card);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--forge-primary) 0%, var(--forge-secondary) 100%);
        color: white !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {
        background: var(--forge-primary) !important;
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: var(--forge-card);
        border-color: var(--forge-border);
        border-radius: 8px;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--forge-card);
        border: 1px solid var(--forge-border);
        border-radius: 12px;
    }
    
    /* ===== ZONE CREATION STYLES ===== */
    .zone-marker {
        position: absolute;
        border: 2px dashed var(--forge-primary);
        background: rgba(255, 107, 0, 0.2);
        border-radius: 8px;
    }
    
    /* ===== ADMIN PANEL ===== */
    .admin-section {
        background: var(--forge-card);
        border: 1px solid var(--forge-border);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .admin-section h3 {
        color: var(--forge-primary);
        border-bottom: 1px solid var(--forge-border);
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: slideIn 0.5s ease-out forwards;
    }
    
    /* ===== ALERT BANNER ===== */
    .alert-banner {
        background: linear-gradient(135deg, var(--forge-danger) 0%, #cc0033 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        animation: alertPulse 2s infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 51, 102, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 51, 102, 0.8); }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'api_url': 'http://localhost:8000',
        'current_page': 'dashboard',
        'zones': [],
        'zone_creation_mode': False,
        'zone_start_point': None,
        'alerts': [],
        'events': [],
        'last_frame': None,
        'settings': {
            'fall_detection': True,
            'gesture_detection': True,
            'medicine_monitoring': True,
            'object_detection': True,
            'alert_cooldown': 30,
            'fall_sensitivity': 0.7,
            'gesture_hold_time': 2.0,
            'video_quality': 80,
            'auto_refresh': True,
            'refresh_rate': 1.0,
            'dark_mode': True,
            'sound_alerts': True,
            'email_alerts': False,
            'sms_alerts': False,
        },
        'admin_authenticated': False,
        'admin_password': 'forge2024',  # Default admin password
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# API COMMUNICATION
# ============================================================================

def api_request(endpoint, method='GET', data=None, timeout=5):
    """Make API request with error handling."""
    try:
        url = f"{st.session_state.api_url}{endpoint}"
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        elif method == 'PATCH':
            response = requests.patch(url, json=data, timeout=timeout)
        elif method == 'DELETE':
            response = requests.delete(url, timeout=timeout)
        
        if response.ok:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return None

def get_system_status():
    """Get current system status from API."""
    return api_request('/api/status')

def get_recent_events(count=20):
    """Get recent events from API."""
    return api_request(f'/api/events?count={count}')

def get_recent_alerts(count=10):
    """Get recent alerts from API."""
    return api_request(f'/api/alerts?count={count}')

def update_config(config_data):
    """Update system configuration."""
    return api_request('/api/config', method='PATCH', data=config_data)

def create_zone(zone_data):
    """Create a new monitoring zone."""
    return api_request('/api/zones', method='POST', data=zone_data)

def delete_zone(zone_name):
    """Delete a monitoring zone."""
    return api_request(f'/api/zones/{zone_name}', method='DELETE')

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="forge-header">
        <h1 class="forge-title">🔥 FORGE-Guard</h1>
        <p class="forge-subtitle">Elderly Safety Monitoring System • Real-Time AI Protection</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_cards(status):
    """Render status indicator cards."""
    detectors = status.get('detectors', {}) if status else {}
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fall_status = detectors.get('fall_detection', {})
        is_active = fall_status.get('enabled', False) if fall_status else False
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">🚨</div>
            <div class="status-label">Fall Detection</div>
            <div class="status-value {'status-active' if is_active else 'status-warning'}">
                {'ACTIVE' if is_active else 'OFFLINE'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gesture_status = detectors.get('gesture_detection', {})
        is_active = gesture_status.get('enabled', False) if gesture_status else False
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">🖐️</div>
            <div class="status-label">Gesture Detection</div>
            <div class="status-value {'status-active' if is_active else 'status-warning'}">
                {'ACTIVE' if is_active else 'OFFLINE'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        medicine_status = detectors.get('medicine_monitor', {})
        is_active = medicine_status.get('enabled', False) if medicine_status else False
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">💊</div>
            <div class="status-label">Medicine Monitor</div>
            <div class="status-value {'status-active' if is_active else 'status-warning'}">
                {'ACTIVE' if is_active else 'OFFLINE'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        object_status = detectors.get('object_detection', {})
        is_active = object_status.get('enabled', False) if object_status else False
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">👁️</div>
            <div class="status-label">Object Detection</div>
            <div class="status-value {'status-active' if is_active else 'status-warning'}">
                {'ACTIVE' if is_active else 'OFFLINE'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        is_running = status.get('running', False) if status else False
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">📡</div>
            <div class="status-label">System Status</div>
            <div class="status-value {'status-active' if is_running else 'status-danger'}">
                {'ONLINE' if is_running else 'OFFLINE'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_video_feed():
    """Render the video feed section with zone creation."""
    st.markdown("### 📹 Live Video Feed")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video feed placeholder
        video_placeholder = st.empty()
        
        # Try to get video frame
        try:
            response = requests.get(f"{st.session_state.api_url}/stream", 
                                   stream=True, timeout=2)
            # For now, show placeholder
            video_placeholder.markdown("""
            <div class="video-container">
                <div style="height: 480px; display: flex; align-items: center; 
                            justify-content: center; background: #000; border-radius: 12px;">
                    <div style="text-align: center;">
                        <p style="font-size: 4rem; margin: 0;">📹</p>
                        <p style="color: #888; margin-top: 16px;">Live Feed Active</p>
                        <p style="color: #666; font-size: 0.8rem;">
                            Open <a href="http://localhost:8000/stream" target="_blank" 
                                   style="color: #ff6b00;">Full Stream</a> in new tab
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except:
            video_placeholder.markdown("""
            <div class="video-container">
                <div style="height: 480px; display: flex; align-items: center; 
                            justify-content: center; background: #111; border-radius: 12px;">
                    <div style="text-align: center;">
                        <p style="font-size: 4rem; margin: 0;">⚠️</p>
                        <p style="color: #ff6b00; margin-top: 16px;">Camera Feed Unavailable</p>
                        <p style="color: #666; font-size: 0.8rem;">Make sure the API server is running</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Zone Controls")
        
        # Zone creation mode toggle
        if st.button("➕ Create Zone", use_container_width=True):
            st.session_state.zone_creation_mode = not st.session_state.zone_creation_mode
        
        if st.session_state.zone_creation_mode:
            st.info("🎯 Click on video to set zone corners")
            zone_name = st.text_input("Zone Name", placeholder="e.g., Medicine Cabinet")
            
            col_a, col_b = st.columns(2)
            with col_a:
                x = st.number_input("X", min_value=0, max_value=1280, value=100)
                width = st.number_input("Width", min_value=10, max_value=500, value=200)
            with col_b:
                y = st.number_input("Y", min_value=0, max_value=720, value=100)
                height = st.number_input("Height", min_value=10, max_value=500, value=200)
            
            if st.button("✅ Save Zone", use_container_width=True):
                if zone_name:
                    result = create_zone({
                        'name': zone_name,
                        'x': x, 'y': y,
                        'width': width, 'height': height
                    })
                    if result:
                        st.success(f"Zone '{zone_name}' created!")
                        st.session_state.zone_creation_mode = False
                    else:
                        st.error("Failed to create zone")
                else:
                    st.warning("Please enter a zone name")
        
        st.markdown("---")
        st.markdown("#### Active Zones")
        
        zones = api_request('/api/zones') or []
        if zones:
            for zone in zones:
                with st.expander(f"📍 {zone['name']}"):
                    st.text(f"Position: ({zone['x']}, {zone['y']})")
                    st.text(f"Size: {zone['width']}x{zone['height']}")
                    if st.button(f"🗑️ Delete", key=f"del_{zone['name']}"):
                        delete_zone(zone['name'])
                        st.rerun()
        else:
            st.caption("No zones defined yet")

def render_event_log():
    """Render the event log section."""
    st.markdown("### 📋 Event Log")
    
    events = get_recent_events(15) or []
    
    if events:
        st.markdown('<div class="event-log">', unsafe_allow_html=True)
        for event in reversed(events[-10:]):
            timestamp = datetime.fromtimestamp(event.get('timestamp', 0))
            time_str = timestamp.strftime("%H:%M:%S")
            source = event.get('source', 'system')
            message = event.get('message', '')
            
            st.markdown(f"""
            <div class="event-item">
                <span class="event-time">{time_str}</span>
                <span class="event-source">{source}</span>
                <span class="event-message">{message}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No events recorded yet. Events will appear here when detected.")

def render_alerts_section():
    """Render active alerts section."""
    st.markdown("### 🚨 Active Alerts")
    
    alerts = get_recent_alerts(5) or []
    
    if alerts:
        for alert in alerts:
            priority = alert.get('priority', 'MEDIUM')
            message = alert.get('message', 'Unknown alert')
            timestamp = datetime.fromtimestamp(alert.get('timestamp', 0))
            
            if priority == 'CRITICAL':
                st.markdown(f"""
                <div class="alert-banner">
                    <span style="font-size: 1.5rem; margin-right: 16px;">🚨</span>
                    <div>
                        <strong>CRITICAL ALERT</strong><br>
                        <span>{message}</span>
                        <span style="font-size: 0.8rem; opacity: 0.8;"> • {timestamp.strftime("%H:%M:%S")}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ {message} ({timestamp.strftime('%H:%M:%S')})")
    else:
        st.success("✅ No active alerts - All systems normal")

# ============================================================================
# ADMIN PANEL
# ============================================================================

def render_admin_panel():
    """Render the admin settings panel."""
    st.markdown("## ⚙️ Admin Panel")
    st.markdown("Configure all system settings from here.")
    
    # Admin authentication
    if not st.session_state.admin_authenticated:
        st.markdown("### 🔐 Admin Login")
        with st.form("admin_login"):
            password = st.text_input("Enter Admin Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if password == st.session_state.admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("✅ Authenticated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
        return
    
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # Settings tabs
    admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5 = st.tabs([
        "🎯 Detectors", "🔔 Alerts", "📹 Video", "🌐 Network", "🔒 Security"
    ])
    
    with admin_tab1:
        st.markdown("### Detection Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fall Detection")
            fall_enabled = st.toggle("Enable Fall Detection", 
                                    value=st.session_state.settings['fall_detection'],
                                    key="fall_toggle")
            fall_sensitivity = st.slider("Sensitivity", 0.1, 1.0, 
                                        st.session_state.settings['fall_sensitivity'],
                                        help="Higher = more sensitive")
            st.caption("Detects when a person falls using pose estimation")
            
            st.markdown("#### Gesture Detection")
            gesture_enabled = st.toggle("Enable Gesture Detection",
                                       value=st.session_state.settings['gesture_detection'],
                                       key="gesture_toggle")
            gesture_hold = st.slider("Hold Time (seconds)", 0.5, 5.0,
                                    st.session_state.settings['gesture_hold_time'])
            st.caption("Recognizes SOS and help gestures")
        
        with col2:
            st.markdown("#### Medicine Monitoring")
            medicine_enabled = st.toggle("Enable Medicine Monitor",
                                        value=st.session_state.settings['medicine_monitoring'],
                                        key="medicine_toggle")
            st.caption("Tracks medicine taking in defined zones")
            
            st.markdown("#### Object Detection")
            object_enabled = st.toggle("Enable Object Detection",
                                      value=st.session_state.settings['object_detection'],
                                      key="object_toggle")
            st.caption("YOLO-based object and person detection")
        
        if st.button("💾 Save Detection Settings", use_container_width=True):
            st.session_state.settings['fall_detection'] = fall_enabled
            st.session_state.settings['gesture_detection'] = gesture_enabled
            st.session_state.settings['medicine_monitoring'] = medicine_enabled
            st.session_state.settings['object_detection'] = object_enabled
            st.session_state.settings['fall_sensitivity'] = fall_sensitivity
            st.session_state.settings['gesture_hold_time'] = gesture_hold
            
            # Update API
            update_config({
                'fall_detection_enabled': fall_enabled,
                'gesture_detection_enabled': gesture_enabled,
                'medicine_monitoring_enabled': medicine_enabled,
                'object_detection_enabled': object_enabled,
            })
            st.success("✅ Settings saved!")
    
    with admin_tab2:
        st.markdown("### Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Alert Cooldown")
            cooldown = st.slider("Cooldown (seconds)", 5, 120, 
                               st.session_state.settings['alert_cooldown'],
                               help="Minimum time between alerts from same source")
            
            st.markdown("#### Sound Alerts")
            sound_enabled = st.toggle("Enable Sound Alerts",
                                     value=st.session_state.settings['sound_alerts'])
        
        with col2:
            st.markdown("#### Notification Channels")
            
            email_enabled = st.toggle("Email Notifications",
                                     value=st.session_state.settings['email_alerts'])
            if email_enabled:
                st.text_input("Email Address", placeholder="alert@example.com")
            
            sms_enabled = st.toggle("SMS Notifications",
                                   value=st.session_state.settings['sms_alerts'])
            if sms_enabled:
                st.text_input("Phone Number", placeholder="+1234567890")
        
        if st.button("💾 Save Alert Settings", use_container_width=True):
            st.session_state.settings['alert_cooldown'] = cooldown
            st.session_state.settings['sound_alerts'] = sound_enabled
            st.session_state.settings['email_alerts'] = email_enabled
            st.session_state.settings['sms_alerts'] = sms_enabled
            st.success("✅ Settings saved!")
    
    with admin_tab3:
        st.markdown("### Video Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Quality")
            video_quality = st.slider("JPEG Quality", 30, 100,
                                     st.session_state.settings['video_quality'])
            st.caption("Higher = better quality, more bandwidth")
            
            st.markdown("#### Camera")
            camera_source = st.selectbox("Camera Source", 
                                        ["Default (0)", "Camera 1", "Camera 2", "RTSP Stream"])
            if camera_source == "RTSP Stream":
                st.text_input("RTSP URL", placeholder="rtsp://...")
        
        with col2:
            st.markdown("#### Frame Rate")
            target_fps = st.slider("Target FPS", 5, 30, 15)
            
            st.markdown("#### Resolution")
            resolution = st.selectbox("Resolution", 
                                     ["640x480", "1280x720", "1920x1080"])
        
        if st.button("💾 Save Video Settings", use_container_width=True):
            st.session_state.settings['video_quality'] = video_quality
            st.success("✅ Settings saved!")
    
    with admin_tab4:
        st.markdown("### Network Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### API Server")
            api_host = st.text_input("Host", value="0.0.0.0")
            api_port = st.number_input("Port", value=8000, min_value=1, max_value=65535)
            
        with col2:
            st.markdown("#### Dashboard Server")
            dash_port = st.number_input("Dashboard Port", value=8501, 
                                       min_value=1, max_value=65535)
        
        st.markdown("#### Twilio Configuration (SMS/Calls)")
        with st.expander("Configure Twilio"):
            twilio_sid = st.text_input("Account SID", type="password")
            twilio_token = st.text_input("Auth Token", type="password")
            twilio_phone = st.text_input("Twilio Phone Number", placeholder="+1234567890")
            emergency_contact = st.text_input("Emergency Contact", placeholder="+1234567890")
        
        if st.button("💾 Save Network Settings", use_container_width=True):
            st.success("✅ Settings saved! Restart required for changes to take effect.")
    
    with admin_tab5:
        st.markdown("### Security Settings")
        
        st.markdown("#### Change Admin Password")
        with st.form("change_password"):
            current_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Change Password"):
                if current_pass == st.session_state.admin_password:
                    if new_pass == confirm_pass:
                        if len(new_pass) >= 6:
                            st.session_state.admin_password = new_pass
                            st.success("✅ Password changed successfully!")
                        else:
                            st.error("Password must be at least 6 characters")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Current password is incorrect")
        
        st.markdown("---")
        st.markdown("#### System Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart System", use_container_width=True):
                st.warning("System restart initiated...")
        
        with col2:
            if st.button("📊 Export Logs", use_container_width=True):
                events = get_recent_events(100) or []
                if events:
                    json_str = json.dumps(events, indent=2)
                    st.download_button("📥 Download", json_str, "forge_guard_logs.json")
        
        with col3:
            if st.button("🗑️ Clear Events", use_container_width=True):
                st.warning("Event log cleared!")

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #ff6b00; margin: 0;">🔥 FORGE</h2>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">Guard System v2.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()
        
        if st.button("⚙️ Admin Panel", use_container_width=True):
            st.session_state.current_page = 'admin'
            st.rerun()
        
        if st.button("📖 Manual", use_container_width=True):
            st.session_state.current_page = 'manual'
            st.rerun()
        
        st.markdown("---")
        
        # Quick status
        st.markdown("### Quick Status")
        status = get_system_status()
        
        if status and status.get('running'):
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Offline")
        
        # Connection info
        st.markdown("---")
        st.markdown("### Connections")
        st.caption(f"🌐 API: localhost:8000")
        st.caption(f"📊 Dashboard: localhost:8501")
        
        # Auto-refresh toggle
        st.markdown("---")
        auto_refresh = st.toggle("Auto Refresh", value=st.session_state.settings['auto_refresh'])
        st.session_state.settings['auto_refresh'] = auto_refresh
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (s)", 0.5, 5.0, 
                                    st.session_state.settings['refresh_rate'])
            st.session_state.settings['refresh_rate'] = refresh_rate
            
            # Use st.rerun with a timer instead of blocking sleep
            # Store last refresh time in session state
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = time.time()
            
            # Only rerun if enough time has passed
            if time.time() - st.session_state.last_refresh >= refresh_rate:
                st.session_state.last_refresh = time.time()
                st.rerun()

def render_manual():
    """Render the manual/help page."""
    st.markdown("## 📖 User Manual")
    
    st.markdown("""
    <div class="glass-card">
        <h3>Welcome to FORGE-Guard</h3>
        <p>FORGE-Guard is an AI-powered elderly safety monitoring system designed to keep your loved ones safe.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🚨 Fall Detection", expanded=True):
        st.markdown("""
        **How it works:** Uses MediaPipe pose estimation to analyze body position in real-time.
        
        **Triggers when:**
        - Person's center of mass drops rapidly
        - Body position becomes horizontal
        - Person doesn't get up within threshold time
        
        **Configuration:**
        - Adjust sensitivity in Admin Panel → Detectors
        - Higher sensitivity = more alerts, potentially more false positives
        """)
    
    with st.expander("🖐️ Gesture Recognition"):
        st.markdown("""
        **Supported Gestures:**
        - **SOS Signal:** Wave both hands above head
        - **Help Request:** Raise one hand and hold
        
        **Tips:**
        - Ensure hands are visible to camera
        - Hold gesture for configured time (default 2s)
        """)
    
    with st.expander("💊 Medicine Monitoring"):
        st.markdown("""
        **Setup:**
        1. Create a zone around medicine cabinet/location
        2. System captures reference image
        3. Alerts when medicine is moved/taken
        
        **Creating Zones:**
        - Use the Zone Controls in Dashboard
        - Enter coordinates or click on video
        """)
    
    with st.expander("👁️ Object Detection"):
        st.markdown("""
        **Detects:**
        - People (for occupancy)
        - Specific objects based on configuration
        
        **Uses:**
        - Track room occupancy
        - Detect unusual situations
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Camera not working?**
        - Check camera is connected
        - Verify no other app is using camera
        - Try restarting the system
        
        **Detectors offline?**
        - MediaPipe requires Python 3.10-3.11
        - Run `setup.bat` to reinstall dependencies
        
        **Getting too many false alerts?**
        - Reduce sensitivity in Admin Panel
        - Increase alert cooldown
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize
    inject_custom_css()
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render current page
    if st.session_state.current_page == 'dashboard':
        render_header()
        
        # Get system status
        status = get_system_status()
        
        # Status cards
        render_status_cards(status)
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📹 Live Feed", "📋 Events", "🚨 Alerts"])
        
        with tab1:
            render_video_feed()
        
        with tab2:
            render_event_log()
        
        with tab3:
            render_alerts_section()
    
    elif st.session_state.current_page == 'admin':
        render_admin_panel()
    
    elif st.session_state.current_page == 'manual':
        render_manual()

if __name__ == "__main__":
    main()
