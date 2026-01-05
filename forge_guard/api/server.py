"""
FORGE-Guard FastAPI Server
Backend API with video streaming, WebSocket events, and REST endpoints.
"""

import asyncio
import cv2
import json
import time
import threading
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import config
from ..pipeline import VideoPipeline, ProcessedFrame
from ..detectors import FallDetector, MedicineMonitor, GestureDetector, ObjectDetector
from ..alerts import NotificationManager, AlertPriority, EventLogger, Event


# ============================================================================
# Pydantic Models
# ============================================================================

class ZoneCreate(BaseModel):
    """Request model for creating a zone."""
    name: str
    x: int
    y: int
    width: int
    height: int


class ZoneUpdate(BaseModel):
    """Request model for updating a zone."""
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ConfigUpdate(BaseModel):
    """Request model for configuration updates."""
    fall_detection_enabled: Optional[bool] = None
    medicine_monitoring_enabled: Optional[bool] = None
    gesture_detection_enabled: Optional[bool] = None
    object_detection_enabled: Optional[bool] = None


class AlertCreate(BaseModel):
    """Request model for manual alert creation."""
    message: str
    priority: str = "MEDIUM"


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        with self._lock:
            connections = self.active_connections.copy()
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected
        with self._lock:
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
    
    @property
    def connection_count(self) -> int:
        with self._lock:
            return len(self.active_connections)


# ============================================================================
# FORGE-Guard API Class
# ============================================================================

class ForgeGuardAPI:
    """
    Main API handler for FORGE-Guard system.
    Integrates video pipeline, detectors, and notifications.
    """
    
    def __init__(self):
        self.pipeline: Optional[VideoPipeline] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.event_logger: Optional[EventLogger] = None
        self.ws_manager = ConnectionManager()
        
        # Detectors
        self.fall_detector: Optional[FallDetector] = None
        self.medicine_monitor: Optional[MedicineMonitor] = None
        self.gesture_detector: Optional[GestureDetector] = None
        self.object_detector: Optional[ObjectDetector] = None
        
        # State
        self._latest_frame: Optional[ProcessedFrame] = None
        self._running = False
    
    def initialize(self):
        """Initialize all components."""
        print("[API] Initializing FORGE-Guard components...")
        
        # Initialize event logger
        self.event_logger = EventLogger()
        
        # Initialize notification manager
        self.notification_manager = NotificationManager(
            on_alert=self._on_alert
        )
        self.notification_manager.start()
        
        # Initialize detectors
        self.fall_detector = FallDetector()
        self.medicine_monitor = MedicineMonitor()
        self.gesture_detector = GestureDetector()
        self.object_detector = ObjectDetector()
        
        # Initialize video pipeline
        self.pipeline = VideoPipeline(
            on_frame_processed=self._on_frame_processed
        )
        
        # Register detectors
        self.pipeline.register_detector(self.fall_detector)
        self.pipeline.register_detector(self.medicine_monitor)
        self.pipeline.register_detector(self.gesture_detector)
        self.pipeline.register_detector(self.object_detector)
        
        # Start pipeline
        self.pipeline.start()
        self._running = True
        
        self.event_logger.log(
            "FORGE-Guard system initialized",
            source="api",
            level="INFO"
        )
        
        print("[API] Initialization complete")
    
    def shutdown(self):
        """Shutdown all components."""
        print("[API] Shutting down...")
        self._running = False
        
        if self.pipeline:
            self.pipeline.stop()
        
        if self.notification_manager:
            self.notification_manager.stop()
        
        # Cleanup detectors
        for detector in [self.fall_detector, self.medicine_monitor, 
                        self.gesture_detector, self.object_detector]:
            if detector:
                detector.cleanup()
        
        print("[API] Shutdown complete")
    
    def _on_frame_processed(self, frame: ProcessedFrame):
        """Callback for processed frames."""
        self._latest_frame = frame
        
        # Check for alerts
        for detector_name, result in frame.detections.items():
            if result.detected and result.alert_level.value >= 3:  # HIGH or CRITICAL
                self._trigger_alert(detector_name, result)
    
    def _trigger_alert(self, source: str, result):
        """Trigger notification for detection."""
        from ..detectors.base_detector import AlertLevel
        
        priority_map = {
            AlertLevel.LOW: AlertPriority.LOW,
            AlertLevel.MEDIUM: AlertPriority.MEDIUM,
            AlertLevel.HIGH: AlertPriority.HIGH,
            AlertLevel.CRITICAL: AlertPriority.CRITICAL,
        }
        
        priority = priority_map.get(result.alert_level, AlertPriority.MEDIUM)
        
        alert = self.notification_manager.send_alert(
            message=result.message,
            source=source,
            priority=priority,
            details=result.details
        )
        
        if alert and self.event_logger:
            self.event_logger.alert(source, result.message, result.details)
    
    def _on_alert(self, alert):
        """Callback for new alerts - broadcast via WebSocket."""
        asyncio.create_task(self.ws_manager.broadcast({
            "type": "alert",
            "data": alert.to_dict()
        }))
    
    def get_latest_frame_bytes(self) -> Optional[bytes]:
        """Get latest processed frame as JPEG bytes."""
        if self._latest_frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', self._latest_frame.frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            return buffer.tobytes()
        return None
    
    async def generate_video_stream(self):
        """Generator for MJPEG video stream."""
        while self._running:
            frame_bytes = self.get_latest_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(1/30)  # ~30 FPS
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            "running": self._running,
            "pipeline": self.pipeline.stats() if self.pipeline else None,
            "detectors": {
                "fall_detection": self.fall_detector.stats() if self.fall_detector else None,
                "medicine_monitor": self.medicine_monitor.stats() if self.medicine_monitor else None,
                "gesture_detection": self.gesture_detector.stats() if self.gesture_detector else None,
                "object_detection": self.object_detector.stats() if self.object_detector else None,
            },
            "notifications": self.notification_manager.stats() if self.notification_manager else None,
            "websocket_clients": self.ws_manager.connection_count
        }


# ============================================================================
# Global API Instance
# ============================================================================

forge_api = ForgeGuardAPI()


# ============================================================================
# FastAPI Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    forge_api.initialize()
    yield
    forge_api.shutdown()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="FORGE-Guard API",
    description="Elderly Monitoring System - Real-time Computer Vision Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - returns simple status page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FORGE-Guard API</title>
        <style>
            body { 
                font-family: 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                min-height: 100vh;
                margin: 0;
                padding: 40px;
            }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #ff8000; }
            .status { 
                background: rgba(255,128,0,0.1); 
                border: 1px solid #ff8000;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            a { color: #00c8ff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔥 FORGE-Guard API</h1>
            <div class="status">
                <h3>API is running</h3>
                <p>Video Stream: <a href="/stream">/stream</a></p>
                <p>API Docs: <a href="/docs">/docs</a></p>
                <p>Status: <a href="/api/status">/api/status</a></p>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/stream")
async def video_stream():
    """MJPEG video stream endpoint."""
    return StreamingResponse(
        forge_api.generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return forge_api.get_status()


@app.get("/api/events")
async def get_events(count: int = 20):
    """Get recent events."""
    if forge_api.event_logger:
        events = forge_api.event_logger.get_recent(count)
        return [e.to_dict() for e in events]
    return []


@app.get("/api/alerts")
async def get_alerts(count: int = 20):
    """Get recent alerts."""
    if forge_api.notification_manager:
        alerts = forge_api.notification_manager.get_recent_alerts(count)
        return [a.to_dict() for a in alerts]
    return []


@app.post("/api/alerts")
async def create_alert(alert: AlertCreate):
    """Manually create an alert."""
    if not forge_api.notification_manager:
        raise HTTPException(status_code=503, detail="Notification manager not available")
    
    priority = AlertPriority[alert.priority.upper()]
    result = forge_api.notification_manager.send_alert(
        message=alert.message,
        source="manual",
        priority=priority
    )
    
    if result:
        return result.to_dict()
    raise HTTPException(status_code=429, detail="Alert rate limited")


# ============================================================================
# Zone Management
# ============================================================================

@app.get("/api/zones")
async def get_zones():
    """Get all defined zones."""
    if forge_api.medicine_monitor:
        return [{"name": z.name, "x": z.x, "y": z.y, 
                "width": z.width, "height": z.height} 
               for z in forge_api.medicine_monitor.zones]
    return []


@app.post("/api/zones")
async def create_zone(zone: ZoneCreate):
    """Create a new monitoring zone."""
    if not forge_api.medicine_monitor:
        raise HTTPException(status_code=503, detail="Medicine monitor not available")
    
    forge_api.medicine_monitor.add_zone(
        name=zone.name,
        x=zone.x,
        y=zone.y,
        width=zone.width,
        height=zone.height
    )
    
    if forge_api.event_logger:
        forge_api.event_logger.user_action(
            f"Created zone '{zone.name}'",
            {"zone": zone.dict()}
        )
    
    return {"status": "created", "zone": zone.dict()}


@app.delete("/api/zones/{zone_name}")
async def delete_zone(zone_name: str):
    """Delete a monitoring zone."""
    if not forge_api.medicine_monitor:
        raise HTTPException(status_code=503, detail="Medicine monitor not available")
    
    forge_api.medicine_monitor.remove_zone(zone_name)
    
    if forge_api.event_logger:
        forge_api.event_logger.user_action(f"Deleted zone '{zone_name}'")
    
    return {"status": "deleted", "zone": zone_name}


@app.post("/api/zones/{zone_name}/capture")
async def capture_zone_reference(zone_name: str):
    """Capture reference image for a zone."""
    if not forge_api.medicine_monitor or not forge_api._latest_frame:
        raise HTTPException(status_code=503, detail="Not available")
    
    forge_api.medicine_monitor.capture_reference(
        forge_api._latest_frame.original_frame,
        zone_name
    )
    
    return {"status": "captured", "zone": zone_name}


# ============================================================================
# Configuration
# ============================================================================

@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "fall_detection_enabled": forge_api.fall_detector.enabled if forge_api.fall_detector else False,
        "medicine_monitoring_enabled": forge_api.medicine_monitor.enabled if forge_api.medicine_monitor else False,
        "gesture_detection_enabled": forge_api.gesture_detector.enabled if forge_api.gesture_detector else False,
        "object_detection_enabled": forge_api.object_detector.enabled if forge_api.object_detector else False,
        "video": {
            "width": config.video.width,
            "height": config.video.height,
            "fps": config.video.fps
        },
        "detection": {
            "fall_ratio_threshold": config.detection.fall_ratio_threshold,
            "gesture_hold_seconds": config.detection.gesture_hold_seconds,
            "medicine_change_threshold": config.detection.medicine_change_threshold
        }
    }


@app.patch("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration."""
    updates = {}
    
    if update.fall_detection_enabled is not None and forge_api.fall_detector:
        forge_api.fall_detector.enabled = update.fall_detection_enabled
        updates["fall_detection_enabled"] = update.fall_detection_enabled
    
    if update.medicine_monitoring_enabled is not None and forge_api.medicine_monitor:
        forge_api.medicine_monitor.enabled = update.medicine_monitoring_enabled
        updates["medicine_monitoring_enabled"] = update.medicine_monitoring_enabled
    
    if update.gesture_detection_enabled is not None and forge_api.gesture_detector:
        forge_api.gesture_detector.enabled = update.gesture_detection_enabled
        updates["gesture_detection_enabled"] = update.gesture_detection_enabled
    
    if update.object_detection_enabled is not None and forge_api.object_detector:
        forge_api.object_detector.enabled = update.object_detection_enabled
        updates["object_detection_enabled"] = update.object_detection_enabled
    
    if forge_api.event_logger:
        forge_api.event_logger.user_action("Updated configuration", updates)
    
    return {"status": "updated", "changes": updates}


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await forge_api.ws_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        forge_api.ws_manager.disconnect(websocket)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "pipeline_running": forge_api._running
    }
