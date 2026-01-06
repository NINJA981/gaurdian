"""
FORGE-Guard Object Detector
Production-ready object detection using YOLOv8-nano for walking aids, wheelchairs, doors.
"""

from typing import Optional, List, Tuple, Dict, Any
import time
import threading
import logging

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_numpy, get_yolo

# Setup logging
logger = logging.getLogger(__name__)

# Get safe module handles
cv2 = get_cv2()
np = get_numpy()
YOLO = get_yolo()
YOLO_AVAILABLE = YOLO is not None


class ObjectDetector(BaseDetector):
    """
    Object detection using YOLOv8-nano.
    
    Detects: persons, walking sticks/canes, wheelchairs, doors, chairs, beds.
    Uses the lightweight YOLOv8-nano model (3MB) for edge device compatibility.
    
    Features:
    - Person absence detection with configurable timeout
    - Custom class filtering
    - Thread-safe state management
    - Graceful degradation
    """
    
    # COCO class IDs we're interested in
    COCO_CLASSES = {
        0: "person",
        56: "chair",
        59: "bed",
        57: "couch",
        62: "tv",
        63: "laptop",
    }
    
    # Classes we want to track
    TARGET_CLASSES = ["person", "chair", "bed", "couch"]
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: Optional[float] = None,
        custom_classes: Optional[List[str]] = None,
        absence_alert_minutes: float = 5.0
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detection
            custom_classes: Additional classes to track
            absence_alert_minutes: Minutes without person to trigger alert
        """
        super().__init__(name="object_detector")
        
        self.model_path = model_path
        self.confidence_threshold = (
            confidence_threshold or config.detection.object_confidence_threshold
        )
        self.custom_classes = custom_classes or []
        self.absence_alert_minutes = absence_alert_minutes
        self._absence_alert_seconds = absence_alert_minutes * 60
        
        # Model initialization
        self.model = None
        self._model_loaded = False
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self._model_loaded = True
                logger.info(f"[OBJECTS] Loaded YOLOv8 model: {model_path}")
            except Exception as e:
                logger.error(f"[OBJECTS] Failed to load YOLO model: {e}")
        else:
            logger.warning("[OBJECTS] YOLO not available - Object detection DISABLED")
            self._enabled = False
        
        # Detection state
        self._lock = threading.Lock()
        self._last_detections: List[dict] = []
        self._person_in_frame = False
        self._person_last_seen: Optional[float] = None
        self._person_absent_start: Optional[float] = None
        self._total_person_detections = 0
        self._absence_alerts_sent = 0
    
    def _process_frame(self, frame) -> DetectionResult:
        """Process frame for object detection."""
        if not YOLO_AVAILABLE or self.model is None or not self._model_loaded:
            return self._fallback_detection(frame)
        
        if np is None:
            return DetectionResult(
                detected=False,
                message="Object detection offline - NumPy unavailable",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled"}
            )
        
        try:
            current_time = time.time()
            
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )[0]
            
            # Parse detections
            detections = []
            bboxes = []
            person_detected = False
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name
                class_name = results.names.get(cls_id, f"class_{cls_id}")
                
                # Only track target classes
                if class_name in self.TARGET_CLASSES or class_name in self.custom_classes:
                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 3),
                        "bbox": (x1, y1, x2, y2)
                    })
                    bboxes.append((x1, y1, x2, y2))
                    
                    if class_name == "person":
                        person_detected = True
            
            with self._lock:
                self._last_detections = detections
                
                # Track person presence for absence detection
                if person_detected:
                    self._person_in_frame = True
                    self._person_last_seen = current_time
                    self._person_absent_start = None
                    self._total_person_detections += 1
                else:
                    if self._person_in_frame:
                        # Person just left frame
                        self._person_absent_start = current_time
                    self._person_in_frame = False
                
                # Check for extended absence
                absence_alert = False
                absence_duration = 0.0
                
                if self._person_absent_start is not None:
                    absence_duration = current_time - self._person_absent_start
                    if absence_duration > self._absence_alert_seconds:
                        absence_alert = True
                        self._absence_alerts_sent += 1
            
            # Build response
            if absence_alert:
                return DetectionResult(
                    detected=True,
                    alert_level=AlertLevel.HIGH,
                    confidence=1.0,
                    message=f"⚠️ Person not detected for {absence_duration/60:.1f} minutes!",
                    details={
                        "objects": detections,
                        "person_detected": person_detected,
                        "absence_duration_minutes": round(absence_duration / 60, 1),
                        "status": "person_absent"
                    },
                    bounding_boxes=bboxes
                )
            
            # Normal detection summary
            class_counts: Dict[str, int] = {}
            for d in detections:
                cls = d["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            summary = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
            if not summary:
                summary = "No target objects"
            
            return DetectionResult(
                detected=len(detections) > 0,
                alert_level=AlertLevel.NONE,
                confidence=max((d["confidence"] for d in detections), default=0),
                message=f"Objects: {summary}",
                details={
                    "objects": detections,
                    "class_counts": class_counts,
                    "person_detected": person_detected,
                    "status": "monitoring"
                },
                bounding_boxes=bboxes
            )
            
        except Exception as e:
            logger.error(f"[OBJECTS] Detection error: {e}")
            return DetectionResult(
                detected=False,
                message=f"Detection error: {str(e)}",
                alert_level=AlertLevel.NONE,
                details={"status": "error", "error": str(e)}
            )
    
    def _fallback_detection(self, frame) -> DetectionResult:
        """Fallback detection when YOLO is not available."""
        return DetectionResult(
            detected=False,
            message="Object detection offline (YOLO not installed)",
            alert_level=AlertLevel.NONE,
            details={"status": "disabled", "fallback": True}
        )
    
    def draw_overlay(self, frame, result: DetectionResult):
        """Draw object detection overlays."""
        try:
            overlay = frame.copy()
            h, w = frame.shape[:2]
            
            if not self._model_loaded:
                cv2.putText(overlay, "Object Detection: OFFLINE", 
                           (w - 300, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return overlay

            # Color palette for different classes
            class_colors = {
                "person": (0, 255, 0),      # Green
                "chair": (255, 165, 0),      # Orange
                "bed": (147, 112, 219),      # Purple
                "couch": (255, 192, 203),    # Pink
                "walking_stick": (255, 255, 0),  # Yellow
                "wheelchair": (0, 191, 255),  # Deep sky blue
            }
            default_color = (128, 128, 128)  # Gray
            
            with self._lock:
                for detection in self._last_detections:
                    cls = detection["class"]
                    conf = detection["confidence"]
                    x1, y1, x2, y2 = detection["bbox"]
                    
                    color = class_colors.get(cls, default_color)
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label = f"{cls}: {conf:.0%}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    cv2.rectangle(
                        overlay,
                        (x1, y1 - 22),
                        (x1 + label_size[0] + 10, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        overlay,
                        label,
                        (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
            
            # Draw absence alert if triggered
            if result.alert_level == AlertLevel.HIGH:
                cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 100, 200), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
                
                cv2.putText(
                    overlay,
                    result.message,
                    (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            
            # Detection count overlay
            with self._lock:
                count_text = f"Objects: {len(self._last_detections)}"
            
            cv2.putText(
                overlay,
                count_text,
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            return overlay
            
        except Exception as e:
            logger.error(f"[OBJECTS] Overlay error: {e}")
            return frame
    
    @property
    def last_detections(self) -> List[dict]:
        """Get last detection results."""
        with self._lock:
            return self._last_detections.copy()
    
    @property
    def person_in_frame(self) -> bool:
        """Check if a person is currently in frame."""
        with self._lock:
            return self._person_in_frame
    
    @property
    def person_last_seen(self) -> Optional[float]:
        """Get timestamp of last person detection."""
        with self._lock:
            return self._person_last_seen
    
    def reset_absence_timer(self):
        """Manually reset the absence timer."""
        with self._lock:
            self._person_absent_start = None
            logger.info("[OBJECTS] Absence timer reset")
    
    def reset(self):
        """Reset detector state."""
        super().reset()
        with self._lock:
            self._last_detections.clear()
            self._person_in_frame = False
            self._person_absent_start = None
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        base_stats = super().stats()
        with self._lock:
            base_stats.update({
                "yolo_available": YOLO_AVAILABLE,
                "model_loaded": self._model_loaded,
                "model_path": self.model_path,
                "person_in_frame": self._person_in_frame,
                "person_last_seen": self._person_last_seen,
                "total_person_detections": self._total_person_detections,
                "absence_alerts_sent": self._absence_alerts_sent,
                "absence_threshold_minutes": self.absence_alert_minutes
            })
        return base_stats
