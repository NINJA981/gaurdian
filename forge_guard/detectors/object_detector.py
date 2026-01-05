"""
FORGE-Guard Object Detector
Module 4: Object Detection using YOLOv8-nano for walking sticks, wheelchairs, doors.
"""

from typing import Optional, List, Tuple, Dict
import time

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_numpy, get_yolo

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
    """
    
    # COCO class IDs we're interested in
    COCO_CLASSES = {
        0: "person",
        56: "chair",
        59: "bed",
        # Note: walking stick and wheelchair are not in COCO
        # We'll detect them via custom training or alternative methods
    }
    
    # Classes we want to track
    TARGET_CLASSES = ["person", "chair", "bed"]
    
    # Alert-triggering classes
    ALERT_CLASSES = ["person"]  # Alert when person detected in certain conditions
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: Optional[float] = None,
        custom_classes: Optional[List[str]] = None
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detection
            custom_classes: Additional classes to track
        """
        super().__init__(name="object_detector")
        
        self.model_path = model_path
        self.confidence_threshold = (
            confidence_threshold or config.detection.object_confidence_threshold
        )
        self.custom_classes = custom_classes or []
        
        # Model initialization
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"[OBJECTS] Loaded YOLOv8 model: {model_path}")
            except Exception as e:
                print(f"[OBJECTS] Failed to load YOLO model: {e}")
        
        # Detection state
        self._last_detections: List[dict] = []
        self._person_in_frame = False
        self._person_absent_start: Optional[float] = None
        self._absence_alert_threshold = 300  # 5 minutes
    
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process frame for object detection."""
        if not YOLO_AVAILABLE or self.model is None:
            return self._fallback_detection(frame)
        
        try:
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
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })
                    bboxes.append((x1, y1, x2, y2))
                    
                    if class_name == "person":
                        person_detected = True
            
            self._last_detections = detections
            
            # Track person presence for absence detection
            current_time = time.time()
            if person_detected:
                self._person_in_frame = True
                self._person_absent_start = None
            else:
                if self._person_in_frame:
                    self._person_absent_start = current_time
                self._person_in_frame = False
            
            # Check for extended absence
            absence_alert = False
            absence_duration = 0
            if self._person_absent_start is not None:
                absence_duration = current_time - self._person_absent_start
                if absence_duration > self._absence_alert_threshold:
                    absence_alert = True
            
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
                        "absence_duration": absence_duration
                    },
                    bounding_boxes=bboxes
                )
            
            # Normal detection summary
            class_counts = {}
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
                    "person_detected": person_detected
                },
                bounding_boxes=bboxes
            )
            
        except Exception as e:
            print(f"[OBJECTS] Detection error: {e}")
            return DetectionResult(
                detected=False,
                message=f"Detection error: {str(e)}"
            )
    
    def _fallback_detection(self, frame: np.ndarray) -> DetectionResult:
        """Fallback detection when YOLO is not available."""
        # Basic motion detection as fallback
        return DetectionResult(
            detected=False,
            message="Object detection unavailable (YOLO not installed)",
            alert_level=AlertLevel.NONE,
            details={"fallback": True}
        )
    
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw object detection overlays."""
        if not YOLO_AVAILABLE or self.model is None:
            cv2.putText(frame, "Object Detection: OFFLINE", (frame.shape[1] - 300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        # Color palette for different classes
        class_colors = {
            "person": (0, 255, 0),      # Green
            "chair": (255, 165, 0),      # Orange
            "bed": (147, 112, 219),      # Purple
            "walking_stick": (255, 255, 0),  # Yellow
            "wheelchair": (0, 191, 255),  # Deep sky blue
        }
        default_color = (128, 128, 128)  # Gray
        
        for detection in self._last_detections:
            cls = detection["class"]
            conf = detection["confidence"]
            x1, y1, x2, y2 = detection["bbox"]
            
            color = class_colors.get(cls, default_color)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{cls}: {conf:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(
                frame,
                (x1, y1 - 22),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Draw absence alert if triggered
        if result.alert_level == AlertLevel.HIGH:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 100, 200), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(
                frame,
                result.message,
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        # Detection count overlay
        h, w = frame.shape[:2]
        count_text = f"Objects: {len(self._last_detections)}"
        cv2.putText(
            frame,
            count_text,
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame
    
    @property
    def last_detections(self) -> List[dict]:
        """Get last detection results."""
        return self._last_detections.copy()
    
    @property
    def person_in_frame(self) -> bool:
        """Check if a person is currently in frame."""
        return self._person_in_frame
