"""
FORGE-Guard Medicine Monitor
Production-ready medicine box monitoring using ROI tracking and background subtraction.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import time
import threading
import logging

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_numpy

# Setup logging
logger = logging.getLogger(__name__)

# Get safe module handles
cv2 = get_cv2()
np = get_numpy()


@dataclass
class ROIZone:
    """Region of Interest zone definition."""
    name: str
    x: int
    y: int
    width: int
    height: int
    created_at: float = field(default_factory=time.time)
    last_activity: Optional[float] = None
    activity_count: int = 0
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if point is inside ROI."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "activity_count": self.activity_count
        }


class MedicineMonitor(BaseDetector):
    """
    Medicine box monitoring using ROI-based background subtraction.
    
    Detection Logic:
    1. User defines ROI zone for medicine area
    2. Capture background reference of the zone
    3. Compare current frame's ROI with reference
    4. Trigger alert if pixel change > threshold
    
    Features:
    - Multiple zone support
    - Automatic reference capture
    - Configurable sensitivity
    - Alert cooldown per zone
    - Thread-safe operations
    """
    
    def __init__(
        self,
        change_threshold: Optional[float] = None,
        update_interval: float = 0.5,
        cooldown_seconds: float = 30.0,
        blur_kernel: int = 5,
        diff_threshold: int = 25
    ):
        """
        Initialize medicine monitor.
        
        Args:
            change_threshold: Percentage of pixel change to trigger alert (0-1)
            update_interval: Minimum seconds between change checks
            cooldown_seconds: Seconds between alerts for same zone
            blur_kernel: Gaussian blur kernel size
            diff_threshold: Pixel difference threshold for detection
        """
        super().__init__(name="medicine_monitor")
        
        self.change_threshold = change_threshold or config.detection.medicine_change_threshold
        self.update_interval = update_interval
        self.cooldown_seconds = cooldown_seconds
        self.blur_kernel = blur_kernel
        self.diff_threshold = diff_threshold
        
        # ROI zones
        self._zones: List[ROIZone] = []
        self._zone_references: Dict[str, Any] = {}  # Zone name -> reference image
        self._last_check_time = 0.0
        self._lock = threading.Lock()
        
        # Detection state
        self._last_change_percent: Dict[str, float] = {}
        self._alert_cooldown: Dict[str, float] = {}  # Zone name -> last alert time
        self._total_activities = 0
    
    def add_zone(self, name: str, x: int, y: int, width: int, height: int) -> ROIZone:
        """
        Add a new ROI zone for monitoring.
        
        Args:
            name: Unique name for the zone
            x, y: Top-left corner coordinates
            width, height: Zone dimensions
            
        Returns:
            Created ROIZone
        """
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise ValueError("Zone width and height must be positive")
        
        zone = ROIZone(name=name, x=max(0, x), y=max(0, y), 
                      width=width, height=height)
        
        with self._lock:
            # Remove existing zone with same name
            self._zones = [z for z in self._zones if z.name != name]
            self._zones.append(zone)
            
            # Clear reference for re-capture
            if name in self._zone_references:
                del self._zone_references[name]
        
        logger.info(f"[MEDICINE] Added zone '{name}' at ({x}, {y}) size {width}x{height}")
        return zone
    
    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name. Returns True if found and removed."""
        with self._lock:
            original_count = len(self._zones)
            self._zones = [z for z in self._zones if z.name != name]
            
            if name in self._zone_references:
                del self._zone_references[name]
            if name in self._last_change_percent:
                del self._last_change_percent[name]
            if name in self._alert_cooldown:
                del self._alert_cooldown[name]
            
            removed = len(self._zones) < original_count
            if removed:
                logger.info(f"[MEDICINE] Removed zone '{name}'")
            return removed
    
    def clear_zones(self):
        """Remove all zones."""
        with self._lock:
            count = len(self._zones)
            self._zones.clear()
            self._zone_references.clear()
            self._last_change_percent.clear()
            self._alert_cooldown.clear()
        logger.info(f"[MEDICINE] Cleared {count} zones")
    
    def capture_reference(self, frame, zone_name: Optional[str] = None) -> int:
        """
        Capture reference image for zone(s).
        
        Args:
            frame: Current frame to use as reference
            zone_name: Specific zone to update, or None for all zones
            
        Returns:
            Number of references captured
        """
        if np is None:
            logger.error("[MEDICINE] NumPy not available for reference capture")
            return 0
        
        captured = 0
        with self._lock:
            for zone in self._zones:
                if zone_name is None or zone.name == zone_name:
                    x1, y1, x2, y2 = zone.bounds
                    
                    # Bounds check
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    roi = frame[y1:y2, x1:x2].copy()
                    
                    # Convert to grayscale for comparison
                    if len(roi.shape) == 3:
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Gaussian blur to reduce noise
                    roi = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
                    
                    self._zone_references[zone.name] = roi
                    captured += 1
                    logger.info(f"[MEDICINE] Captured reference for zone '{zone.name}'")
        
        return captured
    
    def _calculate_change(self, current_roi, reference) -> float:
        """
        Calculate percentage of changed pixels between ROIs.
        
        Returns:
            Percentage of changed pixels (0-1)
        """
        if np is None:
            return 0.0
        
        try:
            if current_roi.shape != reference.shape:
                # Resize if needed
                current_roi = cv2.resize(current_roi, 
                                        (reference.shape[1], reference.shape[0]))
            
            # Absolute difference
            diff = cv2.absdiff(current_roi, reference)
            
            # Threshold to binary
            _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate percentage of changed (white) pixels
            changed_pixels = np.sum(thresh > 0)
            total_pixels = thresh.size
            
            return changed_pixels / total_pixels if total_pixels > 0 else 0.0
            
        except Exception as e:
            logger.error(f"[MEDICINE] Change calculation error: {e}")
            return 0.0
    
    def _process_frame(self, frame) -> DetectionResult:
        """Process frame for medicine box changes."""
        current_time = time.time()
        
        if np is None:
            return DetectionResult(
                detected=False,
                message="Medicine monitoring offline - NumPy unavailable",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled"}
            )
        
        with self._lock:
            # Rate limit checks
            if current_time - self._last_check_time < self.update_interval:
                # Return cached result
                if any(p > self.change_threshold for p in self._last_change_percent.values()):
                    return DetectionResult(
                        detected=True,
                        alert_level=AlertLevel.HIGH,
                        message="Medicine area change detected",
                        details={"changes": dict(self._last_change_percent)}
                    )
                return DetectionResult(
                    detected=False, 
                    message=f"Monitoring {len(self._zones)} medicine zone(s)",
                    details={"zones": len(self._zones)}
                )
            
            self._last_check_time = current_time
            
            if not self._zones:
                return DetectionResult(
                    detected=False,
                    message="No medicine zones defined. Use 'Add Zone' to define an area.",
                    alert_level=AlertLevel.NONE,
                    details={"status": "no_zones"}
                )
            
            # Auto-capture reference if not exists
            for zone in self._zones:
                if zone.name not in self._zone_references:
                    self.capture_reference(frame, zone.name)
            
            # Check each zone
            alerts = []
            self._last_change_percent = {}
            
            for zone in self._zones:
                if zone.name not in self._zone_references:
                    continue
                
                x1, y1, x2, y2 = zone.bounds
                
                # Bounds checking
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract current ROI
                current_roi = frame[y1:y2, x1:x2]
                if len(current_roi.shape) == 3:
                    current_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                current_roi = cv2.GaussianBlur(current_roi, 
                                               (self.blur_kernel, self.blur_kernel), 0)
                
                # Calculate change
                reference = self._zone_references[zone.name]
                change_percent = self._calculate_change(current_roi, reference)
                self._last_change_percent[zone.name] = change_percent
                
                # Check threshold with cooldown
                if change_percent > self.change_threshold:
                    last_alert = self._alert_cooldown.get(zone.name, 0)
                    if current_time - last_alert > self.cooldown_seconds:
                        self._alert_cooldown[zone.name] = current_time
                        zone.last_activity = current_time
                        zone.activity_count += 1
                        self._total_activities += 1
                        
                        alerts.append({
                            "zone": zone.name,
                            "change_percent": round(change_percent * 100, 1),
                            "bounds": zone.bounds,
                            "activity_count": zone.activity_count
                        })
        
        if alerts:
            return DetectionResult(
                detected=True,
                alert_level=AlertLevel.HIGH,
                confidence=max(a["change_percent"] for a in alerts) / 100,
                message=f"💊 Medicine activity in {len(alerts)} zone(s)!",
                details={
                    "alerts": alerts,
                    "all_changes": {k: round(v * 100, 1) for k, v in self._last_change_percent.items()}
                },
                bounding_boxes=[tuple(a["bounds"]) for a in alerts]
            )
        
        return DetectionResult(
            detected=False,
            message=f"Monitoring {len(self._zones)} medicine zone(s)",
            details={
                "zones": len(self._zones),
                "changes": {k: round(v * 100, 1) for k, v in self._last_change_percent.items()}
            }
        )
    
    def draw_overlay(self, frame, result: DetectionResult):
        """Draw medicine zone overlays."""
        try:
            overlay = frame.copy()
            
            with self._lock:
                for zone in self._zones:
                    x1, y1, x2, y2 = zone.bounds
                    change = self._last_change_percent.get(zone.name, 0)
                    
                    # Color based on change level
                    if change > self.change_threshold:
                        color = (0, 0, 255)  # Red - alert
                        thickness = 3
                    elif change > self.change_threshold * 0.5:
                        color = (0, 165, 255)  # Orange - warning
                        thickness = 2
                    else:
                        color = (0, 255, 0)  # Green - normal
                        thickness = 2
                    
                    # Draw zone rectangle
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw zone label
                    label = f"{zone.name}: {change*100:.1f}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # Label background
                    cv2.rectangle(
                        overlay,
                        (x1, y1 - 22),
                        (x1 + label_size[0] + 10, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        overlay,
                        label,
                        (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
            
            # Draw alert banner if detected
            if result.detected:
                h, w = frame.shape[:2]
                cv2.rectangle(overlay, (w - 350, 10), (w - 10, 60), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
                
                cv2.putText(
                    overlay,
                    "MEDICINE ACTIVITY",
                    (w - 340, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            
            return overlay
            
        except Exception as e:
            logger.error(f"[MEDICINE] Overlay error: {e}")
            return frame
    
    @property
    def zones(self) -> List[ROIZone]:
        """Get all defined zones."""
        with self._lock:
            return self._zones.copy()
    
    def get_zone(self, name: str) -> Optional[ROIZone]:
        """Get zone by name."""
        with self._lock:
            for zone in self._zones:
                if zone.name == name:
                    return zone
            return None
    
    def reset(self):
        """Reset detector state (keeps zones but resets references)."""
        super().reset()
        with self._lock:
            self._zone_references.clear()
            self._last_change_percent.clear()
            self._alert_cooldown.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        base_stats = super().stats()
        with self._lock:
            base_stats.update({
                "zones_count": len(self._zones),
                "zones": [z.to_dict() for z in self._zones],
                "total_activities": self._total_activities,
                "change_threshold": self.change_threshold,
                "cooldown_seconds": self.cooldown_seconds
            })
        return base_stats
