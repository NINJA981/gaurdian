"""
FORGE-Guard Medicine Monitor
Module 2: Medicine Box Monitoring using ROI tracking and background subtraction.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import time

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_numpy

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
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside ROI."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


class MedicineMonitor(BaseDetector):
    """
    Medicine box monitoring using ROI-based background subtraction.
    
    Detection Logic:
    1. User defines ROI zone for medicine area
    2. Capture background reference of the zone
    3. Compare current frame's ROI with reference
    4. Trigger alert if pixel change > threshold
    """
    
    def __init__(
        self,
        change_threshold: Optional[float] = None,
        update_interval: float = 0.5
    ):
        """
        Initialize medicine monitor.
        
        Args:
            change_threshold: Percentage of pixel change to trigger alert (0-1)
            update_interval: Minimum seconds between change checks
        """
        super().__init__(name="medicine_monitor")
        
        self.change_threshold = change_threshold or config.detection.medicine_change_threshold
        self.update_interval = update_interval
        
        # ROI zones
        self._zones: List[ROIZone] = []
        self._zone_references: dict = {}  # Zone name -> reference image
        self._last_check_time = 0.0
        
        # Detection state
        self._last_change_percent: dict = {}  # Zone name -> change percent
        self._alert_cooldown: dict = {}  # Zone name -> last alert time
        self._cooldown_seconds = 30.0
    
    def add_zone(self, name: str, x: int, y: int, width: int, height: int):
        """
        Add a new ROI zone for monitoring.
        
        Args:
            name: Unique name for the zone
            x, y: Top-left corner coordinates
            width, height: Zone dimensions
        """
        zone = ROIZone(name=name, x=x, y=y, width=width, height=height)
        
        # Remove existing zone with same name
        self._zones = [z for z in self._zones if z.name != name]
        self._zones.append(zone)
        
        # Clear reference for re-capture
        if name in self._zone_references:
            del self._zone_references[name]
        
        print(f"[MEDICINE] Added zone '{name}' at ({x}, {y}) size {width}x{height}")
    
    def remove_zone(self, name: str):
        """Remove a zone by name."""
        self._zones = [z for z in self._zones if z.name != name]
        if name in self._zone_references:
            del self._zone_references[name]
        print(f"[MEDICINE] Removed zone '{name}'")
    
    def clear_zones(self):
        """Remove all zones."""
        self._zones.clear()
        self._zone_references.clear()
        self._last_change_percent.clear()
        print("[MEDICINE] All zones cleared")
    
    def capture_reference(self, frame: np.ndarray, zone_name: Optional[str] = None):
        """
        Capture reference image for zone(s).
        
        Args:
            frame: Current frame to use as reference
            zone_name: Specific zone to update, or None for all zones
        """
        for zone in self._zones:
            if zone_name is None or zone.name == zone_name:
                x1, y1, x2, y2 = zone.bounds
                roi = frame[y1:y2, x1:x2].copy()
                
                # Convert to grayscale for comparison
                if len(roi.shape) == 3:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                roi = cv2.GaussianBlur(roi, (5, 5), 0)
                
                self._zone_references[zone.name] = roi
                print(f"[MEDICINE] Captured reference for zone '{zone.name}'")
    
    def _calculate_change(self, current_roi: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate percentage of changed pixels between ROIs.
        
        Returns:
            Percentage of changed pixels (0-1)
        """
        if current_roi.shape != reference.shape:
            # Resize if needed (shouldn't happen normally)
            current_roi = cv2.resize(current_roi, (reference.shape[1], reference.shape[0]))
        
        # Absolute difference
        diff = cv2.absdiff(current_roi, reference)
        
        # Threshold to binary
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed (white) pixels
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        
        return changed_pixels / total_pixels if total_pixels > 0 else 0
    
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process frame for medicine box changes."""
        current_time = time.time()
        
        # Rate limit checks
        if current_time - self._last_check_time < self.update_interval:
            # Return cached result
            if any(p > self.change_threshold for p in self._last_change_percent.values()):
                return DetectionResult(
                    detected=True,
                    alert_level=AlertLevel.HIGH,
                    message="Medicine area change detected",
                    details=self._last_change_percent.copy()
                )
            return DetectionResult(detected=False, message="Monitoring medicine zones")
        
        self._last_check_time = current_time
        
        if not self._zones:
            return DetectionResult(
                detected=False,
                message="No medicine zones defined. Use 'Set Zone' to define an area.",
                alert_level=AlertLevel.NONE
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
            current_roi = cv2.GaussianBlur(current_roi, (5, 5), 0)
            
            # Calculate change
            reference = self._zone_references[zone.name]
            change_percent = self._calculate_change(current_roi, reference)
            self._last_change_percent[zone.name] = change_percent
            
            # Check threshold with cooldown
            if change_percent > self.change_threshold:
                last_alert = self._alert_cooldown.get(zone.name, 0)
                if current_time - last_alert > self._cooldown_seconds:
                    self._alert_cooldown[zone.name] = current_time
                    alerts.append({
                        "zone": zone.name,
                        "change_percent": change_percent,
                        "bounds": zone.bounds
                    })
        
        if alerts:
            return DetectionResult(
                detected=True,
                alert_level=AlertLevel.HIGH,
                confidence=max(a["change_percent"] for a in alerts),
                message=f"💊 Medicine area activity detected in {len(alerts)} zone(s)!",
                details={
                    "alerts": alerts,
                    "all_changes": self._last_change_percent
                },
                bounding_boxes=[a["bounds"] for a in alerts]
            )
        
        return DetectionResult(
            detected=False,
            message=f"Monitoring {len(self._zones)} medicine zone(s)",
            details={"changes": self._last_change_percent}
        )
    
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw medicine zone overlays."""
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw zone label
            label = f"{zone.name}: {change*100:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - 20),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Draw alert banner if detected
        if result.detected:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 350, 10), (w - 10, 60), (0, 0, 200), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(
                frame,
                "💊 MEDICINE ACTIVITY",
                (w - 340, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        return frame
    
    @property
    def zones(self) -> List[ROIZone]:
        """Get all defined zones."""
        return self._zones.copy()
    
    def get_zone(self, name: str) -> Optional[ROIZone]:
        """Get zone by name."""
        for zone in self._zones:
            if zone.name == name:
                return zone
        return None
