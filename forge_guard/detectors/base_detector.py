"""
FORGE-Guard Base Detector
Abstract base class for all detection modules.
Production-ready with proper threading, statistics, and lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import logging

# Use safe import for numpy
from ..utils.safe_imports import get_numpy

np = get_numpy()
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    
    def __ge__(self, other):
        if isinstance(other, AlertLevel):
            return self.value >= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, AlertLevel):
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, AlertLevel):
            return self.value <= other.value
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, AlertLevel):
            return self.value < other.value
        return NotImplemented


@dataclass
class DetectionResult:
    """Container for detection results."""
    detected: bool = False
    alert_level: AlertLevel = AlertLevel.NONE
    confidence: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # Single box (x1, y1, x2, y2)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    keypoints: List[Tuple[int, int]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detected": self.detected,
            "alert_level": self.alert_level.name,
            "alert_level_value": self.alert_level.value,
            "confidence": round(self.confidence, 3),
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "bounding_box": self.bounding_box,
        }
    
    def is_alert(self, min_level: AlertLevel = AlertLevel.HIGH) -> bool:
        """Check if result warrants an alert at the given minimum level."""
        return self.detected and self.alert_level >= min_level


class BaseDetector(ABC):
    """
    Abstract base class for detection modules.
    All detectors must implement _process_frame() and draw_overlay() methods.
    
    Features:
    - Thread-safe statistics tracking
    - Automatic timing measurement
    - Enable/disable support
    - Lifecycle management (cleanup)
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize base detector.
        
        Args:
            name: Unique name for this detector
            enabled: Whether detection is active
        """
        self._name = name
        self._enabled = enabled
        self._last_result: Optional[DetectionResult] = None
        self._detection_count = 0
        self._positive_count = 0
        self._total_processing_time = 0.0
        self._min_processing_time = float('inf')
        self._max_processing_time = 0.0
        self._last_processing_time = 0.0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()
        self._created_at = time.time()
    
    @property
    def name(self) -> str:
        """Detector name."""
        return self._name
    
    @property
    def enabled(self) -> bool:
        """Whether detector is enabled."""
        with self._lock:
            return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Set enabled state."""
        with self._lock:
            self._enabled = value
            logger.info(f"[{self._name}] {'Enabled' if value else 'Disabled'}")
    
    @property
    def last_result(self) -> Optional[DetectionResult]:
        """Last detection result."""
        with self._lock:
            return self._last_result
    
    @abstractmethod
    def _process_frame(self, frame) -> DetectionResult:
        """
        Process a single frame for detection.
        Must be implemented by subclasses.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            DetectionResult with detection details
        """
        pass
    
    @abstractmethod
    def draw_overlay(self, frame, result: DetectionResult):
        """
        Draw detection overlay on frame.
        Must be implemented by subclasses.
        
        Args:
            frame: Input frame to draw on
            result: Detection result to visualize
            
        Returns:
            Frame with overlay drawn
        """
        pass
    
    def detect(self, frame) -> DetectionResult:
        """
        Public detection method with timing and state management.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            DetectionResult
        """
        # Check if enabled
        with self._lock:
            if not self._enabled:
                return DetectionResult(
                    detected=False, 
                    message=f"{self._name} disabled",
                    details={"status": "disabled"}
                )
        
        # Process with timing
        start_time = time.time()
        try:
            result = self._process_frame(frame)
            processing_time = time.time() - start_time
            
            # Update statistics thread-safely
            with self._lock:
                self._last_result = result
                self._detection_count += 1
                if result.detected:
                    self._positive_count += 1
                self._total_processing_time += processing_time
                self._last_processing_time = processing_time
                self._min_processing_time = min(self._min_processing_time, processing_time)
                self._max_processing_time = max(self._max_processing_time, processing_time)
            
            # Add processing time to result details
            result.details["processing_time_ms"] = round(processing_time * 1000, 2)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            with self._lock:
                self._error_count += 1
                self._last_error = str(e)
                self._detection_count += 1
            
            logger.error(f"[{self._name}] Detection error: {e}")
            
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message=f"Error: {str(e)}",
                alert_level=AlertLevel.NONE,
                details={
                    "status": "error",
                    "error": str(e),
                    "processing_time_ms": round(processing_time * 1000, 2)
                }
            )
    
    def reset(self):
        """Reset detector state. Override in subclasses for custom reset."""
        with self._lock:
            self._last_result = None
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self._detection_count = 0
            self._positive_count = 0
            self._total_processing_time = 0.0
            self._min_processing_time = float('inf')
            self._max_processing_time = 0.0
            self._last_processing_time = 0.0
            self._error_count = 0
            self._last_error = None
    
    def stats(self) -> dict:
        """Get detector statistics."""
        with self._lock:
            avg_time = (self._total_processing_time / self._detection_count * 1000
                       if self._detection_count > 0 else 0)
            detection_rate = (self._positive_count / self._detection_count * 100
                            if self._detection_count > 0 else 0)
            error_rate = (self._error_count / self._detection_count * 100
                        if self._detection_count > 0 else 0)
            
            return {
                "name": self._name,
                "enabled": self._enabled,
                "detection_count": self._detection_count,
                "positive_count": self._positive_count,
                "detection_rate": round(detection_rate, 2),
                "error_count": self._error_count,
                "error_rate": round(error_rate, 2),
                "last_error": self._last_error,
                "avg_processing_time_ms": round(avg_time, 2),
                "min_processing_time_ms": round(self._min_processing_time * 1000, 2) if self._min_processing_time != float('inf') else 0,
                "max_processing_time_ms": round(self._max_processing_time * 1000, 2),
                "last_processing_time_ms": round(self._last_processing_time * 1000, 2),
                "uptime_seconds": round(time.time() - self._created_at, 1)
            }
    
    def cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}', enabled={self._enabled})"
