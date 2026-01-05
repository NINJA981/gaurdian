"""
FORGE-Guard Base Detector
Abstract base class for all detection modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import numpy as np


class AlertLevel(Enum):
    """Alert severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """Container for detection results."""
    detected: bool = False
    alert_level: AlertLevel = AlertLevel.NONE
    confidence: float = 0.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    keypoints: List[Tuple[int, int]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detected": self.detected,
            "alert_level": self.alert_level.name,
            "confidence": self.confidence,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details
        }


class BaseDetector(ABC):
    """
    Abstract base class for detection modules.
    All detectors must implement detect() and draw_overlay() methods.
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
        self._total_processing_time = 0.0
    
    @property
    def name(self) -> str:
        """Detector name."""
        return self._name
    
    @property
    def enabled(self) -> bool:
        """Whether detector is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Set enabled state."""
        self._enabled = value
    
    @property
    def last_result(self) -> Optional[DetectionResult]:
        """Last detection result."""
        return self._last_result
    
    @abstractmethod
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
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
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
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
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Public detection method with timing and state management.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            DetectionResult
        """
        if not self._enabled:
            return DetectionResult(detected=False, message=f"{self._name} disabled")
        
        start_time = time.time()
        result = self._process_frame(frame)
        processing_time = time.time() - start_time
        
        self._last_result = result
        self._detection_count += 1
        self._total_processing_time += processing_time
        
        result.details["processing_time_ms"] = processing_time * 1000
        
        return result
    
    def reset(self):
        """Reset detector state."""
        self._last_result = None
        self._detection_count = 0
        self._total_processing_time = 0.0
    
    def stats(self) -> dict:
        """Get detector statistics."""
        avg_time = (self._total_processing_time / self._detection_count * 1000
                   if self._detection_count > 0 else 0)
        return {
            "name": self._name,
            "enabled": self._enabled,
            "detection_count": self._detection_count,
            "avg_processing_time_ms": avg_time
        }
    
    def cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        pass
