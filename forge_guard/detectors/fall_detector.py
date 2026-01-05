"""
FORGE-Guard Fall Detector
Production-ready fall detection using MediaPipe Pose estimation.
Includes graceful degradation when MediaPipe is unavailable.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

from .base_detector import BaseDetector, DetectionResult, AlertLevel

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# SAFE MEDIAPIPE IMPORT
# ============================================================================

MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_drawing = None

try:
    import mediapipe as mp
    # Check if solutions attribute exists (compatibility check)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        logger.info("[FALL] MediaPipe Pose loaded successfully")
    else:
        logger.warning("[FALL] MediaPipe solutions not available")
except ImportError as e:
    logger.warning(f"[FALL] MediaPipe not installed: {e}")
except Exception as e:
    logger.warning(f"[FALL] MediaPipe initialization error: {e}")

# ============================================================================
# FALL DETECTOR
# ============================================================================

@dataclass
class PoseMetrics:
    """Metrics extracted from pose estimation."""
    center_y: float = 0.0
    hip_height: float = 0.0
    shoulder_height: float = 0.0
    body_angle: float = 0.0  # Degrees from vertical
    is_horizontal: bool = False
    confidence: float = 0.0


class FallDetector(BaseDetector):
    """
    Detects falls using MediaPipe Pose estimation.
    
    Detection Logic:
    1. Track vertical position of body center over time
    2. Detect rapid drops in height
    3. Check if body becomes horizontal
    4. Confirm fall if person stays down
    
    Features:
    - Graceful degradation when MediaPipe unavailable
    - Configurable sensitivity and thresholds
    - False positive reduction via confirmation time
    """
    
    def __init__(
        self,
        fall_ratio_threshold: float = 0.5,
        fall_speed_threshold: float = 0.3,
        confirmation_time: float = 2.0,
        sensitivity: float = 0.7,
        min_confidence: float = 0.5
    ):
        """
        Initialize Fall Detector.
        
        Args:
            fall_ratio_threshold: Height ratio below which is considered fallen
            fall_speed_threshold: Rate of height change to trigger detection
            confirmation_time: Seconds person must stay down to confirm fall
            sensitivity: Detection sensitivity (0.1 to 1.0)
            min_confidence: Minimum pose confidence to process
        """
        super().__init__(name="fall_detector")
        
        # Configuration
        self.fall_ratio_threshold = fall_ratio_threshold
        self.fall_speed_threshold = fall_speed_threshold
        self.confirmation_time = confirmation_time
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.min_confidence = min_confidence
        
        # State tracking
        self._pose = None
        self._height_history: list = []
        self._history_max_size = 30  # ~1 second at 30fps
        self._initial_height: Optional[float] = None
        self._fall_detected_time: Optional[float] = None
        self._last_metrics: Optional[PoseMetrics] = None
        
        # Statistics
        self._total_falls_detected = 0
        self._false_positive_count = 0
        
        # Initialize MediaPipe Pose
        self._init_pose()
    
    def _init_pose(self):
        """Initialize MediaPipe Pose with error handling."""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("[FALL] MediaPipe not available - Fall detection DISABLED")
            self._enabled = False
            return
        
        try:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=lite, 1=full, 2=heavy
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self._enabled = True
            logger.info("[FALL] Pose estimator initialized successfully")
        except Exception as e:
            logger.error(f"[FALL] Failed to initialize Pose: {e}")
            self._enabled = False
    
    def process(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a frame for fall detection.
        
        Args:
            frame: BGR image from camera (numpy array)
            
        Returns:
            DetectionResult with detection status and metadata
        """
        self._frames_processed += 1
        current_time = time.time()
        
        # Check if detector is enabled
        if not self._enabled or self._pose is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message="Fall detection offline",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled", "reason": "MediaPipe unavailable"}
            )
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
            
            # Process frame
            results = self._pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return DetectionResult(
                    detected=False,
                    confidence=0.0,
                    message="No person detected",
                    alert_level=AlertLevel.NONE,
                    details={"status": "no_pose"}
                )
            
            # Extract pose metrics
            metrics = self._extract_metrics(results.pose_landmarks, frame.shape)
            self._last_metrics = metrics
            
            if metrics.confidence < self.min_confidence:
                return DetectionResult(
                    detected=False,
                    confidence=metrics.confidence,
                    message="Low confidence pose",
                    alert_level=AlertLevel.NONE,
                    details={"status": "low_confidence"}
                )
            
            # Update height history
            self._update_height_history(metrics.center_y, current_time)
            
            # Set initial height reference
            if self._initial_height is None and len(self._height_history) >= 10:
                self._initial_height = self._calculate_standing_height()
            
            # Check for fall
            fall_detected, fall_reason = self._check_for_fall(metrics, current_time)
            
            if fall_detected:
                self._detections_count += 1
                self._last_detection_time = current_time
                
                # Check confirmation time
                if self._fall_detected_time is None:
                    self._fall_detected_time = current_time
                
                time_down = current_time - self._fall_detected_time
                
                if time_down >= self.confirmation_time:
                    # Confirmed fall!
                    self._total_falls_detected += 1
                    return DetectionResult(
                        detected=True,
                        confidence=metrics.confidence,
                        message=f"🚨 FALL DETECTED - Person down for {time_down:.1f}s",
                        alert_level=AlertLevel.CRITICAL,
                        bounding_box=self._get_bounding_box(results.pose_landmarks, frame.shape),
                        details={
                            "status": "fall_confirmed",
                            "reason": fall_reason,
                            "time_down": time_down,
                            "metrics": self._metrics_to_dict(metrics),
                            "total_falls": self._total_falls_detected
                        }
                    )
                else:
                    # Potential fall - waiting for confirmation
                    return DetectionResult(
                        detected=True,
                        confidence=metrics.confidence * 0.7,
                        message=f"⚠️ Potential fall - confirming ({time_down:.1f}s)",
                        alert_level=AlertLevel.HIGH,
                        bounding_box=self._get_bounding_box(results.pose_landmarks, frame.shape),
                        details={
                            "status": "potential_fall",
                            "reason": fall_reason,
                            "time_down": time_down,
                            "metrics": self._metrics_to_dict(metrics)
                        }
                    )
            else:
                # No fall - reset detection state
                self._fall_detected_time = None
                
                return DetectionResult(
                    detected=False,
                    confidence=metrics.confidence,
                    message="Person standing/moving normally",
                    alert_level=AlertLevel.NONE,
                    bounding_box=self._get_bounding_box(results.pose_landmarks, frame.shape),
                    details={
                        "status": "normal",
                        "metrics": self._metrics_to_dict(metrics)
                    }
                )
                
        except Exception as e:
            logger.error(f"[FALL] Processing error: {e}")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message=f"Detection error: {str(e)}",
                alert_level=AlertLevel.NONE,
                details={"status": "error", "error": str(e)}
            )
    
    def _extract_metrics(self, landmarks, frame_shape) -> PoseMetrics:
        """Extract pose metrics from landmarks."""
        h, w = frame_shape[:2]
        
        # Key landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        NOSE = 0
        
        # Get landmark positions (normalized 0-1)
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return (lm.x, lm.y, lm.visibility)
        
        left_shoulder = get_point(LEFT_SHOULDER)
        right_shoulder = get_point(RIGHT_SHOULDER)
        left_hip = get_point(LEFT_HIP)
        right_hip = get_point(RIGHT_HIP)
        nose = get_point(NOSE)
        
        # Calculate averages
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        center_y = (shoulder_y + hip_y) / 2
        
        # Calculate body angle (vertical = 0 degrees)
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                          (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                     (left_hip[1] + right_hip[1]) / 2)
        
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        
        if abs(dy) > 0.001:
            angle = abs(np.degrees(np.arctan(dx / dy)))
        else:
            angle = 90.0  # Horizontal
        
        # Average visibility as confidence
        confidence = (left_shoulder[2] + right_shoulder[2] + 
                     left_hip[2] + right_hip[2]) / 4
        
        return PoseMetrics(
            center_y=center_y,
            hip_height=hip_y,
            shoulder_height=shoulder_y,
            body_angle=angle,
            is_horizontal=angle > 60,
            confidence=confidence
        )
    
    def _update_height_history(self, height: float, timestamp: float):
        """Update height history for tracking."""
        self._height_history.append((height, timestamp))
        
        # Trim to max size
        if len(self._height_history) > self._history_max_size:
            self._height_history = self._height_history[-self._history_max_size:]
    
    def _calculate_standing_height(self) -> float:
        """Calculate reference standing height from history."""
        if len(self._height_history) < 5:
            return 0.3  # Default
        
        # Use minimum height (highest position in frame) as standing reference
        heights = [h for h, t in self._height_history]
        return min(heights)
    
    def _check_for_fall(self, metrics: PoseMetrics, current_time: float) -> Tuple[bool, str]:
        """
        Check if current pose indicates a fall.
        
        Returns:
            Tuple of (fall_detected, reason)
        """
        reasons = []
        
        # Adjust thresholds based on sensitivity
        height_threshold = self.fall_ratio_threshold * (2.0 - self.sensitivity)
        angle_threshold = 45 + (15 * (1 - self.sensitivity))
        
        # Check 1: Body is horizontal
        if metrics.body_angle > angle_threshold:
            reasons.append(f"horizontal_body({metrics.body_angle:.1f}°)")
        
        # Check 2: Height dropped significantly from standing
        if self._initial_height is not None:
            height_ratio = metrics.center_y / max(self._initial_height, 0.01)
            if height_ratio > (1 + height_threshold):  # Lower in frame = larger y
                reasons.append(f"low_position({height_ratio:.2f}x)")
        
        # Check 3: Rapid height change
        if len(self._height_history) >= 5:
            recent_heights = [h for h, t in self._height_history[-5:]]
            height_change = max(recent_heights) - min(recent_heights)
            if height_change > self.fall_speed_threshold * self.sensitivity:
                reasons.append(f"rapid_drop({height_change:.2f})")
        
        # Fall detected if multiple indicators present
        is_fall = len(reasons) >= 2 or (len(reasons) >= 1 and metrics.is_horizontal)
        
        return is_fall, ", ".join(reasons) if reasons else "none"
    
    def _get_bounding_box(self, landmarks, frame_shape) -> Tuple[int, int, int, int]:
        """Get bounding box around detected pose."""
        h, w = frame_shape[:2]
        
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        
        x_min = int(min(xs) * w)
        x_max = int(max(xs) * w)
        y_min = int(min(ys) * h)
        y_max = int(max(ys) * h)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def _metrics_to_dict(self, metrics: PoseMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "center_y": round(metrics.center_y, 3),
            "hip_height": round(metrics.hip_height, 3),
            "shoulder_height": round(metrics.shoulder_height, 3),
            "body_angle": round(metrics.body_angle, 1),
            "is_horizontal": metrics.is_horizontal,
            "confidence": round(metrics.confidence, 3)
        }
    
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection overlay on frame."""
        if not MEDIAPIPE_AVAILABLE or self._pose is None:
            # Draw "OFFLINE" indicator
            cv2 = __import__('cv2')
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 100), -1)
            cv2.putText(overlay, "FALL DETECT: OFFLINE", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            return overlay
        
        # Use OpenCV for drawing
        try:
            cv2 = __import__('cv2')
            overlay = frame.copy()
            
            # Draw bounding box if detected
            if result.bounding_box:
                x1, y1, x2, y2 = result.bounding_box
                color = (0, 0, 255) if result.detected else (0, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw status indicator
            status_color = (0, 0, 255) if result.detected else (0, 255, 0)
            cv2.rectangle(overlay, (10, 10), (250, 60), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (250, 60), status_color, 2)
            
            status_text = "FALL DETECTED!" if result.detected else "Monitoring..."
            cv2.putText(overlay, status_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            return overlay
            
        except Exception:
            return frame
    
    def reset(self):
        """Reset detector state."""
        self._height_history.clear()
        self._initial_height = None
        self._fall_detected_time = None
        self._last_metrics = None
    
    def cleanup(self):
        """Clean up resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        base_stats = super().stats()
        base_stats.update({
            "total_falls_detected": self._total_falls_detected,
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "sensitivity": self.sensitivity,
            "confirmation_time": self.confirmation_time
        })
        return base_stats
