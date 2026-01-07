"""
FORGE-Guard Fall Detector
Production-ready fall detection using MediaPipe Pose estimation.
Includes graceful degradation, calibration support, and false positive reduction.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..utils.safe_imports import get_numpy, get_mediapipe

# Setup logging
logger = logging.getLogger(__name__)

# Get numpy
np = get_numpy()

# ============================================================================
# SAFE MEDIAPIPE IMPORT
# ============================================================================

MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_drawing = None

try:
    mp = get_mediapipe()
    if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        logger.info("[FALL] MediaPipe Pose loaded successfully")
    else:
        logger.warning("[FALL] MediaPipe solutions not available")
except Exception as e:
    logger.warning(f"[FALL] MediaPipe initialization error: {e}")


# ============================================================================
# DATA CLASSES
# ============================================================================

class FallState(Enum):
    """Fall detection state machine states."""
    NORMAL = "normal"
    MONITORING = "monitoring"  # Person visible, tracking
    POTENTIAL_FALL = "potential_fall"  # Fall indicators detected
    CONFIRMED_FALL = "confirmed_fall"  # Fall confirmed
    RECOVERY = "recovery"  # Person getting up


@dataclass
class PoseMetrics:
    """Metrics extracted from pose estimation."""
    center_y: float = 0.0
    hip_height: float = 0.0
    shoulder_height: float = 0.0
    body_angle: float = 0.0  # Degrees from vertical
    is_horizontal: bool = False
    confidence: float = 0.0
    head_y: float = 0.0
    velocity_y: float = 0.0  # Vertical velocity


# ============================================================================
# FALL DETECTOR
# ============================================================================

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
    - Standing height calibration
    - Recovery detection
    """
    
    def __init__(
        self,
        fall_ratio_threshold: float = 0.5,
        fall_speed_threshold: float = 0.3,
        confirmation_time: float = 2.0,
        recovery_time: float = 3.0,
        sensitivity: float = 0.7,
        min_confidence: float = 0.5,
        calibration_frames: int = 30
    ):
        """
        Initialize Fall Detector.
        
        Args:
            fall_ratio_threshold: Height ratio below which is considered fallen
            fall_speed_threshold: Rate of height change to trigger detection
            confirmation_time: Seconds person must stay down to confirm fall
            recovery_time: Seconds of standing to reset after fall
            sensitivity: Detection sensitivity (0.1 to 1.0)
            min_confidence: Minimum pose confidence to process
            calibration_frames: Frames needed for initial calibration
        """
        super().__init__(name="fall_detector")
        
        # Configuration
        self.fall_ratio_threshold = fall_ratio_threshold
        self.fall_speed_threshold = fall_speed_threshold
        self.confirmation_time = confirmation_time
        self.recovery_time = recovery_time
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.min_confidence = min_confidence
        self.calibration_frames = calibration_frames
        
        # State tracking
        self._pose = None
        self._height_history: List[Tuple[float, float]] = []  # (height, timestamp)
        self._history_max_size = 60  # ~2 seconds at 30fps
        self._standing_height: Optional[float] = None
        self._standing_height_samples: List[float] = []
        self._fall_state = FallState.NORMAL
        self._state_start_time: Optional[float] = None
        self._last_metrics: Optional[PoseMetrics] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._total_falls_detected = 0
        self._false_positives_avoided = 0
        self._calibrated = False
        
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
    
    def calibrate(self, standing_height: float):
        """
        Manually calibrate standing height reference.
        
        Args:
            standing_height: Normalized Y position (0-1) of person's center when standing
        """
        with self._lock:
            self._standing_height = standing_height
            self._calibrated = True
            logger.info(f"[FALL] Calibrated standing height: {standing_height:.3f}")
    
    def _process_frame(self, frame) -> DetectionResult:
        """
        Process a frame for fall detection.
        
        Args:
            frame: BGR image from camera (numpy array)
            
        Returns:
            DetectionResult with detection status and metadata
        """
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
        
        if np is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message="Fall detection offline",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled", "reason": "NumPy unavailable"}
            )
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = frame[:, :, ::-1] if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            
            # Process frame
            results = self._pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                # No person detected - check if we were tracking a fall
                with self._lock:
                    if self._fall_state == FallState.CONFIRMED_FALL:
                        # Person might have been helped up or left frame
                        return DetectionResult(
                            detected=True,
                            confidence=0.8,
                            message="⚠️ Person left frame during fall alert",
                            alert_level=AlertLevel.HIGH,
                            details={"status": "person_lost", "fall_state": self._fall_state.value}
                        )
                    self._fall_state = FallState.NORMAL
                
                return DetectionResult(
                    detected=False,
                    confidence=0.0,
                    message="No person detected",
                    alert_level=AlertLevel.NONE,
                    details={"status": "no_pose"}
                )
            
            # Extract pose metrics
            metrics = self._extract_metrics(results.pose_landmarks, frame.shape, current_time)
            
            with self._lock:
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
            
            # Calibration phase - collect standing height samples
            if not self._calibrated:
                return self._handle_calibration(metrics, results.pose_landmarks, frame.shape)
            
            # Check for fall using state machine
            return self._update_fall_state(metrics, results.pose_landmarks, frame.shape, current_time)
            
        except Exception as e:
            logger.error(f"[FALL] Processing error: {e}")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message=f"Detection error: {str(e)}",
                alert_level=AlertLevel.NONE,
                details={"status": "error", "error": str(e)}
            )
    
    def _handle_calibration(self, metrics: PoseMetrics, landmarks, frame_shape) -> DetectionResult:
        """Handle calibration phase to establish standing height."""
        with self._lock:
            # Only collect samples when person appears to be standing (body angle < 30°)
            if metrics.body_angle < 30 and metrics.confidence > 0.6:
                self._standing_height_samples.append(metrics.center_y)
            
            if len(self._standing_height_samples) >= self.calibration_frames:
                # Calculate standing height as the median of samples (more robust than min)
                self._standing_height = sorted(self._standing_height_samples)[len(self._standing_height_samples) // 2]
                self._calibrated = True
                logger.info(f"[FALL] Auto-calibrated standing height: {self._standing_height:.3f}")
                
                return DetectionResult(
                    detected=False,
                    confidence=metrics.confidence,
                    message="✅ Calibration complete - Monitoring active",
                    alert_level=AlertLevel.NONE,
                    bounding_box=self._get_bounding_box(landmarks, frame_shape),
                    details={
                        "status": "calibrated",
                        "standing_height": self._standing_height,
                        "samples": len(self._standing_height_samples)
                    }
                )
            
            progress = len(self._standing_height_samples) / self.calibration_frames * 100
            return DetectionResult(
                detected=False,
                confidence=metrics.confidence,
                message=f"📐 Calibrating... {progress:.0f}% (stand normally)",
                alert_level=AlertLevel.NONE,
                bounding_box=self._get_bounding_box(landmarks, frame_shape),
                details={
                    "status": "calibrating",
                    "progress": progress,
                    "samples": len(self._standing_height_samples)
                }
            )
    
    def _update_fall_state(self, metrics: PoseMetrics, landmarks, frame_shape, current_time: float) -> DetectionResult:
        """Update fall detection state machine."""
        with self._lock:
            # Check for fall indicators
            is_falling, fall_reasons = self._check_for_fall(metrics, current_time)
            
            # State machine logic
            if self._fall_state == FallState.NORMAL or self._fall_state == FallState.MONITORING:
                if is_falling:
                    self._fall_state = FallState.POTENTIAL_FALL
                    self._state_start_time = current_time
                else:
                    self._fall_state = FallState.MONITORING
                    
            elif self._fall_state == FallState.POTENTIAL_FALL:
                if is_falling:
                    time_in_state = current_time - (self._state_start_time or current_time)
                    
                    if time_in_state >= self.confirmation_time:
                        # CONFIRMED FALL
                        self._fall_state = FallState.CONFIRMED_FALL
                        self._total_falls_detected += 1
                        self._state_start_time = current_time
                        
                        return DetectionResult(
                            detected=True,
                            confidence=metrics.confidence,
                            message=f"🚨 FALL DETECTED - Person down for {time_in_state:.1f}s",
                            alert_level=AlertLevel.CRITICAL,
                            bounding_box=self._get_bounding_box(landmarks, frame_shape),
                            details={
                                "status": "fall_confirmed",
                                "reason": fall_reasons,
                                "time_down": time_in_state,
                                "metrics": self._metrics_to_dict(metrics),
                                "total_falls": self._total_falls_detected,
                                "fall_state": self._fall_state.value
                            }
                        )
                    else:
                        # Potential fall - waiting for confirmation
                        return DetectionResult(
                            detected=True,
                            confidence=metrics.confidence * 0.7,
                            message=f"⚠️ Potential fall - confirming ({time_in_state:.1f}s)",
                            alert_level=AlertLevel.HIGH,
                            bounding_box=self._get_bounding_box(landmarks, frame_shape),
                            details={
                                "status": "potential_fall",
                                "reason": fall_reasons,
                                "time_down": time_in_state,
                                "confirmation_needed": self.confirmation_time - time_in_state,
                                "metrics": self._metrics_to_dict(metrics),
                                "fall_state": self._fall_state.value
                            }
                        )
                else:
                    # False positive - person recovered
                    self._fall_state = FallState.MONITORING
                    self._false_positives_avoided += 1
                    self._state_start_time = None
                    
            elif self._fall_state == FallState.CONFIRMED_FALL:
                if not is_falling:
                    # Person may be recovering
                    self._fall_state = FallState.RECOVERY
                    self._state_start_time = current_time
                else:
                    # Still down - continue alert
                    time_down = current_time - (self._state_start_time or current_time)
                    return DetectionResult(
                        detected=True,
                        confidence=metrics.confidence,
                        message=f"🚨 FALL ALERT ACTIVE - Person down for {time_down:.1f}s",
                        alert_level=AlertLevel.CRITICAL,
                        bounding_box=self._get_bounding_box(landmarks, frame_shape),
                        details={
                            "status": "fall_active",
                            "time_down": time_down,
                            "metrics": self._metrics_to_dict(metrics),
                            "fall_state": self._fall_state.value
                        }
                    )
                    
            elif self._fall_state == FallState.RECOVERY:
                if is_falling:
                    # Back to confirmed fall
                    self._fall_state = FallState.CONFIRMED_FALL
                else:
                    time_recovering = current_time - (self._state_start_time or current_time)
                    if time_recovering >= self.recovery_time:
                        # Fully recovered
                        self._fall_state = FallState.MONITORING
                        self._state_start_time = None
                        logger.info("[FALL] Person recovered from fall")
                    else:
                        return DetectionResult(
                            detected=True,
                            confidence=metrics.confidence * 0.5,
                            message=f"Person recovering ({time_recovering:.1f}s)",
                            alert_level=AlertLevel.MEDIUM,
                            bounding_box=self._get_bounding_box(landmarks, frame_shape),
                            details={
                                "status": "recovering",
                                "time_recovering": time_recovering,
                                "fall_state": self._fall_state.value
                            }
                        )
            
            # Normal monitoring state
            return DetectionResult(
                detected=False,
                confidence=metrics.confidence,
                message="Person standing/moving normally",
                alert_level=AlertLevel.NONE,
                bounding_box=self._get_bounding_box(landmarks, frame_shape),
                details={
                    "status": "normal",
                    "metrics": self._metrics_to_dict(metrics),
                    "fall_state": self._fall_state.value
                }
            )
    
    def _extract_metrics(self, landmarks, frame_shape, current_time: float) -> PoseMetrics:
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
        
        # Calculate velocity from history
        velocity_y = 0.0
        with self._lock:
            if len(self._height_history) >= 2:
                prev_height, prev_time = self._height_history[-1]
                dt = current_time - prev_time
                if dt > 0:
                    velocity_y = (center_y - prev_height) / dt
        
        # Average visibility as confidence
        confidence = (left_shoulder[2] + right_shoulder[2] + 
                     left_hip[2] + right_hip[2]) / 4
        
        return PoseMetrics(
            center_y=center_y,
            hip_height=hip_y,
            shoulder_height=shoulder_y,
            body_angle=angle,
            is_horizontal=angle > 60,
            confidence=confidence,
            head_y=nose[1],
            velocity_y=velocity_y
        )
    
    def _update_height_history(self, height: float, timestamp: float):
        """Update height history for tracking."""
        with self._lock:
            self._height_history.append((height, timestamp))
            
            # Trim to max size
            if len(self._height_history) > self._history_max_size:
                self._height_history = self._height_history[-self._history_max_size:]
    
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
        velocity_threshold = self.fall_speed_threshold * self.sensitivity
        
        # Check 1: Body is horizontal
        if metrics.body_angle > angle_threshold:
            reasons.append(f"horizontal_body({metrics.body_angle:.1f}°)")
        
        # Check 2: Height dropped significantly from standing
        if self._standing_height is not None:
            height_ratio = metrics.center_y / max(self._standing_height, 0.01)
            # Higher center_y means lower position in frame
            if height_ratio > (1 + height_threshold):
                reasons.append(f"low_position({height_ratio:.2f}x)")
        
        # Check 3: Rapid downward velocity
        if metrics.velocity_y > velocity_threshold:
            reasons.append(f"rapid_drop(v={metrics.velocity_y:.2f})")
        
        # Check 4: Height change over recent history
        with self._lock:
            if len(self._height_history) >= 10:
                recent_heights = [h for h, t in self._height_history[-10:]]
                height_range = max(recent_heights) - min(recent_heights)
                if height_range > self.fall_speed_threshold:
                    reasons.append(f"height_variance({height_range:.2f})")
        
        # Fall detected if multiple indicators present OR body is horizontal
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
            "confidence": round(metrics.confidence, 3),
            "velocity_y": round(metrics.velocity_y, 3)
        }
    
    def draw_overlay(self, frame, result: DetectionResult):
        """Draw detection overlay on frame."""
        cv2 = __import__('cv2')
        
        if not MEDIAPIPE_AVAILABLE or self._pose is None:
            # Draw "OFFLINE" indicator
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 100), -1)
            cv2.putText(overlay, "FALL DETECT: OFFLINE", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            return overlay
        
        try:
            overlay = frame.copy()
            
            # Draw bounding box if detected
            if result.bounding_box:
                x1, y1, x2, y2 = result.bounding_box
                if result.alert_level == AlertLevel.CRITICAL:
                    color = (0, 0, 255)  # Red
                elif result.alert_level == AlertLevel.HIGH:
                    color = (0, 165, 255)  # Orange
                elif result.detected:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw status indicator
            status_color = (0, 0, 255) if result.detected else (0, 255, 0)
            cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (280, 70), status_color, 2)
            
            # Status text
            state = result.details.get('fall_state', 'unknown')
            cv2.putText(overlay, f"Fall Detector: {state}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show calibration or confidence
            if not self._calibrated:
                progress = result.details.get('progress', 0)
                cv2.putText(overlay, f"Calibrating: {progress:.0f}%", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(overlay, f"Conf: {result.confidence:.1%}", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            return overlay
            
        except Exception:
            return frame
    
    def reset(self):
        """Reset detector state."""
        with self._lock:
            self._height_history.clear()
            self._standing_height = None
            self._standing_height_samples.clear()
            self._fall_state = FallState.NORMAL
            self._state_start_time = None
            self._last_metrics = None
            self._calibrated = False
    
    def cleanup(self):
        """Clean up resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        base_stats = super().stats()
        with self._lock:
            base_stats.update({
                "total_falls_detected": self._total_falls_detected,
                "false_positives_avoided": self._false_positives_avoided,
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "sensitivity": self.sensitivity,
                "confirmation_time": self.confirmation_time,
                "calibrated": self._calibrated,
                "standing_height": self._standing_height,
                "current_state": self._fall_state.value
            })
        return base_stats
