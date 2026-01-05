"""
FORGE-Guard Fall Detector
Module 1: Fall Detection using MediaPipe Pose and geometry analysis.
"""

import time
from collections import deque
from typing import Optional, Tuple

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_mediapipe, get_numpy

# Get safe module handles
cv2 = get_cv2()
mp = get_mediapipe()
np = get_numpy()


class FallDetector(BaseDetector):
    """
    Fall detection using MediaPipe Pose.
    
    Detection Logic:
    1. Calculate bounding box ratio (width/height) from pose landmarks
    2. Track mid-hip velocity across frames
    3. Monitor shoulder-to-ankle distance changes
    4. Trigger alert if ratio < 0.8 sustained for 5 frames
    """
    
    # MediaPipe Pose landmarks indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    def __init__(
        self,
        ratio_threshold: Optional[float] = None,
        frame_threshold: Optional[int] = None,
        velocity_threshold: Optional[float] = None
    ):
        """
        Initialize fall detector.
        
        Args:
            ratio_threshold: Width/height ratio threshold for fall detection
            frame_threshold: Number of frames to confirm fall
            velocity_threshold: Mid-hip velocity threshold
        """
        super().__init__(name="fall_detector")
        
        self.ratio_threshold = ratio_threshold or config.detection.fall_ratio_threshold
        self.frame_threshold = frame_threshold or config.detection.fall_frame_threshold
        self.velocity_threshold = velocity_threshold or config.detection.fall_velocity_threshold
        
        # Initialize MediaPipe Pose
        try:
            if mp and hasattr(mp.solutions, "pose"):
                self.mp_pose = mp.solutions.pose
                self.mp_draw = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,  # 0=lite, 1=full, 2=heavy
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.ml_available = True
            else:
                raise ImportError("MediaPipe not available via safe_imports")
        except Exception as e:
            print(f"[FALL] MediaPipe Pose init failed: {e}. Fall detection DISABLED.")
            self.pose = None
            self.ml_available = False
        
        # State tracking
        self._fall_frame_count = 0
        self._hip_positions: deque = deque(maxlen=10)
        self._last_pose_result = None
        self._fall_detected_time: Optional[float] = None
        self._cooldown_seconds = 10.0  # Cooldown after fall detected
    
    def _calculate_bounding_box(self, landmarks) -> Tuple[int, int, int, int, float]:
        """
        Calculate bounding box from pose landmarks.
        
        Returns:
            (x_min, y_min, x_max, y_max, width/height ratio)
        """
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        width = x_max - x_min
        height = y_max - y_min
        
        ratio = width / height if height > 0 else 0
        
        return x_min, y_min, x_max, y_max, ratio
    
    def _get_mid_hip(self, landmarks) -> Tuple[float, float]:
        """Calculate mid-hip position."""
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        
        mid_x = (left_hip.x + right_hip.x) / 2
        mid_y = (left_hip.y + right_hip.y) / 2
        
        return mid_x, mid_y
    
    def _calculate_hip_velocity(self, current_hip: Tuple[float, float]) -> float:
        """Calculate vertical velocity of mid-hip."""
        if len(self._hip_positions) < 2:
            self._hip_positions.append(current_hip)
            return 0.0
        
        prev_hip = self._hip_positions[-1]
        self._hip_positions.append(current_hip)
        
        # Vertical velocity (positive = moving down)
        velocity = (current_hip[1] - prev_hip[1]) * 1000  # Scale for visibility
        
        return velocity
    
    def _get_shoulder_ankle_distance(self, landmarks) -> float:
        """Calculate average distance between shoulders and ankles."""
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_ankle = landmarks[self.LEFT_ANKLE]
        right_ankle = landmarks[self.RIGHT_ANKLE]
        
        # Average shoulder Y
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        # Average ankle Y
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        # Distance (should be large when standing, small when fallen)
        return abs(ankle_y - shoulder_y)
    
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process frame for fall detection."""
        # Check cooldown
        if not self.ml_available or self.pose is None:
            return DetectionResult(
                detected=False,
                message="ML Offline (Fall Detection Disabled)",
                alert_level=AlertLevel.NONE,
                details={"error": "mediapipe_missing"}
            )

        # Check cooldown
        if self._fall_detected_time is not None:
            if time.time() - self._fall_detected_time < self._cooldown_seconds:
                return DetectionResult(
                    detected=False,
                    message="Fall cooldown active",
                    alert_level=AlertLevel.NONE
                )
            else:
                self._fall_detected_time = None
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        self._last_pose_result = results
        
        if not results.pose_landmarks:
            self._fall_frame_count = 0
            return DetectionResult(
                detected=False,
                message="No person detected",
                alert_level=AlertLevel.NONE
            )
        
        landmarks = results.pose_landmarks.landmark
        
        # Calculate metrics
        x_min, y_min, x_max, y_max, ratio = self._calculate_bounding_box(landmarks)
        mid_hip = self._get_mid_hip(landmarks)
        hip_velocity = self._calculate_hip_velocity(mid_hip)
        shoulder_ankle_dist = self._get_shoulder_ankle_distance(landmarks)
        
        # Fall detection logic
        is_falling = False
        fall_confidence = 0.0
        
        # Check 1: Bounding box ratio (width > height suggests horizontal position)
        if ratio > (1 / self.ratio_threshold):  # Inverted because low ratio = vertical
            is_falling = True
            fall_confidence += 0.4
        
        # Check 2: Rapid downward hip movement
        if hip_velocity > self.velocity_threshold:
            is_falling = True
            fall_confidence += 0.3
        
        # Check 3: Small shoulder-to-ankle distance
        if shoulder_ankle_dist < 0.3:  # Thresholdfor compressed pose
            is_falling = True
            fall_confidence += 0.3
        
        # Frame count for sustained detection
        if is_falling:
            self._fall_frame_count += 1
        else:
            self._fall_frame_count = max(0, self._fall_frame_count - 1)
        
        # Confirm fall after threshold frames
        fall_confirmed = self._fall_frame_count >= self.frame_threshold
        
        if fall_confirmed:
            self._fall_detected_time = time.time()
            self._fall_frame_count = 0
            
            return DetectionResult(
                detected=True,
                alert_level=AlertLevel.CRITICAL,
                confidence=min(fall_confidence, 1.0),
                message="⚠️ FALL DETECTED! Immediate attention required!",
                details={
                    "ratio": ratio,
                    "hip_velocity": hip_velocity,
                    "shoulder_ankle_distance": shoulder_ankle_dist,
                    "consecutive_frames": self.frame_threshold
                },
                bounding_boxes=[(
                    int(x_min * frame.shape[1]),
                    int(y_min * frame.shape[0]),
                    int(x_max * frame.shape[1]),
                    int(y_max * frame.shape[0])
                )]
            )
        
        return DetectionResult(
            detected=False,
            confidence=fall_confidence,
            message=f"Monitoring - Ratio: {ratio:.2f}, Velocity: {hip_velocity:.1f}",
            details={
                "ratio": ratio,
                "hip_velocity": hip_velocity,
                "shoulder_ankle_distance": shoulder_ankle_dist,
                "fall_frame_count": self._fall_frame_count
            }
        )
    
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw pose skeleton and fall detection overlay."""
        if not self.ml_available:
            cv2.putText(frame, "Fall Detection: OFFLINE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        if self._last_pose_result and self._last_pose_result.pose_landmarks:
            # Draw pose landmarks
            self.mp_draw.draw_landmarks(
                frame,
                self._last_pose_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(0, 255, 255),  # Cyan
                    thickness=2,
                    circle_radius=3
                ),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(255, 128, 0),  # Orange
                    thickness=2
                )
            )
        
        h, w = frame.shape[:2]
        
        # Draw status
        if result.detected:
            # Draw alert overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 180), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(
                frame,
                "🚨 FALL DETECTED!",
                (w // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )
            
            # Draw bounding boxes
            for bbox in result.bounding_boxes:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        else:
            # Show monitoring status
            ratio = result.details.get("ratio", 0)
            velocity = result.details.get("hip_velocity", 0)
            
            cv2.putText(
                frame,
                f"Fall Monitor: Ratio={ratio:.2f} Vel={velocity:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return frame
    
    def cleanup(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
