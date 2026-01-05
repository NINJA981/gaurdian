"""
FORGE-Guard Gesture Detector
Module 3: Emergency Gesture Detection using MediaPipe Hands.
"""

import time
from typing import Optional, Tuple, List

from .base_detector import BaseDetector, DetectionResult, AlertLevel
from ..config import config
from ..utils.safe_imports import get_cv2, get_mediapipe, get_numpy

# Get safe module handles
cv2 = get_cv2()
mp = get_mediapipe()
np = get_numpy()


class GestureDetector(BaseDetector):
    """
    Emergency gesture detection using MediaPipe Hands.
    
    Detection Logic:
    1. Detect open palm (SOS gesture)
    2. Calculate thumb-to-pinky distance
    3. Check all fingers are extended
    4. Require 3-second continuous detection for confirmation
    """
    
    # Hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    
    def __init__(
        self,
        hold_seconds: Optional[float] = None,
        palm_threshold: Optional[float] = None
    ):
        """
        Initialize gesture detector.
        
        Args:
            hold_seconds: Seconds to hold gesture for confirmation
            palm_threshold: Distance threshold for open palm detection
        """
        super().__init__(name="gesture_detector")
        
        self.hold_seconds = hold_seconds or config.detection.gesture_hold_seconds
        self.palm_threshold = palm_threshold or config.detection.palm_distance_threshold
        
        # Initialize MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,  # Lite model for speed
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            self.ml_available = True
        except Exception as e:
            print(f"[GESTURE] MediaPipe Hands init failed: {e}. Gesture detection DISABLED.")
            self.hands = None
            self.ml_available = False
        
        # State tracking
        self._sos_start_time: Optional[float] = None
        self._last_hands_result = None
        self._gesture_confirmed = False
        self._alert_cooldown_time: Optional[float] = None
        self._cooldown_seconds = 30.0
    
    def _calculate_distance(
        self, 
        landmarks, 
        idx1: int, 
        idx2: int
    ) -> float:
        """Calculate Euclidean distance between two landmarks."""
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def _is_finger_extended(
        self, 
        landmarks, 
        tip_idx: int, 
        mcp_idx: int
    ) -> bool:
        """Check if a finger is extended (tip above MCP)."""
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        # Y increases downward, so tip.y < mcp.y means extended
        return tip.y < mcp.y
    
    def _is_thumb_extended(self, landmarks) -> bool:
        """Check if thumb is extended (special case for thumb)."""
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]
        
        # Thumb extended means tip is further from wrist than MCP
        tip_dist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
        mcp_dist = np.sqrt((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2)
        
        return tip_dist > mcp_dist
    
    def _is_open_palm(self, landmarks) -> Tuple[bool, float]:
        """
        Detect open palm gesture.
        
        Returns:
            (is_open_palm, confidence)
        """
        # Check all fingers extended
        fingers_extended = [
            self._is_thumb_extended(landmarks),
            self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP),
            self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP),
            self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP),
            self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP)
        ]
        
        extended_count = sum(fingers_extended)
        
        # Calculate palm spread (thumb to pinky distance)
        thumb_pinky_dist = self._calculate_distance(
            landmarks, self.THUMB_TIP, self.PINKY_TIP
        )
        
        # Calculate hand size for normalization
        wrist_middle_dist = self._calculate_distance(
            landmarks, self.WRIST, self.MIDDLE_TIP
        )
        
        # Normalized spread ratio
        spread_ratio = thumb_pinky_dist / wrist_middle_dist if wrist_middle_dist > 0 else 0
        
        # Open palm conditions:
        # 1. At least 4 fingers extended
        # 2. Good spread ratio (fingers apart)
        is_open = extended_count >= 4 and spread_ratio > 0.8
        
        # Confidence based on how well conditions are met
        confidence = (extended_count / 5) * min(spread_ratio / 1.0, 1.0)
        
        return is_open, confidence
    
    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process frame for gesture detection."""
        current_time = time.time()
        
        if not self.ml_available or self.hands is None:
             return DetectionResult(
                detected=False,
                message="ML Offline (Gesture Detection Disabled)",
                alert_level=AlertLevel.NONE,
                details={"error": "mediapipe_missing"}
            )

        # Check cooldown
        if self._alert_cooldown_time is not None:
            if current_time - self._alert_cooldown_time < self._cooldown_seconds:
                return DetectionResult(
                    detected=False,
                    message="SOS cooldown active",
                    alert_level=AlertLevel.NONE
                )
            else:
                self._alert_cooldown_time = None
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        self._last_hands_result = results
        
        if not results.multi_hand_landmarks:
            self._sos_start_time = None
            self._gesture_confirmed = False
            return DetectionResult(
                detected=False,
                message="No hands detected",
                alert_level=AlertLevel.NONE
            )
        
        # Check each detected hand for SOS gesture
        sos_detected = False
        best_confidence = 0.0
        hand_positions = []
        
        for hand_landmarks, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            landmarks = hand_landmarks.landmark
            is_open, confidence = self._is_open_palm(landmarks)
            
            if is_open and confidence > best_confidence:
                sos_detected = True
                best_confidence = confidence
                
                # Get hand bounding box
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                h, w = frame.shape[:2]
                
                hand_positions.append((
                    int(min(x_coords) * w),
                    int(min(y_coords) * h),
                    int(max(x_coords) * w),
                    int(max(y_coords) * h)
                ))
        
        if sos_detected:
            # Start or continue timing
            if self._sos_start_time is None:
                self._sos_start_time = current_time
            
            elapsed = current_time - self._sos_start_time
            remaining = self.hold_seconds - elapsed
            
            if elapsed >= self.hold_seconds:
                # SOS confirmed!
                self._sos_start_time = None
                self._alert_cooldown_time = current_time
                
                return DetectionResult(
                    detected=True,
                    alert_level=AlertLevel.CRITICAL,
                    confidence=best_confidence,
                    message="🆘 SOS GESTURE DETECTED! Emergency help requested!",
                    details={
                        "gesture": "open_palm",
                        "hold_time": self.hold_seconds
                    },
                    bounding_boxes=hand_positions
                )
            else:
                # Still holding
                return DetectionResult(
                    detected=False,
                    confidence=best_confidence,
                    message=f"✋ Hold SOS gesture: {remaining:.1f}s remaining",
                    alert_level=AlertLevel.MEDIUM,
                    details={
                        "gesture": "open_palm",
                        "elapsed": elapsed,
                        "remaining": remaining
                    },
                    bounding_boxes=hand_positions
                )
        else:
            self._sos_start_time = None
            return DetectionResult(
                detected=False,
                message="Monitoring for SOS gesture (open palm)",
                alert_level=AlertLevel.NONE
            )
    
    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw hand landmarks and gesture status overlay."""
        if not self.ml_available:
             cv2.putText(frame, "Gesture Detection: OFFLINE", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
             return frame

        h, w = frame.shape[:2]
        
        if self._last_hands_result and self._last_hands_result.multi_hand_landmarks:
            for hand_landmarks in self._last_hands_result.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 128),  # Bright green
                        thickness=2,
                        circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(128, 128, 255),  # Light purple
                        thickness=2
                    )
                )
        
        # Draw SOS progress/status
        if result.alert_level == AlertLevel.MEDIUM:
            # Holding gesture - show progress
            remaining = result.details.get("remaining", 0)
            progress = 1 - (remaining / self.hold_seconds)
            
            # Progress bar
            bar_width = 300
            bar_height = 30
            bar_x = (w - bar_width) // 2
            bar_y = h - 80
            
            # Background
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Progress fill
            fill_width = int(bar_width * progress)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                (0, 200, 255),  # Orange
                -1
            )
            
            # Text
            cv2.putText(
                frame,
                f"✋ SOS: {remaining:.1f}s",
                (bar_x + bar_width // 2 - 60, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        elif result.detected:
            # SOS confirmed!
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 180), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(
                frame,
                "🆘 SOS EMERGENCY!",
                (w // 2 - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3
            )
            
            # Draw bounding boxes
            for bbox in result.bounding_boxes:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        else:
            # Normal monitoring state
            cv2.putText(
                frame,
                "✋ SOS: Show open palm for 3s",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
        
        return frame
    
    def cleanup(self):
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()
