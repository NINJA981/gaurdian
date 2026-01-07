"""
FORGE-Guard Gesture Detector
Production-ready gesture recognition using MediaPipe Hands.
Detects SOS and help gestures for elderly assistance.
"""

import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

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
mp_hands = None
mp_drawing = None

try:
    mp = get_mediapipe()
    if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        logger.info("[GESTURE] MediaPipe Hands loaded successfully")
    else:
        logger.warning("[GESTURE] MediaPipe solutions not available")
except Exception as e:
    logger.warning(f"[GESTURE] MediaPipe initialization error: {e}")


# ============================================================================
# GESTURE TYPES
# ============================================================================

class GestureType(Enum):
    """Recognized gesture types."""
    NONE = "none"
    WAVE = "wave"
    RAISED_HAND = "raised_hand"
    SOS = "sos"
    HELP = "help"
    THUMBS_UP = "thumbs_up"
    OPEN_PALM = "open_palm"
    FIST = "fist"


@dataclass
class HandInfo:
    """Information about a detected hand."""
    landmarks: Any
    handedness: str  # "Left" or "Right"
    confidence: float
    is_raised: bool = False
    fingers_extended: int = 0
    gesture: GestureType = GestureType.NONE


# ============================================================================
# GESTURE DETECTOR
# ============================================================================

class GestureDetector(BaseDetector):
    """
    Detects hand gestures for elderly assistance.
    
    Supported Gestures:
    - SOS: Both hands raised and waving
    - HELP: One hand raised above head
    - WAVE: Hand waving motion
    - THUMBS_UP: Acknowledgment gesture
    
    Features:
    - Multi-hand tracking
    - Gesture hold time for confirmation
    - Graceful degradation when MediaPipe unavailable
    - Thread-safe state management
    """
    
    def __init__(
        self,
        hold_time: float = 2.0,
        min_confidence: float = 0.5,
        max_hands: int = 2
    ):
        """
        Initialize Gesture Detector.
        
        Args:
            hold_time: Seconds gesture must be held to confirm
            min_confidence: Minimum detection confidence
            max_hands: Maximum hands to track
        """
        super().__init__(name="gesture_detector")
        
        self.hold_time = hold_time
        self.min_confidence = min_confidence
        self.max_hands = max_hands
        
        # State tracking
        self._hands = None
        self._current_gesture: GestureType = GestureType.NONE
        self._gesture_start_time: Optional[float] = None
        self._last_hands: List[HandInfo] = []
        self._wave_history: List[float] = []  # Track x positions for wave
        self._state_lock = threading.Lock()
        
        # Statistics tracking
        self._gesture_detections_count = 0
        self._last_gesture_time: Optional[float] = None
        self._gesture_counts: Dict[str, int] = {g.value: 0 for g in GestureType}
        
        # Initialize MediaPipe Hands
        self._init_hands()
    
    def _init_hands(self):
        """Initialize MediaPipe Hands with error handling."""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("[GESTURE] MediaPipe not available - Gesture detection DISABLED")
            self._enabled = False
            return
        
        try:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self._enabled = True
            logger.info("[GESTURE] Hand detector initialized successfully")
        except Exception as e:
            logger.error(f"[GESTURE] Failed to initialize Hands: {e}")
            self._enabled = False
    
    def _process_frame(self, frame) -> DetectionResult:
        """
        Process a frame for gesture detection.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            DetectionResult with detection status
        """
        current_time = time.time()
        
        if not self._enabled or self._hands is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message="Gesture detection offline",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled", "reason": "MediaPipe unavailable"}
            )
        
        if np is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message="Gesture detection offline",
                alert_level=AlertLevel.NONE,
                details={"status": "disabled", "reason": "NumPy unavailable"}
            )
        
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1] if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            
            # Process frame
            results = self._hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                self._reset_gesture_state()
                return DetectionResult(
                    detected=False,
                    confidence=0.0,
                    message="No hands detected",
                    alert_level=AlertLevel.NONE,
                    details={"status": "no_hands"}
                )
            
            # Analyze hands
            hands_info = self._analyze_hands(
                results.multi_hand_landmarks,
                results.multi_handedness,
                frame.shape
            )
            
            with self._state_lock:
                self._last_hands = hands_info
            
            # Detect gesture
            detected_gesture = self._classify_gesture(hands_info, frame.shape)
            
            # Handle gesture state
            return self._handle_gesture_state(detected_gesture, hands_info, current_time)
            
        except Exception as e:
            logger.error(f"[GESTURE] Processing error: {e}")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                message=f"Error: {str(e)}",
                alert_level=AlertLevel.NONE,
                details={"status": "error", "error": str(e)}
            )
    
    def _handle_gesture_state(self, detected_gesture: GestureType, 
                              hands_info: List[HandInfo], current_time: float) -> DetectionResult:
        """Handle gesture state machine and return appropriate result."""
        with self._state_lock:
            if detected_gesture != GestureType.NONE:
                if detected_gesture == self._current_gesture:
                    # Same gesture continues
                    if self._gesture_start_time is None:
                        self._gesture_start_time = current_time
                    
                    hold_duration = current_time - self._gesture_start_time
                    
                    if hold_duration >= self.hold_time:
                        # Gesture confirmed!
                        self._gesture_detections_count += 1
                        self._last_gesture_time = current_time
                        self._gesture_counts[detected_gesture.value] += 1
                        
                        alert_level = self._get_gesture_alert_level(detected_gesture)
                        message = self._get_gesture_message(detected_gesture)
                        
                        return DetectionResult(
                            detected=True,
                            confidence=max((h.confidence for h in hands_info), default=0.0),
                            message=message,
                            alert_level=alert_level,
                            details={
                                "gesture": detected_gesture.value,
                                "hold_time": round(hold_duration, 2),
                                "hands_detected": len(hands_info),
                                "confirmed": True
                            }
                        )
                    else:
                        # Still holding - not yet confirmed
                        return DetectionResult(
                            detected=True,
                            confidence=max((h.confidence for h in hands_info), default=0.0) * 0.7,
                            message=f"Holding {detected_gesture.value} ({hold_duration:.1f}s / {self.hold_time}s)",
                            alert_level=AlertLevel.MEDIUM,
                            details={
                                "gesture": detected_gesture.value,
                                "hold_time": round(hold_duration, 2),
                                "required_time": self.hold_time,
                                "progress": round(hold_duration / self.hold_time * 100, 1),
                                "confirmed": False
                            }
                        )
                else:
                    # New gesture detected - start timing
                    self._current_gesture = detected_gesture
                    self._gesture_start_time = current_time
                    
                    return DetectionResult(
                        detected=True,
                        confidence=max((h.confidence for h in hands_info), default=0.0) * 0.5,
                        message=f"Gesture detected: {detected_gesture.value}",
                        alert_level=AlertLevel.LOW,
                        details={
                            "gesture": detected_gesture.value,
                            "status": "detecting"
                        }
                    )
            else:
                # No gesture - reset state
                self._reset_gesture_state_locked()
                return DetectionResult(
                    detected=False,
                    confidence=max((h.confidence for h in hands_info), default=0.0) if hands_info else 0.0,
                    message=f"Hands visible ({len(hands_info)})",
                    alert_level=AlertLevel.NONE,
                    details={
                        "hands_detected": len(hands_info),
                        "status": "monitoring"
                    }
                )
    
    def _analyze_hands(self, landmarks_list, handedness_list, frame_shape) -> List[HandInfo]:
        """Analyze detected hands and extract information."""
        hands = []
        h, w = frame_shape[:2]
        
        for i, (landmarks, handedness) in enumerate(zip(landmarks_list, handedness_list)):
            hand_label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            
            # Get wrist landmark
            wrist = landmarks.landmark[0]
            
            # Check if hand is raised (wrist above middle of frame)
            is_raised = wrist.y < 0.5
            
            # Count extended fingers
            fingers_extended = self._count_extended_fingers(landmarks)
            
            # Classify gesture for this hand
            gesture = self._classify_single_hand(landmarks, is_raised, fingers_extended)
            
            hands.append(HandInfo(
                landmarks=landmarks,
                handedness=hand_label,
                confidence=confidence,
                is_raised=is_raised,
                fingers_extended=fingers_extended,
                gesture=gesture
            ))
        
        return hands
    
    def _count_extended_fingers(self, landmarks) -> int:
        """Count number of extended fingers."""
        # Finger tip and pip landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [2, 6, 10, 14, 18]
        
        extended = 0
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]
            
            # Finger is extended if tip is above (smaller y) pip
            if tip.y < pip.y:
                extended += 1
        
        return extended
    
    def _classify_single_hand(self, landmarks, is_raised: bool, fingers: int) -> GestureType:
        """Classify gesture for a single hand."""
        if not is_raised:
            return GestureType.NONE
        
        # Open palm: all fingers extended
        if fingers >= 4:
            return GestureType.OPEN_PALM
        
        # Fist: no fingers extended
        if fingers <= 1:
            # Check for thumbs up (thumb extended, others closed)
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            if thumb_tip.y < thumb_ip.y and fingers == 1:
                return GestureType.THUMBS_UP
            return GestureType.FIST
        
        return GestureType.RAISED_HAND
    
    def _classify_gesture(self, hands: List[HandInfo], frame_shape) -> GestureType:
        """Classify overall gesture from all hands."""
        if not hands:
            return GestureType.NONE
        
        raised_hands = [h for h in hands if h.is_raised]
        
        # SOS: Both hands raised with open palms above head
        if len(raised_hands) >= 2:
            open_palms = sum(1 for h in raised_hands if h.gesture == GestureType.OPEN_PALM)
            if open_palms >= 2:
                # Check for wave motion
                if self._detect_wave_motion(hands):
                    return GestureType.SOS
                return GestureType.SOS  # Even without wave, two raised palms = SOS
        
        # HELP: One hand raised high with open palm
        if len(raised_hands) == 1:
            hand = raised_hands[0]
            if hand.gesture == GestureType.OPEN_PALM:
                # Check if hand is high (above 1/3 of frame)
                wrist_y = hand.landmarks.landmark[0].y
                if wrist_y < 0.35:
                    return GestureType.HELP
                return GestureType.RAISED_HAND
            elif hand.gesture == GestureType.THUMBS_UP:
                return GestureType.THUMBS_UP
        
        return GestureType.NONE
    
    def _detect_wave_motion(self, hands: List[HandInfo]) -> bool:
        """Detect waving motion from hand position history."""
        if not hands:
            return False
        
        # Get average x position of raised hands
        raised = [h for h in hands if h.is_raised]
        if not raised:
            return False
        
        avg_x = sum(h.landmarks.landmark[0].x for h in raised) / len(raised)
        
        # Add to history
        with self._state_lock:
            self._wave_history.append(avg_x)
            if len(self._wave_history) > 20:
                self._wave_history = self._wave_history[-20:]
            
            # Detect oscillation in history
            if len(self._wave_history) >= 10:
                direction_changes = 0
                for i in range(2, len(self._wave_history)):
                    prev_dir = self._wave_history[i-1] - self._wave_history[i-2]
                    curr_dir = self._wave_history[i] - self._wave_history[i-1]
                    if prev_dir * curr_dir < 0:  # Direction changed
                        direction_changes += 1
                
                # Wave detected if multiple direction changes
                return direction_changes >= 3
        
        return False
    
    def _reset_gesture_state(self):
        """Reset gesture tracking state (acquires lock)."""
        with self._state_lock:
            self._reset_gesture_state_locked()
    
    def _reset_gesture_state_locked(self):
        """Reset gesture tracking state (assumes lock is held)."""
        self._current_gesture = GestureType.NONE
        self._gesture_start_time = None
    
    def _get_gesture_alert_level(self, gesture: GestureType) -> AlertLevel:
        """Get alert level for a gesture."""
        alert_map = {
            GestureType.SOS: AlertLevel.CRITICAL,
            GestureType.HELP: AlertLevel.HIGH,
            GestureType.WAVE: AlertLevel.MEDIUM,
            GestureType.RAISED_HAND: AlertLevel.LOW,
            GestureType.THUMBS_UP: AlertLevel.LOW,
        }
        return alert_map.get(gesture, AlertLevel.NONE)
    
    def _get_gesture_message(self, gesture: GestureType) -> str:
        """Get message for a gesture."""
        messages = {
            GestureType.SOS: "🆘 SOS SIGNAL DETECTED - Immediate assistance needed!",
            GestureType.HELP: "🖐️ HELP REQUEST - Person requesting attention",
            GestureType.WAVE: "👋 Wave detected",
            GestureType.RAISED_HAND: "✋ Hand raised",
            GestureType.THUMBS_UP: "👍 Thumbs up - All OK",
        }
        return messages.get(gesture, "Gesture detected")
    
    def draw_overlay(self, frame, result: DetectionResult):
        """Draw gesture overlay on frame."""
        try:
            cv2 = __import__('cv2')
            overlay = frame.copy()
            
            # Draw status box
            if result.detected:
                color = (0, 0, 255) if result.alert_level.value >= 3 else (0, 255, 255)
            else:
                color = (0, 255, 0)
            
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (w - 260, 10), (w - 10, 70), (0, 0, 0), -1)
            cv2.rectangle(overlay, (w - 260, 10), (w - 10, 70), color, 2)
            
            # Status text
            gesture_name = result.details.get('gesture', 'monitoring')
            cv2.putText(overlay, f"Gesture: {gesture_name}", 
                       (w - 250, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Progress bar for hold time
            if 'progress' in result.details:
                progress = result.details['progress'] / 100
                bar_width = int(230 * progress)
                cv2.rectangle(overlay, (w - 250, 50), (w - 250 + bar_width, 60), color, -1)
                cv2.rectangle(overlay, (w - 250, 50), (w - 20, 60), color, 1)
            
            # Draw hand landmarks if available
            if MEDIAPIPE_AVAILABLE:
                with self._state_lock:
                    for hand in self._last_hands:
                        if hand.landmarks:
                            mp_drawing.draw_landmarks(
                                overlay,
                                hand.landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )
            
            return overlay
            
        except Exception as e:
            logger.error(f"[GESTURE] Overlay error: {e}")
            return frame
    
    def reset(self):
        """Reset detector state."""
        super().reset()
        self._reset_gesture_state()
        with self._state_lock:
            self._wave_history.clear()
            self._last_hands.clear()
    
    def cleanup(self):
        """Clean up resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        base_stats = super().stats()
        with self._state_lock:
            base_stats.update({
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "hold_time": self.hold_time,
                "current_gesture": self._current_gesture.value,
                "gesture_confirmations": self._gesture_detections_count,
                "gesture_counts": dict(self._gesture_counts)
            })
        return base_stats
