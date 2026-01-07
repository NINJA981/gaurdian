"""
FORGE-Guard Video Pipeline
Production-ready multi-threaded producer-consumer architecture for real-time video processing.
"""

import threading
import time
import logging
from typing import List, Optional, Callable, Any, Dict
from dataclasses import dataclass, field

from .frame_buffer import FrameBuffer, FrameData
from ..config import config
from ..utils.safe_imports import get_cv2, get_numpy

# Setup logging
logger = logging.getLogger(__name__)

# Get safe modules
cv2 = get_cv2()
np = get_numpy()


@dataclass
class ProcessedFrame:
    """Container for processed frame with detection results."""
    frame: Any  # np.ndarray
    original_frame: Any  # np.ndarray
    frame_id: int
    timestamp: float
    detections: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without frame data)."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "processing_time_ms": round(self.processing_time * 1000, 2),
            "detection_count": len(self.detections)
        }


class FrameProducer(threading.Thread):
    """
    Producer thread that captures frames from camera
    and pushes them to the frame buffer.
    
    Features:
    - Automatic camera reconnection
    - FPS calculation
    - Graceful error handling
    """
    
    def __init__(
        self,
        buffer: FrameBuffer,
        camera_index: int = 0,
        target_fps: int = 30,
        width: int = 1280,
        height: int = 720,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 5
    ):
        """
        Initialize frame producer.
        
        Args:
            buffer: Frame buffer to push frames to
            camera_index: Camera device index (or RTSP URL)
            target_fps: Target frames per second
            width: Frame width
            height: Frame height
            reconnect_delay: Seconds to wait before reconnection attempt
            max_reconnect_attempts: Maximum reconnection attempts before giving up
        """
        super().__init__(name="FrameProducer", daemon=True)
        self.buffer = buffer
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._running = False
        self._cap = None
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._lock = threading.Lock()
        
        # Statistics
        self._actual_fps = 0.0
        self._last_frame_time = 0.0
        self._frames_captured = 0
        self._errors_count = 0
        self._camera_opened = False
        self._start_time: Optional[float] = None
    
    def _open_camera(self) -> bool:
        """Attempt to open the camera."""
        try:
            if self._cap is not None:
                self._cap.release()
            
            self._cap = cv2.VideoCapture(self.camera_index)
            
            if not self._cap.isOpened():
                logger.error(f"[PRODUCER] Failed to open camera {self.camera_index}")
                return False
            
            # Configure camera
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            self._camera_opened = True
            logger.info(f"[PRODUCER] Camera initialized: {self.width}x{self.height} @ {self.target_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"[PRODUCER] Camera initialization error: {e}")
            return False
    
    def run(self):
        """Main producer loop."""
        self._running = True
        self._start_time = time.time()
        reconnect_attempts = 0
        
        # Initial camera open
        while self._running and not self._open_camera():
            reconnect_attempts += 1
            if reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("[PRODUCER] Max reconnection attempts reached. Stopping.")
                self._running = False
                return
            logger.warning(f"[PRODUCER] Retrying camera connection ({reconnect_attempts}/{self.max_reconnect_attempts})...")
            time.sleep(self.reconnect_delay)
        
        if not self._running:
            return
        
        reconnect_attempts = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while self._running:
            loop_start = time.time()
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self._errors_count += 1
                    logger.warning("[PRODUCER] Failed to read frame")
                    
                    # Try to reconnect
                    reconnect_attempts += 1
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error("[PRODUCER] Max reconnection attempts reached. Stopping.")
                        self._running = False
                        break
                    
                    time.sleep(self.reconnect_delay)
                    if self._open_camera():
                        reconnect_attempts = 0
                    continue
                
                # Reset reconnect counter on successful read
                reconnect_attempts = 0
                
                # Push frame to buffer
                self.buffer.push(frame)
                
                with self._lock:
                    self._last_frame_time = time.time()
                    self._frames_captured += 1
                
                # Calculate actual FPS
                fps_counter += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    with self._lock:
                        self._actual_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Frame rate limiting
                processing_time = time.time() - loop_start
                sleep_time = self._frame_interval - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self._errors_count += 1
                logger.error(f"[PRODUCER] Frame capture error: {e}")
                time.sleep(0.1)
        
        # Cleanup
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._camera_opened = False
        logger.info("[PRODUCER] Stopped")
    
    def stop(self):
        """Stop the producer."""
        self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if producer is running."""
        return self._running
    
    @property
    def actual_fps(self) -> float:
        """Get actual frames per second."""
        with self._lock:
            return self._actual_fps
    
    def stats(self) -> dict:
        """Get producer statistics."""
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0
            return {
                "running": self._running,
                "camera_opened": self._camera_opened,
                "frames_captured": self._frames_captured,
                "actual_fps": round(self._actual_fps, 1),
                "target_fps": self.target_fps,
                "errors_count": self._errors_count,
                "uptime_seconds": round(uptime, 1)
            }


class FrameConsumer(threading.Thread):
    """
    Consumer thread that processes frames from the buffer
    using registered detectors.
    
    Features:
    - Multiple detector support
    - Processing time tracking
    - Callback for processed frames
    """
    
    def __init__(
        self,
        buffer: FrameBuffer,
        detectors: Optional[List[Any]] = None,
        on_frame_processed: Optional[Callable[[ProcessedFrame], None]] = None
    ):
        """
        Initialize frame consumer.
        
        Args:
            buffer: Frame buffer to consume from
            detectors: List of detector instances
            on_frame_processed: Callback for processed frames
        """
        super().__init__(name="FrameConsumer", daemon=True)
        self.buffer = buffer
        self._detectors: List[Any] = detectors or []
        self.on_frame_processed = on_frame_processed
        
        self._running = False
        self._lock = threading.Lock()
        
        # Statistics
        self._frames_processed = 0
        self._total_processing_time = 0.0
        self._last_processing_time = 0.0
        self._start_time: Optional[float] = None
    
    def register_detector(self, detector):
        """Add a detector to the processing pipeline."""
        with self._lock:
            if detector not in self._detectors:
                self._detectors.append(detector)
                logger.info(f"[CONSUMER] Registered detector: {detector.name}")
    
    def unregister_detector(self, detector):
        """Remove a detector from the pipeline."""
        with self._lock:
            if detector in self._detectors:
                self._detectors.remove(detector)
                logger.info(f"[CONSUMER] Unregistered detector: {detector.name}")
    
    def run(self):
        """Main consumer loop."""
        self._running = True
        self._start_time = time.time()
        logger.info("[CONSUMER] Started")
        
        while self._running:
            # Get frame from buffer
            frame_data = self.buffer.pop(timeout=1.0)
            
            if frame_data is None:
                continue
            
            start_time = time.time()
            
            try:
                # Run all detectors
                detections = {}
                processed_frame = frame_data.frame.copy() if np is not None else frame_data.frame
                
                with self._lock:
                    detectors = self._detectors.copy()
                
                for detector in detectors:
                    try:
                        if detector.enabled:
                            result = detector.detect(frame_data.frame)
                            detections[detector.name] = result
                            
                            # Draw overlay
                            if result is not None:
                                processed_frame = detector.draw_overlay(processed_frame, result)
                    except Exception as e:
                        logger.error(f"[CONSUMER] Detector {detector.name} error: {e}")
                        detections[detector.name] = None
                
                processing_time = time.time() - start_time
                
                # Create processed frame
                result = ProcessedFrame(
                    frame=processed_frame,
                    original_frame=frame_data.frame,
                    frame_id=frame_data.frame_id,
                    timestamp=frame_data.timestamp,
                    detections=detections,
                    processing_time=processing_time
                )
                
                # Update statistics
                with self._lock:
                    self._frames_processed += 1
                    self._total_processing_time += processing_time
                    self._last_processing_time = processing_time
                
                # Callback
                if self.on_frame_processed:
                    try:
                        self.on_frame_processed(result)
                    except Exception as e:
                        logger.error(f"[CONSUMER] Callback error: {e}")
                        
            except Exception as e:
                logger.error(f"[CONSUMER] Processing error: {e}")
        
        logger.info("[CONSUMER] Stopped")
    
    def stop(self):
        """Stop the consumer."""
        self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running
    
    def stats(self) -> dict:
        """Get consumer statistics."""
        with self._lock:
            avg_time = (self._total_processing_time / self._frames_processed * 1000
                       if self._frames_processed > 0 else 0)
            uptime = time.time() - self._start_time if self._start_time else 0
            return {
                "running": self._running,
                "frames_processed": self._frames_processed,
                "avg_processing_time_ms": round(avg_time, 2),
                "last_processing_time_ms": round(self._last_processing_time * 1000, 2),
                "detector_count": len(self._detectors),
                "uptime_seconds": round(uptime, 1)
            }


class VideoPipeline:
    """
    Main video processing pipeline.
    Manages producer, consumer, and detectors.
    
    Features:
    - Easy start/stop lifecycle
    - Detector registration
    - Comprehensive statistics
    """
    
    def __init__(
        self,
        camera_index: Optional[int] = None,
        buffer_size: Optional[int] = None,
        target_fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        on_frame_processed: Optional[Callable[[ProcessedFrame], None]] = None
    ):
        """
        Initialize video pipeline.
        
        Args:
            camera_index: Camera device index (default from config)
            buffer_size: Frame buffer size (default from config)
            target_fps: Target FPS (default from config)
            width: Frame width (default from config)
            height: Frame height (default from config)
            on_frame_processed: Callback for processed frames
        """
        # Use config defaults
        self.camera_index = camera_index if camera_index is not None else config.video.camera_index
        self.buffer_size = buffer_size if buffer_size is not None else config.video.buffer_size
        self.target_fps = target_fps if target_fps is not None else config.video.fps
        self.width = width if width is not None else config.video.width
        self.height = height if height is not None else config.video.height
        self.on_frame_processed = on_frame_processed
        
        # Create components
        self._buffer = FrameBuffer(max_size=self.buffer_size)
        self._producer: Optional[FrameProducer] = None
        self._consumer: Optional[FrameConsumer] = None
        self._detectors: List[Any] = []
        self._running = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
    
    def register_detector(self, detector):
        """Register a detector with the pipeline."""
        with self._lock:
            if detector not in self._detectors:
                self._detectors.append(detector)
                if self._consumer:
                    self._consumer.register_detector(detector)
                logger.info(f"[PIPELINE] Registered detector: {detector.name}")
    
    def unregister_detector(self, detector):
        """Unregister a detector from the pipeline."""
        with self._lock:
            if detector in self._detectors:
                self._detectors.remove(detector)
                if self._consumer:
                    self._consumer.unregister_detector(detector)
                logger.info(f"[PIPELINE] Unregistered detector: {detector.name}")
    
    def start(self):
        """Start the video pipeline."""
        if self._running:
            logger.warning("[PIPELINE] Already running")
            return
        
        logger.info("[PIPELINE] Starting...")
        self._start_time = time.time()
        
        # Create producer
        self._producer = FrameProducer(
            buffer=self._buffer,
            camera_index=self.camera_index,
            target_fps=self.target_fps,
            width=self.width,
            height=self.height
        )
        
        # Create consumer with detectors
        self._consumer = FrameConsumer(
            buffer=self._buffer,
            detectors=self._detectors.copy(),
            on_frame_processed=self.on_frame_processed
        )
        
        # Start threads
        self._producer.start()
        self._consumer.start()
        self._running = True
        
        logger.info("[PIPELINE] Started successfully")
    
    def stop(self):
        """Stop the video pipeline."""
        if not self._running:
            return
        
        logger.info("[PIPELINE] Stopping...")
        self._running = False
        
        # Stop producer first
        if self._producer:
            self._producer.stop()
        
        # Stop buffer (releases consumer)
        self._buffer.stop()
        
        # Stop consumer
        if self._consumer:
            self._consumer.stop()
        
        # Wait for threads
        if self._producer and self._producer.is_alive():
            self._producer.join(timeout=2.0)
        if self._consumer and self._consumer.is_alive():
            self._consumer.join(timeout=2.0)
        
        logger.info("[PIPELINE] Stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    @property
    def buffer(self) -> FrameBuffer:
        """Get the frame buffer."""
        return self._buffer
    
    @property
    def detectors(self) -> List[Any]:
        """Get registered detectors."""
        with self._lock:
            return self._detectors.copy()
    
    def stats(self) -> dict:
        """Get pipeline statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "buffer": self._buffer.stats(),
            "producer": self._producer.stats() if self._producer else None,
            "consumer": self._consumer.stats() if self._consumer else None,
            "detector_count": len(self._detectors),
            "detectors": [d.name for d in self._detectors]
        }
