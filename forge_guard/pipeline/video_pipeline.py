"""
FORGE-Guard Video Pipeline
Multi-threaded producer-consumer architecture for real-time video processing.
"""

import cv2
import threading
import time
from typing import List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from .frame_buffer import FrameBuffer, FrameData
from ..config import config


@dataclass
class ProcessedFrame:
    """Container for processed frame with detection results."""
    frame: np.ndarray
    original_frame: np.ndarray
    frame_id: int
    timestamp: float
    detections: dict
    processing_time: float


class FrameProducer(threading.Thread):
    """
    Producer thread that captures frames from camera
    and pushes them to the frame buffer.
    """
    
    def __init__(
        self,
        buffer: FrameBuffer,
        camera_index: int = 0,
        target_fps: int = 30,
        width: int = 1280,
        height: int = 720
    ):
        """
        Initialize frame producer.
        
        Args:
            buffer: Frame buffer to push frames to
            camera_index: Camera device index
            target_fps: Target frames per second
            width: Frame width
            height: Frame height
        """
        super().__init__(name="FrameProducer", daemon=True)
        self.buffer = buffer
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.width = width
        self.height = height
        
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_interval = 1.0 / target_fps
        self._actual_fps = 0.0
        self._last_frame_time = 0.0
    
    def run(self):
        """Main producer loop."""
        self._running = True
        self._cap = cv2.VideoCapture(self.camera_index)
        
        if not self._cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.camera_index}")
            self._running = False
            return
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        print(f"[PRODUCER] Camera initialized: {self.width}x{self.height} @ {self.target_fps}fps")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while self._running:
            loop_start = time.time()
            
            ret, frame = self._cap.read()
            if not ret:
                print("[PRODUCER] Failed to read frame")
                time.sleep(0.01)
                continue
            
            self.buffer.push(frame)
            self._last_frame_time = time.time()
            
            # Calculate actual FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                self._actual_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
            
            # Maintain target FPS
            processing_time = time.time() - loop_start
            sleep_time = self._frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._cleanup()
    
    def stop(self):
        """Stop the producer thread."""
        self._running = False
    
    def _cleanup(self):
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[PRODUCER] Stopped and released camera")
    
    @property
    def actual_fps(self) -> float:
        """Current actual FPS."""
        return self._actual_fps
    
    @property
    def is_running(self) -> bool:
        """Check if producer is running."""
        return self._running


class FrameConsumer(threading.Thread):
    """
    Consumer thread that processes frames from the buffer
    using registered detection modules.
    """
    
    def __init__(
        self,
        buffer: FrameBuffer,
        on_frame_processed: Optional[Callable[[ProcessedFrame], None]] = None
    ):
        """
        Initialize frame consumer.
        
        Args:
            buffer: Frame buffer to consume frames from
            on_frame_processed: Callback for processed frames
        """
        super().__init__(name="FrameConsumer", daemon=True)
        self.buffer = buffer
        self.on_frame_processed = on_frame_processed
        
        self._running = False
        self._detectors: List[Any] = []
        self._actual_fps = 0.0
        self._latest_frame: Optional[ProcessedFrame] = None
        self._frame_lock = threading.Lock()
    
    def register_detector(self, detector):
        """
        Register a detection module.
        
        Args:
            detector: Detector instance implementing BaseDetector interface
        """
        self._detectors.append(detector)
        print(f"[CONSUMER] Registered detector: {detector.__class__.__name__}")
    
    def run(self):
        """Main consumer loop."""
        self._running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        print(f"[CONSUMER] Started with {len(self._detectors)} detectors")
        
        while self._running:
            frame_data = self.buffer.pop(timeout=0.5)
            
            if frame_data is None:
                continue
            
            process_start = time.time()
            
            # Process frame through all detectors
            detections = {}
            display_frame = frame_data.frame.copy()
            
            for detector in self._detectors:
                try:
                    result = detector.detect(frame_data.frame)
                    detections[detector.name] = result
                    
                    # Draw overlay on display frame
                    if result.detected:
                        display_frame = detector.draw_overlay(display_frame, result)
                except Exception as e:
                    print(f"[CONSUMER] Detector {detector.__class__.__name__} error: {e}")
            
            processing_time = time.time() - process_start
            
            # Create processed frame result
            processed = ProcessedFrame(
                frame=display_frame,
                original_frame=frame_data.frame,
                frame_id=frame_data.frame_id,
                timestamp=frame_data.timestamp,
                detections=detections,
                processing_time=processing_time
            )
            
            with self._frame_lock:
                self._latest_frame = processed
            
            # Callback for processed frame
            if self.on_frame_processed:
                try:
                    self.on_frame_processed(processed)
                except Exception as e:
                    print(f"[CONSUMER] Callback error: {e}")
            
            # Calculate actual FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                self._actual_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
        
        print("[CONSUMER] Stopped")
    
    def stop(self):
        """Stop the consumer thread."""
        self._running = False
    
    def get_latest_frame(self) -> Optional[ProcessedFrame]:
        """Get the most recently processed frame."""
        with self._frame_lock:
            return self._latest_frame
    
    @property
    def actual_fps(self) -> float:
        """Current actual processing FPS."""
        return self._actual_fps
    
    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running


class VideoPipeline:
    """
    Main video pipeline orchestrator.
    Manages producer-consumer threads and detection modules.
    """
    
    def __init__(
        self,
        camera_index: Optional[int] = None,
        on_frame_processed: Optional[Callable[[ProcessedFrame], None]] = None
    ):
        """
        Initialize video pipeline.
        
        Args:
            camera_index: Camera device index (uses config if not specified)
            on_frame_processed: Callback for processed frames
        """
        self.camera_index = camera_index or config.video.camera_index
        self.on_frame_processed = on_frame_processed
        
        self.buffer = FrameBuffer(max_size=config.video.buffer_size)
        
        self.producer = FrameProducer(
            buffer=self.buffer,
            camera_index=self.camera_index,
            target_fps=config.video.fps,
            width=config.video.width,
            height=config.video.height
        )
        
        self.consumer = FrameConsumer(
            buffer=self.buffer,
            on_frame_processed=on_frame_processed
        )
        
        self._running = False
    
    def register_detector(self, detector):
        """Register a detection module."""
        self.consumer.register_detector(detector)
    
    def start(self):
        """Start the video pipeline."""
        if self._running:
            print("[PIPELINE] Already running")
            return
        
        print("[PIPELINE] Starting...")
        self._running = True
        self.producer.start()
        self.consumer.start()
        print("[PIPELINE] Started successfully")
    
    def stop(self):
        """Stop the video pipeline."""
        if not self._running:
            return
        
        print("[PIPELINE] Stopping...")
        self._running = False
        
        self.producer.stop()
        self.consumer.stop()
        self.buffer.stop()
        
        # Wait for threads to finish
        if self.producer.is_alive():
            self.producer.join(timeout=2.0)
        if self.consumer.is_alive():
            self.consumer.join(timeout=2.0)
        
        print("[PIPELINE] Stopped")
    
    def get_latest_frame(self) -> Optional[ProcessedFrame]:
        """Get the most recently processed frame."""
        return self.consumer.get_latest_frame()
    
    def stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "buffer": self.buffer.stats(),
            "producer_fps": self.producer.actual_fps,
            "consumer_fps": self.consumer.actual_fps,
            "running": self._running
        }
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
