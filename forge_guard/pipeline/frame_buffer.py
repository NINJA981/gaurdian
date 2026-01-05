"""
FORGE-Guard Frame Buffer
Thread-safe circular buffer for frame management.
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    
    def age(self) -> float:
        """Return age of frame in seconds."""
        return time.time() - self.timestamp


class FrameBuffer:
    """
    Thread-safe circular buffer for video frames.
    Implements a fixed-size buffer that drops oldest frames
    when full to maintain real-time performance.
    """
    
    def __init__(self, max_size: int = 5):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._frame_counter = 0
        self._dropped_frames = 0
        self._running = True
    
    def push(self, frame: np.ndarray) -> bool:
        """
        Push a frame to the buffer.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            True if frame was added, False otherwise
        """
        if not self._running:
            return False
            
        with self._lock:
            # Track if we're dropping a frame
            if len(self._buffer) >= self.max_size:
                self._dropped_frames += 1
            
            self._frame_counter += 1
            frame_data = FrameData(
                frame=frame.copy(),
                timestamp=time.time(),
                frame_id=self._frame_counter
            )
            self._buffer.append(frame_data)
            self._not_empty.notify()
            return True
    
    def pop(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Pop the oldest frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            FrameData if available, None if timeout or stopped
        """
        with self._not_empty:
            if not self._buffer and self._running:
                self._not_empty.wait(timeout)
            
            if self._buffer:
                return self._buffer.popleft()
            return None
    
    def peek(self) -> Optional[FrameData]:
        """
        Peek at the latest frame without removing it.
        
        Returns:
            Latest FrameData if available, None otherwise
        """
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None
    
    def clear(self):
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def stop(self):
        """Stop the buffer and release waiting threads."""
        self._running = False
        with self._not_empty:
            self._not_empty.notify_all()
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.max_size
    
    @property
    def dropped_count(self) -> int:
        """Number of frames dropped due to buffer overflow."""
        return self._dropped_frames
    
    @property
    def total_frames(self) -> int:
        """Total number of frames processed."""
        return self._frame_counter
    
    def stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.max_size,
                "total_frames": self._frame_counter,
                "dropped_frames": self._dropped_frames,
                "drop_rate": (self._dropped_frames / self._frame_counter * 100) 
                            if self._frame_counter > 0 else 0
            }
