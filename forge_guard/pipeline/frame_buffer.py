"""
FORGE-Guard Frame Buffer
Thread-safe circular buffer for frame management.
Production-ready with proper synchronization and resource management.
"""

import threading
import time
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

# Use safe import for numpy
from ..utils.safe_imports import get_numpy

np = get_numpy()


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: 'np.ndarray'  # Type hint as string to avoid import issues
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    
    def age(self) -> float:
        """Return age of frame in seconds."""
        return time.time() - self.timestamp
    
    def is_stale(self, max_age: float = 1.0) -> bool:
        """Check if frame is older than max_age seconds."""
        return self.age() > max_age


class FrameBuffer:
    """
    Thread-safe circular buffer for video frames.
    Implements a fixed-size buffer that drops oldest frames
    when full to maintain real-time performance.
    
    Thread Safety:
    - All public methods are thread-safe
    - Uses condition variable for efficient waiting
    - Proper cleanup on stop()
    """
    
    def __init__(self, max_size: int = 5):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to buffer (must be >= 1)
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._not_empty = threading.Condition(self._lock)
        self._frame_counter = 0
        self._dropped_frames = 0
        self._running = True
        self._last_push_time: Optional[float] = None
        self._last_pop_time: Optional[float] = None
    
    def push(self, frame) -> bool:
        """
        Push a frame to the buffer.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            True if frame was added, False if buffer is stopped
        """
        if frame is None:
            return False
            
        with self._lock:
            if not self._running:
                return False
            
            # Track if we're dropping a frame (deque handles removal automatically)
            if len(self._buffer) >= self.max_size:
                self._dropped_frames += 1
            
            self._frame_counter += 1
            self._last_push_time = time.time()
            
            # Copy frame to prevent external modification
            np_module = get_numpy()
            if np_module is not None:
                frame_copy = np_module.copy(frame)
            else:
                frame_copy = frame  # Fallback - may have issues
            
            frame_data = FrameData(
                frame=frame_copy,
                timestamp=self._last_push_time,
                frame_id=self._frame_counter
            )
            self._buffer.append(frame_data)
            self._not_empty.notify()
            return True
    
    def pop(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        Pop the oldest frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            FrameData if available, None if timeout or stopped
        """
        with self._not_empty:
            # Wait for frame with timeout
            end_time = time.time() + timeout
            while not self._buffer and self._running:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)
            
            if self._buffer:
                self._last_pop_time = time.time()
                return self._buffer.popleft()
            return None
    
    def pop_latest(self) -> Optional[FrameData]:
        """
        Pop the latest (newest) frame, discarding older frames.
        Useful for getting the most recent frame when latency matters.
        
        Returns:
            Latest FrameData if available, None otherwise
        """
        with self._lock:
            if not self._buffer:
                return None
            
            # Get latest, count others as dropped
            old_count = len(self._buffer) - 1
            if old_count > 0:
                self._dropped_frames += old_count
            
            latest = self._buffer[-1]
            self._buffer.clear()
            self._last_pop_time = time.time()
            return latest
    
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
    
    def peek_oldest(self) -> Optional[FrameData]:
        """
        Peek at the oldest frame without removing it.
        
        Returns:
            Oldest FrameData if available, None otherwise
        """
        with self._lock:
            if self._buffer:
                return self._buffer[0]
            return None
    
    def clear(self):
        """Clear all frames from the buffer."""
        with self._lock:
            cleared = len(self._buffer)
            self._buffer.clear()
            return cleared
    
    def stop(self):
        """Stop the buffer and release waiting threads."""
        with self._not_empty:
            self._running = False
            self._not_empty.notify_all()
    
    def start(self):
        """Restart the buffer after stopping."""
        with self._lock:
            self._running = True
    
    def reset_stats(self):
        """Reset frame and drop counters."""
        with self._lock:
            self._frame_counter = 0
            self._dropped_frames = 0
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.max_size
    
    @property
    def is_running(self) -> bool:
        """Check if buffer is accepting frames."""
        with self._lock:
            return self._running
    
    @property
    def dropped_count(self) -> int:
        """Number of frames dropped due to buffer overflow."""
        with self._lock:
            return self._dropped_frames
    
    @property
    def total_frames(self) -> int:
        """Total number of frames pushed."""
        with self._lock:
            return self._frame_counter
    
    def stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.max_size,
                "total_frames": self._frame_counter,
                "dropped_frames": self._dropped_frames,
                "drop_rate": (self._dropped_frames / max(self._frame_counter, 1) * 100),
                "running": self._running,
                "last_push": self._last_push_time,
                "last_pop": self._last_pop_time
            }
    
    def __len__(self) -> int:
        """Support len() operator."""
        return self.size
    
    def __bool__(self) -> bool:
        """Support bool() operator - True if not empty."""
        return not self.is_empty
