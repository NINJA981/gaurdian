"""
FORGE-Guard Event Logger
Local logging system with WebSocket broadcast capability.
"""

import json
import os
import time
import threading
from typing import Optional, List, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler

from ..config import config


class EventType(Enum):
    """Event type categories."""
    SYSTEM = "system"
    DETECTION = "detection"
    ALERT = "alert"
    USER_ACTION = "user_action"
    ERROR = "error"


@dataclass
class Event:
    """Event data container."""
    id: str
    type: EventType
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level,
            "details": self.details
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventLogger:
    """
    Event logging system with file persistence and real-time broadcast.
    
    Features:
    - JSON-formatted event logs
    - Rotating file handler for storage management
    - WebSocket broadcast for real-time dashboard updates
    - Thread-safe operations
    - Event filtering and querying
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_size_mb: Optional[int] = None,
        on_event: Optional[Callable[[Event], None]] = None
    ):
        """
        Initialize event logger.
        
        Args:
            log_file: Path to log file
            max_size_mb: Maximum log file size in MB
            on_event: Callback for new events
        """
        self.log_file = log_file or config.alerts.log_file_path
        self.max_size_mb = max_size_mb or config.alerts.max_log_size_mb
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self._logger = logging.getLogger("forge_guard")
        self._logger.setLevel(logging.DEBUG)
        
        # Rotating file handler
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_size_mb * 1024 * 1024,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s')
        )
        self._logger.addHandler(console_handler)
        
        # State
        self._event_counter = 0
        self._lock = threading.Lock()
        self._recent_events: List[Event] = []
        self._max_recent = 100
        
        # Callbacks
        self._callbacks: List[Callable[[Event], None]] = []
        if on_event:
            self._callbacks.append(on_event)
    
    def add_callback(self, callback: Callable[[Event], None]):
        """Add callback for new events."""
        self._callbacks.append(callback)
    
    def log(
        self,
        message: str,
        source: str = "system",
        event_type: EventType = EventType.SYSTEM,
        level: str = "INFO",
        details: Optional[dict] = None
    ) -> Event:
        """
        Log an event.
        
        Args:
            message: Event message
            source: Event source identifier
            event_type: Type of event
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            details: Additional event details
            
        Returns:
            Created Event object
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"evt_{self._event_counter}_{int(time.time())}"
        
        event = Event(
            id=event_id,
            type=event_type,
            source=source,
            message=message,
            level=level,
            details=details or {}
        )
        
        # Write to log file
        self._logger.log(
            getattr(logging, level, logging.INFO),
            event.to_json()
        )
        
        # Store in recent events
        with self._lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent:
                self._recent_events = self._recent_events[-self._max_recent:]
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(f"Event callback error: {e}")
        
        return event
    
    def detection(self, source: str, message: str, details: Optional[dict] = None) -> Event:
        """Log a detection event."""
        return self.log(
            message=message,
            source=source,
            event_type=EventType.DETECTION,
            level="INFO",
            details=details
        )
    
    def alert(self, source: str, message: str, details: Optional[dict] = None) -> Event:
        """Log an alert event."""
        return self.log(
            message=message,
            source=source,
            event_type=EventType.ALERT,
            level="WARNING",
            details=details
        )
    
    def error(self, source: str, message: str, details: Optional[dict] = None) -> Event:
        """Log an error event."""
        return self.log(
            message=message,
            source=source,
            event_type=EventType.ERROR,
            level="ERROR",
            details=details
        )
    
    def user_action(self, action: str, details: Optional[dict] = None) -> Event:
        """Log a user action event."""
        return self.log(
            message=action,
            source="user",
            event_type=EventType.USER_ACTION,
            level="INFO",
            details=details
        )
    
    def get_recent(self, count: int = 20) -> List[Event]:
        """Get recent events."""
        with self._lock:
            return self._recent_events[-count:]
    
    def get_by_type(self, event_type: EventType, count: int = 20) -> List[Event]:
        """Get recent events of a specific type."""
        with self._lock:
            filtered = [e for e in self._recent_events if e.type == event_type]
            return filtered[-count:]
    
    def get_by_source(self, source: str, count: int = 20) -> List[Event]:
        """Get recent events from a specific source."""
        with self._lock:
            filtered = [e for e in self._recent_events if e.source == source]
            return filtered[-count:]
    
    def clear_recent(self):
        """Clear recent events buffer."""
        with self._lock:
            self._recent_events.clear()
    
    def stats(self) -> dict:
        """Get logger statistics."""
        with self._lock:
            type_counts = {}
            for event in self._recent_events:
                t = event.type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            
            return {
                "total_events": self._event_counter,
                "recent_count": len(self._recent_events),
                "type_counts": type_counts,
                "log_file": self.log_file
            }


# Global logger instance
event_logger = EventLogger()
