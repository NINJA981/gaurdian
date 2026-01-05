import json
import os
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class JournalManager:
    """
    Manages persistent logging of system events (Journal).
    Events are saved to a JSON file in the logs directory.
    """
    
    def __init__(self, log_dir="logs", filename="journal.json"):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self._ensure_log_dir()
        
    def _ensure_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # Create file if it doesn't exist
        if not os.path.exists(self.filepath):
            self._save_events([])

    def _load_events(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    content = f.read()
                    if not content:
                        return []
                    return json.loads(content)
            return []
        except Exception as e:
            logger.error(f"Failed to load journal: {e}")
            return []

    def _save_events(self, events):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save journal: {e}")

    def log_event(self, source, message, level="INFO", metadata=None):
        """
        Log a new event to the journal.
        
        Args:
            source (str): Origin of the event (e.g., 'FallDetector', 'System')
            message (str): Description of the event
            level (str): 'INFO', 'WARNING', 'CRITICAL'
            metadata (dict): Optional additional data
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "source": source,
            "message": message,
            "level": level,
            "metadata": metadata or {}
        }
        
        events = self._load_events()
        events.append(event)
        
        # Limit size (keep last 1000 events)
        if len(events) > 1000:
            events = events[-1000:]
            
        self._save_events(events)
        return event

    def get_events(self, limit=50, level_filter=None):
        """Get recent events, optionally filtered by level."""
        events = self._load_events()
        
        if level_filter:
            events = [e for e in events if e.get("level") == level_filter]
            
        # Return most recent first
        return sorted(events, key=lambda x: x["timestamp"], reverse=True)[:limit]

# Global instance
journal = JournalManager()
