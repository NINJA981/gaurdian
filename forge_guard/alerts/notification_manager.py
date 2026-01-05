"""
FORGE-Guard Notification Manager
Module 5: Alert System with Twilio SMS/Call integration and cooldown logic.
"""

import threading
import time
import queue
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import json

from ..config import config

# Try to import Twilio
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARNING] Twilio not installed. SMS/Call alerts disabled.")


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Alert data container."""
    id: str
    priority: AlertPriority
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    details: Dict = field(default_factory=dict)
    sent_sms: bool = False
    sent_call: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "priority": self.priority.name,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "details": self.details,
            "sent_sms": self.sent_sms,
            "sent_call": self.sent_call
        }


class NotificationManager:
    """
    Centralized notification manager with Twilio integration.
    
    Features:
    - SMS and Voice call alerts via Twilio
    - Cooldown timer to prevent alert spam
    - Priority-based notification routing
    - Thread-safe alert queue
    - Callback system for UI updates
    """
    
    def __init__(
        self,
        cooldown_seconds: Optional[int] = None,
        on_alert: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize notification manager.
        
        Args:
            cooldown_seconds: Minimum seconds between alerts (per source)
            on_alert: Callback function for new alerts
        """
        self.cooldown_seconds = cooldown_seconds or config.alerts.cooldown_seconds
        self.on_alert = on_alert
        
        # Twilio setup
        self._twilio_client: Optional[TwilioClient] = None
        if TWILIO_AVAILABLE and config.twilio.is_configured:
            try:
                self._twilio_client = TwilioClient(
                    config.twilio.account_sid,
                    config.twilio.auth_token
                )
                print("[NOTIFICATIONS] Twilio client initialized")
            except Exception as e:
                print(f"[NOTIFICATIONS] Twilio initialization failed: {e}")
        
        # State management
        self._alert_queue: queue.Queue = queue.Queue()
        self._alert_history: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}  # source -> last time
        self._alert_counter = 0
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks for UI
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        if on_alert:
            self._alert_callbacks.append(on_alert)
    
    def start(self):
        """Start the notification worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_alerts,
            name="NotificationWorker",
            daemon=True
        )
        self._worker_thread.start()
        print("[NOTIFICATIONS] Manager started")
    
    def stop(self):
        """Stop the notification manager."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        print("[NOTIFICATIONS] Manager stopped")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for new alerts."""
        self._alert_callbacks.append(callback)
    
    def send_alert(
        self,
        message: str,
        source: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        details: Optional[Dict] = None
    ) -> Optional[Alert]:
        """
        Queue an alert for processing.
        
        Args:
            message: Alert message text
            source: Source detector name
            priority: Alert priority level
            details: Additional alert details
            
        Returns:
            Alert object if queued, None if rate-limited
        """
        current_time = time.time()
        
        # Check cooldown (unless CRITICAL)
        if priority != AlertPriority.CRITICAL:
            last_time = self._last_alert_time.get(source, 0)
            if current_time - last_time < self.cooldown_seconds:
                remaining = self.cooldown_seconds - (current_time - last_time)
                print(f"[NOTIFICATIONS] Rate limited for {source}: {remaining:.1f}s remaining")
                return None
        
        # Create alert
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(current_time)}"
        
        alert = Alert(
            id=alert_id,
            priority=priority,
            message=message,
            source=source,
            timestamp=current_time,
            details=details or {}
        )
        
        self._last_alert_time[source] = current_time
        self._alert_queue.put(alert)
        
        return alert
    
    def _process_alerts(self):
        """Worker thread to process alert queue."""
        while self._running:
            try:
                alert = self._alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                self._handle_alert(alert)
            except Exception as e:
                print(f"[NOTIFICATIONS] Alert processing error: {e}")
    
    def _handle_alert(self, alert: Alert):
        """Process a single alert."""
        print(f"[NOTIFICATIONS] Processing alert: {alert.priority.name} - {alert.message}")
        
        # Store in history
        with self._lock:
            self._alert_history.append(alert)
            # Keep only last 100 alerts
            if len(self._alert_history) > 100:
                self._alert_history = self._alert_history[-100:]
        
        # Send notifications based on priority
        if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]:
            # Send SMS for high priority
            if self._twilio_client and config.twilio.emergency_contact:
                self._send_sms(alert)
        
        if alert.priority == AlertPriority.CRITICAL:
            # Make voice call for critical alerts
            if self._twilio_client and config.twilio.emergency_contact:
                self._make_call(alert)
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"[NOTIFICATIONS] Callback error: {e}")
    
    def _send_sms(self, alert: Alert) -> bool:
        """Send SMS alert via Twilio."""
        if not self._twilio_client:
            print("[NOTIFICATIONS] SMS skipped - Twilio not configured")
            return False
        
        try:
            message = self._twilio_client.messages.create(
                body=f"🚨 FORGE-Guard Alert: {alert.message}",
                from_=config.twilio.phone_number,
                to=config.twilio.emergency_contact
            )
            alert.sent_sms = True
            print(f"[NOTIFICATIONS] SMS sent: {message.sid}")
            return True
        except Exception as e:
            print(f"[NOTIFICATIONS] SMS failed: {e}")
            return False
    
    def _make_call(self, alert: Alert) -> bool:
        """Make voice call via Twilio."""
        if not self._twilio_client:
            print("[NOTIFICATIONS] Call skipped - Twilio not configured")
            return False
        
        try:
            # TwiML for the call - speaks the alert message
            twiml = f"""
            <Response>
                <Say voice="alice">
                    Emergency alert from FORGE Guard monitoring system.
                    {alert.message}
                    Please check on the person immediately.
                </Say>
                <Pause length="2"/>
                <Say voice="alice">
                    Repeating: {alert.message}
                </Say>
            </Response>
            """
            
            call = self._twilio_client.calls.create(
                twiml=twiml,
                from_=config.twilio.phone_number,
                to=config.twilio.emergency_contact
            )
            alert.sent_call = True
            print(f"[NOTIFICATIONS] Call initiated: {call.sid}")
            return True
        except Exception as e:
            print(f"[NOTIFICATIONS] Call failed: {e}")
            return False
    
    @property
    def alert_history(self) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return self._alert_history.copy()
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get most recent alerts."""
        with self._lock:
            return self._alert_history[-count:]
    
    def clear_history(self):
        """Clear alert history."""
        with self._lock:
            self._alert_history.clear()
    
    @property
    def is_twilio_configured(self) -> bool:
        """Check if Twilio is configured and ready."""
        return self._twilio_client is not None
    
    def stats(self) -> dict:
        """Get notification statistics."""
        with self._lock:
            return {
                "total_alerts": len(self._alert_history),
                "twilio_configured": self.is_twilio_configured,
                "cooldown_seconds": self.cooldown_seconds,
                "queue_size": self._alert_queue.qsize()
            }
