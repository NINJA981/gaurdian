"""
Tests for Notification Manager module.
"""

import pytest
import time
import threading

from forge_guard.alerts.notification_manager import (
    NotificationManager, 
    AlertPriority, 
    Alert
)


class TestNotificationManager:
    """Test cases for NotificationManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a notification manager instance."""
        nm = NotificationManager(cooldown_seconds=1)  # Short cooldown for tests
        nm.start()
        yield nm
        nm.stop()
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.cooldown_seconds == 1
        assert manager._running is True
    
    def test_send_alert(self, manager):
        """Test sending an alert."""
        alert = manager.send_alert(
            message="Test alert",
            source="test",
            priority=AlertPriority.MEDIUM
        )
        
        assert alert is not None
        assert alert.message == "Test alert"
        assert alert.source == "test"
        assert alert.priority == AlertPriority.MEDIUM
    
    def test_alert_cooldown(self, manager):
        """Test that cooldown prevents rapid alerts."""
        # First alert should succeed
        alert1 = manager.send_alert(
            message="Alert 1",
            source="test_source",
            priority=AlertPriority.MEDIUM
        )
        assert alert1 is not None
        
        # Second alert from same source should be rate-limited
        alert2 = manager.send_alert(
            message="Alert 2",
            source="test_source",
            priority=AlertPriority.MEDIUM
        )
        assert alert2 is None  # Rate limited
    
    def test_critical_bypasses_cooldown(self, manager):
        """Test that CRITICAL alerts bypass cooldown."""
        # First alert
        alert1 = manager.send_alert(
            message="Alert 1",
            source="test_source",
            priority=AlertPriority.MEDIUM
        )
        assert alert1 is not None
        
        # CRITICAL should bypass cooldown
        alert2 = manager.send_alert(
            message="Critical Alert",
            source="test_source",
            priority=AlertPriority.CRITICAL
        )
        assert alert2 is not None
    
    def test_different_sources_not_rate_limited(self, manager):
        """Test that different sources are not affected by each other's cooldown."""
        alert1 = manager.send_alert(
            message="Alert from source A",
            source="source_a",
            priority=AlertPriority.MEDIUM
        )
        
        alert2 = manager.send_alert(
            message="Alert from source B",
            source="source_b",
            priority=AlertPriority.MEDIUM
        )
        
        assert alert1 is not None
        assert alert2 is not None
    
    def test_callback_invoked(self, manager):
        """Test that callbacks are invoked for alerts."""
        callback_received = []
        
        def callback(alert):
            callback_received.append(alert)
        
        manager.add_callback(callback)
        
        alert = manager.send_alert(
            message="Callback test",
            source="test",
            priority=AlertPriority.HIGH
        )
        
        # Wait for worker thread to process
        time.sleep(0.5)
        
        assert len(callback_received) > 0
    
    def test_alert_history(self, manager):
        """Test alert history tracking."""
        manager.send_alert(
            message="History test",
            source="test",
            priority=AlertPriority.LOW
        )
        
        # Wait for processing
        time.sleep(0.5)
        
        history = manager.alert_history
        assert len(history) > 0
    
    def test_stats(self, manager):
        """Test manager statistics."""
        stats = manager.stats()
        
        assert "total_alerts" in stats
        assert "twilio_configured" in stats
        assert "cooldown_seconds" in stats
    
    def test_alert_to_dict(self):
        """Test Alert serialization."""
        alert = Alert(
            id="test_123",
            priority=AlertPriority.HIGH,
            message="Test message",
            source="test"
        )
        
        data = alert.to_dict()
        
        assert data["id"] == "test_123"
        assert data["priority"] == "HIGH"
        assert data["message"] == "Test message"
        assert data["source"] == "test"
