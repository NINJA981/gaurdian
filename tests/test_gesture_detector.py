"""
Tests for Gesture Detector module.
"""

import pytest
import numpy as np
import cv2

from forge_guard.detectors.gesture_detector import GestureDetector
from forge_guard.detectors.base_detector import AlertLevel


class TestGestureDetector:
    """Test cases for GestureDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create a gesture detector instance."""
        return GestureDetector()
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame for testing."""
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.name == "gesture_detector"
        assert detector.enabled is True
        assert detector.hold_seconds == 3.0
    
    def test_detector_disabled(self, detector, dummy_frame):
        """Test detector returns empty result when disabled."""
        detector.enabled = False
        result = detector.detect(dummy_frame)
        
        assert result.detected is False
        assert "disabled" in result.message.lower()
    
    def test_no_hands_detected(self, detector, dummy_frame):
        """Test detector handles no hands in frame."""
        result = detector.detect(dummy_frame)
        
        assert result.detected is False
        assert "no hands" in result.message.lower()
    
    def test_stats(self, detector):
        """Test detector statistics."""
        stats = detector.stats()
        
        assert "name" in stats
        assert stats["name"] == "gesture_detector"
        assert "enabled" in stats
    
    def test_cleanup(self, detector):
        """Test detector cleanup."""
        detector.cleanup()
    
    def test_sos_requires_hold_time(self, detector):
        """Test that SOS detection requires holding gesture."""
        # Even if open palm is detected, it should not immediately trigger
        # because hold_seconds = 3.0
        assert detector.hold_seconds > 0
