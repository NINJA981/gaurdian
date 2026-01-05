"""
Tests for Fall Detector module.
"""

import pytest
import numpy as np
import cv2

from forge_guard.detectors.fall_detector import FallDetector
from forge_guard.detectors.base_detector import AlertLevel


class TestFallDetector:
    """Test cases for FallDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create a fall detector instance."""
        return FallDetector()
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame for testing."""
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.name == "fall_detector"
        assert detector.enabled is True
        assert detector.ratio_threshold == 0.8
        assert detector.frame_threshold == 5
    
    def test_detector_disabled(self, detector, dummy_frame):
        """Test detector returns empty result when disabled."""
        detector.enabled = False
        result = detector.detect(dummy_frame)
        
        assert result.detected is False
        assert "disabled" in result.message.lower()
    
    def test_no_person_detected(self, detector, dummy_frame):
        """Test detector handles no person in frame."""
        result = detector.detect(dummy_frame)
        
        assert result.detected is False
        assert "no person" in result.message.lower()
    
    def test_stats(self, detector):
        """Test detector statistics."""
        stats = detector.stats()
        
        assert "name" in stats
        assert stats["name"] == "fall_detector"
        assert "enabled" in stats
        assert "detection_count" in stats
    
    def test_cleanup(self, detector):
        """Test detector cleanup."""
        # Should not raise any exceptions
        detector.cleanup()
    
    def test_draw_overlay_no_detection(self, detector, dummy_frame):
        """Test draw overlay with no detection."""
        from forge_guard.detectors.base_detector import DetectionResult
        
        result = DetectionResult(detected=False, message="Test")
        frame = detector.draw_overlay(dummy_frame.copy(), result)
        
        # Frame should be modified (has status text)
        assert frame.shape == dummy_frame.shape


class TestFallDetectorIntegration:
    """Integration tests for FallDetector (requires MediaPipe)."""
    
    @pytest.fixture
    def detector(self):
        """Create a fall detector instance."""
        return FallDetector()
    
    def test_process_real_frame(self, detector):
        """Test processing a frame with a person-like shape."""
        # Create a frame with a simple person-like shape
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw a simple standing figure (vertical rectangle)
        cv2.rectangle(frame, (600, 100), (680, 600), (255, 255, 255), -1)
        cv2.circle(frame, (640, 80), 40, (255, 255, 255), -1)  # Head
        
        result = detector.detect(frame)
        
        # Should process without error
        assert result is not None
        assert hasattr(result, 'detected')
        assert hasattr(result, 'alert_level')
