"""
FORGE-Guard Drawing Utilities
Overlay drawing utilities for skeletons, bounding boxes, and zones.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Colors:
    """Color palette for FORGE-Guard UI (BGR format)."""
    # Primary colors
    PRIMARY = (255, 128, 0)       # Orange - FORGE brand
    SECONDARY = (0, 200, 255)     # Cyan
    ACCENT = (180, 100, 255)      # Purple
    
    # Status colors
    SUCCESS = (0, 255, 100)       # Green
    WARNING = (0, 200, 255)       # Orange-yellow
    DANGER = (0, 0, 255)          # Red
    INFO = (255, 200, 0)          # Cyan-blue
    
    # Neutral colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (50, 50, 50)
    
    # Detection-specific colors
    SKELETON = (255, 128, 0)      # Orange for pose
    HAND = (0, 255, 128)          # Green for hands
    FACE = (255, 100, 200)        # Pink for face
    OBJECT = (100, 255, 255)      # Cyan for objects
    ZONE = (0, 255, 0)            # Green for ROI zones
    
    @staticmethod
    def with_alpha(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        """Blend color towards black with alpha."""
        return tuple(int(c * alpha) for c in color)


class DrawingUtils:
    """
    Utility class for drawing overlays on video frames.
    Provides consistent styling across the FORGE-Guard interface.
    """
    
    @staticmethod
    def draw_text(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = Colors.WHITE,
        font_scale: float = 0.6,
        thickness: int = 2,
        background: bool = False,
        bg_color: Tuple[int, int, int] = Colors.DARK_GRAY,
        bg_padding: int = 5
    ) -> np.ndarray:
        """
        Draw text with optional background.
        
        Args:
            frame: Frame to draw on
            text: Text to draw
            position: (x, y) position for text
            color: Text color (BGR)
            font_scale: Font size scale
            thickness: Text thickness
            background: Whether to draw background
            bg_color: Background color
            bg_padding: Background padding
            
        Returns:
            Frame with text drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if background:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x, y = position
            cv2.rectangle(
                frame,
                (x - bg_padding, y - text_size[1] - bg_padding),
                (x + text_size[0] + bg_padding, y + bg_padding),
                bg_color,
                -1
            )
        
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
    
    @staticmethod
    def draw_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = Colors.PRIMARY,
        thickness: int = 2,
        label: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw bounding box with optional label.
        
        Args:
            frame: Frame to draw on
            bbox: (x1, y1, x2, y2) bounding box
            color: Box color
            thickness: Line thickness
            label: Optional label text
            confidence: Optional confidence score
            
        Returns:
            Frame with box drawn
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        if label:
            text = label
            if confidence is not None:
                text = f"{label}: {confidence:.0%}"
            
            DrawingUtils.draw_text(
                frame,
                text,
                (x1, y1 - 5),
                color=Colors.WHITE,
                font_scale=0.5,
                thickness=1,
                background=True,
                bg_color=color
            )
        
        return frame
    
    @staticmethod
    def draw_zone(
        frame: np.ndarray,
        zone_bounds: Tuple[int, int, int, int],
        name: str,
        active: bool = False,
        alert: bool = False
    ) -> np.ndarray:
        """
        Draw ROI zone with styling.
        
        Args:
            frame: Frame to draw on
            zone_bounds: (x1, y1, x2, y2) zone bounds
            name: Zone name
            active: Whether zone is actively detecting
            alert: Whether zone has triggered an alert
            
        Returns:
            Frame with zone drawn
        """
        x1, y1, x2, y2 = zone_bounds
        
        # Choose color based on state
        if alert:
            color = Colors.DANGER
            thickness = 3
        elif active:
            color = Colors.WARNING
            thickness = 2
        else:
            color = Colors.ZONE
            thickness = 2
        
        # Draw dashed rectangle
        for i in range(4):
            if i == 0:  # Top
                pts = [(x1 + j, y1) for j in range(0, x2 - x1, 10)]
            elif i == 1:  # Right
                pts = [(x2, y1 + j) for j in range(0, y2 - y1, 10)]
            elif i == 2:  # Bottom
                pts = [(x2 - j, y2) for j in range(0, x2 - x1, 10)]
            else:  # Left
                pts = [(x1, y2 - j) for j in range(0, y2 - y1, 10)]
            
            for j, pt in enumerate(pts):
                if j % 2 == 0 and j + 1 < len(pts):
                    cv2.line(frame, pt, pts[j + 1], color, thickness)
        
        # Draw label
        DrawingUtils.draw_text(
            frame,
            f"📦 {name}",
            (x1 + 5, y1 + 20),
            color=color,
            font_scale=0.5,
            thickness=1,
            background=True,
            bg_color=Colors.DARK_GRAY
        )
        
        return frame
    
    @staticmethod
    def draw_status_bar(
        frame: np.ndarray,
        fps: float,
        modules_active: int,
        alert_count: int
    ) -> np.ndarray:
        """
        Draw status bar at bottom of frame.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS
            modules_active: Number of active detection modules
            alert_count: Number of recent alerts
            
        Returns:
            Frame with status bar
        """
        h, w = frame.shape[:2]
        bar_height = 35
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, h - bar_height),
            (w, h),
            Colors.DARK_GRAY,
            -1
        )
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Draw status items
        y = h - 10
        
        # FPS
        fps_color = Colors.SUCCESS if fps >= 25 else Colors.WARNING if fps >= 15 else Colors.DANGER
        DrawingUtils.draw_text(frame, f"FPS: {fps:.1f}", (10, y), fps_color, 0.5, 1)
        
        # Modules
        DrawingUtils.draw_text(
            frame, 
            f"Modules: {modules_active}", 
            (120, y), 
            Colors.INFO, 
            0.5, 
            1
        )
        
        # Alerts
        alert_color = Colors.DANGER if alert_count > 0 else Colors.GRAY
        DrawingUtils.draw_text(
            frame, 
            f"Alerts: {alert_count}", 
            (250, y), 
            alert_color, 
            0.5, 
            1
        )
        
        # FORGE branding
        DrawingUtils.draw_text(
            frame,
            "FORGE-Guard",
            (w - 120, y),
            Colors.PRIMARY,
            0.5,
            2
        )
        
        return frame
    
    @staticmethod
    def draw_alert_banner(
        frame: np.ndarray,
        message: str,
        level: str = "WARNING"
    ) -> np.ndarray:
        """
        Draw alert banner at top of frame.
        
        Args:
            frame: Frame to draw on
            message: Alert message
            level: Alert level (INFO, WARNING, CRITICAL)
            
        Returns:
            Frame with alert banner
        """
        h, w = frame.shape[:2]
        banner_height = 50
        
        # Choose color based on level
        if level == "CRITICAL":
            bg_color = Colors.DANGER
        elif level == "WARNING":
            bg_color = Colors.WARNING
        else:
            bg_color = Colors.INFO
        
        # Draw banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        # Draw message
        DrawingUtils.draw_text(
            frame,
            message,
            (w // 2 - len(message) * 6, 32),
            Colors.WHITE,
            0.8,
            2
        )
        
        return frame
    
    @staticmethod
    def blend_overlay(
        frame: np.ndarray,
        overlay_color: Tuple[int, int, int],
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Blend a color overlay onto the frame.
        
        Args:
            frame: Frame to blend onto
            overlay_color: Color to overlay
            alpha: Blend factor (0-1)
            
        Returns:
            Blended frame
        """
        overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
