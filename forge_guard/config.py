"""
FORGE-Guard Configuration Module
Centralized configuration management using pydantic-settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class TwilioSettings(BaseSettings):
    """Twilio API configuration for SMS/Call alerts."""
    account_sid: str = Field(default="", env="TWILIO_ACCOUNT_SID")
    auth_token: str = Field(default="", env="TWILIO_AUTH_TOKEN")
    phone_number: str = Field(default="", env="TWILIO_PHONE_NUMBER")
    emergency_contact: str = Field(default="", env="EMERGENCY_CONTACT_NUMBER")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.phone_number)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class VideoSettings(BaseSettings):
    """Video capture and processing settings."""
    width: int = Field(default=1280, env="VIDEO_WIDTH")
    height: int = Field(default=720, env="VIDEO_HEIGHT")
    fps: int = Field(default=30, env="VIDEO_FPS")
    buffer_size: int = Field(default=5, env="VIDEO_BUFFER_SIZE")
    camera_index: int = Field(default=0, env="CAMERA_INDEX")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class DetectionSettings(BaseSettings):
    """Detection module thresholds and parameters."""
    # Fall Detection
    fall_ratio_threshold: float = Field(default=0.8, env="FALL_RATIO_THRESHOLD")
    fall_frame_threshold: int = Field(default=5, env="FALL_FRAME_THRESHOLD")
    fall_velocity_threshold: float = Field(default=50.0, env="FALL_VELOCITY_THRESHOLD")
    
    # Gesture Detection
    gesture_hold_seconds: float = Field(default=3.0, env="GESTURE_HOLD_SECONDS")
    palm_distance_threshold: float = Field(default=0.15, env="PALM_DISTANCE_THRESHOLD")
    
    # Medicine Monitor
    medicine_change_threshold: float = Field(default=0.2, env="MEDICINE_CHANGE_THRESHOLD")
    
    # Object Detection
    object_confidence_threshold: float = Field(default=0.5, env="OBJECT_CONFIDENCE_THRESHOLD")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class AlertSettings(BaseSettings):
    """Alert and notification settings."""
    cooldown_seconds: int = Field(default=300, env="ALERT_COOLDOWN_SECONDS")
    log_file_path: str = Field(default="logs/forge_guard.log", env="LOG_FILE_PATH")
    max_log_size_mb: int = Field(default=10, env="MAX_LOG_SIZE_MB")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    
    class Config:
        env_file = ".env"
        extra = "ignore"


class ServerSettings(BaseSettings):
    """Web server configuration."""
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class ForgeGuardConfig:
    """Main configuration container for FORGE-Guard system."""
    
    _instance: Optional['ForgeGuardConfig'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all configuration settings."""
        self.twilio = TwilioSettings()
        self.video = VideoSettings()
        self.detection = DetectionSettings()
        self.alerts = AlertSettings()
        self.server = ServerSettings()
    
    def reload(self):
        """Reload configuration from environment."""
        self._initialize()


# Global configuration instance
config = ForgeGuardConfig()
