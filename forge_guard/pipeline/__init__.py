"""Pipeline module initialization."""

from .video_pipeline import VideoPipeline, FrameProducer, FrameConsumer, ProcessedFrame
from .frame_buffer import FrameBuffer

__all__ = ['VideoPipeline', 'FrameProducer', 'FrameConsumer', 'FrameBuffer', 'ProcessedFrame']

