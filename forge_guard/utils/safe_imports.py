"""
FORGE-Guard Safe Imports Utility
Wraps risky imports (cv2, mediapipe, numpy, ultralytics) to prevent crashes on incompatible systems.
Production-ready with proper lazy loading and error handling.
"""

import sys
import importlib
import logging
from typing import Optional, Any
from functools import lru_cache

# Setup logger
logger = logging.getLogger(__name__)

# ============================================================================
# Safe Import Helper
# ============================================================================

class SafeImport:
    """Helper to lazy-load modules or return mocks on failure."""
    
    def __init__(self, module_name: str, strict: bool = False):
        self.module_name = module_name
        self.strict = strict
        self._module = None
        self._tried = False
        self._error: Optional[Exception] = None
        self._lock = None  # Lazy init to avoid import issues

    def _ensure_lock(self):
        """Ensure thread lock is initialized."""
        if self._lock is None:
            import threading
            self._lock = threading.Lock()

    @property
    def module(self) -> Optional[Any]:
        """Get the module, loading it if necessary."""
        self._ensure_lock()
        with self._lock:
            if not self._tried:
                try:
                    self._module = importlib.import_module(self.module_name)
                    logger.info(f"[SafeImport] Successfully loaded '{self.module_name}'")
                except ImportError as e:
                    self._error = e
                    logger.warning(f"[SafeImport] Module '{self.module_name}' not installed: {e}")
                except Exception as e:
                    self._error = e
                    logger.warning(f"[SafeImport] Failed to load '{self.module_name}': {e}")
                finally:
                    self._tried = True
            return self._module

    @property
    def is_available(self) -> bool:
        """Check if module is available."""
        return self.module is not None

    @property
    def error(self) -> Optional[Exception]:
        """Get the error that occurred during import, if any."""
        return self._error


# ============================================================================
# Module Singletons (lazy initialized)
# ============================================================================

_cv2: Optional[SafeImport] = None
_mediapipe: Optional[SafeImport] = None
_numpy: Optional[SafeImport] = None
_ultralytics: Optional[SafeImport] = None


def _get_cv2_import() -> SafeImport:
    """Get or create cv2 SafeImport singleton."""
    global _cv2
    if _cv2 is None:
        _cv2 = SafeImport("cv2")
    return _cv2


def _get_mediapipe_import() -> SafeImport:
    """Get or create mediapipe SafeImport singleton."""
    global _mediapipe
    if _mediapipe is None:
        _mediapipe = SafeImport("mediapipe")
    return _mediapipe


def _get_numpy_import() -> SafeImport:
    """Get or create numpy SafeImport singleton."""
    global _numpy
    if _numpy is None:
        _numpy = SafeImport("numpy")
    return _numpy


def _get_ultralytics_import() -> SafeImport:
    """Get or create ultralytics SafeImport singleton."""
    global _ultralytics
    if _ultralytics is None:
        _ultralytics = SafeImport("ultralytics")
    return _ultralytics


# ============================================================================
# Mock Classes for Graceful Degradation
# ============================================================================

class CV2Mock:
    """Mock cv2 module for when OpenCV is not available."""
    
    # Common constants
    FONT_HERSHEY_SIMPLEX = 0
    FONT_HERSHEY_PLAIN = 1
    FONT_HERSHEY_DUPLEX = 2
    COLOR_BGR2RGB = 4
    COLOR_RGB2BGR = 4
    COLOR_BGR2GRAY = 6
    THRESH_BINARY = 0
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1
    
    @staticmethod
    def rectangle(*args, **kwargs):
        """Mock rectangle drawing."""
        pass
    
    @staticmethod
    def putText(*args, **kwargs):
        """Mock text drawing."""
        pass
    
    @staticmethod
    def addWeighted(src1, alpha, src2, beta, gamma, *args, **kwargs):
        """Mock addWeighted - return src1 if available."""
        return src1 if src1 is not None else src2
    
    @staticmethod
    def cvtColor(src, code, *args, **kwargs):
        """Mock color conversion - return input."""
        return src
    
    @staticmethod
    def GaussianBlur(src, ksize, sigmaX, *args, **kwargs):
        """Mock blur - return input."""
        return src
    
    @staticmethod
    def resize(src, dsize, *args, **kwargs):
        """Mock resize - return input."""
        return src
    
    @staticmethod
    def absdiff(src1, src2, *args, **kwargs):
        """Mock absdiff."""
        np = get_numpy()
        if np is not None:
            return np.abs(src1.astype(float) - src2.astype(float)).astype(src1.dtype)
        return src1
    
    @staticmethod
    def threshold(src, thresh, maxval, type_, *args, **kwargs):
        """Mock threshold."""
        np = get_numpy()
        if np is not None:
            result = (src > thresh).astype(src.dtype) * maxval
            return maxval, result
        return maxval, src
    
    @staticmethod
    def imencode(ext, img, params=None):
        """Mock imencode."""
        return False, None
    
    @staticmethod
    def getTextSize(text, fontFace, fontScale, thickness):
        """Mock getTextSize."""
        return (len(text) * 10, 20), 5
    
    class VideoCapture:
        """Mock VideoCapture."""
        def __init__(self, *args, **kwargs):
            self._opened = False
        
        def isOpened(self) -> bool:
            return self._opened
        
        def read(self):
            return False, None
        
        def set(self, propId, value):
            pass
        
        def release(self):
            pass


# ============================================================================
# Public Module Accessors
# ============================================================================

def get_cv2() -> Any:
    """Get cv2 module or a safe mock."""
    cv2_import = _get_cv2_import()
    if cv2_import.is_available:
        return cv2_import.module
    return CV2Mock()


def get_mediapipe() -> Optional[Any]:
    """Get mediapipe module or None."""
    return _get_mediapipe_import().module


def get_numpy() -> Optional[Any]:
    """Get numpy module or None."""
    return _get_numpy_import().module


def get_yolo() -> Optional[Any]:
    """Get Ultralytics YOLO class or None."""
    ul = _get_ultralytics_import().module
    if ul and hasattr(ul, "YOLO"):
        return ul.YOLO
    return None


def check_safe_imports() -> dict:
    """Return status of all safe imports."""
    return {
        "cv2": _get_cv2_import().is_available,
        "mediapipe": _get_mediapipe_import().is_available,
        "numpy": _get_numpy_import().is_available,
        "ultralytics": _get_ultralytics_import().is_available
    }


def get_import_errors() -> dict:
    """Return any import errors that occurred."""
    return {
        "cv2": str(_get_cv2_import().error) if _get_cv2_import().error else None,
        "mediapipe": str(_get_mediapipe_import().error) if _get_mediapipe_import().error else None,
        "numpy": str(_get_numpy_import().error) if _get_numpy_import().error else None,
        "ultralytics": str(_get_ultralytics_import().error) if _get_ultralytics_import().error else None,
    }
