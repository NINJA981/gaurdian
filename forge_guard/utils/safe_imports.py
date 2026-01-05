"""
FORGE-Guard Safe Imports Utility
Wraps risky imports (cv2, mediapipe, numpy, ultralytics) to prevent crashes on incompatible systems.
"""

import sys
import importlib
from typing import Optional, Any
from unittest.mock import MagicMock

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
        self._error = None

    @property
    def module(self) -> Optional[Any]:
        if not self._tried:
            try:
                self._module = importlib.import_module(self.module_name)
            except Exception as e:
                self._error = e
                # Special handling for numpy/cv2 MINGW crashes (often abort process, but if we are here we caught it)
                print(f"⚠️ [SafeImport] Failed to load '{self.module_name}': {e}")
            self._tried = True
        return self._module

    @property
    def is_available(self) -> bool:
        return self.module is not None

# ============================================================================
# Specific Module Accessors
# ============================================================================

_cv2 = SafeImport("cv2")
_mediapipe = SafeImport("mediapipe")
_numpy = SafeImport("numpy")
_ultralytics = SafeImport("ultralytics")

def get_cv2() -> Any:
    """Get cv2 module or a safe mock."""
    if _cv2.is_available:
        return _cv2.module
    
    # Return a mock that allows basic constants but warns on usage
    mock = MagicMock()
    # Mock specific constants used in code to prevent AttributeErrors
    mock.FONT_HERSHEY_SIMPLEX = 0
    mock.COLOR_BGR2RGB = 1
    mock.rectangle = MagicMock()
    mock.putText = MagicMock()
    mock.addWeighted = MagicMock()
    return mock

def get_mediapipe() -> Any:
    """Get mediapipe module or None."""
    return _mediapipe.module

def get_numpy() -> Any:
    """Get numpy module or None."""
    return _numpy.module

def get_yolo() -> Any:
    """Get Ultralytics YOLO class or None."""
    ul = _ultralytics.module
    if ul and hasattr(ul, "YOLO"):
        return ul.YOLO
    return None

def check_safe_imports() -> dict:
    """Return status of all safe imports."""
    return {
        "cv2": _cv2.is_available,
        "mediapipe": _mediapipe.is_available,
        "numpy": _numpy.is_available,
        "ultralytics": _ultralytics.is_available
    }
