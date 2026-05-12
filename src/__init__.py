"""
HARD NDT — Acoustic Surface Tomography Pipeline
src package initializer.

Exposes the four pipeline classes and all custom exceptions at the
package level so main.py can do clean top-level imports:

    from src import AudioDSP, VisionTracker, DataFusion, Renderer
    from src import VideoSourceError, SensorDesyncError, ...
"""


# Pipeline classes
from src.audio_dsp       import AudioDSP
from src.vision_tracker  import VisionTracker
from src.data_fusion     import DataFusion
from src.renderer        import Renderer

# Custom exceptions — imported here so callers never need to know
# which sub-module an exception lives in.
from src.exceptions import (
    HardNDTError,
    VideoSourceError,
    AudioExtractionError,
    AmbientNoiseError,
    CalibrationError,
    SensorDesyncError,
    TrackingError,
)

__all__ = [
    # Classes
    "AudioDSP",
    "VisionTracker",
    "DataFusion",
    "Renderer",
    # Exceptions
    "HardNDTError",
    "VideoSourceError",
    "AudioExtractionError",
    "AmbientNoiseError",
    "CalibrationError",
    "SensorDesyncError",
    "TrackingError",
]