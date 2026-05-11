class HardNDTError(Exception):
    """Base class for all HARD NDT pipeline errors."""

class VideoSourceError(HardNDTError):
    """Raised when the video file cannot be opened or decoded."""

class AudioExtractionError(HardNDTError):
    """Raised when FFmpeg fails to extract the WAV audio track."""

class AmbientNoiseError(HardNDTError):
    """Raised when background noise exceeds the threshold for a valid scan."""

class CalibrationError(HardNDTError):
    """Raised when fewer than the required baseline knocks are detected."""

class SensorDesyncError(HardNDTError):
    """Raised when audio and visual timestamps cannot be matched within tolerance."""

class TrackingError(HardNDTError):
    """Raised when the CV tracker finds no valid marker frames."""