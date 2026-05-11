import os
import subprocess
import logging
import cv2
from pathlib import Path
from src.exceptions import AudioExtractionError, VideoSourceError

log = logging.getLogger(__name__)

class VideoAudioHandler:
    """
    Handles FFmpeg audio extraction and OpenCV video writer initialization.
    """
    
    def __init__(self, video_path: str | Path, external_audio: str | Path | None = None) -> None:
        """
        Initializes the handler with a video path and an optional external audio path.
        
        Inputs:
            video_path (str | Path): Path to the primary video file.
            external_audio (str | Path | None): Optional path to a high-quality external audio file.
            
        Outputs:
            None
            
        Raises:
            None
        """
        self.video_path = Path(video_path)
        self.external_audio = Path(external_audio) if external_audio else None
        self.audio_source: Path | None = None

    def resolve_audio(self) -> Path:
        """
        Resolves the audio source, prioritizing external audio over video extraction.
        
        Inputs:
            None
            
        Outputs:
            Path: Path to the resolved audio file.
            
        Raises:
            AudioExtractionError: If extraction fails or no audio source is valid.
        """
        if self.external_audio and self.external_audio.exists():
            log.info(f"Using High-Quality External Audio: {self.external_audio}")
            self.audio_source = self.external_audio
        else:
            log.info("No external audio found. Extracting from video...")
            self.audio_source = self.extract_audio_from_video()
            
        return self.audio_source

    def extract_audio_from_video(self) -> Path:
        """
        Extracts high-fidelity audio from the video using FFmpeg for FFT analysis.
        
        Inputs:
            None
            
        Outputs:
            Path: Path to the temporary extracted .wav file.
            
        Raises:
            AudioExtractionError: If FFmpeg fails or the video has no audio track.
        """
        temp_audio = self.video_path.with_name(f"{self.video_path.stem}_extracted.wav")
        
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "48000", "-ac", "1", str(temp_audio)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"FFmpeg failed:\n{result.stderr}")
            raise AudioExtractionError("Failed to extract audio from video using FFmpeg. Video may have no audio track.")
            
        if not temp_audio.exists():
            raise AudioExtractionError("FFmpeg reported success but the output audio file was not found.")
            
        return temp_audio

    def get_writer(self, output_path: str | Path, width: int, height: int, fps: float = 30.0) -> cv2.VideoWriter:
        """
        Creates an OpenCV VideoWriter with the appropriate FourCC codec based on extension.
        
        Inputs:
            output_path (str | Path): Path to the output video file.
            width (int): Frame width.
            height (int): Frame height.
            fps (float): Frames per second.
            
        Outputs:
            cv2.VideoWriter: Initialized OpenCV VideoWriter object.
            
        Raises:
            VideoSourceError: If the writer fails to open.
        """
        output_path = Path(output_path)
        ext = output_path.suffix.lower().lstrip('.')
        
        # 'avc1' or 'X264' is hardware-accelerated for Ryzen encoders
        codecs = {
            'mp4':  'avc1', 
            'mov':  'apcn', # ProRes 
            'mkv':  'h264', 
            'webm': 'vp09', # VP9
            'h265': 'hevc', # H.265 / HEVC
            'hevc': 'hevc'
        }
        
        fourcc_str = codecs.get(ext, 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise VideoSourceError(f"Failed to initialize VideoWriter for {output_path}")
            
        return writer
