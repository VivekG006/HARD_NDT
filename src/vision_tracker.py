import numpy as np
import cv2
import subprocess
import json
import logging
from pathlib import Path
from src.exceptions import VideoSourceError, TrackingError

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 0
HSV_CHANNELS = 3

class VisionTracker:
    """
    HSV-based marker tracker for the impact probe's yellow tape band.
    """

    def __init__(self, config: dict) -> None:
        """
        Reads HSV bounds, kernel size, tip offset from config.
        
        Inputs:
            config (dict): Vision configuration dictionary.
            
        Outputs:
            None
            
        Raises:
            None
        """
        self._config = config
        self._hsv_lower = np.array(self._config.get("hsv_lower", [20, 100, 100]), dtype=np.uint8)
        self._hsv_upper = np.array(self._config.get("hsv_upper", [40, 255, 255]), dtype=np.uint8)

    def get_fps(self, video_path: str) -> float:
        """
        Tries FFprobe first for exact FPS, falls back to cv2.CAP_PROP_FPS.
        
        Inputs:
            video_path (str): Path to the .mp4 video.
            
        Outputs:
            float: Frames per second.
            
        Raises:
            VideoSourceError: If cv2 cannot open the video for fallback.
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            r_frame_rate = info["streams"][0]["r_frame_rate"]
            num, den = map(int, r_frame_rate.split('/'))
            if den > 0:
                fps = num / den
                log.info(f"FFprobe detected FPS: {fps:.2f}")
                return fps
        except Exception as e:
            log.warning(f"FFprobe failed ({e}). Falling back to cv2.CAP_PROP_FPS.")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoSourceError(f"Cannot open video for FPS check: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            log.warning(f"cv2 returned invalid FPS. Defaulting to {DEFAULT_FPS} as last resort.")
            fps = DEFAULT_FPS
        return float(fps)

    def calibrate_from_reference_image(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads a reference photo of the tracking tape, samples the center region,
        and dynamically sets the HSV tracking bounds based on actual room lighting.
        
        Inputs:
            image_path (str | Path): Filepath to the calibration photo (e.g., 'tape_ref.jpg').
            
        Outputs:
            tuple[np.ndarray, np.ndarray]: The computed lower and upper HSV bounds.
            
        Raises:
            VideoSourceError: If the image cannot be loaded.
            TrackingError: If the image is invalid or completely dark.
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        
        # Fallback for Apple HEIC format since OpenCV lacks native support
        if img is None and image_path.suffix.lower() in ['.heic', '.heif']:
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except ImportError:
                log.error("pillow-heif is required to read .heic files.")
                
        if img is None:
            log.error(f"Failed to load reference image: {image_path}")
            raise VideoSourceError(f"Could not load calibration image at {image_path}")
            
        log.warning("WIP QoL update: Reference image color auto-calibration is currently disabled.")
        log.warning("Falling back to manual HSV bounds from config.yaml.")
        
        # Revert: Return the default config bounds without attempting to process the image
        return self._hsv_lower, self._hsv_upper

    def extract_tracking_data(self, video_path: str) -> list[dict]:
        """
        Main frame loop. Blurs, masks, and tracks the marker.
        
        Inputs:
            video_path (str): Path to the video.
            
        Outputs:
            list[dict]: List containing timestamp, x, and y for each tracked frame.
            
        Raises:
            VideoSourceError: If the video cannot be opened.
            TrackingError: If zero valid frames are extracted.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoSourceError(f"Cannot open video {video_path}")
            
        fps = self.get_fps(video_path)
        
        kernel_size = int(self._config.get("morph_kernel_size", 5))
        dilation_iters = int(self._config.get("dilation_iterations", 1))
        tip_offset = int(self._config.get("tip_y_offset_px", 50))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        tracking_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            frame_idx += 1
            
            blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if dilation_iters > 0:
                mask = cv2.dilate(mask, kernel, iterations=dilation_iters)
                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    cy += tip_offset
                    
                    tracking_data.append({
                        "timestamp": float(timestamp),
                        "x": int(cx),
                        "y": int(cy)
                    })
                    
        cap.release()
        
        if not tracking_data:
            raise TrackingError("Zero valid frames extracted; marker never found.")
            
        log.info(f"Extracted {len(tracking_data)} tracked frames.")
        return tracking_data