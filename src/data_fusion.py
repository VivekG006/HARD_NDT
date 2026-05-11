import numpy as np
import scipy.interpolate as interpolate
import logging
from src.exceptions import SensorDesyncError

log = logging.getLogger(__name__)

GRID_MARGIN_FACTOR = 0.05
FILL_VALUE = 0.0
Z_SOLID = 0.0
Z_HOLLOW = 1.0
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

class DataFusion:
    """
    Fuses sparse acoustic point cloud with spatial tracking data,
    then interpolates a dense 2D hollowness grid.
    """

    def __init__(self, config: dict) -> None:
        """
        Reads thresholds, tolerance, delay, and grid resolution from config.
        
        Inputs:
            config (dict): Fusion configuration dictionary.
            
        Outputs:
            None
            
        Raises:
            None
        """
        self._config = config

    def fuse(self, tracking_data: list[dict], acoustic_data: list[dict]) -> list[dict]:
        """
        Matches each acoustic event to its corresponding delayed visual frame.
        
        Inputs:
            tracking_data (list[dict]): Spatial data from VisionTracker.
            acoustic_data (list[dict]): Acoustic events from AudioDSP.
            
        Outputs:
            list[dict]: Fused datapoints containing x, y, and normalized h.
            
        Raises:
            SensorDesyncError: If zero points match within time tolerance.
        """
        time_tol = float(self._config.get("time_tolerance_seconds", 0.1))
        delay = float(self._config.get("visual_sample_delay_seconds", 0.050))
        solid_thresh = float(self._config.get("solid_thresh", 0.005))
        hollow_thresh = float(self._config.get("hollow_thresh", 0.025))
        
        fused_data = []
        
        # Array of raw video timestamps (no delay applied here)
        track_ts = np.array([float(d["timestamp"]) for d in tracking_data])
        
        for ac in acoustic_data:
            ac_ts = float(ac["timestamp"])
            raw_h = float(ac["h_metric"])
            
            # The exact time we expect the stick to be resting on the floor (prior to audio reaching mic)
            target_video_time = ac_ts - delay
            
            # Find the closest physical frame to the target time
            idx = (np.abs(track_ts - target_video_time)).argmin()
            nearest_ts = float(track_ts[idx])
            
            if abs(nearest_ts - target_video_time) <= time_tol:
                h_norm = (raw_h - solid_thresh) / (hollow_thresh - solid_thresh)
                h_norm = float(max(LOWER_BOUND, min(UPPER_BOUND, h_norm)))
                
                z_val = Z_SOLID + h_norm * (Z_HOLLOW - Z_SOLID)
                
                fused_data.append({
                    "timestamp": ac_ts,
                    "x": tracking_data[idx]["x"],
                    "y": tracking_data[idx]["y"],
                    "raw_h": raw_h,
                    "h": float(z_val)
                })
                
        if not fused_data:
            log.error("No acoustic points matched with vision tracking data.")
            raise SensorDesyncError("Zero points matched during fusion. Check visual_sample_delay_seconds.")
            
        log.info(f"Fused {len(fused_data)} points successfully.")
        return fused_data

    def interpolate(self, fused_points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts fused point cloud to dense grid via scipy.interpolate.griddata.
        
        Inputs:
            fused_points (list[dict]): The mapped acoustic/spatial data.
            
        Outputs:
            tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y, and Z grids.
            
        Raises:
            None
        """
        points = np.array([[p["x"], p["y"]] for p in fused_points])
        values = np.array([p["h"] for p in fused_points])
        
        grid_res = int(self._config.get("grid_resolution", 100))
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        margin_x = (x_max - x_min) * GRID_MARGIN_FACTOR
        margin_y = (y_max - y_min) * GRID_MARGIN_FACTOR
        
        grid_x, grid_y = np.mgrid[
            x_min - margin_x : x_max + margin_x : complex(0, grid_res),
            y_min - margin_y : y_max + margin_y : complex(0, grid_res)
        ]
        
        grid_z = interpolate.griddata(points, values, (grid_x, grid_y), method='linear', fill_value=FILL_VALUE)
        
        return grid_x.astype(np.float32), grid_y.astype(np.float32), grid_z.astype(np.float32)