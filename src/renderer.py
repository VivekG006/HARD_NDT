import logging
import cv2
import numpy as np
import pyvista as pv
from src.exceptions import VideoSourceError, HardNDTError

log = logging.getLogger(__name__)

# Constants for plane topology
VERTICES_PER_FACE = 4

class Renderer:
    """
    AR-style 3D acoustic tomography viewer using PyVista.
    Projects the interpolated mesh structure directly onto the 2D video frame.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes rendering configuration parameters from the config dictionary.
        
        Inputs:
            config (dict): Configuration dictionary for rendering thresholds.
            
        Outputs:
            None
        """
        self._config = config
        self.z_depth = float(self._config.get("z_depth_pixels", 140.0))
        self.opacity = float(self._config.get("surface_opacity", 0.78))
        self.lift = float(self._config.get("surface_lift_pixels", 2.0))
        
        window_size = self._config.get("window_size", [1400, 900])
        self.window_width = int(window_size[0])
        self.window_height = int(window_size[1])

    def load_mesh(self, mesh_path: str) -> tuple[pv.StructuredGrid, float]:
        """
        Loads mesh_data.npz and converts raw coordinate arrays into a PyVista grid.
        
        Inputs:
            mesh_path (str): Filepath to the .npz archive.
            
        Outputs:
            tuple[pv.StructuredGrid, float]: The constructed 3D surface and the background frame timestamp.
            
        Raises:
            HardNDTError: If the arrays cannot be loaded or have mismatched shapes.
        """
        try:
            data = np.load(mesh_path)
            grid_x = data["x"].astype(np.float32)
            grid_y = data["y"].astype(np.float32)
            grid_z = data["z"].astype(np.float32)
            timestamp_seconds = float(data.get("timestamp", 2.0))
        except Exception as e:
            log.error(f"Failed to load mesh data: {e}")
            raise HardNDTError(f"Mesh data corruption: {e}")

        if grid_x.shape != grid_y.shape or grid_x.shape != grid_z.shape:
            raise HardNDTError("mesh_data.npz arrays must have matching dimensions.")

        # Scale physical crater depth
        z_pixels = (grid_z * self.z_depth) + self.lift

        surface = pv.StructuredGrid(grid_x, grid_y, z_pixels)

        # Generate positive scalar for heatmap projection (0.0=Solid, 1.0=Hollow)
        hollowness = np.clip(np.abs(grid_z), 0.0, 1.0)
        surface["Hollowness"] = hollowness.ravel(order="F")

        log.info("Successfully constructed PyVista StructuredGrid.")
        return surface, timestamp_seconds

    def capture_frame(self, video_path: str, timestamp_seconds: float) -> np.ndarray:
        """
        Captures a single RGB frame from the H.264 source video to act as the base plane.
        
        Inputs:
            video_path (str): Filepath to the source video.
            timestamp_seconds (float): Specific timestamp to seek and decode.
            
        Outputs:
            np.ndarray: The decoded RGB frame.
            
        Raises:
            VideoSourceError: If OpenCV cannot open the file or decode the timestamp.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoSourceError(f"Could not open video file: {video_path}")

        # Seek by timestamp instead of frame index
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok or frame_bgr is None:
            raise VideoSourceError(f"Could not capture background frame at {timestamp_seconds:.3f}s")

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _make_textured_video_plane(self, frame_rgb: np.ndarray) -> tuple[pv.PolyData, pv.Texture, int, int]:
        """
        Builds a flat rectangular PyVista plane mapped exactly to the video dimensions.
        
        Inputs:
            frame_rgb (np.ndarray): The decoded background frame.
            
        Outputs:
            tuple[pv.PolyData, pv.Texture, int, int]: The plane, texture, and boundary dimensions.
        """
        height, width, _ = frame_rgb.shape

        points = np.array([
            [0.0, 0.0, 0.0],
            [float(width), 0.0, 0.0],
            [float(width), float(height), 0.0],
            [0.0, float(height), 0.0],
        ], dtype=np.float32)

        faces = np.array([VERTICES_PER_FACE, 0, 1, 2, 3])
        plane = pv.PolyData(points, faces)

        # Invert V coordinate to match OpenCV orientation
        plane.active_texture_coordinates = np.array([
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
        ], dtype=np.float32)

        texture = pv.Texture(frame_rgb)
        return plane, texture, width, height

    def build_and_render_scene(self, surface: pv.StructuredGrid, frame_rgb: np.ndarray) -> None:
        """
        Assembles the PyVista scene and launches the interactive viewer.
        
        Inputs:
            surface (pv.StructuredGrid): The 3D topography mesh.
            frame_rgb (np.ndarray): The 2D background image.
            
        Outputs:
            None
            
        Raises:
            None
        """
        video_plane, video_texture, frame_width, frame_height = self._make_textured_video_plane(frame_rgb)

        plotter = pv.Plotter(window_size=(self.window_width, self.window_height))
        plotter.set_background("black")

        plotter.add_mesh(video_plane, texture=video_texture, smooth_shading=False, show_edges=False)

        plotter.add_mesh(
            surface,
            scalars="Hollowness",
            cmap="coolwarm",
            clim=[0.0, 1.0],
            opacity=self.opacity,
            smooth_shading=True,
            show_edges=False,
            scalar_bar_args={"title": "Hollowness", "vertical": True, "position_x": 0.88, "position_y": 0.22}
        )

        plotter.add_mesh(
            surface.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False),
            color="white", line_width=2, opacity=0.85
        )

        # Contextual oblique camera perspective
        center_x, center_y = frame_width * 0.5, frame_height * 0.5
        max_dim = max(frame_width, frame_height)
        
        plotter.camera_position = [
            (center_x, center_y + max_dim * 1.25, max_dim * 0.85),
            (center_x, center_y, -self.z_depth * 0.35),
            (0.0, 0.0, 1.0),
        ]

        plotter.enable_anti_aliasing()
        plotter.add_axes()
        
        log.info("Launching interactive 3D acoustic tomography view.")
        plotter.show(title="HARD NDT - AR Acoustic Tomography")