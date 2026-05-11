import argparse
import logging
import subprocess
import sys
import os
import yaml
import numpy as np
from pathlib import Path

from src.audio_dsp import AudioDSP
from src.vision_tracker import VisionTracker
from src.data_fusion import DataFusion
from src.renderer import Renderer
from src.exceptions import HardNDTError, CalibrationError, SensorDesyncError
from src.media_handler import VideoAudioHandler

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger("main")

    parser = argparse.ArgumentParser(description="HARD NDT Acoustic Tomography Pipeline")
    parser.add_argument("video", nargs="?", help="Path to the input .mp4 scan file (optional if set in config)")
    parser.add_argument("--audio", help="Optional path to external high-fidelity audio (.wav)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--render-only", action="store_true",
                        help="Skip DSP/CV, load existing mesh_data.npz and launch viewer")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    video_path = Path(args.video) if args.video else Path(cfg.get("paths", {}).get("video_input")) if cfg.get("paths", {}).get("video_input") else None
    external_audio = Path(args.audio) if args.audio else Path(cfg.get("paths", {}).get("external_audio")) if cfg.get("paths", {}).get("external_audio") else None
    
    # Auto-detect external audio if a matching .wav exists next to the video
    if not external_audio and video_path:
        auto_audio_path = video_path.with_suffix(".wav")
        if auto_audio_path.exists():
            external_audio = auto_audio_path
            
    if not video_path:
        log.error("Video path not provided. Pass it via CLI or set 'paths.video_input' in config.yaml")
        sys.exit(1)

    try:
        if not args.render_only:
            # Step 1: Resolve audio source
            log.info("Resolving audio source...")
            handler = VideoAudioHandler(video_path, external_audio)
            wav_path = str(handler.resolve_audio())

            # Step 2: DSP — calibration + grid scan
            dsp = AudioDSP(cfg["audio"])
            dsp.check_ambient_noise(wav_path)
            log.info("Calibrating baseline...")
            baseline_fft = dsp.calibrate_baseline(wav_path)
            log.info("Processing grid scan...")
            acoustic_results = dsp.process_grid_scan(wav_path, baseline_fft)

            # Step 3: CV tracking
            tracker = VisionTracker(cfg["vision"])
            
            ref_photo_str = cfg["vision"].get("reference_photo_path")
            if ref_photo_str:
                ref_photo = Path(ref_photo_str)
                # Fallback to other extensions if exact file doesn't exist
                if not ref_photo.exists():
                    for ext in [".png", ".heic", ".HEIC", ".PNG", ".jpeg", ".JPEG", ".jpg"]:
                        candidate = ref_photo.with_suffix(ext)
                        if candidate.exists():
                            ref_photo = candidate
                            break
                            
                if ref_photo.exists():
                    log.info(f"Found reference photo at {ref_photo}, calibrating HSV bounds...")
                    tracker.calibrate_from_reference_image(ref_photo)
                
            log.info("Extracting marker tracking data...")
            tracking_data = tracker.extract_tracking_data(str(video_path))

            # Step 4: Fuse + interpolate
            fusion = DataFusion(cfg["fusion"])
            fused = fusion.fuse(tracking_data, acoustic_results)
            grid_x, grid_y, grid_z = fusion.interpolate(fused)

            # Step 5: Save mesh
            first_ts = acoustic_results[0]["timestamp"] if acoustic_results else 2.0
            np.savez(cfg["paths"]["mesh_output"],
                     x=grid_x, y=grid_y, z=grid_z, timestamp=first_ts)
            log.info(f"Mesh saved to {cfg['paths']['mesh_output']}")

        # Step 6: Render
        renderer = Renderer(cfg["renderer"])
        mesh_data, timestamp = renderer.load_mesh(cfg["paths"]["mesh_output"])
        frame = renderer.capture_frame(str(video_path), timestamp)
        renderer.build_and_render_scene(
            mesh_data,
            frame
        )

    except CalibrationError as e:
        log.error(f"Calibration failed: {e}")
        sys.exit(1)
    except SensorDesyncError as e:
        log.error(f"A/V sync failed: {e}. Check visual_sample_delay_seconds in config.")
        sys.exit(1)
    except HardNDTError as e:
        log.error(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()