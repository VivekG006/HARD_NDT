# System Architecture

The HARD, NDT (Hollowness using Acoustic Relative Detection, Non-Destructive Testing) pipeline is designed as a modular, decoupled data processing system. 

It was developed through an iterative AI-assisted workflow, which deeply influenced its strict separation of concerns. By keeping the modules isolated, we were able to independently prompt the AI to solve specific domain problems (e.g., OpenCV spatial mapping vs. PyVista 3D rendering) without cross-contaminating the logic.

## Core Modules

The pipeline executes synchronously via `main.py` and relies on four primary modules located in `src/`:

### 1. `MediaHandler`
Responsible for file ingestion.
*   Uses `FFmpeg` via subprocess to dynamically extract high-fidelity `.wav` tracks from compressed `.mp4` video inputs.
*   Prioritizes external `.wav` files if they are provided, ensuring the downstream DSP engine always receives the highest quality uncompressed audio.

### 2. `AudioDSP`
The mathematical core of the acoustic analysis.
*   Detects impact transients.
*   Slices the "decay tail" of the impact, bypassing the initial broadband noise of the strike.
*   Performs FFT (Fast Fourier Transform) analysis and bandpass filtering ($100Hz \rightarrow 800Hz$) to isolate the resonant frequencies that characterize "hollowness".
*   Produces a normalized scalar value ($0.0 = Solid, 1.0 = Hollow$) for each strike.

### 3. `VisionTracker`
The spatial mapping engine.
*   Uses OpenCV to decode the video frame-by-frame.
*   Applies an HSV color mask to isolate the yellow (default) tracking tape on the acoustic probe.
*   Performs morphological operations (erosion/dilation) and contour extraction to find the centroid of the tape.
*   Applies a configurable Y-pixel offset to estimate the exact point where the stick contacts the floor.

### 4. `DataFusion`
The temporal synchronizer.
*   Aligns the asynchronous data streams from `AudioDSP` and `VisionTracker`.
*   Crucially, it **compensates for the speed of sound**. Because light travels faster than sound, the visual impact occurs slightly before the audio transient is recorded by the microphone. `DataFusion` subtracts a calculated delay from the audio timestamp to ensure the tracker captures the stick exactly at the moment of impact, rather than during its rebound.
*   Interpolates the discrete impact points into a continuous $X, Y, Z$ Cartesian mesh.

### 5. `Renderer`
The visualization engine.
*   Uses `PyVista` (backed by `VTK`) to project the generated 3D topography directly onto a decoded 2D background frame from the video.
*   Uses the hollowness metric to colorize the mesh (via a heat map) and deform the Z-axis (creating physical "mountains" where the surface is hollow).

## AI-Assisted Design Philosophy

Because this project was heavily assisted by AI (AntiGravity), the architecture strictly enforces:
1.  **Immutability of Data:** Modules do not mutate each other's state. They return clean, typed data structures.
2.  **Configuration-Driven Logic:** Magic numbers are banned. All thresholds, delays, and filter constraints are injected via `config.yaml`.
3.  **Explicit Boundaries:** The AI was constrained by a master ruleset (`AGENTS.md`), which mandated strict PEP 257 docstrings, type hinting, and custom exceptions (`src/exceptions.py`). This prevented the AI from writing monolithic "spaghetti" scripts during rapid prototyping.
