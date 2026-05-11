# HARD NDT Pipeline: Mathematical Reference

This document serves as the mathematical foundation for the HARD NDT Acoustic Tomography pipeline. It details the exact formulas and mappings used to translate an audio recording and an MP4 video into a 3D topographic heat map.

## Acoustic DSP (Spectral Hollowness)

The acoustic metric measures how significantly a strike's spectral resonance deviates from a solid concrete baseline. It is strictly amplitude-invariant.

**1. Time-Domain Slicing:** For each detected transient peak at time $T_{impact}$, a "tail window" is isolated to ignore the broadband initial strike and capture only the resonant decay:

$$Window = [T_{impact} + 10ms, \quad T_{impact} + 150ms]$$

**2. Frequency-Domain Transformation:** The window is filtered through a Butterworth bandpass ($100Hz \rightarrow 800Hz$), then transformed to the frequency domain using a real-valued Fast Fourier Transform:

$$S(f) = |FFT(Window_{filtered})|$$

**3. Unit-Energy Normalization:** To achieve amplitude invariance (so a hard strike and a soft strike produce the same metric if they hit the same material), the target frequency band is normalized so the total area under the curve equals $1.0$:

$$\hat{S}_{band} = \frac{S(100 \dots 800Hz)}{\sum S_{band}}$$

**4. The $H$ Metric:** Hollowness is calculated as the absolute area of difference between the current strike's normalized spectrum and the baseline's normalized spectrum:

$$H_{raw} = \sum |\hat{S}_{current} - \hat{S}_{baseline}|$$

## Temporal Synchronization

Because sound travels through air slower than light hits the camera sensor, an audio impact recorded by the microphone is temporally delayed from the visual strike seen in the video. The pipeline applies a reverse-delay to map the acoustic timestamp ($T_{audio}$) back to the correct physical video frame ($T_{video}$):

$$T_{video} = T_{audio} - Delay_{acoustic}$$

This ensures the spatial tracking grabs the $X, Y$ coordinate precisely when the stick's tip is touching the concrete, rather than when it is rebounding in the air.

## Topographic Math (Hollowness to Z-Axis)

Once the raw metrics and spatial coordinates are fused, the data is interpolated onto a dense 3D grid.

**1. Normalization:** $H_{raw}$ is mapped to a normalized range $[0.0, 1.0]$ using configuration thresholds, and clamped to avoid extreme outliers:

$$H_{norm} = \max\left(0.0, \quad \min\left(1.0, \quad \frac{H_{raw} - Threshold_{solid}}{Threshold_{hollow} - Threshold_{solid}}\right)\right)$$

**2. Physical Z-Mapping:** The normalized metric translates to a topological $Z$ coordinate.

$$Z_{val} = Z_{solid} + H_{norm} \times (Z_{hollow} - Z_{solid})$$

Where $Z_{solid} = 0.0$ and $Z_{hollow} = 1.0$. Thus, a highly hollow point produces $Z_{val} = 1.0$.

**3. Dense Interpolation:** The sparse point cloud $(X, Y, Z_{val})$ is extrapolated into a continuous surface using `scipy.interpolate.griddata` with a `linear` interpolation methodology.

## Render Projection (PyVista AR Overlay)

The final step is translating the mathematical grid into a textured 3D environment that perfectly matches the user's physical perspective.

**1. Spatial Coordinates:** OpenCV tracks $X$ (left-to-right) and $Y$ (top-to-bottom). PyVista maps the video texture onto a 3D plane mapping $Y=0$ to the top of the room and $Y=Height$ to the bottom of the room.

**2. Pixel Scaling:** The raw topological coordinate ($Z_{val}$) is scaled into visual screen pixels for the 3D camera:

$$Z_{pixels} = (Z_{val} \times Depth_{pixels}) + Lift_{pixels}$$

A fully hollow mountain ($Z_{val} = 1.0$) with a depth config of 140 and lift of 2 will project to $Z_{pixels} = 142.0$.

**3. Visual Camera Matrix:** To accurately represent physical space without mirroring left/right or top/bottom, the PyVista virtual camera is anchored at the mathematical bottom of the room, looking upwards towards the center:

$$
\begin{aligned}
Camera_{X} &= Center_{X} \\
Camera_{Y} &= Center_{Y} + (MaxDim \times 1.25) \\
Camera_{Z} &= MaxDim \times 0.85 \\
Up_{Vector} &= (0.0, \quad 0.0, \quad 1.0)
\end{aligned}
$$

*(Because $Y$ increases downwards, adding to $Y$ pushes the camera to the physical bottom edge).*