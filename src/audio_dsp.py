import numpy as np
import librosa
import scipy.signal as signal
import logging
from src.exceptions import AudioExtractionError, AmbientNoiseError, CalibrationError

log = logging.getLogger(__name__)

TARGET_SR = 44100
HOP_LENGTH = 512
MS_TO_SEC = 1000.0
CALIBRATION_TIME_MARGIN_FACTOR = 2.5
NYQUIST_FACTOR = 0.5

class AudioDSP:
    """
    Acoustic signal processor for HARD NDT.
    """

    def __init__(self, config: dict) -> None:
        """
        Reads all audio parameters from config. No side effects.
        
        Inputs:
            config (dict): Audio configuration dictionary.
            
        Outputs:
            None
            
        Raises:
            None
        """
        self._config = config

    def check_ambient_noise(self, wav_path: str) -> None:
        """
        Samples the first N seconds of the WAV and computes RMS.
        
        Inputs:
            wav_path (str): Path to the audio file.
            
        Outputs:
            None
            
        Raises:
            AudioExtractionError: If the file cannot be loaded.
            AmbientNoiseError: If the RMS exceeds the config threshold.
        """
        sample_time = float(self._config.get("ambient_noise_sample_seconds", 1.0))
        threshold = float(self._config.get("ambient_noise_rms_threshold", 0.05))
        
        try:
            y, _ = librosa.load(wav_path, sr=TARGET_SR, duration=sample_time, mono=True)
        except Exception as e:
            log.error(f"Failed to load audio for ambient check: {e}")
            raise AudioExtractionError(f"Could not load {wav_path}: {e}")
            
        rms = float(np.sqrt(np.mean(y**2)))
        log.info(f"Ambient RMS: {rms:.4f} (threshold: {threshold})")
        
        if rms > threshold:
            log.error("Ambient noise is too high.")
            raise AmbientNoiseError(f"Ambient RMS {rms:.4f} > {threshold}")

    def calibrate_baseline(self, wav_path: str) -> np.ndarray:
        """
        Detects the N calibration knocks, processes each with _analyze_strike(),
        and returns the mean normalized FFT magnitude.
        
        Inputs:
            wav_path (str): Path to the audio file.
            
        Outputs:
            np.ndarray: The mean normalized FFT magnitude representing a solid baseline.
            
        Raises:
            AudioExtractionError: If the file cannot be loaded.
            CalibrationError: If fewer than the expected number of knocks are found.
        """
        try:
            y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            log.error(f"Failed to load audio for calibration: {e}")
            raise AudioExtractionError(f"Could not load {wav_path}: {e}")
            
        pre_max = float(self._config.get("calibration_onset_pre_max", 0.5))
        post_max = float(self._config.get("calibration_onset_post_max", 0.5))
        delta = float(self._config.get("calibration_onset_delta", 0.1))
        min_gap = float(self._config.get("calibration_min_gap_seconds", 0.5))
        expected_knocks = int(self._config.get("calibration_knock_count", 5))
        
        calib_end_time = expected_knocks * min_gap * CALIBRATION_TIME_MARGIN_FACTOR
        calib_y = y[:int(calib_end_time * sr)]
        
        onsets = self._detect_onsets(calib_y, sr, min_gap, pre_max, post_max, delta)
        log.info(f"Found {len(onsets)} calibration knocks (expected {expected_knocks})")
        
        if len(onsets) < expected_knocks:
            err_msg = f"Insufficient calibration knocks. Expected {expected_knocks}, got {len(onsets)}"
            log.error(err_msg)
            raise CalibrationError(err_msg)
            
        spectra = []
        for t in onsets[:expected_knocks]:
            spec = self._analyze_strike(y, sr, t)
            if spec is not None:
                spectra.append(spec)
                
        if not spectra:
            raise CalibrationError("All calibration knocks produced empty spectra.")
            
        min_len = min(len(s) for s in spectra)
        spectra = [s[:min_len] for s in spectra]
        
        mean_spec = np.mean(spectra, axis=0)
        energy = np.sum(mean_spec)
        return mean_spec / energy if energy > 0 else mean_spec

    def process_grid_scan(self, wav_path: str, baseline_fft: np.ndarray) -> list[dict]:
        """
        Detects all grid strike events, computes H metric for each.
        
        Inputs:
            wav_path (str): Path to the audio file.
            baseline_fft (np.ndarray): The normalized baseline spectrum.
            
        Outputs:
            list[dict]: A list of dictionaries containing timestamp and h_metric.
            
        Raises:
            AudioExtractionError: If the file cannot be loaded.
        """
        try:
            y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            raise AudioExtractionError(f"Could not load {wav_path}: {e}")
            
        pre_max = float(self._config.get("grid_onset_pre_max", 0.1))
        post_max = float(self._config.get("grid_onset_post_max", 0.1))
        delta = float(self._config.get("grid_onset_delta", 0.05))
        min_gap = float(self._config.get("grid_min_gap_seconds", 0.2))
        
        calib_gap = float(self._config.get("calibration_min_gap_seconds", 0.5))
        calib_count = int(self._config.get("calibration_knock_count", 5))
        calib_end_time = calib_count * calib_gap * CALIBRATION_TIME_MARGIN_FACTOR
        
        onsets = self._detect_onsets(y, sr, min_gap, pre_max, post_max, delta)
        grid_onsets = [t for t in onsets if t > calib_end_time]
        log.info(f"Found {len(grid_onsets)} grid scan events")
        
        results = []
        for t in grid_onsets:
            spec = self._analyze_strike(y, sr, t)
            if spec is not None:
                compare_len = min(len(spec), len(baseline_fft))
                spec_comp = spec[:compare_len]
                base_comp = baseline_fft[:compare_len]
                
                h_metric = float(np.sum(np.abs(spec_comp - base_comp)))
                results.append({"timestamp": float(t), "h_metric": h_metric})
                
        return results

    def _detect_onsets(self, y: np.ndarray, sr: int, min_gap: float,
                       pre_max: float, post_max: float, delta: float) -> list[float]:
        """
        Shared onset detection with gap filtering.
        
        Inputs:
            y (np.ndarray): Audio time series.
            sr (int): Sample rate.
            min_gap (float): Minimum gap between onsets in seconds.
            pre_max (float): Pre-max filter config.
            post_max (float): Post-max filter config.
            delta (float): Delta threshold.
            
        Outputs:
            list[float]: Clean onset times in seconds.
            
        Raises:
            None
        """
        wait_frames = int(min_gap * sr / float(HOP_LENGTH))
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, pre_max=pre_max, post_max=post_max, delta=delta, wait=wait_frames
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        clean_onsets = []
        last_t = -min_gap
        for t in onset_times:
            if t - last_t >= min_gap:
                clean_onsets.append(float(t))
                last_t = t
        return clean_onsets

    def _analyze_strike(self, y: np.ndarray, sr: int, impact_time: float) -> np.ndarray | None:
        """
        Extracts the decay tail, applies a bandpass filter, and computes
        an amplitude-invariant frequency spectrum for the target band.
        
        Inputs:
            y (np.ndarray): Full audio time series.
            sr (int): Sample rate in Hz.
            impact_time (float): Timestamp of the transient peak in seconds.
            
        Outputs:
            np.ndarray | None: Normalized frequency magnitude array, or None if invalid.
            
        Raises:
            None
        """
        tail_start_ms = float(self._config.get("tail_start_offset_ms", 10.0))
        tail_end_ms = float(self._config.get("tail_end_offset_ms", 150.0))
        
        start_idx = int((impact_time + tail_start_ms / MS_TO_SEC) * sr)
        end_idx = int((impact_time + tail_end_ms / MS_TO_SEC) * sr)
        
        if start_idx >= len(y) or end_idx <= start_idx:
            return None
            
        tail = y[start_idx:min(end_idx, len(y))]
        
        lowcut = float(self._config.get("bandpass_low_hz", 100.0))
        highcut = float(self._config.get("bandpass_high_hz", 800.0))
        
        # Time-domain Butterworth filter
        filtered_tail = self._apply_bandpass(tail, lowcut, highcut, sr)
        
        # Transform to frequency domain
        fft_result = np.fft.rfft(filtered_tail)
        fft_freqs = np.fft.rfftfreq(len(filtered_tail), 1/sr)
        fft_magnitude = np.abs(fft_result)
        
        # Strict slicing: Isolate the target resonance band BEFORE normalization
        band_indices = np.where((fft_freqs >= lowcut) & (fft_freqs <= highcut))[0]
        sliced_magnitude = fft_magnitude[band_indices]
        
        # Amplitude Invariance: Normalize target band to unit energy
        energy = np.sum(sliced_magnitude)
        if energy == 0.0:
            return sliced_magnitude
            
        return sliced_magnitude / energy

    def _apply_bandpass(self, data: np.ndarray, lowcut: float,
                        highcut: float, sr: int) -> np.ndarray:
        """
        Butterworth bandpass filter.
        
        Inputs:
            data (np.ndarray): Input audio data.
            lowcut (float): Lower frequency bound in Hz.
            highcut (float): Upper frequency bound in Hz.
            sr (int): Sample rate.
            
        Outputs:
            np.ndarray: Bandpass filtered data.
            
        Raises:
            None
        """
        nyq = NYQUIST_FACTOR * sr
        low = lowcut / nyq
        high = highcut / nyq
        order = int(self._config.get("filter_order", 5))
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)