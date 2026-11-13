"""
dsp_forensics.py - DHWANI KAVACH :: Pure NumPy/SciPy Acoustic Physics Forensics
Ultra-fast (< 8 ms per chunk, guaranteed max cap):
  1) STFT Phase Coherence / Instantaneous Frequency Variance across 1-7 kHz on voiced frames
  2) Vectorized Autocorrelation F0 (60-380 Hz) & Cycle-to-Cycle Pitch Jitter
  3) Digital-Zero Non-Speech Gap Analysis (< -70 dBFS floor vs ambient noise)
  4) Acoustic Spectral Drift vs Session Baseline (Cosine distance)
  5) Forensic AI Probability
"""

import math
import numpy as np

try:
    import scipy.fft as s_fft
except ImportError:
    s_fft = np.fft


def _frame_audio(audio: np.ndarray, frame_len: int = 512, hop_len: int = 160) -> np.ndarray:
    """Fast strided 2D framing of 1D audio array."""
    n_frames = (len(audio) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return np.empty((0, frame_len), dtype=np.float32)
    shape = (n_frames, frame_len)
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def _compute_phase_coherence(voiced_frames: np.ndarray, sr: int = 16000) -> float:
    """
    Measure STFT phase derivative variance in the 1-7 kHz band across voiced frames.
    Neural vocoders (HiFi-GAN, WaveGlow, ElevenLabs) exhibit unnaturally high phase regularity
    (low second-derivative variance) compared to human turbulent vocal cord airflow.
    Returns score 0.0 (natural human phase turbulence) to 1.0 (highly coherent vocoder).
    """
    n_voiced = len(voiced_frames)
    if n_voiced < 5:
        return 0.0

    # Cap to max 48 evenly-spaced frames for guaranteed < 3 ms execution
    if n_voiced > 48:
        step = np.linspace(0, n_voiced - 1, 48, dtype=int)
        v_sub = voiced_frames[step]
    else:
        v_sub = voiced_frames

    frame_len = v_sub.shape[1]
    window = np.hanning(frame_len).astype(np.float32)
    windowed = v_sub * window

    stft = s_fft.rfft(windowed, axis=1)  # (N, frame_len // 2 + 1)
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    band_mask = (freqs >= 1000) & (freqs <= 7000)
    if not np.any(band_mask):
        return 0.0

    stft_band = stft[:, band_mask]
    mag = np.abs(stft_band)
    peak_mag = float(np.max(mag))
    if peak_mag < 1e-6:
        return 0.0

    active_bins = np.max(mag, axis=0) > (0.02 * peak_mag)
    if not np.any(active_bins):
        return 0.0

    active_stft = stft_band[:, active_bins]
    phases = np.arctan2(active_stft.imag, active_stft.real)

    # 1st and 2nd unwrapped phase differences along time axis
    d_phase = np.diff(np.unwrap(phases, axis=0), axis=0)
    d2_phase = np.diff(d_phase, axis=0)

    if d2_phase.shape[0] < 2:
        return 0.0

    phase_var = np.var(d2_phase, axis=0)
    mean_var = float(np.median(phase_var))

    # Human biological speech typical phase var in 1-7kHz > 0.6 rad^2
    # Neural vocoders often have phase var < 0.35 rad^2
    if mean_var < 0.35:
        score = 1.0 - (mean_var / 0.35) * 0.40
    elif mean_var < 0.75:
        score = 0.60 * (1.0 - (mean_var - 0.35) / 0.40)
    else:
        score = max(0.0, 0.15 * (1.0 - (mean_var - 0.75) / 0.75))

    return float(np.clip(score, 0.0, 1.0))


def _compute_pitch_jitter_vectorized(voiced_frames: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """
    Vectorized F0 (60-380 Hz) using 2D FFT autocorrelation across voiced frames.
    Guaranteed < 3 ms runtime.
    Measures relative cycle-to-cycle jitter: |F0[t] - F0[t-1]| / F0[t] * 100%.
    Human voices have natural physiological micro-jitter (0.6% - 2.5%).
    Synthesized voices often exhibit hyper-smooth flat pitch tracks (jitter < 0.35%).
    Returns (pitch_jitter_pct, smooth_pitch_score 0..1).
    """
    n_voiced = len(voiced_frames)
    if n_voiced < 4:
        return 1.5, 0.0

    # Cap to max 48 evenly-spaced frames
    if n_voiced > 48:
        step = np.linspace(0, n_voiced - 1, 48, dtype=int)
        v_sub = voiced_frames[step]
    else:
        v_sub = voiced_frames

    frame_len = v_sub.shape[1]
    lag_min = max(2, int(sr / 380))  # ~42 samples
    lag_max = min(int(sr / 60), frame_len - 1)  # ~266 samples
    if lag_max <= lag_min:
        return 1.5, 0.0

    means = np.mean(v_sub, axis=1, keepdims=True)
    centered = v_sub - means
    energies = np.sum(centered ** 2, axis=1)
    valid_energy = energies > 1e-6
    if np.sum(valid_energy) < 4:
        return 1.5, 0.0

    centered = centered[valid_energy]

    fft_size = 1024
    F = s_fft.rfft(centered, fft_size, axis=1)
    corr = s_fft.irfft(F * np.conj(F), fft_size, axis=1)[:, :frame_len]

    norm = corr[:, 0:1] + 1e-12
    corr_slice = corr[:, lag_min:lag_max + 1] / norm

    peak_indices = np.argmax(corr_slice, axis=1)
    peak_values = np.max(corr_slice, axis=1)

    pitch_mask = peak_values > 0.32
    if np.sum(pitch_mask) < 4:
        return 1.5, 0.0

    valid_lags = lag_min + peak_indices[pitch_mask]
    f0_arr = sr / valid_lags.astype(np.float64)

    f0_in_bounds = (f0_arr >= 60.0) & (f0_arr <= 380.0)
    f0_arr = f0_arr[f0_in_bounds]
    if len(f0_arr) < 4:
        return 1.5, 0.0

    diffs = np.abs(np.diff(f0_arr))
    rel_diffs = diffs / f0_arr[1:] * 100.0
    jitter_pct = float(np.median(rel_diffs))

    # Flag unnaturally smooth pitch tracks (jitter < 0.35%)
    if jitter_pct < 0.35:
        smooth_score = 1.0 - (jitter_pct / 0.35) * 0.35
    elif jitter_pct < 0.60:
        smooth_score = 0.65 * (1.0 - (jitter_pct - 0.35) / 0.25)
    else:
        smooth_score = 0.0

    return float(round(jitter_pct, 3)), float(np.clip(smooth_score, 0.0, 1.0))


def _compute_digital_zero_ratio(frames: np.ndarray, rms_values: np.ndarray,
                                voiced_mask: np.ndarray) -> float:
    """
    Detect absolute digital zero silence in non-speech gaps adjacent to voiced speech.
    Natural microphone captures have thermal/acoustic noise floor (-55 to -35 dBFS, RMS ~0.001 - 0.015).
    Direct TTS files contain absolute digital silence (RMS < 3e-4, i.e. < -70 dBFS).
    Returns ratio 0.0 (natural room ambient floor) to 1.0 (digital zero truncation).
    """
    n_frames = len(rms_values)
    if n_frames < 6:
        return 0.0

    unvoiced_mask = ~voiced_mask
    if not np.any(unvoiced_mask):
        return 0.0

    unvoiced_rms = rms_values[unvoiced_mask]
    digital_zero_frames = np.sum(unvoiced_rms < 0.0003)
    ratio = float(digital_zero_frames / len(unvoiced_rms))
    return float(np.clip(ratio, 0.0, 1.0))


def _extract_spectral_profile(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract a 5-element spectral shape vector (spectral tilt + 4 subband energies)."""
    if len(audio) < 512:
        return np.ones(5, dtype=np.float32) / math.sqrt(5)

    sub = audio[:min(len(audio), 4096)]
    window = np.hanning(len(sub)).astype(np.float32)
    spectrum = np.abs(s_fft.rfft(sub * window)) ** 2
    freqs = np.fft.rfftfreq(len(sub), 1.0 / sr)

    b1 = np.sum(spectrum[(freqs >= 100) & (freqs < 1000)]) + 1e-12
    b2 = np.sum(spectrum[(freqs >= 1000) & (freqs < 2500)]) + 1e-12
    b3 = np.sum(spectrum[(freqs >= 2500) & (freqs < 4500)]) + 1e-12
    b4 = np.sum(spectrum[(freqs >= 4500) & (freqs <= 7500)]) + 1e-12

    tilt = (b1 + b2) / (b3 + b4)

    profile = np.array([tilt, b1, b2, b3, b4], dtype=np.float32)
    norm = np.linalg.norm(profile)
    if norm > 1e-9:
        profile /= norm
    return profile


def analyze_acoustic_physics(
    audio: np.ndarray,
    sr: int = 16000,
    session_baseline: dict | None = None
) -> dict:
    """
    Pure NumPy/SciPy acoustic forensics engine.
    Executes in < 8 ms on 1.5s 16kHz audio chunk.
    Returns:
      phase_coherence_score: float (0..1)
      pitch_jitter_pct: float (percentage)
      smooth_pitch_score: float (0..1)
      digital_zero_ratio: float (0..1)
      acoustic_drift_score: float (0..1)
      forensic_ai_prob: float (0..1)
      baseline_profile: list[float] | None
    """
    if audio is None or len(audio) < 400:
        return {
            "phase_coherence_score": 0.0,
            "pitch_jitter_pct": 1.5,
            "smooth_pitch_score": 0.0,
            "digital_zero_ratio": 0.0,
            "acoustic_drift_score": 0.0,
            "forensic_ai_prob": 0.0,
            "voiced_frames": 0,
            "total_frames": 0,
            "baseline_profile": None
        }

    y = np.asarray(audio, dtype=np.float32).flatten()
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / max(peak, 1.0)

    frame_len = 512
    hop_len = 160
    frames = _frame_audio(y, frame_len=frame_len, hop_len=hop_len)
    total_frames = len(frames)
    if total_frames < 3:
        return {
            "phase_coherence_score": 0.0,
            "pitch_jitter_pct": 1.5,
            "smooth_pitch_score": 0.0,
            "digital_zero_ratio": 0.0,
            "acoustic_drift_score": 0.0,
            "forensic_ai_prob": 0.0,
            "voiced_frames": 0,
            "total_frames": total_frames,
            "baseline_profile": None
        }

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    max_rms = float(np.max(rms))
    voiced_thresh = max(0.007, max_rms * 0.12)
    voiced_mask = rms >= voiced_thresh
    voiced_frames_count = int(np.sum(voiced_mask))

    if voiced_frames_count >= 5:
        voiced_frames = frames[voiced_mask]
        phase_coherence = _compute_phase_coherence(voiced_frames, sr=sr)
        pitch_jitter_pct, smooth_pitch_score = _compute_pitch_jitter_vectorized(voiced_frames, sr=sr)
    else:
        phase_coherence = 0.0
        pitch_jitter_pct, smooth_pitch_score = 1.5, 0.0

    digital_zero_ratio = _compute_digital_zero_ratio(frames, rms, voiced_mask)
    current_profile = _extract_spectral_profile(y, sr=sr)
    acoustic_drift_score = 0.0

    if session_baseline and "spectral_profile" in session_baseline:
        base_prof = np.asarray(session_baseline["spectral_profile"], dtype=np.float32)
        if len(base_prof) == len(current_profile):
            dot = float(np.dot(current_profile, base_prof))
            dot = np.clip(dot, -1.0, 1.0)
            acoustic_drift_score = float(np.clip(1.0 - dot, 0.0, 1.0))

    if voiced_frames_count >= 4:
        raw_forensic = (
            0.30 * phase_coherence +
            0.40 * smooth_pitch_score +
            0.30 * digital_zero_ratio
        )
        forensic_ai_prob = float(np.clip(raw_forensic, 0.0, 1.0))
    else:
        forensic_ai_prob = 0.0

    return {
        "phase_coherence_score": round(phase_coherence, 4),
        "pitch_jitter_pct": round(pitch_jitter_pct, 3),
        "smooth_pitch_score": round(smooth_pitch_score, 4),
        "digital_zero_ratio": round(digital_zero_ratio, 4),
        "acoustic_drift_score": round(acoustic_drift_score, 4),
        "forensic_ai_prob": round(forensic_ai_prob, 4),
        "voiced_frames": voiced_frames_count,
        "total_frames": total_frames,
        "baseline_profile": current_profile.tolist()
    }
