# channel_calibration.py
# Channel-Degradation & Uncertainty Calibration for DHWANI-KAVACH

import numpy as np
import librosa


def evaluate_channel_and_calibrate(y: np.ndarray, sr: int, raw_ai_prob: float) -> dict:
    """Evaluate audio channel degradation (telephony / noise / band-limiting)
    and calibrate risk tier and uncertainty note accordingly.

    Args:
        y: Audio signal array (1D or 2D).
        sr: Sampling rate in Hz.
        raw_ai_prob: Raw synthetic speech probability [0.0, 1.0].

    Returns:
        dict:
            channel_quality: "TELEPHONY_DEGRADED" | "WIDEBAND_CLEAN"
            calibrated_prob: float
            tier: "ALLOW" | "WARN" | "HOLD"
            uncertainty_note: str
            channel_metrics: {"high_band_ratio": float, "snr_db": float}
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    n_fft = 512
    hop_length = 160

    if len(y) == 0:
        high_band_ratio = 0.0
        snr_db = 0.0
    else:
        # 1. STFT power spectrum (n_fft=512, hop_length=160)
        # Compare mean energy in 300-3400 Hz (e_low) vs 4000-7800 Hz (e_high)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        power_spec = np.abs(D) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        low_mask = (freqs >= 300) & (freqs <= 3400)
        high_mask = (freqs >= 4000) & (freqs <= 7800)

        e_low = float(np.mean(power_spec[low_mask])) if np.any(low_mask) else 0.0
        e_high = float(np.mean(power_spec[high_mask])) if np.any(high_mask) else 0.0
        high_band_ratio = float(e_high / (e_low + 1e-10))

        # 2. Frame RMS SNR (20 * log10(percentile(rms, 90) / (percentile(rms, 15) + 1e-8)))
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        if len(rms) > 0:
            p90 = float(np.percentile(rms, 90))
            p15 = float(np.percentile(rms, 15))
            val = max(p90, 1e-8) / (max(p15, 0.0) + 1e-8)
            snr_db = float(20.0 * np.log10(val))
        else:
            snr_db = 0.0

    # 3. Degradation criteria: sr <= 8000 or high_band_ratio < 0.0012
    is_degraded = bool((sr <= 8000) or (high_band_ratio < 0.0012))
    channel_quality = "TELEPHONY_DEGRADED" if is_degraded else "WIDEBAND_CLEAN"

    # 4. Calibrate risk tier ("ALLOW" | "WARN" | "HOLD") and uncertainty_note
    prob = float(raw_ai_prob)
    if is_degraded:  # TELEPHONY_DEGRADED
        if prob >= 0.82:
            tier = "HOLD"
            calibrated_prob = round(max(prob, 0.86), 4)
            uncertainty_note = "Degraded 8kHz/noisy channel • Strong synthetic artifact overrides noise"
        else:
            tier = "ALLOW"
            calibrated_prob = round(min(prob * 0.25, 0.22), 4)
            uncertainty_note = "Degraded mic/channel bandwidth • Calibrated against DSP false-positives"
    else:  # WIDEBAND_CLEAN
        if prob >= 0.42:
            tier = "HOLD"
            calibrated_prob = round(max(prob, 0.86), 4)
            uncertainty_note = "Wideband harmonics intact • High-confidence synthetic signature"
        else:
            tier = "ALLOW"
            calibrated_prob = round(min(prob, 0.16), 4)
            uncertainty_note = "Wideband harmonics intact • Natural human vocal profile"

    return {
        "channel_quality": channel_quality,
        "calibrated_prob": calibrated_prob,
        "tier": tier,
        "uncertainty_note": uncertainty_note,
        "channel_metrics": {
            "high_band_ratio": round(float(high_band_ratio), 4),
            "snr_db": round(float(snr_db), 2),
        },
    }

