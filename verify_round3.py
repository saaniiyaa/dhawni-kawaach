# verify_round3.py
# Automated End-to-End Verification for Round 3 (DHWANI-KAVACH)

import io
import sys
import numpy as np
import requests
import soundfile as sf

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("\n--- 1. Testing /api/health ---")
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Health Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("ok") is True or data.get("status") == "ONLINE", "Backend not online"
    print("PASS: /api/health is HTTP 200 and ONLINE")

def test_short_human_burst():
    print("\n--- 2. Testing Short 0.4s 'Hello' Burst (source_type='mic') ---")
    sr = 16000
    dur = 1.5
    total_samples = int(dur * sr)
    audio = np.random.normal(0, 0.002, total_samples).astype(np.float32)

    # 0.4s voiced 'Hello' formant burst (F0 ~ 130 Hz + harmonics)
    burst_len = int(0.40 * sr)
    t = np.arange(burst_len) / float(sr)
    burst = (
        0.18 * np.sin(2 * np.pi * 130 * t) +
        0.10 * np.sin(2 * np.pi * 260 * t) +
        0.06 * np.sin(2 * np.pi * 390 * t) +
        0.03 * np.sin(2 * np.pi * 800 * t)
    ).astype(np.float32)
    # Window to avoid abrupt click
    window = np.hanning(burst_len).astype(np.float32)
    start_idx = int(0.35 * sr)
    audio[start_idx : start_idx + burst_len] += burst * window

    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_io.getvalue()

    # Reset session first
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    files = {"file": ("live_chunk.wav", wav_bytes, "audio/wav")}
    data = {"session_id": "test_mic_hello_01", "mode": "live", "source_type": "mic"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}'")
    assert tier == "ALLOW", f"Expected tier ALLOW, got {tier}"
    assert prob <= 0.22, f"Expected prob <= 0.22, got {prob}"
    print("PASS: 0.4s 'Hello' burst on mic correctly returns ALLOW (<= 0.22)")

def test_scam_ai_sample():
    print("\n--- 3. Testing scam_ai_sample.wav (source_type='upload') ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    with open("scam_ai_sample.wav", "rb") as fh:
        files = {"file": ("scam_ai_sample.wav", fh.read(), "audio/wav")}
    data = {"mode": "upload", "source_type": "upload"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}'")
    assert tier == "HOLD", f"Expected tier HOLD, got {tier}"
    assert prob >= 0.85, f"Expected prob >= 0.85, got {prob}"
    print("PASS: scam_ai_sample.wav correctly detects synthetic voice (HOLD, >= 0.85)")

def test_scam_neural_mp3():
    print("\n--- 4. Testing scam_neural_in.mp3 (source_type='upload') ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    with open("scam_neural_in.mp3", "rb") as fh:
        files = {"file": ("scam_neural_in.mp3", fh.read(), "audio/mpeg")}
    data = {"mode": "upload", "source_type": "upload"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}'")
    assert tier == "HOLD", f"Expected tier HOLD, got {tier}"
    assert prob >= 0.85, f"Expected prob >= 0.85, got {prob}"
    print("PASS: scam_neural_in.mp3 correctly detects synthetic voice (HOLD, >= 0.85)")

if __name__ == "__main__":
    try:
        test_health()
        test_short_human_burst()
        test_scam_ai_sample()
        test_scam_neural_mp3()
        print("\n==============================================")
        print("ALL 4 END-TO-END VERIFICATION CHECKS PASSED!")
        print("==============================================")
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test run failed: {e}")
        sys.exit(1)
