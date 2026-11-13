# verify_round3.py
# Automated End-to-End Verification for Round 3 (DHWANI-KAVACH)

import io
import re
import sys
import numpy as np
import requests
import soundfile as sf

BASE_URL = "http://127.0.0.1:8000"

REQUIRED_SCORE_KEYS = [
    "tier",
    "calibrated_prob",
    "raw_prob",
    "channel_status",
    "uncertainty_note",
    "transcript",
    "nlp_coercion",
    "session_id",
    "sha256_seal",
]

def test_health():
    print("\n--- 1. Testing /api/health ---")
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Health Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("ok") is True or data.get("status") == "ONLINE", "Backend not online"
    print("PASS: /api/health is HTTP 200 and ONLINE")

def test_reset():
    print("\n--- 2. Testing /api/reset ---")
    resp = requests.post(f"{BASE_URL}/api/reset", timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Reset Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("ok") is True, f"Expected ok: True, got {data}"
    print("PASS: /api/reset is HTTP 200 and reset session state")

def test_short_human_burst():
    print("\n--- 3. Testing Short 0.4s 'Hello' Burst (source_type='mic') ---")
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
    window = np.hanning(burst_len).astype(np.float32)
    start_idx = int(0.35 * sr)
    audio[start_idx : start_idx + burst_len] += burst * window

    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_io.getvalue()

    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    files = {"file": ("live_chunk.wav", wav_bytes, "audio/wav")}
    data = {"session_id": "test_mic_hello_01", "mode": "live", "source_type": "mic"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res, f"Missing required key '{k}' in /api/score response"

    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    seal = res.get("sha256_seal")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}', Seal: {seal[:16]}...")
    assert tier == "ALLOW", f"Expected tier ALLOW, got {tier}"
    assert prob <= 0.22, f"Expected prob <= 0.22, got {prob}"
    assert len(seal) == 64, f"Expected 64-char hex SHA-256 seal, got {len(seal)}"
    print("PASS: 0.4s 'Hello' burst on mic correctly returns ALLOW (<= 0.22) with all contract keys")

def test_scam_ai_sample():
    print("\n--- 4. Testing scam_ai_sample.wav (source_type='upload') ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    with open("scam_ai_sample.wav", "rb") as fh:
        files = {"file": ("scam_ai_sample.wav", fh.read(), "audio/wav")}
    data = {"mode": "upload", "source_type": "upload"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res, f"Missing required key '{k}' in /api/score response"

    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    seal = res.get("sha256_seal")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}', Seal: {seal[:16]}...")
    assert tier == "HOLD", f"Expected tier HOLD, got {tier}"
    assert prob >= 0.85, f"Expected prob >= 0.85, got {prob}"
    assert len(seal) == 64, f"Expected 64-char hex SHA-256 seal, got {len(seal)}"
    print("PASS: scam_ai_sample.wav correctly detects synthetic voice (HOLD, >= 0.85)")

def test_scam_neural_mp3():
    print("\n--- 5. Testing scam_neural_in.mp3 (source_type='upload') ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    with open("scam_neural_in.mp3", "rb") as fh:
        files = {"file": ("scam_neural_in.mp3", fh.read(), "audio/mpeg")}
    data = {"mode": "upload", "source_type": "upload"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res, f"Missing required key '{k}' in /api/score response"

    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    seal = res.get("sha256_seal")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}', Seal: {seal[:16]}...")
    assert tier == "HOLD", f"Expected tier HOLD, got {tier}"
    assert prob >= 0.85, f"Expected prob >= 0.85, got {prob}"
    assert len(seal) == 64, f"Expected 64-char hex SHA-256 seal, got {len(seal)}"
    print("PASS: scam_neural_in.mp3 correctly detects synthetic voice (HOLD, >= 0.85)")

def test_npci_freeze():
    print("\n--- 6. Testing /api/npci/freeze ---")
    payload = {"risk_tier": "HOLD", "session_id": "test_e2e_freeze"}
    resp = requests.post(f"{BASE_URL}/api/npci/freeze", json=payload, timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Freeze Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("success") is True, "Expected success: True"
    assert data.get("status") == "FROZEN", f"Expected status FROZEN, got {data.get('status')}"
    assert data.get("protected_amount_inr") == 50000, f"Expected 50000 protected amount, got {data.get('protected_amount_inr')}"
    assert "freeze" in data and "reference" in data["freeze"], "Missing freeze receipt reference"
    print("PASS: /api/npci/freeze engaged and returned HTTP 200 with ₹50,000 protected receipt")

def test_challenge_verify():
    print("\n--- 7. Testing /api/challenge/verify (Safe-Word Challenge) ---")
    # 1. Invalid safe-word -> verified: False, status: FROZEN
    bad_payload = {"safe_word": "WRONG_ATTEMPT"}
    resp_bad = requests.post(f"{BASE_URL}/api/challenge/verify", json=bad_payload, timeout=5)
    print(f"Bad Word Status Code: {resp_bad.status_code}")
    bad_data = resp_bad.json()
    print(f"Bad Word Response: {bad_data}")
    assert resp_bad.status_code == 200, f"Expected 200, got {resp_bad.status_code}"
    assert bad_data.get("verified") is False, "Expected verified: False for bad safe-word"
    assert bad_data.get("status") == "FROZEN", "Expected status: FROZEN"

    # 2. Valid safe-word "KAVACH-2026" -> verified: True, status: UNFROZEN
    good_payload = {"safe_word": "KAVACH-2026"}
    resp_good = requests.post(f"{BASE_URL}/api/challenge/verify", json=good_payload, timeout=5)
    print(f"Good Word ('KAVACH-2026') Status Code: {resp_good.status_code}")
    good_data = resp_good.json()
    print(f"Good Word Response: {good_data}")
    assert resp_good.status_code == 200, f"Expected 200, got {resp_good.status_code}"
    assert good_data.get("verified") is True, "Expected verified: True for 'KAVACH-2026'"
    assert good_data.get("status") == "UNFROZEN", "Expected status: UNFROZEN"

    # 3. Valid lowercase safe-word "safe-home" -> verified: True
    lower_payload = {"safe_word": "safe-home"}
    resp_lower = requests.post(f"{BASE_URL}/api/challenge/verify", json=lower_payload, timeout=5)
    assert resp_lower.json().get("verified") is True, "Expected case-insensitive match for 'safe-home'"

    print("PASS: /api/challenge/verify correctly rejects invalid words and unfreezes on valid safe-word")

def test_evidence_dossier():
    print("\n--- 8. Testing /api/evidence/dossier (Section 63 BSA 2023 Digital Evidence Seal) ---")
    payload = {
        "caller_number": "+91-98765-43210",
        "risk_tier": "HOLD",
        "radar_score": 92.5,
        "transcript": "This is CBI officer, transfer money immediately to avoid arrest warrant",
        "session_id": "sess_dossier_test_01",
        "channel_quality": "WIDEBAND_CLEAN"
    }
    resp = requests.post(f"{BASE_URL}/api/evidence/dossier", json=payload, timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Dossier Response: success={data.get('success')}, case_id={data.get('dossier', {}).get('case_id')}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("success") is True, "Expected success: True"
    assert "dossier" in data, "Expected 'dossier' object in response"
    dossier = data["dossier"]
    assert "case_id" in dossier and str(dossier["case_id"]).startswith("DK-"), "Invalid case_id"
    assert "record_sha256" in dossier and len(dossier["record_sha256"]) == 64, "Invalid record_sha256"
    seal = data.get("sha256_seal") or data.get("record_sha256")
    assert seal and len(seal) == 64, f"Invalid 64-char sha256_seal: {seal}"
    print(f"PASS: /api/evidence/dossier generated valid Case ID '{dossier['case_id']}' with SHA-256 seal: {seal}")

if __name__ == "__main__":
    try:
        test_health()
        test_reset()
        test_short_human_burst()
        test_scam_ai_sample()
        test_scam_neural_mp3()
        test_npci_freeze()
        test_challenge_verify()
        test_evidence_dossier()
        print("\n=======================================================")
        print("ALL 8 END-TO-END ROUND 3 INTEGRATION CHECKS PASSED 100%!")
        print("=======================================================")
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test run failed: {e}")
        sys.exit(1)
