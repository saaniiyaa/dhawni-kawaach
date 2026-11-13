# verify_round3.py
# Automated End-to-End Verification for Round 3 (DHWANI-KAVACH)
# Tests: ONNX health, empty/corrupt safe handling, mic 'Hello', 4-chunk continuous mic,
# synthetic ElevenLabs-style forensic fusion, scam WAV/MP3, NPCI freeze, Safe-Word, and BSA Evidence Dossier.

import io
import math
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
    "forensics",
]

REQUIRED_FORENSICS_KEYS = [
    "pitch_jitter_pct",
    "digital_zero_ratio",
    "phase_coherence_score",
    "acoustic_drift_score",
    "forensic_ai_prob",
]


def test_health():
    print("\n--- 1. Testing /api/health (ONNX inference active) ---")
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    print(f"Health Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert data.get("ok") is True or data.get("status") == "ONLINE", "Backend not online"
    assert "inference_backend" in data, "Missing 'inference_backend' in /api/health"
    backend = data.get("inference_backend")
    print(f"Active inference backend: {backend}")
    assert backend in ("onnx", "pytorch"), f"Unexpected backend: {backend}"
    assert data.get("onnx_active") is True, f"Expected onnx_active: True, got {data.get('onnx_active')}"
    print("PASS: /api/health is HTTP 200, ONLINE, and ONNX execution provider active")


def test_empty_and_corrupt_bytes():
    print("\n--- 2. Testing Empty and Corrupt Bytes (HTTP 200 Safe Handling) ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    # 1. Empty bytes
    files_empty = {"file": ("empty.wav", b"", "audio/wav")}
    resp_empty = requests.post(f"{BASE_URL}/api/score", files=files_empty, data={"session_id": "test_empty"}, timeout=5)
    print(f"Empty bytes Status Code: {resp_empty.status_code}")
    assert resp_empty.status_code == 200, f"Expected 200 for empty bytes, got {resp_empty.status_code}"
    res_e = resp_empty.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res_e, f"Missing key '{k}' in empty bytes response"
    assert res_e["tier"] == "ALLOW", f"Expected tier ALLOW for empty bytes, got {res_e['tier']}"
    assert res_e["prob"] <= 0.20, f"Expected safe prob <= 0.20 for empty bytes, got {res_e['prob']}"

    # 2. Corrupt / invalid bytes
    files_corrupt = {"file": ("corrupt.bin", b"\x00\xff\xfe\x12\x34\x56\x78\x9a\xbc\xde", "application/octet-stream")}
    resp_corrupt = requests.post(f"{BASE_URL}/api/score", files=files_corrupt, data={"session_id": "test_corrupt"}, timeout=5)
    print(f"Corrupt bytes Status Code: {resp_corrupt.status_code}")
    assert resp_corrupt.status_code == 200, f"Expected 200 for corrupt bytes, got {resp_corrupt.status_code}"
    res_c = resp_corrupt.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res_c, f"Missing key '{k}' in corrupt bytes response"
    assert res_c["tier"] == "ALLOW", f"Expected tier ALLOW for corrupt bytes, got {res_c['tier']}"
    assert res_c["prob"] <= 0.20, f"Expected safe prob <= 0.20 for corrupt bytes, got {res_c['prob']}"

    print("PASS: Empty and corrupt bytes safely handled with HTTP 200 and standard contract")


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
    for fk in REQUIRED_FORENSICS_KEYS:
        assert fk in res["forensics"], f"Missing forensics key '{fk}'"

    prob = res.get("prob") or res.get("score")
    tier = res.get("tier")
    note = res.get("uncertainty_note")
    seal = res.get("sha256_seal")
    print(f"Score: {prob:.4f}, Tier: {tier}, Note: '{note}', Seal: {seal[:16]}...")
    assert tier == "ALLOW", f"Expected tier ALLOW, got {tier}"
    assert prob <= 0.22, f"Expected prob <= 0.22, got {prob}"
    assert len(seal) == 64, f"Expected 64-char hex SHA-256 seal, got {len(seal)}"
    print("PASS: 0.4s 'Hello' burst on mic correctly returns ALLOW (<= 0.22) with all contract keys")


def test_four_consecutive_mic_speech_chunks():
    print("\n--- 4. Testing 4 Consecutive Continuous Mic Speech Chunks (Zero False Positives) ---")
    sr = 16000
    dur = 1.5
    requests.post(f"{BASE_URL}/api/reset", timeout=5)
    sid = "sess_mic_continuous_4chunks"

    for chunk_idx in range(4):
        # Generate natural human voice chunk with pitch tremor (jitter > 0.8%) and natural mic noise floor
        audio = np.random.normal(0, 0.003, int(dur * sr)).astype(np.float32)
        t = np.arange(int(1.2 * sr)) / float(sr)
        f0_track = 135.0 + 3.0 * np.sin(2 * np.pi * 7 * t) + np.random.normal(0, 1.8, len(t))
        phase = 2 * np.pi * np.cumsum(f0_track) / float(sr)
        voice = (0.24 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)).astype(np.float32)
        audio[int(0.15 * sr) : int(0.15 * sr) + len(voice)] += voice

        wav_io = io.BytesIO()
        sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {"session_id": sid, "mode": "live", "source_type": "mic"}
        resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
        assert resp.status_code == 200
        res = resp.json()
        print(f"Chunk #{chunk_idx+1}: Tier={res['tier']}, Score={res['score']:.4f}, Forensics_AI_Prob={res['forensics']['forensic_ai_prob']:.4f}")
        assert res["tier"] == "ALLOW", f"Chunk #{chunk_idx+1} falsely triggered HOLD!"
        assert res["score"] <= 0.22, f"Chunk #{chunk_idx+1} score too high: {res['score']}"

    print("PASS: 4 consecutive continuous speech chunks on mic remained 100% ALLOW (zero false alarms)")


def test_synthetic_forensic_fusion_upload():
    print("\n--- 5. Testing Synthetic Smooth-F0 + Digital-Zero ElevenLabs-style Clip on Upload ---")
    requests.post(f"{BASE_URL}/api/reset", timeout=5)

    sr = 16000
    dur = 2.5
    total_samples = int(dur * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Synthetic tone with flat pitch (jitter = 0%) and digital zero gap
    seg_len = int(0.8 * sr)
    t = np.arange(seg_len) / float(sr)
    tone = (
        0.30 * np.sin(2 * np.pi * 150 * t) +
        0.20 * np.sin(2 * np.pi * 300 * t) +
        0.15 * np.sin(2 * np.pi * 450 * t) +
        0.10 * np.sin(2 * np.pi * 600 * t) +
        0.08 * np.sin(2 * np.pi * 1500 * t) +
        0.05 * np.sin(2 * np.pi * 3000 * t) +
        0.04 * np.sin(2 * np.pi * 5000 * t)
    ).astype(np.float32)

    audio[int(0.1 * sr) : int(0.1 * sr) + seg_len] = tone
    # Digital zero silence gap between 0.9s and 1.3s
    audio[int(1.3 * sr) : int(1.3 * sr) + seg_len] = tone

    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_io.getvalue()

    files = {"file": ("elevenlabs_synth.wav", wav_bytes, "audio/wav")}
    data = {"mode": "upload", "source_type": "upload", "session_id": "test_synth_fusion"}
    resp = requests.post(f"{BASE_URL}/api/score", files=files, data=data, timeout=30)
    print(f"Status Code: {resp.status_code}")
    res = resp.json()
    for k in REQUIRED_SCORE_KEYS:
        assert k in res, f"Missing required key '{k}'"

    forensic_prob = res["forensics"]["forensic_ai_prob"]
    print(f"Synth Clip: Tier={res['tier']}, Prob={res['prob']:.4f}, Forensic_AI_Prob={forensic_prob:.4f}")
    assert forensic_prob >= 0.68, f"Expected forensic_ai_prob >= 0.68, got {forensic_prob}"
    assert res["tier"] == "HOLD", f"Expected tier HOLD via forensic fusion, got {res['tier']}"
    assert res["prob"] >= 0.78, f"Expected prob >= 0.78, got {res['prob']}"
    print("PASS: Synthetic smooth-F0 + digital-zero clip correctly caught at HOLD (>= 0.78) via forensic fusion")


def test_scam_ai_sample():
    print("\n--- 6. Testing scam_ai_sample.wav (source_type='upload') ---")
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
    print("\n--- 7. Testing scam_neural_in.mp3 (source_type='upload') ---")
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
    print("\n--- 8. Testing /api/npci/freeze ---")
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
    print("\n--- 9. Testing /api/challenge/verify (Safe-Word Challenge) ---")
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
    print("\n--- 10. Testing /api/evidence/dossier (Section 63 BSA 2023 Digital Evidence Seal) ---")
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
        test_empty_and_corrupt_bytes()
        test_short_human_burst()
        test_four_consecutive_mic_speech_chunks()
        test_synthetic_forensic_fusion_upload()
        test_scam_ai_sample()
        test_scam_neural_mp3()
        test_npci_freeze()
        test_challenge_verify()
        test_evidence_dossier()
        print("\n========================================================")
        print("ALL 10 END-TO-END ROUND 3 INTEGRATION CHECKS PASSED 100%!")
        print("========================================================")
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test run failed: {e}")
        sys.exit(1)
