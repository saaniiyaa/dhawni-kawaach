import os
import sys
import time
import subprocess
import numpy as np
import librosa
import soundfile as sf
import requests

API_URL = "http://127.0.0.1:8000/api/score"


def generate_test_clips():
    # 1. Realistic Indian-accent neural speech with prosody shifts
    cmd = [
        "edge-tts",
        "--voice", "en-IN-PrabhatNeural",
        "--rate=+6%",
        "--pitch=+3Hz",
        "--text",
        "Hello Papa, wait a second... can you hear me? My phone is dying, please send twenty-five thousand rupees on UPI right now.",
        "--write-media", "raw_ai_conv.mp3",
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # Fallback to python -m edge_tts if edge-tts not in PATH
        subprocess.run([sys.executable, "-m"] + cmd, check=True)

    y, sr = librosa.load("raw_ai_conv.mp3", sr=16000)

    # Clip A: Wideband clean AI with a 1.5s mid-sentence turn-taking pause
    mid = len(y) // 2
    pause = np.zeros(int(1.5 * sr), dtype=np.float32)
    y_paused = np.concatenate([y[:mid], pause, y[mid:]])
    sf.write("bench_1_ai_turn_taking_wideband.wav", y_paused, sr)

    # Clip B: 8kHz narrowband telephony codec + background line noise
    y_8k, _ = librosa.load("raw_ai_conv.mp3", sr=8000)
    rng = np.random.default_rng(42)
    noisy_8k = y_8k + rng.normal(0, 0.005, size=y_8k.shape).astype(np.float32)
    y_codec = librosa.resample(noisy_8k, orig_sr=8000, target_sr=16000)
    sf.write("bench_2_ai_8khz_degraded.wav", y_codec, 16000)


def run_benchmark(clip_paths):
    print(f"{'CLIP':<38} | {'LATENCY':<9} | {'CHANNEL':<18} | {'SCORE':<7} | {'TIER':<6} | NOTE")
    print("-" * 125)
    for path in clip_paths:
        if not os.path.exists(path):
            continue
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            resp = requests.post(API_URL, files={"file": (path, f, "audio/wav")}, timeout=15)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        data = resp.json()

        score = data.get("calibrated_prob", data.get("score", data.get("ai_prob", "N/A")))
        channel = data.get("channel_quality", "MISSING")
        tier = data.get("calibrated_tier", data.get("tier", data.get("action", "N/A")))
        note = data.get("uncertainty_note", "")
        print(f"{path:<38} | {ms:>6} ms | {channel:<18} | {str(score):<7} | {tier:<6} | {note}")


if __name__ == "__main__":
    generate_test_clips()
    clips = [
        "bench_1_ai_turn_taking_wideband.wav",
        "bench_2_ai_8khz_degraded.wav",
    ]
    # Add any real human recording in folder to verify 0% false HOLD
    for cand in ["results/demo/demo_real_studio.wav", "data/my_voice.wav"]:
        if os.path.exists(cand):
            clips.append(cand)
            break
    run_benchmark(clips)
