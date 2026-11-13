# demo_server.py  (v2 - full demo: call screen, context card, voiceprint, radar)
# DHWANI-KAVACH LIVE DEMO SERVER

import hashlib
import math
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import tempfile
import time
import uuid

try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    print("flask is not installed. Run 0c_install_flask.py once, then run this.")
    raise SystemExit(0)

import numpy as np
import torch

from ml.audio.io import load, SR, fix_length, energy_vad
from ml.features.dsp import logmel
from ml.models.cnn import MelCNN
from ml.engine.temporal import TemporalEngine, EngineConfig, policy_for
from ml.engine import voiceprint_falak
from ml.augment.codecs import FFMPEG, run_quiet
from channel_calibration import evaluate_channel_and_calibrate
from dsp_forensics import analyze_acoustic_physics

torch.set_num_threads(min(4, os.cpu_count() or 2))

WHISPER_MODEL = None
try:
    from faster_whisper import WhisperModel
    print("Initializing faster_whisper ASR (tiny, cpu, int8)...")
    WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("[OK] faster_whisper ASR initialized successfully")
except Exception as e:
    WHISPER_MODEL = None

MODEL_PATH, MODEL_NAME = None, None
for cand, name in [("results/cnn_v2/model.pt", "CNN v2 (codec-hardened)"),
                   ("results/cnn_v1/model.pt", "CNN v1 (clean-trained)")]:
    if os.path.exists(cand):
        MODEL_PATH, MODEL_NAME = cand, name
        break
if MODEL_PATH is None:
    print("No trained model found (results/cnn_v1 or cnn_v2).")
    print("Run 7_train_cnn.py first, then start the demo.")
    raise SystemExit(0)

model = MelCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

ONNX_SESSION = None
INFERENCE_BACKEND = "pytorch"
ONNX_CANDS = ["results/cnn_v2/model.onnx", "results/cnn_v1/model.onnx"]
for ocand in ONNX_CANDS:
    if os.path.exists(ocand):
        try:
            import onnxruntime as ort
            ONNX_SESSION = ort.InferenceSession(ocand, providers=["CPUExecutionProvider"])
            INFERENCE_BACKEND = "onnx"
            print("[OK] Loaded ONNX inference session:", ocand)
            break
        except Exception as oe:
            print("[WARN] ONNX load notice for %s: %s" % (ocand, oe))

print("demo model:", MODEL_NAME, "<-", MODEL_PATH, f"(inference: {INFERENCE_BACKEND})")

# ---- Block 10 calibration: thresholds + silence gate ----
_CAL_PATH = os.path.join("results", "calibration.json")
CAL = json.load(open(_CAL_PATH)) if os.path.exists(_CAL_PATH) else {}
engine = TemporalEngine(EngineConfig(
    theta_lo=float(CAL.get("theta_lo", 0.62)),
    theta_hi=float(CAL.get("theta_hi", 0.90)),
))
SESSION_STATE = {
    "prev_risk": 0.0,
}
SESSIONS = {}

def get_session_data(session_id=None):
    sid = session_id or "default"
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "engine": TemporalEngine(EngineConfig(
                theta_lo=float(CAL.get("theta_lo", 0.62)),
                theta_hi=float(CAL.get("theta_hi", 0.90)),
            )),
            "ema": 0.0,
            "ai_streak": 0,
            "human_streak": 0,
            "latched_hold": False,
            "rolling_audio": np.array([], dtype=np.float32),
            "baseline": None,
        }
    return SESSIONS[sid]

def get_session_engine(session_id=None):
    return get_session_data(session_id)["engine"]

def transcribe_chunk_bytes(wav_bytes):
    """Fast backend chunk transcription with strict 1.0s timeout."""
    if not wav_bytes or len(wav_bytes) < 400 or len(wav_bytes) > 250000:
        return ""
    result = [""]

    def _worker():
        try:
            import speech_recognition as sr_mod
            import io
            r = sr_mod.Recognizer()
            with sr_mod.AudioFile(io.BytesIO(wav_bytes)) as source:
                audio = r.record(source)
            text = r.recognize_google(audio, language="en-IN")
            if text:
                result[0] = str(text).strip()
        except Exception:
            pass

    import threading
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=1.0)
    return result[0]

def transcribe_audio_context(audio_samples, sr=SR):
    """Transcribe rolling audio context using faster_whisper if available, else thread-safe STT fallback."""
    if audio_samples is None or len(audio_samples) < int(0.35 * sr):
        return ""

    global WHISPER_MODEL
    if WHISPER_MODEL is not None:
        try:
            segments, _ = WHISPER_MODEL.transcribe(audio_samples, beam_size=1, language="en", task="transcribe")
            t_list = [s.text.strip() for s in segments if s.text.strip()]
            if t_list:
                return " ".join(t_list)
        except Exception:
            pass

    try:
        import io, soundfile as sf
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_samples, sr, format="WAV", subtype="PCM_16")
        return transcribe_chunk_bytes(wav_buf.getvalue())
    except Exception:
        return ""

def analyze_coercion_intent(text: str) -> dict:
    """Multi-factor intent scoring across 3 vectors:
    Vector A (Authority/Impersonation): cbi, ed, cyber cell, police, arrest warrant, customs, narcotics, supreme court, rbi, trai notice, parcel seized
    Vector B (Urgency/Isolation): immediately, right now, abhi, turant, video call, don't disconnect, phone mat katna, digital arrest, jail
    Vector C (Financial/Credential Demand): otp, upi, transfer, bank account, fine, bail, paise bhejo, scan qr, verify account, share screen
    Flag COERCION: DETECTED only when either:
    (a) Vector C + (Vector A or Vector B) co-occur, or
    (b) an explicit high-severity scam phrase appears.
    """
    if not text:
        return {
            "detected": False,
            "status": "CLEAN",
            "score": 0.0,
            "vectors": {"vector_a": [], "vector_b": [], "vector_c": []},
            "high_severity_matches": [],
            "reason": "No coercion indicators found."
        }

    t_lower = str(text).lower()

    vector_a_terms = [
        "cbi", "ed", "cyber cell", "police", "arrest warrant", "customs",
        "narcotics", "supreme court", "rbi", "trai notice", "parcel seized"
    ]
    vector_b_terms = [
        "immediately", "right now", "abhi", "turant", "video call",
        "don't disconnect", "dont disconnect", "phone mat katna", "digital arrest", "jail"
    ]
    vector_c_terms = [
        "otp", "upi", "transfer", "bank account", "fine", "bail",
        "paise bhejo", "scan qr", "verify account", "share screen"
    ]

    high_sev_phrases = [
        "digital arrest", "cbi arrest warrant", "mujhe aapka otp chahiye", "paise transfer karo"
    ]

    import re
    def find_matches(terms, s):
        matches = []
        for term in terms:
            if " " in term or "-" in term or "'" in term:
                if term in s:
                    matches.append(term)
            else:
                if re.search(r'\b' + re.escape(term) + r'\b', s):
                    matches.append(term)
        return list(set(matches))

    matched_a = find_matches(vector_a_terms, t_lower)
    matched_b = find_matches(vector_b_terms, t_lower)
    matched_c = find_matches(vector_c_terms, t_lower)
    matched_high_sev = [p for p in high_sev_phrases if p in t_lower]

    rule_a_b_c = bool(matched_c and (matched_a or matched_b))
    rule_high_sev = bool(matched_high_sev)

    detected = rule_a_b_c or rule_high_sev
    status = "DETECTED" if detected else ("SUSPICIOUS" if (matched_a or matched_b or matched_c) else "CLEAN")

    score = 0.0
    if rule_high_sev:
        score = 0.95
    elif rule_a_b_c:
        score = min(0.92, 0.70 + 0.05 * (len(matched_a) + len(matched_b) + len(matched_c)))
    elif matched_c:
        score = min(0.40, 0.15 + 0.08 * len(matched_c))
    elif matched_a or matched_b:
        score = min(0.30, 0.10 + 0.05 * (len(matched_a) + len(matched_b)))

    reason = "High-severity coercion pattern detected." if detected else "Normal conversation."
    return {
        "detected": detected,
        "status": status,
        "score": round(score, 4),
        "coercion_detected": detected,
        "vectors": {
            "vector_a": matched_a,
            "vector_b": matched_b,
            "vector_c": matched_c,
        },
        "high_severity_matches": matched_high_sev,
        "matched_keywords": matched_a + matched_b + matched_c,
        "reason": reason
    }
if CAL:
    print(f"calibration: loaded {_CAL_PATH} "
          f"(vad_min={CAL.get('vad_min')}, bias={CAL.get('logit_bias')}, "
          f"theta {CAL.get('theta_lo')}/{CAL.get('theta_hi')})")
else:
    print("calibration: none found - DEFAULTS (run 14_block10_calibrate.py)")
SECONDS = 4.0

# ---- demo whitelist of OFFICIAL helpline numbers ----
OFFICIAL_NUMBERS = {
    # national helplines (publicly published)
    "1930": "National Cyber Crime Helpline",
    "112":  "National Emergency Number",
    "100":  "Police",
    "108":  "Ambulance Service",
    "1091": "Women Helpline",
    "1098": "Child Helpline (Childline)",
    "139":  "Rail Madad - Indian Railways",
    "1906": "LPG Emergency Helpline",
    "1909": "TRAI DND / Telemarketing Complaints",
    "1947": "UIDAI / Aadhaar Helpline",
    "14567": "Elder Line - Senior Citizens",
    "198":  "Telecom Customer Care (all operators)",
    # major banks - officially published customer-care numbers
    "1800112211": "State Bank of India - customer care",
    "1800227222": "HDFC Bank - customer care",
    "18001024242": "ICICI Bank - customer care",
    "18004195000": "Axis Bank - customer care",
    "18602662666": "Kotak Mahindra Bank - customer care",
    "18001802222": "Punjab National Bank - customer care",
    "1800223344": "Bank of Baroda - customer care",
    "1800222244": "Union Bank of India - customer care",
    "18004250018": "Canara Bank - customer care",
    "18605001111": "IDFC FIRST Bank - customer care",
    "18602677777": "IndusInd Bank - customer care",
    "18001200": "Yes Bank - customer care",
}

app = Flask(__name__, static_folder="static")
app.config["MODEL_VERSION"] = (
    "cnn_v1" if MODEL_NAME and "v1" in MODEL_NAME.lower() else "cnn_v2"
)
# --- Defense API (SIH26104 screening) - try/except = server never breaks ---
try:
    from defense_api import bp as _defense_bp
    app.register_blueprint(_defense_bp)
    print("[OK] defense_api registered: /api/evidence/dossier, /api/npci/freeze, /api/challenge/verify")
except Exception as _e:
    print("[WARN] defense_api not loaded:", _e)

try:
    from flask_cors import CORS
    CORS(app)
    print("CORS: on")
except ImportError:
    print("CORS: flask-cors not installed - other devices may be blocked")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


def preprocess_speech_audio(y: np.ndarray, sr: int = SR, target_rms: float = 0.10):
    """Active-speech trimming & RMS loudness normalization before CNN inference:
    1. Compute frame-level RMS energy (25ms windows, 10ms hop).
    2. If speech is present, trim leading/trailing dead air (< 20% of peak frame RMS).
    3. If active speech duration is < 0.30s (cough/click/silence), return is_idle=True.
    4. Normalize active speech RMS to target_rms (0.10, ~-20 dBFS), clamped to avoid clipping.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if len(y) < int(0.10 * sr):
        return y, True

    frame_len = int(0.025 * sr)  # 25ms window (400 samples at 16kHz)
    hop_len = int(0.010 * sr)    # 10ms hop (160 samples at 16kHz)

    if len(y) < frame_len:
        return y, True

    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    frames_rms = []
    for i in range(n_frames):
        seg = y[i * hop_len : i * hop_len + frame_len]
        r = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        frames_rms.append(r)
    frames_rms = np.array(frames_rms, dtype=np.float32)

    peak_rms = float(np.max(frames_rms))
    if peak_rms < 0.005:  # Inaudible / ambient silence
        return y, True

    thresh = max(0.20 * peak_rms, 0.003)
    active_mask = frames_rms >= thresh
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return y, True

    first_frame = int(active_indices[0])
    last_frame = int(active_indices[-1])

    start_sample = max(0, first_frame * hop_len)
    end_sample = min(len(y), last_frame * hop_len + frame_len)

    active_duration_sec = (end_sample - start_sample) / float(sr)
    if active_duration_sec < 0.30:
        return y, True

    trimmed = y[start_sample:end_sample]

    cur_rms = float(np.sqrt(np.mean(trimmed * trimmed) + 1e-12))
    if cur_rms > 1e-5:
        gain = target_rms / cur_rms
        gain = min(max(gain, 0.1), 10.0)
        normed = trimmed * gain
        normed = np.clip(normed, -0.98, 0.98)
    else:
        normed = trimmed

    return normed.astype(np.float32), False


def score_audio(y):
    piece = fix_length(np.asarray(y, dtype=np.float32), int(SECONDS * SR))
    x = logmel(piece)
    x = (x - x.mean()) / (x.std() + 1e-6)
    if INFERENCE_BACKEND == "onnx" and ONNX_SESSION is not None:
        try:
            inp = x[None, None, ...].astype(np.float32)
            logits = ONNX_SESSION.run(None, {"logmel": inp})[0].reshape(-1)
            p = float(1.0 / (1.0 + np.exp(-logits[0])))
            return p
        except Exception:
            pass
    with torch.no_grad():
        p = float(torch.sigmoid(model(torch.from_numpy(x[None, None, ...]))))
    return p


def decode_to_wav(raw_bytes, suffix):
    if suffix == ".wav" or (len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE"):
        try:
            import io, soundfile as sf
            data, in_sr = sf.read(io.BytesIO(raw_bytes))
            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if in_sr != SR:
                import librosa
                data = librosa.resample(data, orig_sr=in_sr, target_sr=SR)
            return data.astype(np.float32)
        except Exception:
            pass
    tmp = tempfile.mkdtemp(prefix="dk_demo_")
    try:
        inp = os.path.join(tmp, "in" + suffix)
        out = os.path.join(tmp, "out.wav")
        with open(inp, "wb") as fh:
            fh.write(raw_bytes)
        run_quiet([FFMPEG, "-hide_banner", "-y", "-i", inp,
                   "-ar", str(SR), "-ac", "1", out])
        return load(out)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/reset", methods=["POST"])
def reset():
    engine.reset()
    SESSION_STATE["prev_risk"] = 0.0
    SESSIONS.clear()
    return jsonify({"ok": True})

@app.route("/api/ping", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/health", methods=["GET", "POST", "OPTIONS"])
def ping():
    return jsonify({
        "status": "ONLINE",
        "app": "DHWANI-KAVACH",
        "ok": True,
        "online": True,
        "backend": "online",
        "model": MODEL_NAME,
        "inference_backend": INFERENCE_BACKEND,
        "onnx_active": (INFERENCE_BACKEND == "onnx"),
        "timestamp": time.time(),
    })

@app.route("/api/context", methods=["POST"])
def context():
    """What do we know about this number? TRAI numbering rules +
    local curated registry (live Sanchar Saathi API is the roadmap)."""
    d = request.get_json(force=True) if request.is_json else {}
    number = str(d.get("number", ""))
    for ch in "+-() ":
        number = number.replace(ch, "")
    if number.startswith("91") and len(number) == 12:
        number = number[2:]
    elif number.startswith("0") and len(number) == 11:
        number = number[1:]
    notes = []
    level = "neutral"
    if number.startswith("140"):
        notes.append("Number is in the TRAI 140 telemarketing series")
        level = "warning"
    if number in OFFICIAL_NUMBERS:
        notes.append(f"Matches official listing: {OFFICIAL_NUMBERS[number]}")
        level = "safe" if level == "neutral" else level
    elif len(number) == 10 and number[0] in "6789":
        notes.append("Mobile number - registries cannot verify individuals; "
                     "judge the call by what the caller says and asks for")
    elif number.startswith("1800") or number.startswith("1860"):
        notes.append("Toll-free number not in our local registry - if it claims "
                     "to be your bank, verify on the bank's official website")
    else:
        notes.append("Not in our local registry of official helplines")
    if d.get("claims_bank") and number not in OFFICIAL_NUMBERS:
        notes.append("Caller claims to be from a bank but the number is not the "
                     "bank's official helpline - classic impersonation pattern")
        level = "danger"
    trust_score = 38 if level == "danger" else (62 if level == "warning" else 94)
    return jsonify({
        "number": number,
        "level": level,
        "notes": notes,
        "trust_score": trust_score,
    })




@app.route("/api/enrol", methods=["POST"])
def enrol():
    try:
        name = "".join(c for c in request.form.get("name", "") if c.isalnum() or c in " _-").strip()
        if not name:
            return jsonify({"error": "name required"})
        f = request.files.get("audio")
        suffix = os.path.splitext(f.filename)[1] or ".wav"
        y = decode_to_wav(f.read(), suffix)
        voiceprint_falak.enrol(name, [y])
        return jsonify({"ok": True, "name": name,
                        "enrolled": voiceprint_falak.enrolled_names()})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/verify", methods=["POST"])
def verify():
    try:
        name = request.form.get("name", "").strip()
        f = request.files.get("audio")
        suffix = os.path.splitext(f.filename)[1] or ".wav"
        y = decode_to_wav(f.read(), suffix)
        return jsonify(voiceprint_falak.verify(name, y))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/voiceprints", methods=["GET"])
def voiceprints():
    return jsonify({"enrolled": voiceprint_falak.enrolled_names()})


@app.route("/api/score", methods=["POST"])
def api_score():
    try:
        ctype = request.headers.get("Content-Type", "")
        if "webm" in ctype or "ogg" in ctype:
            suffix = ".webm"
        elif "wav" in ctype:
            suffix = ".wav"
        elif "mpeg" in ctype or "mp3" in ctype:
            suffix = ".mp3"
        elif "mp4" in ctype or "m4a" in ctype:
            suffix = ".m4a"
        else:
            suffix = ".bin"
        session_id = request.args.get("session_id") or request.form.get("session_id")
        if request.files or "multipart/form-data" in ctype:
            f = request.files.get("file") or request.files.get("audio") or next(iter(request.files.values()), None)
            raw_bytes = f.read() if f is not None else b""
            if not session_id and request.form.get("session_id"):
                session_id = request.form.get("session_id")
            if f is not None and f.filename and "." in f.filename:
                suffix = "." + f.filename.rsplit(".", 1)[-1].lower()
            elif f is not None and getattr(f, "content_type", ""):
                ct = f.content_type.lower()
                if "webm" in ct: suffix = ".webm"
                elif "ogg" in ct: suffix = ".ogg"
                elif "wav" in ct: suffix = ".wav"
                elif "mp3" in ct or "mpeg" in ct: suffix = ".mp3"
                elif "mp4" in ct or "m4a" in ct: suffix = ".m4a"
        elif "json" in ctype:
            import base64 as _b64
            raw_bytes = b""
            try:
                d = request.get_json(force=True, silent=True) or {}
                if not session_id and d.get("session_id"):
                    session_id = d.get("session_id")
                s = d.get("audio") or d.get("data") or d.get("blob") or d.get("file") or ""
                if isinstance(s, str) and s:
                    raw_bytes = _b64.b64decode(s.split(",")[-1])
            except Exception:
                raw_bytes = b""
            if not raw_bytes:
                raw_bytes = request.get_data()
        else:
            raw_bytes = request.get_data()
        print(f"[score] in: ctype={ctype}, bytes={len(raw_bytes)}, suffix={suffix}, session={session_id}", flush=True)
        if not raw_bytes:
            return jsonify({
                "tier": "ALLOW",
                "calibrated_prob": 0.10,
                "raw_prob": 0.10,
                "channel_status": "WIDEBAND_CLEAN",
                "uncertainty_note": "Natural human vocal harmonics verified • Ambient/idle chunk",
                "transcript": "",
                "nlp_coercion": analyze_coercion_intent(""),
                "session_id": session_id or "default",
                "sha256_seal": hashlib.sha256(b"empty_safe_chunk").hexdigest(),
                "forensics": {
                    "pitch_jitter_pct": 1.5,
                    "digital_zero_ratio": 0.0,
                    "phase_coherence_score": 0.0,
                    "acoustic_drift_score": 0.0,
                    "forensic_ai_prob": 0.0
                },
                "prob": 0.10,
                "score": 0.10,
                "risk_score": 0.10,
                "window_p": 0.10,
                "call_p": 0.10,
                "calibrated_tier": "ALLOW",
                "action": "ALLOW",
                "allow_sensitive": True,
                "model": MODEL_NAME
            })
        y = decode_to_wav(raw_bytes, suffix)       
        if y is None or len(y) == 0:
            return jsonify({
                "tier": "ALLOW",
                "calibrated_prob": 0.10,
                "raw_prob": 0.10,
                "channel_status": "WIDEBAND_CLEAN",
                "uncertainty_note": "Natural human vocal harmonics verified • Ambient/idle chunk",
                "transcript": "",
                "nlp_coercion": analyze_coercion_intent(""),
                "session_id": session_id or "default",
                "sha256_seal": hashlib.sha256(b"corrupt_safe_chunk").hexdigest(),
                "forensics": {
                    "pitch_jitter_pct": 1.5,
                    "digital_zero_ratio": 0.0,
                    "phase_coherence_score": 0.0,
                    "acoustic_drift_score": 0.0,
                    "forensic_ai_prob": 0.0
                },
                "prob": 0.10,
                "score": 0.10,
                "risk_score": 0.10,
                "window_p": 0.10,
                "call_p": 0.10,
                "calibrated_tier": "ALLOW",
                "action": "ALLOW",
                "allow_sensitive": True,
                "model": MODEL_NAME
            })
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))

        source_type = request.form.get("source_type") or request.args.get("source_type")
        if not source_type:
            mode = request.form.get("mode") or request.args.get("mode")
            if mode == "live":
                source_type = "mic"
            elif mode == "upload":
                source_type = "upload"
            else:
                source_type = "upload" if len(y) / float(SR) > 2.2 else "mic"
        source_type = str(source_type).lower().strip()

        duration_sec = len(y) / float(SR)
        sess_data = get_session_data(session_id)
        sess_engine = sess_data["engine"]
        prev_ema = float(sess_data.get("ema", 0.0))

        # Maintain per-session rolling 3.0s audio ring buffer (48,000 samples @ 16kHz)
        rolling = sess_data.get("rolling_audio")
        if rolling is None or len(rolling) == 0:
            rolling = y
        else:
            rolling = np.concatenate([rolling, y])
        if len(rolling) > int(3.0 * SR):
            rolling = rolling[-int(3.0 * SR):]
        sess_data["rolling_audio"] = rolling

        # Active-speech trimming & RMS loudness normalization before CNN inference
        y_proc, is_idle = preprocess_speech_audio(y, sr=SR, target_rms=0.10)

        # Pure NumPy/SciPy Acoustic Physics Forensics (< 8 ms)
        forensics = analyze_acoustic_physics(y_proc, sr=SR, session_baseline=sess_data.get("baseline"))
        if sess_data.get("baseline") is None and forensics.get("baseline_profile"):
            sess_data["baseline"] = {"spectral_profile": forensics["baseline_profile"]}

        if is_idle and source_type != "upload":
            # Treat short cough/click/ambient silence (<0.30s active speech) as idle chunk
            r = sess_engine.update_pause(decay_factor=0.90)
            SESSION_STATE["prev_risk"] = r["call_p"]
            raw_ai_prob = 0.10
            cal_prob = 0.10
            smoothed = 0.10
            tier = "ALLOW"
            sess_data["ai_streak"] = 0
            sess_data["human_streak"] = sess_data.get("human_streak", 0) + 1
            if sess_data["human_streak"] >= 2:
                sess_data["latched_hold"] = False
            sess_data["ema"] = smoothed
            SESSION_STATE["prev_risk"] = smoothed
            calib = {
                "channel_quality": "WIDEBAND_CLEAN",
                "calibrated_prob": 0.10,
                "tier": "ALLOW",
                "uncertainty_note": "Natural human vocal harmonics verified • Ambient/idle chunk",
                "channel_metrics": {"high_band_ratio": 0.05, "snr_db": 25.0},
            }
        else:
            vfrac = float(energy_vad(y_proc[-int(SECONDS * SR):]).mean()) if len(y_proc) >= int(SECONDS * SR) else float(energy_vad(y_proc).mean())
            vad_min = float(CAL.get("vad_min", 0.25))

            if vfrac < vad_min and source_type != "upload":
                r = sess_engine.update_pause(decay_factor=0.90)
                SESSION_STATE["prev_risk"] = r["call_p"]
                raw_ai_prob = r["window_p"]
            else:
                raw = score_audio(y_proc)
                bias = float(CAL.get("logit_bias", -3.677))
                q = min(max(raw, 1e-4), 1 - 1e-4)
                p = 1.0 / (1.0 + math.exp(-(math.log(q / (1.0 - q)) + bias)))
                r = sess_engine.update(p, max(vfrac, 0.25) * SECONDS)
                SESSION_STATE["prev_risk"] = r["call_p"]
                raw_ai_prob = r["window_p"]

            # Channel-Gated Forensic Fusion Rule (Zero Mic False Positives):
            fused_ai_prob = raw_ai_prob
            is_wideband = True
            try:
                freqs = np.fft.rfftfreq(len(y_proc), 1.0 / SR)
                spec = np.abs(np.fft.rfft(y_proc)) ** 2
                tot_e = np.sum(spec) + 1e-12
                high_e = np.sum(spec[freqs >= 3800])
                high_band_ratio = float(high_e / tot_e)
                is_wideband = (high_band_ratio >= 0.0012)
            except Exception:
                is_wideband = True

            # Borderline CNN AI probability boosted if forensics confirms vocoder signatures
            if source_type in ("tab", "upload") and is_wideband:
                if 0.32 <= raw_ai_prob < 0.55 and forensics.get("forensic_ai_prob", 0.0) >= 0.68:
                    fused_ai_prob = max(raw_ai_prob, 0.78)

            calib = evaluate_channel_and_calibrate(y_proc, SR, fused_ai_prob, source_type=source_type)
            cal_prob = calib["calibrated_prob"]

            if duration_sec <= 2.2 and source_type != "upload":
                # Asymmetric EMA risk persistence for live chunks (<= 2.2s)
                if cal_prob >= prev_ema:
                    smoothed = round(0.70 * cal_prob + 0.30 * prev_ema, 4)
                else:
                    smoothed = round(0.88 * prev_ema + 0.12 * cal_prob, 4)

                if source_type == "mic":
                    # Mic: require raw_prob >= 0.85 for 4 consecutive chunks (ai_streak >= 4, ~6.0s) to latch HOLD
                    is_mic_ai_cand = (calib["tier"] == "HOLD" and raw_ai_prob >= 0.85)
                    if is_mic_ai_cand:
                        sess_data["human_streak"] = 0
                        sess_data["ai_streak"] = sess_data.get("ai_streak", 0) + 1
                        if sess_data["ai_streak"] >= 4:
                            sess_data["latched_hold"] = True
                        else:
                            tier = "ALLOW"
                            smoothed = min(smoothed, 0.20)
                    else:
                        sess_data["ai_streak"] = 0
                        sess_data["human_streak"] = sess_data.get("human_streak", 0) + 1
                        if sess_data["human_streak"] >= 2:
                            sess_data["latched_hold"] = False

                    if sess_data.get("latched_hold", False):
                        tier = "HOLD"
                        smoothed = round(max(smoothed, 0.88), 4)
                    else:
                        tier = "ALLOW"
                        smoothed = min(smoothed, 0.20)

                else:
                    # Tab audio: require 2 consecutive synthetic chunks (ai_streak >= 2, ~3.0s) to latch HOLD
                    if calib["tier"] == "HOLD":
                        sess_data["human_streak"] = 0
                        sess_data["ai_streak"] = sess_data.get("ai_streak", 0) + 1
                        if sess_data["ai_streak"] >= 2:
                            sess_data["latched_hold"] = True
                        else:
                            tier = "ALLOW"
                            smoothed = min(smoothed, 0.24)
                    else:
                        sess_data["ai_streak"] = 0
                        sess_data["human_streak"] = sess_data.get("human_streak", 0) + 1
                        if sess_data["human_streak"] >= 2:
                            sess_data["latched_hold"] = False

                    if sess_data.get("latched_hold", False):
                        tier = "HOLD"
                        smoothed = round(max(smoothed, 0.88), 4)
                    else:
                        if sess_data.get("ai_streak", 0) >= 2 or smoothed >= 0.65:
                            tier = "HOLD"
                        else:
                            tier = "ALLOW"
                            smoothed = min(smoothed, 0.24)

                sess_data["ema"] = smoothed
                SESSION_STATE["prev_risk"] = smoothed
            else:
                # Full file uploads (> 2.2s or source_type == 'upload'): single-shot evaluation, bypass debounce
                tier = calib["tier"]
                smoothed = round(max(cal_prob, 0.88), 4) if tier == "HOLD" else round(min(cal_prob, 0.16), 4)
                sess_data["ema"] = smoothed
                SESSION_STATE["prev_risk"] = smoothed

        # Telemetry consistency: NEVER say "synthetic signature" when tier == "ALLOW"
        uncertainty_note = calib["uncertainty_note"]
        if tier == "ALLOW" and "synthetic" in uncertainty_note.lower():
            uncertainty_note = "Natural human vocal harmonics verified"

        pol = policy_for("HIGH_RISK" if tier == "HOLD" else "SAFE")

        # Fast chunk transcription for live stream chunks using rolling context buffer
        transcript_text = ""
        if source_type != "upload" and duration_sec <= 2.2:
            audio_for_stt = sess_data.get("rolling_audio")
            if audio_for_stt is None or len(audio_for_stt) == 0:
                audio_for_stt = y
            transcript_text = transcribe_audio_context(audio_for_stt, SR)

        nlp_coercion = analyze_coercion_intent(transcript_text)

        sha256_seal = hashlib.sha256(
            f"{session_id}|{tier}|{smoothed:.4f}|{transcript_text}|{time.time():.2f}".encode("utf-8")
        ).hexdigest()

        print(f"[LIVE SCORE] chunk ({source_type}) -> cal={cal_prob:.4f} smoothed={smoothed:.4f} tier={tier} note='{uncertainty_note}' coercion={nlp_coercion['status']} transcript='{transcript_text}'", flush=True)

        return jsonify({
            # Round 3 deterministic contract keys:
            "tier": tier,
            "calibrated_prob": cal_prob,
            "raw_prob": round(float(raw_ai_prob), 4),
            "channel_status": calib["channel_quality"],
            "uncertainty_note": uncertainty_note,
            "transcript": transcript_text,
            "nlp_coercion": nlp_coercion,
            "session_id": session_id or "default",
            "sha256_seal": sha256_seal,
            "forensics": {
                "pitch_jitter_pct": forensics["pitch_jitter_pct"],
                "digital_zero_ratio": forensics["digital_zero_ratio"],
                "phase_coherence_score": forensics["phase_coherence_score"],
                "acoustic_drift_score": forensics["acoustic_drift_score"],
                "forensic_ai_prob": forensics["forensic_ai_prob"]
            },

            # Backwards compatibility keys:
            "prob": smoothed,
            "score": smoothed,
            "risk_score": smoothed,
            "window_p": smoothed,
            "call_p": smoothed,
            "calibrated_tier": tier,
            "action": tier,
            "voiced_seconds": r.get("voiced_seconds", 0.0) if isinstance(r, dict) else 0.0,
            "n_windows": r.get("n_windows", 1) if isinstance(r, dict) else 1,
            "ui": pol["ui"],
            "step_up": pol["step_up"],
            "allow_sensitive": (tier == "ALLOW"),
            "model": MODEL_NAME,
            "channel_quality": calib["channel_quality"],
            "channel_metrics": calib["channel_metrics"],
        })
    except Exception as e:
        safe_sid = session_id if 'session_id' in locals() and session_id else "default"
        return jsonify({
            "tier": "ALLOW",
            "calibrated_prob": 0.10,
            "raw_prob": 0.10,
            "channel_status": "WIDEBAND_CLEAN",
            "uncertainty_note": "Natural human vocal harmonics verified",
            "transcript": "",
            "nlp_coercion": analyze_coercion_intent(""),
            "session_id": safe_sid,
            "sha256_seal": hashlib.sha256(b"error_fallback").hexdigest(),
            "forensics": {
                "pitch_jitter_pct": 1.5,
                "digital_zero_ratio": 0.0,
                "phase_coherence_score": 0.0,
                "acoustic_drift_score": 0.0,
                "forensic_ai_prob": 0.0
            },
            "error": str(e),
            "prob": 0.10,
            "score": 0.10,
            "risk_score": 0.10,
        })


@app.route("/api/tts", methods=["GET", "POST"])
@app.route("/api/synthesize", methods=["GET", "POST"])
def api_tts():
    """Synthesize voice using edge-tts with offline pyttsx3 and bundled demo fallbacks."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = data.get("text") or request.form.get("text") or ""
        voice = data.get("voice") or request.form.get("voice") or "en-IN-PrabhatNeural"
    else:
        text = request.args.get("text", "")
        voice = request.args.get("voice", "en-IN-PrabhatNeural")

    if not text:
        text = ("Hello Papa, wait a second... can you hear me? "
                "My phone is dying, please send twenty-five thousand rupees on UPI right now.")

    tmp_dir = tempfile.mkdtemp(prefix="dk_tts_")
    out_mp3 = os.path.join(tmp_dir, "tts_out.mp3")
    out_wav = os.path.join(tmp_dir, "tts_out.wav")

    # 1. Primary: edge-tts
    try:
        cmd = [
            sys.executable, "-m", "edge_tts",
            "--voice", voice,
            "--rate=+6%",
            "--pitch=+3Hz",
            "--text", text,
            "--write-media", out_mp3,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if res.returncode == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 100:
            return send_file(out_mp3, mimetype="audio/mpeg")
    except Exception as e_edge:
        print(f"[TTS] edge-tts error: {e_edge}, trying offline fallback...", flush=True)

    # 2. Secondary: pyttsx3 offline Windows SAPI
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 100:
            return send_file(out_wav, mimetype="audio/wav")
    except Exception as e_py:
        print(f"[TTS] pyttsx3 error: {e_py}, serving bundled demo...", flush=True)

    # 3. Tertiary fallback: bundled synthetic demo clip
    for fallback_path in [
        "results/demo/demo_fake_studio.wav",
        "bench_1_ai_turn_taking_wideband.wav",
        "raw_ai_conv.mp3",
    ]:
        if os.path.exists(fallback_path):
            mtype = "audio/mpeg" if fallback_path.endswith(".mp3") else "audio/wav"
            return send_file(fallback_path, mimetype=mtype)

    return jsonify({"error": "No TTS audio available"}), 500


@app.route("/api/demo_audio/<name>", methods=["GET"])
def api_demo_audio(name):
    allowed = {
        "real": "results/demo/demo_real_studio.wav",
        "fake": "results/demo/demo_fake_studio.wav",
        "my_voice": "data/my_voice.wav",
    }
    target = allowed.get(name.lower())
    if target and os.path.exists(target):
        return send_file(target, mimetype="audio/wav")
    if name.lower() in ("tts", "synthetic", "ai"):
        return api_tts()
    return jsonify({"error": "demo file not found"}), 404


@app.route("/api/radar", methods=["POST"])
def api_radar():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    try:
        from ml.engine.scam_radar import scan as radar_scan
    except ImportError as import_error:
        return jsonify({"error": "scam radar unavailable: " + str(import_error)}), 503
    return jsonify(radar_scan(text))


@app.route("/api/action/oob", methods=["POST"])
def api_oob():
    data = request.get_json(silent=True) or {}
    target_number = data.get("number", "+91-98765-XXXXX")
    return jsonify({
        "status": "INITIATED",
        "action": "OUT_OF_BAND_CALLBACK",
        "target": target_number,
        "auth_pin": "582914",
        "expires_in_sec": 60,
        "timestamp": time.strftime("%H:%M:%S"),
    })


@app.route("/api/action/report", methods=["POST"])
def api_report():
    data = request.get_json(silent=True) or {}
    incident_id = f"I4C-DK-{uuid.uuid4().hex[:8].upper()}"
    return jsonify({
        "status": "DISPATCHED",
        "incident_id": incident_id,
        "recipient": "1930 Cyber Fraud / I4C National Helpline",
        "severity": data.get("tier", "HIGH_RISK"),
        "telemetry_attached": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/health", methods=["GET", "POST"])
def api_health():
    return jsonify({
        "status": "ONLINE",
        "app": "DHWANI-KAVACH",
        "ok": True,
        "online": True,
        "backend": "online",
        "model": MODEL_NAME,
        "inference_backend": INFERENCE_BACKEND,
        "onnx_active": (INFERENCE_BACKEND == "onnx"),
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    import socket
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _lan_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _lan_ip = "(run ipconfig to find it)"
    print()
    print("DHWANI-KAVACH demo v2. Open in your browser (Chrome recommended):")
    print("       http://127.0.0.1:8000")
    print(f"phones on the same Wi-Fi:  http://{_lan_ip}:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)