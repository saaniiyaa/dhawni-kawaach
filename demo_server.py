# demo_server.py  (v2 - full demo: call screen, context card, voiceprint, radar)
# DHWANI-KAVACH LIVE DEMO SERVER

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

torch.set_num_threads(1)

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
print("demo model:", MODEL_NAME, "<-", MODEL_PATH)

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
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


def score_audio(y):
    piece = fix_length(np.asarray(y, dtype=np.float32), int(SECONDS * SR))
    x = logmel(piece)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        p = float(torch.sigmoid(model(torch.from_numpy(x[None, None, ...]))))
    return p


def decode_to_wav(raw_bytes, suffix):
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
        if request.files or "multipart/form-data" in ctype:
            f = request.files.get("file") or request.files.get("audio") or next(iter(request.files.values()), None)
            raw_bytes = f.read() if f is not None else b""
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
                s = d.get("audio") or d.get("data") or d.get("blob") or d.get("file") or ""
                if isinstance(s, str) and s:
                    raw_bytes = _b64.b64decode(s.split(",")[-1])
            except Exception:
                raw_bytes = b""
            if not raw_bytes:
                raw_bytes = request.get_data()
        else:
            raw_bytes = request.get_data()
        print(f"[score] in: ctype={ctype}, bytes={len(raw_bytes)}, suffix={suffix}")
        if not raw_bytes:
            return jsonify({"error": "no audio received"})
        y = decode_to_wav(raw_bytes, suffix)       
        if len(y) < int(0.75 * SR):
            return jsonify({"error": "too short"})
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))

        chunk_dur = len(y) / SR
        vfrac = float(energy_vad(y[-int(SECONDS * SR):]).mean()) if len(y) >= int(SECONDS * SR) else float(energy_vad(y).mean())
        vad_min = float(CAL.get("vad_min", 0.25))

        # Turn-taking pause chunk: decay accumulated session risk gradually (prev_risk * 0.90)
        # instead of resetting the session risk score to zero
        if (1.0 <= chunk_dur <= 2.0 and vfrac < vad_min) or vfrac < vad_min:
            r = engine.update_pause(decay_factor=0.90)
            SESSION_STATE["prev_risk"] = r["call_p"]
            raw_ai_prob = r["window_p"]
        else:
            raw = score_audio(y)
            bias = float(CAL.get("logit_bias", 0.0))
            q = min(max(raw, 1e-4), 1 - 1e-4)
            p = 1.0 / (1.0 + math.exp(-(math.log(q / (1.0 - q)) + bias)))
            r = engine.update(p, vfrac * SECONDS)
            SESSION_STATE["prev_risk"] = r["call_p"]
            raw_ai_prob = r["window_p"]

        calib = evaluate_channel_and_calibrate(y, SR, raw_ai_prob)
        pol = policy_for(r["tier"])
        return jsonify({
            "window_p": round(r["window_p"], 4),
            "call_p": round(r["call_p"], 4),
            "tier": r["tier"],
            "voiced_seconds": r["voiced_seconds"],
            "n_windows": r["n_windows"],
            "action": pol["action"],
            "ui": pol["ui"],
            "step_up": pol["step_up"],
            "allow_sensitive": pol["allow_sensitive_action"],
            "model": MODEL_NAME,
            "channel_quality": calib["channel_quality"],
            "uncertainty_note": calib["uncertainty_note"],
            "channel_metrics": calib["channel_metrics"],
            "calibrated_prob": calib["calibrated_prob"],
            "calibrated_tier": calib["tier"],
        })
    except Exception as e:
        return jsonify({"error": str(e)})


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
        "model": MODEL_NAME,
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