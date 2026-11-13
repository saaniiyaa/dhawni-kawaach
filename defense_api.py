"""
defense_api.py - DHWANI KAVACH :: Defense API (SIH26104 screening)
NAYI FILE - demo_server.py mein sirf 2-line registration (try/except wrapped).
"""

import hashlib
import json
import time

from flask import Blueprint, current_app, jsonify, request
import numpy as np

from channel_calibration import evaluate_channel_and_calibrate

bp = Blueprint("defense_api", __name__)

MODEL_META = {
    "model": "MelCNN cnn_v1",
    "params": 236141,
    "pipeline": "16kHz -> energy VAD -> 4s windows -> 80-bin log-mel -> CNN",
    "onnx_parity_max_delta": 2.4e-06,
    "onnx_sha256": "c4134066830c4a3cde640795dbaab5775f70211c2eeed15d2abb26ad97d539c5",
}


def _sha256(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@bp.route("/api/evidence/dossier", methods=["POST"])
def evidence_dossier():
    p = request.get_json(silent=True) or {}
    model_meta = dict(MODEL_META)
    model_meta["model"] = current_app.config.get("MODEL_VERSION") or "cnn_v2"
    case = {
        "case_id": "DK-" + format(int(time.time() * 1000), "X"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "law": "Section 63, Bharatiya Sakshya Adhiniyam 2023 (in force 01-Jul-2024); lineage: IT Act s.65B",
        "model": model_meta,
        "caller_number": str(p.get("caller_number", "unknown"))[:24],
        "risk_tier": str(p.get("risk_tier", p.get("tier", "UNKNOWN"))).upper()[:12],
        "radar_score": p.get("radar_score", p.get("score")),
        "transcript": str(p.get("transcript", "")),
        "pattern_categories": p.get("pattern_categories", [])[:20],
        "window_hashes": p.get("window_hashes", [])[:512],
        "chain_of_custody": "hashes computed on-device at analysis time; no raw audio uploaded; certificate generated server-side",
        "disclaimer": "DEMONSTRATION - telephone channel simulated and disclosed; certificate is a template, not legal advice.",
    }
    if "session_id" in p:
        case["session_id"] = p["session_id"]
    if "channel_quality" in p:
        case["channel_quality"] = p["channel_quality"]
    if "uncertainty_note" in p:
        case["uncertainty_note"] = p["uncertainty_note"]
    if "channel_metrics" in p:
        case["channel_metrics"] = p["channel_metrics"]
    case["record_sha256"] = _sha256(case)
    return jsonify({
        "success": True,
        "dossier": case,
        "record_sha256": case["record_sha256"],
        "sha256_seal": case["record_sha256"],
    })


_FREEZE_LOG = []


@bp.route("/api/npci/freeze", methods=["POST"])
def npci_freeze():
    p = request.get_json(silent=True) or {}
    tier = str(p.get("risk_tier") or p.get("tier") or "HOLD").upper()
    receipt = {
        "mode": "SIMULATION - DISCLOSED (NPCI/bank integration = roadmap)",
        "action": "outbound UPI/IMPS cap",
        "cap_inr": 2000,
        "protected_amount_inr": 50000,
        "duration_minutes": 60,
        "golden_hour": True,
        "family_approval_request": True,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "reference": "DKF-" + format(int(time.time() * 1000), "X"),
        "status": "FROZEN",
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    _FREEZE_LOG.append({"t": time.time(), "ref": receipt["reference"]})
    return jsonify({
        "success": True,
        "status": "FROZEN",
        "freeze": receipt,
        "protected_amount_inr": 50000,
        "reference": receipt["reference"],
        "log_len": len(_FREEZE_LOG),
        "note": "I4C golden-hour concept: early reporting/freezing sharply raises recovery odds",
    })


@bp.route("/api/challenge/verify", methods=["POST"])
def challenge_verify():
    p = request.get_json(silent=True) or {}
    safe_word = str(p.get("safe_word") or p.get("safeword") or p.get("challenge_word") or "").strip().upper()
    
    # Check verbal safe-word challenge
    VALID_SAFE_WORDS = {"KAVACH-2026", "SAFE-HOME", "1234"}
    if safe_word:
        if safe_word in VALID_SAFE_WORDS:
            return jsonify({
                "success": True,
                "verified": True,
                "status": "UNFROZEN",
                "message": "Safe-word verified successfully. Outbound UPI limits restored.",
                "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            })
        else:
            return jsonify({
                "success": True,
                "verified": False,
                "status": "FROZEN",
                "message": "Invalid safe-word. Asset protection remains active.",
                "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            })

    # Fallback to latency TTFT verification if provided
    raw = p.get("ttft_ms", p.get("latency_ms", -1))
    try:
        ttft = float(raw)
    except (TypeError, ValueError):
        ttft = -1.0
    if ttft >= 0:
        if ttft < 1000:
            verdict = "HUMAN_FAST"
        elif ttft <= 1600:
            verdict = "BORDERLINE"
        elif ttft <= 2000:
            verdict = "PROXY_SIGNATURE"
        else:
            verdict = "STRONG_PROXY_SIGNATURE"
        verified = (ttft < 1600)
        return jsonify({
            "success": True,
            "verified": verified,
            "status": "UNFROZEN" if verified else "FROZEN",
            "ttft_ms": round(ttft, 1),
            "verdict": verdict,
            "tiers": {"human_fast": "<1000ms", "borderline": "1000-1600ms", "proxy": "1600-2000ms", "strong_proxy": ">2000ms"},
            "caveat": "latency is one layer of five; optimized realtime agents reduce TTFT",
            "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        })

    # Default response
    return jsonify({
        "success": True,
        "verified": False,
        "status": "FROZEN",
        "message": "Safe-word required (e.g. KAVACH-2026)",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
    })
# ═══════════════ v2 APPEND BLOCK — paste at END of defense_api.py ═══════════════
# Telecom CLI-check · I4C schema export · Kin SOS · Forensics (metrics-only) ·
# Time-synced anti-replay token

from urllib.parse import quote as _urlquote

_BFSI_WORDS = ("bank", "sbi", "hdfc", "icici", "axis", "kotak", "pnb",
               "insurance", "policy", "pension", "epfo", "fraud team",
               "bank manager", "verification")

_PHRASES = [
    "kacha papad paka papad",
    "chandu ke chacha ne chai laayi",
    "irish wristwatch swiss wristwatch",
    "bhaiya ne bhabhi ko bye bye bola",
    "she sells sea shells",
]


@bp.route("/api/telecom/cli-check", methods=["POST"])
def telecom_cli_check():
    """TRAI 1600-series rule check (number + claim only; no PII stored)."""
    p = request.get_json(silent=True) or {}
    num = "".join(ch for ch in str(p.get("caller_number", "")) if ch.isdigit())
    if num.startswith("91") and len(num) > 10:
        num = num[2:]
    if num.startswith("0"):
        num = num[1:]
    if not num:
        return jsonify({"success": False, "reason": "caller_number required"}), 400
    claims_bfsi = bool(p.get("claims_bfsi"))
    if num.startswith("160"):
        v, risk = "GENUINE_BFSI_RANGE", "LOW"
        note = "TRAI 1600-series BFSI range (mandate Jan-Mar 2026). Voice-clone check still applies - a genuine range does not verify the caller."
    elif num.startswith("140"):
        v, risk = "TELEMARKETING", "MEDIUM"
        note = "140-series = telemarketing only; banks never make transactional calls from here."
    elif claims_bfsi:
        v, risk = "CRITICAL_TRAI_RULE_VIOLATION", "HIGH"
        note = "Caller claims bank/insurance/pension but the number is NOT 1600-series. Under the TRAI mandate (Jan-Mar 2026) this is a classic impersonation pattern."
    else:
        v, risk = "NORMAL_NUMBER", "INFO"
        note = "No BFSI claim recorded. If the caller later claims to be a bank, re-run with claims_bfsi=true."
    return jsonify({
        "success": True,
        "verdict": v,
        "risk": risk,
        "rule": "TRAI 1600-series BFSI mandate (Jan-Mar 2026)",
        "note": note,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
    })


@bp.route("/api/i4c/export-schema", methods=["POST"])
def i4c_export_schema():
    """Pre-filled complaint schema ALIGNED to NCRP-style fields.
    DEMONSTRATION template - verify fields on cybercrime.gov.in before filing."""
    p = request.get_json(silent=True) or {}
    schema = {
        "template": "NCRP-complaint-aligned (DEMONSTRATION - not an official integration)",
        "complainant": {"name": p.get("victim_name", ""), "mobile": p.get("victim_mobile", "")},
        "incident": {
            "date_of_call": p.get("incident_date", time.strftime("%Y-%m-%d")),
            "time_approx": p.get("incident_time", ""),
            "channel": "voice call",
            "category": p.get("category", "AI voice impersonation / attempted financial fraud"),
        },
        "suspect": {
            "caller_number": p.get("caller_number", ""),
            "claimed_identity": p.get("claimed_identity", ""),
            "cli_range_check": p.get("cli_check", ""),
        },
        "evidence": {
            "risk_tier": p.get("risk_tier", ""),
            "radar_score": p.get("radar_score"),
            "pattern_categories": p.get("pattern_categories", []),
            "record_sha256": p.get("record_sha256", ""),
            "evidence_note": "SHA-256 stamped dossier available via /api/evidence/dossier",
        },
        "financial": {
            "amount_requested": p.get("amount_requested", ""),
            "transfer_made": p.get("transfer_made", ""),
            "upi_reference": p.get("upi_reference", ""),
        },
        "helplines": {
            "golden_hour": "1930",
            "portal": "https://cybercrime.gov.in",
            "dot_chakshu": "https://sancharsaathi.gov.in",
        },
    }
    schema["schema_sha256"] = _sha256(schema)
    return jsonify({"success": True, "complaint_schema": schema})


@bp.route("/api/sos/notify-kin", methods=["POST"])
def sos_notify_kin():
    """Family SOS payload generator.
    Delivery: WhatsApp click-to-chat deep link returned (SIMULATED webhook;
    automated SMS / WhatsApp Business API = roadmap)."""
    p = request.get_json(silent=True) or {}
    num = "".join(ch for ch in str(p.get("kin_number", "")) if ch.isdigit())
    if len(num) == 10:
        num = "91" + num
    if not num:
        return jsonify({"success": False, "reason": "kin_number required"}), 400
    msg = (
        "\U0001F6A8 DHWANI KAVACH - EMERGENCY SOS\n\n"
        "A suspected AI voice-clone scam call was detected on your family member's phone.\n"
        "DO NOT authorize any UPI transfer or share OTPs.\n\n"
        "Risk tier: " + str(p.get("risk_tier", "HIGH")).upper() + "\n"
        "Radar: " + str(p.get("radar_score", "-")) + "/100\n"
        "Time: " + time.strftime("%Y-%m-%d %H:%M") + "\n\n"
        "Call back on the known number. Report: 1930 / cybercrime.gov.in"
    )
    return jsonify({
        "success": True,
        "delivery": "SIMULATED - WhatsApp click-to-chat deep link returned; automated delivery = roadmap",
        "kin_number_masked": "*" * max(0, len(num) - 4) + num[-4:],
        "message": msg,
        "wa_link": "https://wa.me/" + num + "?text=" + _urlquote(msg),
    })


@bp.route("/api/forensics/vitals", methods=["POST"])
def forensics_vitals():
    """Classify LOCALLY-COMPUTED vitals.
    PRIVACY: this endpoint NEVER receives audio - only derived,
    non-identifying metrics computed in-browser (Pack 2)."""
    p = request.get_json(silent=True) or {}

    def _f(k):
        try:
            return float(p.get(k, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    jitter, harm = _f("jitter_pct"), _f("harmonicity")
    pause, run = _f("pause_pct"), _f("longest_voiced_run_s")
    flags = []
    if jitter == jitter:
        if jitter < 0.3:
            flags.append("very low pitch jitter - synthetic hint (natural ~0.3-2%)")
        elif jitter > 2.5:
            flags.append("elevated jitter - noisy channel or stress")
    if harm == harm and harm < 0.45:
        flags.append("low harmonicity - check channel quality")
    if pause == pause and pause < 8:
        flags.append("almost no pauses - natural speech breathes")
    if run == run and run > 8:
        flags.append("8s+ continuous speech without pause - unnatural breathing pattern")
    if p.get("channel_quality") == "TELEPHONY_DEGRADED":
        flags.append("telephony degraded channel (8kHz / band-limited / low SNR)")
    return jsonify({
        "success": True,
        "received_metrics": {
            "jitter_pct": p.get("jitter_pct"),
            "harmonicity": p.get("harmonicity"),
            "pause_pct": p.get("pause_pct"),
            "longest_voiced_run_s": p.get("longest_voiced_run_s"),
        },
        "flags": flags,
        "reference": {
            "jitter_pct": "0.3-2 natural",
            "harmonicity": ">=0.45 typical voiced",
            "pause_pct": ">=8 typical conversational",
            "longest_voiced_run_s": "<=8 typical",
        },
        "privacy": "no audio ever received; metrics computed in-browser and classified here",
        "caveat": "telemetry supports the verdict - never replaces it (combination law)",
    })


@bp.route("/api/challenge/token", methods=["GET"])
def challenge_token():
    """Time-synced anti-replay token: a pre-recorded clip cannot know it."""
    window = int(time.time() // 30)
    digest = hashlib.sha256((str(window) + "|dhwani-kavach-sih26104").encode()).hexdigest()
    token = str(int(digest[:8], 16) % 10000).zfill(4)
    phrase = _PHRASES[int(digest[8:12], 16) % len(_PHRASES)]
    return jsonify({
        "success": True,
        "window_s": 30,
        "expires_in_s": 30 - int(time.time() % 30),
        "token": token,
        "challenge_phrase": phrase,
        "anti_replay": "a pre-recorded clip cannot contain a token generated after the call started",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
    })


@bp.route("/api/calibrate", methods=["POST"])
@bp.route("/api/channel/calibrate", methods=["POST"])
def channel_calibrate():
    """Evaluate channel degradation and calibrate risk tier."""
    p = request.get_json(silent=True) or {}
    raw_prob = float(p.get("raw_ai_prob", p.get("score", p.get("p", 0.5))))
    sr = int(p.get("sr", 16000))
    audio_data = p.get("audio")
    if isinstance(audio_data, list):
        y = np.array(audio_data, dtype=np.float32)
    elif isinstance(audio_data, str) and audio_data:
        import base64
        try:
            raw_bytes = base64.b64decode(audio_data.split(",")[-1])
            y = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0 if len(raw_bytes) % 2 == 0 else np.array([], dtype=np.float32)
        except Exception:
            y = np.array([], dtype=np.float32)
    else:
        y = np.array([], dtype=np.float32)
    res = evaluate_channel_and_calibrate(y, sr, raw_prob)
    return jsonify({"success": True, **res})
