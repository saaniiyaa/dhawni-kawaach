"""
17_export_onnx.py - DHWANI KAVACH :: ONNX export + self-test
Exports results/cnn_v2/model.pt (MelCNN) to results/cnn_v2/model.onnx
(opset 14+, dynamic batch and time axes) and verifies numerical parity (< 1e-4).
"""

import os
import sys
import hashlib
import numpy as np

# Prioritize CNN v2 (codec-hardened), fallback to CNN v1
if os.path.exists(os.path.join("results", "cnn_v2", "model.pt")):
    MODEL_PATH = os.path.join("results", "cnn_v2", "model.pt")
    ONNX_PATH = os.path.join("results", "cnn_v2", "model.onnx")
    MODEL_LABEL = "CNN v2 (codec-hardened)"
else:
    MODEL_PATH = os.path.join("results", "cnn_v1", "model.pt")
    ONNX_PATH = os.path.join("results", "cnn_v1", "model.onnx")
    MODEL_LABEL = "CNN v1 (clean-trained)"

N_MELS = 80
T_REF = 400

def die(msg):
    print("[FAIL] " + msg)
    sys.exit(1)

# ---------- STEP 0: dependencies ----------
try:
    import onnx
except ImportError:
    die("Package 'onnx' missing. Run: pip install onnx onnxruntime")
try:
    import onnxruntime as ort
except ImportError:
    die("Package 'onnxruntime' missing. Run: pip install onnx onnxruntime")
import torch

print("[OK] torch=%s  onnx=%s  onnxruntime=%s" % (torch.__version__, onnx.__version__, ort.__version__))

# ---------- STEP 1: model class ----------
from ml.models.cnn import MelCNN

if not os.path.exists(MODEL_PATH):
    die("Not found: " + MODEL_PATH)

model = MelCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("[OK] %s checkpoint loaded: %s" % (MODEL_LABEL, MODEL_PATH))

# ---------- STEP 2: export to ONNX (opset 14+, dynamic batch and time axes) ----------
dummy = torch.randn(1, 1, N_MELS, T_REF, dtype=torch.float32)
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
dynamic_ok = True

opset = 14
try:
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["logmel"],
        output_names=["logit"],
        dynamic_axes={
            "logmel": {0: "batch", 3: "time"},
            "logit": {0: "batch"}
        },
        opset_version=opset,
        do_constant_folding=True,
    )
except Exception as e:
    print("[WARN] Export with dynamic axes failed (%r), trying default opset 14..." % (e,))
    try:
        torch.onnx.export(
            model,
            dummy,
            ONNX_PATH,
            input_names=["logmel"],
            output_names=["logit"],
            opset_version=opset,
            do_constant_folding=True
        )
    except Exception as e2:
        die("export failed: %r" % (e2,))

print("[OK] exported ONNX (opset %d): %s" % (opset, ONNX_PATH))

# ---------- STEP 3: validate ----------
try:
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("[OK] onnx.checker passed")
except Exception as e:
    die("onnx.checker failed: %r" % (e,))

# ---------- STEP 4: parity self-tests ----------
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

def sig(z):
    return 1.0 / (1.0 + np.exp(-z))

def test_single(t_len, must_pass=True):
    label = "T=%d" % t_len
    try:
        x = torch.randn(1, 1, N_MELS, t_len, dtype=torch.float32)
        with torch.no_grad():
            ref = float(model(x).reshape(-1)[0])
        got = float(sess.run(None, {"logmel": x.numpy()})[0].reshape(-1)[0])
    except Exception as e:
        print("[FAIL] parity %-7s raised: %r" % (label, e))
        return False
    d = abs(ref - got)
    ok = d < 1e-4
    tag = "PASS" if ok else ("FAIL" if must_pass else "INFO")
    print("[%s] parity %-7s torch=%+.6f onnx=%+.6f diff=%.2e  (p_ai: %.4f vs %.4f)"
          % (tag, label, ref, got, d, sig(ref), sig(got)))
    return ok

def test_batch():
    try:
        x = torch.randn(3, 1, N_MELS, T_REF, dtype=torch.float32)
        with torch.no_grad():
            ref = model(x).reshape(-1).numpy()
        got = sess.run(None, {"logmel": x.numpy()})[0].reshape(-1)
        d = float(np.max(np.abs(ref - got)))
    except Exception as e:
        print("[INFO] batch test raised: %r" % (e,))
        return False
    ok = d < 1e-4
    print("[%s] parity batch=3 max_diff=%.2e" % ("PASS" if ok else "INFO", d))
    return ok

r1 = test_single(T_REF, must_pass=True)
r2 = test_single(250, must_pass=True)
r3 = test_batch()

# ---------- STEP 5: evidence report ----------
pt_b = os.path.getsize(MODEL_PATH)
ox_b = os.path.getsize(ONNX_PATH)
sha = hashlib.sha256(open(ONNX_PATH, "rb").read()).hexdigest()
print()
print("=" * 60)
print("ONNX EXPORT VERIFICATION REPORT")
print("=" * 60)
print("Model           : %s" % MODEL_LABEL)
print("PyTorch path    : %s (%d bytes)" % (MODEL_PATH, pt_b))
print("ONNX path       : %s (%d bytes)" % (ONNX_PATH, ox_b))
print("ONNX SHA-256    : %s" % sha)
print("Parity T=400    : %s" % ("PASS" if r1 else "FAIL"))
print("Parity T=250    : %s" % ("PASS" if r2 else "FAIL"))
print("Parity batch=3  : %s" % ("PASS" if r3 else "FAIL"))
print("=" * 60)

if r1 and r2 and r3:
    print("SUCCESS: CNN v2 ONNX model verified with numerical parity (< 1e-4).")
else:
    die("PARITY FAILED: differences exceed 1e-4")
