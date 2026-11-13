# test_channel_and_temporal.py
import unittest
import numpy as np
import json
from channel_calibration import evaluate_channel_and_calibrate
from ml.engine.temporal import TemporalEngine, EngineConfig
from demo_server import app

class TestChannelAndTemporal(unittest.TestCase):
    def test_channel_calibration_wideband_clean(self):
        sr = 16000
        t = np.linspace(0, 4.0, int(4.0 * sr), endpoint=False)
        env = np.where(np.sin(2 * np.pi * 1.5 * t) > 0, 1.0, 0.001)
        # 500 Hz low band + 5000 Hz high band with envelope for high dynamic range SNR
        y = (env * (0.2 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 5000 * t))).astype(np.float32)

        # 1. HOLD (>= 0.70)
        res_hold = evaluate_channel_and_calibrate(y, sr, 0.75)
        self.assertEqual(res_hold["channel_quality"], "WIDEBAND_CLEAN")
        self.assertEqual(res_hold["tier"], "HOLD")
        self.assertEqual(res_hold["uncertainty_note"], "Wideband harmonics intact • High-confidence synthetic signature")
        self.assertIn("high_band_ratio", res_hold["channel_metrics"])
        self.assertIn("snr_db", res_hold["channel_metrics"])
        self.assertGreaterEqual(res_hold["channel_metrics"]["high_band_ratio"], 0.015)
        self.assertGreaterEqual(res_hold["channel_metrics"]["snr_db"], 10.0)

        # 2. WARN (>= 0.40)
        res_warn = evaluate_channel_and_calibrate(y, sr, 0.45)
        self.assertEqual(res_warn["channel_quality"], "WIDEBAND_CLEAN")
        self.assertEqual(res_warn["tier"], "WARN")
        self.assertEqual(res_warn["uncertainty_note"], "Wideband channel • Moderate anomaly — verify caller")

        # 3. ALLOW (< 0.40)
        res_allow = evaluate_channel_and_calibrate(y, sr, 0.20)
        self.assertEqual(res_allow["channel_quality"], "WIDEBAND_CLEAN")
        self.assertEqual(res_allow["tier"], "ALLOW")
        self.assertEqual(res_allow["uncertainty_note"], "Wideband harmonics intact • Natural vocal profile")

    def test_channel_calibration_telephony_degraded(self):
        # Degraded by 8kHz sample rate
        sr = 8000
        t = np.linspace(0, 4.0, int(4.0 * sr), endpoint=False)
        y = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

        # 1. HOLD (>= 0.82)
        res_hold = evaluate_channel_and_calibrate(y, sr, 0.85)
        self.assertEqual(res_hold["channel_quality"], "TELEPHONY_DEGRADED")
        self.assertEqual(res_hold["tier"], "HOLD")
        self.assertEqual(res_hold["uncertainty_note"], "Degraded 8kHz/noisy channel • Strong synthetic artifact overrides noise")

        # 2. WARN (>= 0.38)
        res_warn = evaluate_channel_and_calibrate(y, sr, 0.70)
        self.assertEqual(res_warn["channel_quality"], "TELEPHONY_DEGRADED")
        self.assertEqual(res_warn["tier"], "WARN")
        self.assertEqual(res_warn["uncertainty_note"], "Degraded channel (8kHz/noise) • Uncertain score — trigger Safe-Word / Callback")

        # 3. ALLOW (< 0.38)
        res_allow = evaluate_channel_and_calibrate(y, sr, 0.30)
        self.assertEqual(res_allow["channel_quality"], "TELEPHONY_DEGRADED")
        self.assertEqual(res_allow["tier"], "ALLOW")
        self.assertEqual(res_allow["uncertainty_note"], "Degraded channel • No sustained synthetic pattern")

    def test_turn_taking_pause_persistence(self):
        eng = TemporalEngine(EngineConfig(theta_lo=0.62, theta_hi=0.90))
        # Build up high risk over 4 voiced speech windows
        for _ in range(4):
            r = eng.update(0.95, 1.0)
        self.assertEqual(r["tier"], "HIGH_RISK")
        prev_risk = r["call_p"]

        # Pause 1: decay by 0.90
        p1 = eng.update_pause()
        self.assertAlmostEqual(p1["call_p"], prev_risk * 0.90, places=4)
        self.assertAlmostEqual(p1["window_p"], prev_risk * 0.90, places=4)
        self.assertGreater(p1["call_p"], 0.70)
        self.assertEqual(p1["tier"], "HIGH_RISK")

        # Pause 2: decay by 0.90 again
        p2 = eng.update_pause()
        self.assertAlmostEqual(p2["call_p"], prev_risk * 0.90 * 0.90, places=4)
        self.assertGreater(p2["call_p"], 0.60)

    def test_api_score_includes_calibration_keys(self):
        client = app.test_client()
        client.post('/api/reset')
        with open("results/demo/demo_real_studio.wav", "rb") as fh:
            res = client.post('/api/score', data=fh.read(), headers={'Content-Type': 'audio/wav'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("channel_quality", data)
        self.assertIn("uncertainty_note", data)
        self.assertIn("channel_metrics", data)
        self.assertIn("high_band_ratio", data["channel_metrics"])
        self.assertIn("snr_db", data["channel_metrics"])
        self.assertIn("calibrated_prob", data)
        self.assertIn("window_p", data)
        self.assertIn("call_p", data)
        self.assertIn("tier", data)

    def test_defense_api_calibrate_endpoint(self):
        client = app.test_client()
        res = client.post('/api/channel/calibrate', json={
            "raw_ai_prob": 0.88,
            "sr": 8000
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("channel_quality"), "TELEPHONY_DEGRADED")
        self.assertEqual(data.get("tier"), "HOLD")

if __name__ == '__main__':
    unittest.main()

