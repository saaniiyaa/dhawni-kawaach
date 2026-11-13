# ml/engine/temporal.py
# Converts noisy per-window detector scores into ONE call-level decision.
# This is what makes us a product, not just a classifier.
# Four protections built in:
#   1. refuses to judge before enough speech (INSUFFICIENT_AUDIO)
#   2. evidence accumulates over the whole call (log-odds meter)
#   3. one crazy window cannot dominate (clipping + dwell time)
#   4. harder to LEAVE high risk than enter it (safety-first hysteresis)

from dataclasses import dataclass, field
import math

SAFE, SUSPICIOUS, HIGH_RISK, INSUFFICIENT = "SAFE", "SUSPICIOUS", "HIGH_RISK", "INSUFFICIENT_AUDIO"


@dataclass
class EngineConfig:
    theta_lo: float = 0.78         # suspicious threshold
    theta_hi: float = 0.90         # high-risk threshold
    theta_neutral: float = 0.70    # calibrated decision boundary for neutral evidence
    ema_alpha: float = 0.35        # smoothing for the number shown on screen
    min_voiced_seconds: float = 2.0   # refuse to judge before this much speech
    dwell_windows: int = 2         # consecutive agreeing windows needed to change tier
    llr_clip: float = 3.5          # one window can add at most this much evidence
    llr_decay: float = 0.95        # old evidence slowly forgotten


@dataclass
class EngineState:
    ema: float = 0.0
    llr: float = -1.5              # default to safe baseline
    n_windows: int = 0
    voiced_seconds: float = 0.0
    tier: str = INSUFFICIENT
    _pending: str = INSUFFICIENT
    _pending_count: int = 0
    history: list = field(default_factory=list)
    session_risk: float = 0.0


class TemporalEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg or EngineConfig()
        self.s = EngineState()

    def reset(self):
        self.s = EngineState()

    @staticmethod
    def _llr(p, theta_ref=0.5, eps=1e-6):
        p = min(max(p, eps), 1 - eps)
        theta_ref = min(max(theta_ref, eps), 1 - eps)
        # Log-odds ratio relative to calibrated decision threshold
        return math.log(p / (1 - p)) - math.log(theta_ref / (1 - theta_ref))

    def update(self, p_spoof, voiced_seconds, t=None, is_pause=False):
        c, s = self.cfg, self.s
        s.n_windows += 1
        s.voiced_seconds += voiced_seconds

        # Turn-taking pause chunk: decay accumulated session risk gradually (prev_risk * 0.90)
        # instead of resetting the session risk score to zero
        if is_pause or voiced_seconds < 0.6:
            prev_risk = s.session_risk if (s.n_windows > 1 and s.session_risk > 0) else (1 / (1 + math.exp(-s.llr)))
            decayed = float(prev_risk * 0.90)
            s.session_risk = decayed
            p_clamped = min(max(decayed, 1e-6), 1 - 1e-6)
            s.llr = math.log(p_clamped / (1 - p_clamped))
            p_spoof = decayed
            call_p = decayed
        else:
            step = max(-c.llr_clip, min(c.llr_clip, self._llr(p_spoof, c.theta_neutral)))
            s.llr = s.llr * c.llr_decay + step * min(voiced_seconds, 1.0)  # evidence meter
            call_p = 1 / (1 + math.exp(-s.llr))
            s.session_risk = call_p

        s.ema = p_spoof if s.n_windows == 1 else c.ema_alpha * p_spoof + (1 - c.ema_alpha) * s.ema

        if s.voiced_seconds < c.min_voiced_seconds:
            target = INSUFFICIENT
        elif s.tier == HIGH_RISK:      # hysteresis: easier to stay in high risk
            target = HIGH_RISK if call_p >= c.theta_lo else (
                SUSPICIOUS if call_p >= c.theta_lo * 0.6 else SAFE)
        elif call_p >= c.theta_hi:
            target = HIGH_RISK
        elif call_p >= c.theta_lo:
            target = SUSPICIOUS
        else:
            target = SAFE

        if target == s._pending:
            s._pending_count += 1
        else:
            s._pending, s._pending_count = target, 1
        if s._pending_count >= (1 if target == INSUFFICIENT else self.cfg.dwell_windows):
            s.tier = target

        out = {"t": t, "window_p": float(p_spoof), "ema": float(s.ema),
               "call_p": float(call_p), "llr": float(s.llr), "tier": s.tier,
               "session_risk": float(s.session_risk),
               "voiced_seconds": round(s.voiced_seconds, 2), "n_windows": s.n_windows}
        s.history.append(out)
        return out

    def update_pause(self, decay_factor=0.90, t=None):
        """When a 1.0s-2.0s silent/low-VAD pause chunk arrives (simulating full-duplex
        conversational turn-taking), decay the accumulated session risk gradually
        (prev_risk * 0.90) instead of resetting the session risk score to zero."""
        return self.update(0.0, 0.0, t=t, is_pause=True)


# ---------------- the prevention policy: what the bank/app actually does ----------------
PREVENTION = {
    INSUFFICIENT: dict(action="OBSERVE", ui="Listening... not enough speech to judge yet",
                       allow_sensitive_action=True, step_up=None),
    SAFE:         dict(action="ALLOW", ui="No synthetic-speech indicators detected",
                       allow_sensitive_action=True, step_up=None),
    SUSPICIOUS:   dict(action="WARN", ui="Possible synthetic speech - verify before acting",
                       allow_sensitive_action=True,
                       step_up="Ask a pre-agreed code word OR call back on the saved number"),
    HIGH_RISK:    dict(action="HOLD", ui="High risk of cloned voice - sensitive action paused",
                       allow_sensitive_action=False,
                       step_up="Out-of-band callback to registered number + second approver"),
}


def policy_for(tier):
    return PREVENTION[tier]