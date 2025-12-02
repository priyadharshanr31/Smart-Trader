# core/debate.py
from __future__ import annotations
from typing import List, Dict, Tuple

class Debate:
    """
    Backward-compatible net-vote (for the UI) + new horizon-aware decision.
    Expect votes like:
      [{"agent":"ShortTerm","decision":"BUY","confidence":0.7,"raw":"..."},
       {"agent":"MidTerm","decision":"HOLD","confidence":0.5,"raw":"..."},
       {"agent":"LongTerm","decision":"SELL","confidence":0.6,"raw":"..."}]
    """

    def __init__(self,
                 enter_th: float = 0.60,
                 exit_th: float = 0.45,
                 weights: Dict[str, float] | None = None):
        """
        Initialize the Debate object with configurable thresholds and weights.

        Parameters
        ----------
        enter_th : float, optional
            Confidence threshold to trigger a BUY action. A lower value makes the
            system more willing to take a position. Default is 0.60.
        exit_th : float, optional
            Confidence threshold to trigger a SELL action. A lower value makes the
            system more willing to exit a position. Default is 0.45.
        weights : dict, optional
            Per‑horizon weights used to combine agent votes. If not provided, a
            default weighting favouring short‑term signals is used.

        The default thresholds and weights have been tuned conservatively; the
        caller can pass lower thresholds via settings to make the agent more
        decisive while still respecting risk controls.
        """
        self.enter_th = float(enter_th)
        self.exit_th = float(exit_th)
        # emphasise short‑term signals slightly but allow overrides
        self.weights = weights or {"short": 0.40, "mid": 0.35, "long": 0.25}

    # ------------ legacy output (UI uses this) ------------
    def run(self, votes: List[dict]) -> Tuple[str, float]:
        if not votes:
            return "HOLD", 0.0
        buy = sum(v["confidence"] for v in votes if v["decision"] == "BUY")
        sell = sum(v["confidence"] for v in votes if v["decision"] == "SELL")
        total = sum(v["confidence"] for v in votes) or 1.0
        net = buy - sell
        final_conf = abs(net) / total
        if net > 0 and final_conf >= self.enter_th:
            return "BUY", final_conf
        if net < 0 and final_conf >= self.exit_th:
            return "SELL", final_conf
        return "HOLD", 1.0 - final_conf

    # ------------ new horizon-aware decision ------------
    def horizon_decide(self, votes: List[dict]) -> Dict:
        """
        Returns:
        {
          "action": "BUY"|"SELL"|"HOLD",
          "target_horizon": "short"|"mid"|"long"|None,
          "confidence": float,
          "scores": {"short": float, "mid": float, "long": float}
        }
        """
        horizon_map = {"ShortTerm": "short", "MidTerm": "mid", "LongTerm": "long"}
        scores = {"short": 0.0, "mid": 0.0, "long": 0.0}

        for v in votes:
            h = horizon_map.get(v.get("agent"))
            if not h: continue
            side = v.get("decision", "HOLD")
            conf = float(v.get("confidence", 0.0))
            side_factor = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
            scores[h] += self.weights.get(h, 0.0) * conf * side_factor

        # Aggregate the weighted signals into a net score. A positive net score
        # indicates broad BUY pressure across horizons; a negative score indicates
        # broad SELL pressure. This allows the system to act when multiple
        # horizons align, even if no individual horizon crosses its threshold.
        net_score = sum(scores.values())

        # Identify the strongest individual positive and negative scores
        best_h, best_score = max(scores.items(), key=lambda kv: kv[1])
        worst_h, worst_score = min(scores.items(), key=lambda kv: kv[1])

        # If the aggregate conviction is high enough, act on the overall net.
        if net_score >= self.enter_th:
            return {
                "action": "BUY",
                "target_horizon": best_h,
                "confidence": round(net_score, 3),
                "scores": scores,
            }
        if net_score <= -self.exit_th:
            return {
                "action": "SELL",
                "target_horizon": worst_h,
                "confidence": round(abs(net_score), 3),
                "scores": scores,
            }

        # Fall back to the previous behaviour: if a single horizon is very
        # confident, act on that. Otherwise return HOLD.
        if best_score >= self.enter_th:
            return {
                "action": "BUY",
                "target_horizon": best_h,
                "confidence": round(best_score, 3),
                "scores": scores,
            }
        if abs(worst_score) >= self.exit_th:
            return {
                "action": "SELL",
                "target_horizon": worst_h,
                "confidence": round(abs(worst_score), 3),
                "scores": scores,
            }
        return {
            "action": "HOLD",
            "target_horizon": None,
            "confidence": round(max(abs(best_score), abs(worst_score)), 3),
            "scores": scores,
        }


# ---- concise human reason for UI/logs ----
def summarize_reason_2lines(votes: List[dict], decision: Dict) -> str:
    tag = {"ShortTerm":"S", "MidTerm":"M", "LongTerm":"L"}
    parts = []
    for v in votes:
        parts.append(f"{tag.get(v.get('agent'), v.get('agent','?')[:1])}:{v.get('decision','?')}({float(v.get('confidence',0)):0.2f})")
    votes_line = " | ".join(parts) if parts else "-"
    act = decision.get("action","HOLD")
    hor = decision.get("target_horizon") or "-"
    conf = float(decision.get("confidence", 0.0))
    return f"Final: {act} (horizon={hor}, conf={conf:.2f}). Votes: {votes_line}"
