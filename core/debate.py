from __future__ import annotations
from typing import List, Tuple

class Debate:
    def __init__(self, mean_conf_to_act: float = 0.6):
        self.mean_conf_to_act = mean_conf_to_act

    def run(self, votes: List[dict]) -> Tuple[str, float]:
        if not votes:
            return "HOLD", 0.0
        buy = sum(v['confidence'] for v in votes if v['decision'] == 'BUY')
        sell = sum(v['confidence'] for v in votes if v['decision'] == 'SELL')
        total = sum(v['confidence'] for v in votes)
        if total == 0:
            return "HOLD", 0.0
        net = buy - sell
        final_conf = abs(net) / total
        if net > 0 and final_conf >= self.mean_conf_to_act:
            return "BUY", final_conf
        if net < 0 and final_conf >= self.mean_conf_to_act:
            return "SELL", final_conf
        return "HOLD", 1.0 - final_conf
