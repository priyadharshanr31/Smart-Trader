# core/llm.py
from __future__ import annotations
import os, json, re
from typing import Tuple, Dict, Any, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def _configure_genai(explicit_key: Optional[str]) -> str:
    """
    Configure Gemini once (explicit key > env).
    """
    key = (explicit_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing (and no explicit api_key provided).")
    genai.configure(api_key=key)
    return key


# Use model names your account actually supports (from your ListModels).
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash").strip() or "models/gemini-2.5-flash"
FALLBACK_MODELS: List[str] = [
    PRIMARY_MODEL,
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash-lite",
]


def _parse_vote(text: str) -> Tuple[str, float]:
    """
    Prefer strict JSON like:
      {"vote":"BUY|SELL|HOLD","confidence":0.73,"rationale":"..."}
    Fallback to: "VOTE: BUY ... CONFIDENCE: 0.73"
    """
    text = (text or "").strip()

    # 1) Try to find and load a JSON block
    m_json = re.search(r"\{.*\}", text, re.DOTALL)
    if m_json:
        try:
            obj = json.loads(m_json.group(0))
            vote = str(obj.get("vote") or obj.get("decision") or obj.get("VOTE") or "").upper()
            conf = float(obj.get("confidence") or obj.get("CONFIDENCE") or 0.0)
            if vote in {"BUY", "SELL", "HOLD"} and 0.0 <= conf <= 1.0:
                return vote, conf
        except Exception:
            pass

    # 2) Fallback: "VOTE: X ... CONFIDENCE: y"
    m = re.search(
        r"VOTE\s*:\s*(BUY|SELL|HOLD)\b.*?CONFIDENCE\s*:\s*([01](?:\.\d+)?)",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).upper(), float(m.group(2))

    # 3) Last-ditch: infer a vote word, neutral confidence
    m2 = re.search(r"\b(BUY|SELL|HOLD)\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper(), 0.5

    return "HOLD", 0.5


def _gen_content(model_name: str, system_msg: str, user_text: str) -> str:
    """
    Gemini 2.x: put the prompt in system_instruction at model construction,
    then pass a single user string to generate_content().
    """
    model = genai.GenerativeModel(model_name, system_instruction=system_msg)
    resp = model.generate_content(user_text)

    # Standard path
    if getattr(resp, "text", None):
        return resp.text

    # Older SDK fallbacks
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return ""


class LCTraderLLM:
    """
    vote_structured(system_msg, user_template, variables)
      -> (decision: 'BUY'|'SELL'|'HOLD', confidence: float [0..1], raw_text: str)
    """

    def __init__(self, model: str | None = None, api_key: Optional[str] = None, **_: Any):
        _configure_genai(api_key)
        # call order with fallbacks
        self.model_chain: List[str] = []
        if model:
            self.model_chain.append(model)
        for m in FALLBACK_MODELS:
            if m not in self.model_chain:
                self.model_chain.append(m)

    def vote_structured(
        self,
        system_msg: str,
        user_template: str,
        variables: Dict[str, Any],
    ) -> Tuple[str, float, str]:
        user_text = user_template.format(**variables)
        errors: List[str] = []

        for m in self.model_chain:
            try:
                raw = _gen_content(m, system_msg, user_text)
                vote, conf = _parse_vote(raw)
                # normalize & clamp
                vote = vote if vote in {"BUY", "SELL", "HOLD"} else "HOLD"
                conf = max(0.0, min(1.0, float(conf)))
                return vote, conf, f"[model={m}] {raw}"
            except Exception as e:
                errors.append(f"{m}: {e}")

        # total failure → safe default
        return "HOLD", 0.5, "LLM unavailable: " + " | ".join(errors or ["unknown"])
