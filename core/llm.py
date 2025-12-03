# core/llm.py
from __future__ import annotations
import os, json, re
from typing import Tuple, Dict, Any, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Load .env from the project root (or nearest) and allow it to override any existing env vars.
# Using find_dotenv ensures we pick up the .env file even when the working directory
# differs from the location of this module. Without this, the GEMINI_API_KEY and other
# secrets defined in the repository's `.env` may not be loaded, causing the LLM to
# misconfigure and always return HOLD. Override existing env variables to honour
# settings in the `.env` file.
dotenv_path = find_dotenv()  # search for .env in current and parent directories
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    # fall back to default behaviour
    load_dotenv(override=True)


def _configure_genai(explicit_key: Optional[str]) -> str:
    """
    Configure Gemini once (explicit key > env).

    Priority:
      1. explicit_key passed in
      2. GEMINI_API_KEY from environment (.env after load_dotenv)
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

    Fallbacks:
      - JSON with slightly different keys (decision/VOTE, CONFIDENCE)
      - "VOTE: BUY ... CONFIDENCE: 0.73"
      - bare 'BUY'/'SELL'/'HOLD' with neutral confidence 0.5
    """
    text = (text or "").strip()

    # 1) Try to find and load a JSON block
    m_json = re.search(r"\{.*\}", text, re.DOTALL)
    if m_json:
        try:
            obj = json.loads(m_json.group(0))
            vote = str(
                obj.get("vote")
                or obj.get("decision")
                or obj.get("VOTE")
                or ""
            ).upper()
            conf = float(obj.get("confidence") or obj.get("CONFIDENCE") or 0.0)
            if vote in {"BUY", "SELL", "HOLD"} and 0.0 <= conf <= 1.0:
                return vote, conf
        except Exception:
            pass

    # 2) Fallback: "VOTE: X ... CONFIDENCE: y"
    m = re.search(
        r"VOTE\s*:\s*(BUY|SELL|HOLD)\b.*?CONFIDENCE\s*:\s*([01](?:\.\d+)?)",
        text,
        re.IGNORECASE | re.DOTALL,
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
    Generate text from the Gemini API.

    Historically, we passed the system prompt via the `system_instruction` argument
    on the `GenerativeModel` constructor.  However, some models and API keys
    reject requests containing system instructions (returning a 403 error
    indicating a leaked or unauthorized key).  To improve robustness, we
    instead concatenate the system message and user text into a single string
    and supply it as the sole input to `generate_content()`.  This avoids
    system-instruction restrictions while preserving the full context for
    prompt engineering.
    """
    try:
        model = genai.GenerativeModel(model_name)
    except Exception:
        # In case model instantiation fails, return empty string
        return ""

    # Combine system message and user text.  If either is blank, skip the separator.
    combined_parts: List[str] = []
    if system_msg:
        combined_parts.append(system_msg)
    if user_text:
        combined_parts.append(user_text)
    combined_prompt = "\n\n".join(combined_parts)

    resp = model.generate_content(combined_prompt)

    # Standard path: resp.text is populated on modern SDKs
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
        # configure Gemini client
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
