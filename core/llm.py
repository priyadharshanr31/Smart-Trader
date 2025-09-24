from __future__ import annotations
import os
import re
from typing import Tuple, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, confloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class VoteModel(BaseModel):
    """Structured output for agent votes."""
    decision: Literal["BUY", "SELL", "HOLD"]
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="0.0 to 1.0")
    reason: str = Field(..., description="Short rationale")


class LCTraderLLM:
    """
    LangChain-powered LLM helper.
    - Primary: structured output via Pydantic (VoteModel)
    - Fallback: regex parse if the model ignores schema
    """
    def __init__(self, gemini_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY missing")
        os.environ["GOOGLE_API_KEY"] = gemini_key
        self.model = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    def _fallback_regex(self, text: str) -> Tuple[str, float, str]:
        vote = re.search(r"VOTE:\s*(BUY|SELL|HOLD)", text, re.I)
        conf = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.I)
        decision = vote.group(1).upper() if vote else "HOLD"
        try:
            confidence = float(conf.group(1)) if conf else 0.5
        except Exception:
            confidence = 0.5
        return decision, confidence, text

    def vote_structured(
        self,
        system_msg: str,
        user_template: str,
        variables: Dict[str, Any],
    ) -> Tuple[str, float, str]:
        """
        Build a ChatPromptTemplate → ask Gemini for structured VoteModel.
        Returns (decision, confidence, raw_json_or_text).
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 system_msg
                 + "\nYou MUST follow the schema: {decision: BUY|SELL|HOLD, confidence: 0..1, reason: string}."),
                ("human", user_template),
            ]
        )
        try:
            chain = prompt | self.model.with_structured_output(VoteModel)
            result: VoteModel = chain.invoke(variables)
            # Serialize to a compact string for display
            raw = result.model_dump_json()
            return result.decision, float(result.confidence), raw
        except Exception:
            # Fallback: ask for a strict line, then regex-parse it
            fallback_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     "Return exactly one line: 'VOTE: BUY|SELL|HOLD, CONFIDENCE: 0.0-1.0'. "
                     "Then add one short reason on the next line."),
                    ("human", user_template),
                ]
            )
            text = (fallback_prompt | self.model).invoke(variables).content
            return self._fallback_regex(text)
