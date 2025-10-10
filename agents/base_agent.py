# agents/base_agent.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Protocol

class LLMProtocol(Protocol):
    """
    Minimal interface the agents need from the LLM wrapper.
    Your LCTraderLLM in core/llm.py already implements this.
    """
    def vote_structured(
        self,
        system_msg: str,
        user_template: str,
        variables: Dict[str, Any],
    ) -> Tuple[str, float, str]: ...

class BaseAgent(ABC):
    """
    Base class for all agents.
    Agents receive an LLM object that satisfies LLMProtocol.
    """
    def __init__(self, name: str, llm: LLMProtocol, config: Dict[str, Any] | None = None):
        self.name = name
        self.llm = llm
        self.config = config or {}

    @abstractmethod
    def vote(self, snapshot: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Given a layered snapshot (short/mid/long DataFrames), return:
        (decision: 'BUY'|'SELL'|'HOLD', confidence: float 0..1, raw_text: str)
        """
        raise NotImplementedError
