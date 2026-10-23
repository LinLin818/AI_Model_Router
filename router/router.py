"""Inference-time router.

    from router import Router
    r = Router.load("artifacts/router.joblib")
    d = r.route("Prove that the sum of two even numbers is even.")
    print(d.model, d.p_cheap_ok)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import joblib


@dataclass
class Decision:
    model: str            # model name to call
    routed_to: str        # "cheap" or "strong"
    p_cheap_ok: float     # predicted probability the cheap model is good enough
    threshold: float


class Router:
    def __init__(self, pipeline, threshold: float, cheap_model: str, strong_model: str, meta: Optional[dict] = None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        self.meta = meta or {}

    @classmethod

    def load(cls, path: str) -> "Router":
        a = joblib.load(path)
        return cls(a["pipeline"], a["threshold"], a["cheap_model"], a["strong_model"], a.get("meta"))

    def route_batch(self, prompts: list[str]) -> list[Decision]:
        probs = self.pipeline.predict_proba(prompts)[:, 1]
        out = []
        for p in probs:
            cheap = p >= self.threshold
            out.append(
                Decision(
                    model=self.cheap_model if cheap else self.strong_model,
                    routed_to="cheap" if cheap else "strong",
                    p_cheap_ok=float(p),
                    threshold=self.threshold,
                )
            )
        return out

    def route(self, prompt: str) -> Decision:
        return self.route_batch([prompt])[0]

    def complete(
        self,
        prompt: str,
        call: Callable[[str, str], str],
        escalate_if: Optional[Callable[[str], bool]] = None,
    ) -> tuple[str, Decision]:
        """Route, call the chosen model, and optionally escalate to the strong model.

        call(model_name, prompt) -> response text   (your LLM client goes here)
        escalate_if(response) -> True if a cheap answer looks bad (empty, refusal,

        failed validation, ...). Escalation is the safety net for router mistakes.
        """
        d = self.route(prompt)
        text = call(d.model, prompt)
        if d.routed_to == "cheap" and escalate_if is not None and escalate_if(text):
            text = call(self.strong_model, prompt)
            d = Decision(self.strong_model, "strong", d.p_cheap_ok, d.threshold)
        return text, d