"""Layer 4: LLM-based training analyzer — called RARELY (3-5× per 100 epochs).

Receives a compact summary from the Summarizer and returns a structured
JSON decision: continue / kill_and_retry / kill_and_stop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger("torchmill.brain")

SYSTEM_PROMPT = """\
You are an expert ML training analyst monitoring a live training run.
You receive compact metrics summaries and must decide what to do next.

Respond with VALID JSON ONLY — no markdown fences, no extra text:
{
  "action": "continue",
  "reason": "concise explanation",
  "changes": {}
}

Actions:
  "continue"       — training looks healthy, let it run
  "kill_and_retry" — stop training and restart with new hyperparameters
  "kill_and_stop"  — stop all rounds; no more retries warranted

The "changes" dict (used only with "kill_and_retry") maps hyperparameter
names to new values. Common keys: lr, dropout, weight_decay, batch_size,
hidden_dim, num_layers. Only include keys you want to change.

Decision guidelines:
- Recommend "kill_and_retry" for overfitting, sustained loss plateau,
  gradient explosion, or clearly bad hyperparameters.
- Recommend "continue" when training is progressing or it's too early to judge.
- Recommend "kill_and_stop" only when further retries are clearly futile
  (e.g., after many rounds with no improvement).
- Be conservative — only intervene when there is clear evidence of a problem.
- Keep the "reason" field to 1-2 sentences.
"""


class BrainAnalyzer:
    """Calls LLM provider and parses structured JSON response."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from .providers import get_client
            self._client = get_client(self.provider, self.api_key)
        return self._client

    def analyze(self, summary: str) -> Dict[str, Any]:
        """Send summary to LLM brain and return parsed action dict."""
        logger.info(
            f"Calling LLM brain ({self.provider}/{self.model}) — "
            f"{len(summary)} chars"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": summary},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            logger.debug(f"Brain raw response: {raw}")
            return self._parse(raw)
        except Exception as exc:
            logger.error(f"Brain LLM call failed: {exc}")
            return {
                "action": "continue",
                "reason": f"LLM unavailable: {exc}",
                "changes": {},
            }

    def _parse(self, raw: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        # Strip markdown code fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence:
            raw = fence.group(1)

        # Try full parse
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to extracting the first {...} blob
            blob = re.search(r"\{[\s\S]*\}", raw)
            if blob:
                try:
                    data = json.loads(blob.group(0))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        action = data.get("action", "continue")
        if action not in ("continue", "kill_and_retry", "kill_and_stop"):
            logger.warning(f"Unknown action '{action}' from brain — defaulting to continue")
            action = "continue"

        return {
            "action": action,
            "reason": str(data.get("reason", "No reason provided")),
            "changes": dict(data.get("changes", {})),
        }
