"""
agents/adaptation_agent.py
After each level, analyses flagged vs safe citizens to identify
how risk patterns are changing. Updates memory for the next level.
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from utils.langfuse_manager import get_client, get_callback_handler
from agents.memory_agent import MemoryAgent, RiskPattern


ADAPTATION_SYSTEM_PROMPT = """You are a health intelligence strategist analysing evolving wellbeing risk patterns.

Your job: compare flagged (high-risk) vs safe citizens to extract pattern rules that will help detect risk in future, harder levels.

The challenge increases in complexity across levels — patterns may drift, new signals may emerge.

Respond ONLY with valid JSON:
{
  "evolution_summary": "1-2 sentence summary of what drives risk in this level",
  "new_patterns": [
    {
      "pattern_id": "P001",
      "description": "...",
      "key_signals": ["declining PA trend", "rising env exposure", "..."],
      "pa_trend_threshold": -10.0,
      "sleep_trend_threshold": -5.0,
      "env_trend_threshold": 15.0,
      "requires_specialist": false,
      "age_group": "all",
      "confidence": 0.8
    }
  ],
  "watch_next_level": ["what to look for in the next, harder dataset"],
  "threshold_recommendation": 0.45
}
"""


class AdaptationAgent:
    def __init__(self, memory: MemoryAgent, session_id: str):
        self.memory = memory
        self.session_id = session_id
        self._pattern_counter = 0
        default_fast_model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("FAST_MODEL", default_fast_model),
            temperature=0.2, max_tokens=600,
        )

    def _next_pid(self):
        self._pattern_counter += 1
        return f"P{self._pattern_counter:03d}"

    def _compact_profile(self, p: dict) -> str:
        return (
            f"{p.get('citizen_id')} | age={p.get('age')} | "
            f"pa_trend={p.get('pa_trend'):+.1f} sleep_trend={p.get('sleep_trend'):+.1f} env_trend={p.get('env_trend'):+.1f} | "
            f"pa_recent={p.get('pa_recent')} sleep_recent={p.get('sleep_recent')} env_recent={p.get('env_recent')} | "
            f"specialist={p.get('specialist_count')} followup={p.get('followup_count')} types={p.get('unique_event_types')}"
        )

    @observe()
    def analyse_and_update(
        self,
        level: int,
        flagged_ids: list[str],
        safe_ids: list[str],
        profiles: dict,
        threshold_used: float,
    ):
        """Run post-level adaptation analysis and update memory."""
        print(f"\n[Adaptation] Analysing level {level} results...")
        handler = get_callback_handler()

        flagged_sample = "\n".join(
            self._compact_profile(profiles[c]) for c in flagged_ids[:5] if c in profiles
        ) or "None"
        safe_sample = "\n".join(
            self._compact_profile(profiles[c]) for c in safe_ids[:5] if c in profiles
        ) or "None"

        prior_context = self.memory.get_llm_context()

        prompt = f"""{prior_context}

=== LEVEL {level} RESULTS ===
Flagged citizens ({len(flagged_ids)} total) — sample:
{flagged_sample}

Safe citizens ({len(safe_ids)} total) — sample:
{safe_sample}

Analyse what differentiates flagged from safe. What patterns define risk at this level?
What should the system watch for in more complex future levels?
Keep recommendations concise and grounded in these numeric signals.
Respond ONLY with valid JSON.
"""
        messages = [
            SystemMessage(content=ADAPTATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        result = {}
        try:
            response = self.llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": self.session_id},
                },
            )
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
        except Exception as e:
            print(f"[Adaptation] Analysis failed: {e}")

        # Build pattern objects
        new_patterns = []
        for p in result.get("new_patterns", []):
            try:
                new_patterns.append(RiskPattern(
                    pattern_id=self._next_pid(),
                    description=p.get("description", ""),
                    key_signals=p.get("key_signals", []),
                    pa_trend_threshold=float(p.get("pa_trend_threshold", -5)),
                    sleep_trend_threshold=float(p.get("sleep_trend_threshold", -5)),
                    env_trend_threshold=float(p.get("env_trend_threshold", 10)),
                    requires_specialist=bool(p.get("requires_specialist", False)),
                    age_group=p.get("age_group", "all"),
                    confidence=float(p.get("confidence", 0.5)),
                ))
            except Exception:
                pass

        # Evolution notes
        summary = result.get("evolution_summary", "")
        for note in result.get("watch_next_level", []):
            self.memory.add_evolution_note(f"[L{level}→NEXT] {note}")

        # Threshold recommendation
        rec_threshold = result.get("threshold_recommendation")
        if rec_threshold:
            print(f"[Adaptation] Threshold recommendation for next level: {rec_threshold}")
            self.memory.add_evolution_note(f"[L{level} threshold rec] {rec_threshold}")

        # Persist to memory
        self.memory.record_level(
            level=level,
            flagged=flagged_ids,
            safe=safe_ids,
            summary=summary or f"Level {level} done. {len(flagged_ids)} flagged.",
            threshold_used=threshold_used,
            new_patterns=new_patterns,
        )

        print(f"[Adaptation] Summary: {summary}")
        print(f"[Adaptation] {len(new_patterns)} new patterns stored.")
        return result