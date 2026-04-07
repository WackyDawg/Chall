"""
agents/memory_agent.py
Persists cross-level knowledge: what risk patterns look like,
how they evolve across levels, and which citizens were flagged before.
"""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class RiskPattern:
    pattern_id: str
    description: str
    key_signals: list[str] = field(default_factory=list)
    # Numerical thresholds learned from prior levels
    pa_trend_threshold: float = -5.0
    sleep_trend_threshold: float = -5.0
    env_trend_threshold: float = 10.0
    requires_specialist: bool = False
    age_group: str = "all"          # "elderly", "working_age", "all"
    confidence: float = 0.5
    seen_in_levels: list[int] = field(default_factory=list)


class MemoryAgent:
    def __init__(self):
        self.risk_patterns: list[RiskPattern] = []
        self.level_summaries: dict[int, str] = {}
        self.flagged_citizens: dict[str, list[int]] = {}   # {citizen_id: [levels]}
        self.safe_citizens: dict[str, list[int]] = {}
        self.evolution_notes: list[str] = []
        self.threshold_history: dict[int, float] = {}
        self.current_level: int = 0

    def record_level(self, level: int, flagged: list[str], safe: list[str],
                     summary: str, threshold_used: float,
                     new_patterns: list[RiskPattern] | None = None):
        self.current_level = level
        self.threshold_history[level] = threshold_used
        self.level_summaries[level] = summary

        for cid in flagged:
            self.flagged_citizens.setdefault(cid, []).append(level)
        for cid in safe:
            self.safe_citizens.setdefault(cid, []).append(level)

        if new_patterns:
            for p in new_patterns:
                p.seen_in_levels.append(level)
                self.risk_patterns.append(p)

        print(f"[Memory] Level {level} stored. Flagged: {len(flagged)}, Safe: {len(safe)}")

    def add_evolution_note(self, note: str):
        self.evolution_notes.append(note)

    def get_llm_context(self) -> str:
        lines = ["=== WELLBEING RISK INTELLIGENCE (from prior levels) ==="]

        if self.risk_patterns:
            lines.append(f"\n[Known Risk Patterns] ({len(self.risk_patterns)})")
            for p in sorted(self.risk_patterns, key=lambda x: -x.confidence)[:8]:
                lines.append(f"  • [{p.pattern_id}] {p.description} (conf={p.confidence:.2f})")
                if p.key_signals:
                    lines.append(f"    Signals: {', '.join(p.key_signals)}")

        if self.evolution_notes:
            lines.append("\n[Pattern Evolution Notes]")
            for note in self.evolution_notes[-6:]:
                lines.append(f"  • {note}")

        if self.level_summaries:
            lines.append("\n[Level Summaries]")
            for lvl, s in sorted(self.level_summaries.items()):
                lines.append(f"  Level {lvl}: {s}")

        lines.append(f"\n[Stats] Current level: {self.current_level} | "
                     f"Distinct citizens flagged across all levels: {len(self.flagged_citizens)}")
        return "\n".join(lines)

    def was_previously_flagged(self, cid: str) -> bool:
        return cid in self.flagged_citizens

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "risk_patterns": [asdict(p) for p in self.risk_patterns],
                "level_summaries": self.level_summaries,
                "flagged_citizens": self.flagged_citizens,
                "safe_citizens": self.safe_citizens,
                "evolution_notes": self.evolution_notes,
                "threshold_history": {str(k): v for k, v in self.threshold_history.items()},
                "current_level": self.current_level,
            }, f, indent=2)
        print(f"[Memory] Saved → {path}")

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.risk_patterns = [RiskPattern(**p) for p in d.get("risk_patterns", [])]
        self.level_summaries = {int(k): v for k, v in d.get("level_summaries", {}).items()}
        self.flagged_citizens = d.get("flagged_citizens", {})
        self.safe_citizens = d.get("safe_citizens", {})
        self.evolution_notes = d.get("evolution_notes", [])
        self.threshold_history = {int(k): v for k, v in d.get("threshold_history", {}).items()}
        self.current_level = d.get("current_level", 0)
        print(f"[Memory] Loaded (level {self.current_level})")