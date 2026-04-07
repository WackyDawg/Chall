"""
agents/coordinator.py
Top-level orchestrator. Runs the full pipeline per level:
  1. Load & feature-engineer data
  2. Rule-based pre-scoring (no tokens)
  3. LLM assessment agent (per citizen)
  4. Make final decisions
  5. Write output file
  6. Run adaptation analysis → update memory
"""

import os
import re
from pathlib import Path

from utils.langfuse_manager import generate_session_id, ensure_session_trace, flush, print_cost_summary
from utils.data_loader import load_level_data, build_citizen_profiles
from agents.memory_agent import MemoryAgent
from agents.rule_scorer import apply_rules_to_profiles
from agents.assessment_agent import AssessmentAgent
from agents.adaptation_agent import AdaptationAgent


class Coordinator:
    def __init__(self, output_dir: str = "output"):
        self.session_id = generate_session_id()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.memory_path = self.output_dir / "memory.json"

        self.memory = MemoryAgent()
        self.assessor = AssessmentAgent(self.memory, self.session_id)
        self.adapter = AdaptationAgent(self.memory, self.session_id)

        if self.memory_path.exists():
            self.memory.load(str(self.memory_path))

        print(f"\n{'='*60}")
        print(f"  MirrorEye — Wellbeing Intelligence System")
        print(f"  Session: {self.session_id}")
        print(f"{'='*60}\n")

    def process_level(
        self,
        level: int,
        zip_path: str,
        use_strong: bool = False,
    ) -> tuple[str, str]:
        """
        Full pipeline for one dataset level.
        Returns (session_id, output_file_path).
        """
        print(f"\n{'='*60}")
        print(f"  LEVEL {level} | {'STRONG' if use_strong else 'FAST'} model")
        print(f"{'='*60}")

        # 1. Load data
        data = load_level_data(zip_path)
        profiles = build_citizen_profiles(data)
        print(f"\n[Coordinator] {len(profiles)} citizens to assess")

        # 2. Rule scoring (free — no tokens)
        print("\n[Coordinator] Running rule scorer...")
        rule_scores = apply_rules_to_profiles(profiles)

        # 3. LLM assessment
        locked_l1_mode = level == 1 and os.getenv("L1_LOCKED_MODE", "true").lower() in {"1", "true", "yes"}
        if locked_l1_mode:
            print("[Coordinator] L1 locked mode enabled")
        assessments = self.assessor.assess_all_citizens(
            profiles=profiles,
            rule_scores=rule_scores,
            use_strong=use_strong,
            deterministic_rules_only=locked_l1_mode,
            level=level,
        )

        # 4. Final decisions
        threshold = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.45"))
        flagged, safe = self.assessor.make_final_decisions(assessments)

        print(f"\n[Coordinator] ✓ FLAGGED ({len(flagged)}): {flagged}")
        print(f"[Coordinator] ✓ SAFE    ({len(safe)}): {safe}")

        # 5. Write output
        output_path = self.output_dir / f"level_{level}_output.txt"
        self._write_output(flagged, output_path)

        # 6. Write detailed results log (for debugging / tuning)
        self._write_detail_log(assessments, level)

        # 7. Adaptation analysis (optional; costly, mainly useful across harder levels)
        enable_adaptation = os.getenv("ENABLE_ADAPTATION", "").lower() in {"1", "true", "yes"}
        if level > 1 or enable_adaptation:
            self.adapter.analyse_and_update(
                level=level,
                flagged_ids=flagged,
                safe_ids=safe,
                profiles=profiles,
                threshold_used=threshold,
            )
        else:
            print("[Adaptation] Skipped for L1 (set ENABLE_ADAPTATION=true to enable)")

        # 8. Persist memory + flush Langfuse
        ensure_session_trace(self.session_id)
        self.memory.save(str(self.memory_path))
        flush()
        print_cost_summary(self.session_id)

        print(f"\n{'='*60}")
        print(f"  Level {level} complete!")
        print(f"  Output:     {output_path}")
        print(f"  Session ID: {self.session_id}  ← paste into submission modal")
        print(f"{'='*60}\n")

        return self.session_id, str(output_path)

    def _write_output(self, flagged: list[str], path: Path):
        """Write required ASCII output file — one Citizen ID per line."""
        safe_ids = sorted({
            str(cid).strip().upper()
            for cid in flagged
            if isinstance(cid, str) and re.fullmatch(r"[A-Za-z0-9]{8}", str(cid).strip())
        })
        content = "\n".join(safe_ids)
        if content:
            content += "\n"
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(content)
        print(f"[Coordinator] Output written → {path}")

    def _write_detail_log(self, assessments: dict, level: int):
        """Write human-readable detail log for debugging."""
        log_path = self.output_dir / f"level_{level}_detail.txt"
        with open(log_path, "w") as f:
            f.write(f"=== Level {level} Assessment Detail ===\n\n")
            threshold = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.45"))
            for cid, a in sorted(assessments.items(),
                                  key=lambda x: -x[1]["blended_score"]):
                flag = "FLAGGED" if a["blended_score"] >= threshold else "safe"
                f.write(f"{cid} | {flag} | blended={a['blended_score']:.3f} "
                        f"llm={a['llm_risk_score']} rule={a['rule_score']:.3f}\n")
                f.write(f"  Reasoning: {a['reasoning']}\n\n")

    def get_session_id(self) -> str:
        return self.session_id
