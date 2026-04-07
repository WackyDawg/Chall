"""
agents/assessment_agent.py
LLM agent that deeply analyses each citizen's wellbeing trajectory.
Uses persona text + longitudinal data + prior level context.
Outputs a risk probability and reasoning per citizen.
"""

import os
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from utils.langfuse_manager import get_callback_handler, update_session
from agents.memory_agent import MemoryAgent


SYSTEM_PROMPT = """You are an expert wellbeing intelligence analyst for MirrorLife in Reply Mirror (year 2087).

Your task: review citizen health monitoring data and determine whether each citizen needs a PREVENTIVE SUPPORT PATHWAY ACTIVATED.

OUTPUT = 1 (activate support) when a citizen shows:
- Declining physical activity (sustained downward trend)
- Deteriorating sleep quality
- Increasing environmental exposure risk
- Escalating medical event types (specialist visits, follow-up assessments appearing)
- Behavioral withdrawal signals (reduced mobility, social isolation)
- Compound risk: multiple mild signals together are more dangerous than one strong signal

OUTPUT = 0 (standard monitoring) when:
- Metrics are stable (small fluctuations are normal)
- Any decline is minor and not sustained
- Event types remain routine (check-ups and coaching only)

CRITICAL BALANCE — F1 score is the metric:
- Missing a high-risk citizen (false negative) = missed preventive opportunity
- Flagging healthy citizens (false positive) = resource waste and system credibility loss
- Aim for balanced precision and recall — do NOT flag everyone

Respond ONLY with valid JSON, no prose outside the JSON:
{
  "assessments": [
    {
      "citizen_id": "XXXXXXXX",
      "risk_score": 0.0,
      "decision": 0,
      "reasoning": "brief evidence-based reasoning (2-3 sentences max)"
    }
  ]
}
"""


class AssessmentAgent:
    def __init__(self, memory: MemoryAgent, session_id: str):
        self.memory = memory
        self.session_id = session_id
        api_key = self._clean_env("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"

        default_fast_model = self._resolve_model("OPENROUTER_MODEL", "gpt-4o-mini")
        default_strong_model = self._resolve_model("OPENROUTER_MODEL", "gpt-4o")
        self.llm_enabled = bool(api_key)
        if not self.llm_enabled:
            print("[Assessment] OPENROUTER_API_KEY missing/invalid. Falling back to rules for all citizens.")

        self.fast_llm = ChatOpenAI(
            api_key=api_key, base_url=base_url,
            model=self._resolve_model("FAST_MODEL", default_fast_model),
            temperature=0.1, max_tokens=700,
        )
        self.strong_llm = ChatOpenAI(
            api_key=api_key, base_url=base_url,
            model=self._resolve_model("STRONG_MODEL", default_strong_model),
            temperature=0.1, max_tokens=3000,
        )
        self.threshold = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.45"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.rule_safe_cutoff = float(os.getenv("RULE_SAFE_CUTOFF", "0.20"))
        self.rule_flag_cutoff = float(os.getenv("RULE_FLAG_CUTOFF", "0.75"))

    def _clean_env(self, key: str, default: str = "") -> str:
        raw = os.getenv(key, default)
        if raw is None:
            return default
        cleaned = str(raw).replace("\\n", "\n").strip()
        if "\n" in cleaned:
            cleaned = cleaned.splitlines()[0].strip()
        return cleaned

    def _resolve_model(self, key: str, fallback: str) -> str:
        model = self._clean_env(key, fallback)
        if not model or model.lower() in {"openrouter/free", "free"}:
            return fallback
        return model

    def _normalize_citizen_id(self, cid: str | None) -> str | None:
        if cid is None:
            return None
        norm = str(cid).strip().upper()
        return norm if re.fullmatch(r"[A-Z0-9]{8}", norm) else None

    def _compact_profile_text(self, profile: dict) -> str:
        return (
            f"CitizenID: {profile['citizen_id']}\n"
            f"Age: {profile['age']} | Job: {profile['job']} | City: {profile['city']}\n"
            f"PA recent={profile['pa_recent']} trend={profile['pa_trend']:+.1f} min={profile['pa_min']}\n"
            f"Sleep recent={profile['sleep_recent']} trend={profile['sleep_trend']:+.1f} min={profile['sleep_min']}\n"
            f"Env recent={profile['env_recent']} trend={profile['env_trend']:+.1f} max={profile['env_max']}\n"
            f"Escalation specialist={profile['specialist_count']} followup={profile['followup_count']} unique_types={profile['unique_event_types']}\n"
            f"Mobility geo_spread={profile['geo_spread']} unique_cities={profile['unique_cities']}"
        )

    def _extract_assessments(self, content) -> list[dict]:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            text = "".join(parts).strip()
        else:
            text = str(content).strip()

        if text.startswith("```"):
            chunks = text.split("```")
            candidate = ""
            for chunk in chunks:
                c = chunk.strip()
                if ("{" in c and "}" in c) or ("[" in c and "]" in c):
                    candidate = c
                    break
            text = candidate or text
            if text.startswith("json"):
                text = text[4:].strip()

        if not text.startswith("{") and not text.startswith("["):
            dict_start, dict_end = text.find("{"), text.rfind("}")
            list_start, list_end = text.find("["), text.rfind("]")
            candidates = []
            if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
                candidates.append(text[dict_start:dict_end + 1])
            if list_start != -1 and list_end != -1 and list_end > list_start:
                candidates.append(text[list_start:list_end + 1])
            if candidates:
                text = max(candidates, key=len)

        result = json.loads(text)
        if isinstance(result, dict):
            assessments = result.get("assessments", [])
            return assessments if isinstance(assessments, list) else []
        if isinstance(result, list):
            return result
        return []

    @observe()
    def _call_llm(self, citizen_texts: list[str], use_strong: bool = False) -> list[dict]:
        """Single LLM call for a batch of citizens."""
        handler = get_callback_handler()
        llm = self.strong_llm if use_strong else self.fast_llm

        memory_context = self.memory.get_llm_context()
        citizens_block = "\n\n---\n\n".join(citizen_texts)

        user_msg = f"""{memory_context}

=== CITIZENS TO ASSESS ===

{citizens_block}

Assess each citizen above. Respond ONLY with valid JSON.
"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            update_session(self.session_id)
            response = llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": self.session_id},
                },
            )
            return self._extract_assessments(response.content)
        except Exception as e:
            print(f"[Assessment] LLM call failed: {e}")
            return []

    def assess_all_citizens(
        self,
        profiles: dict,
        rule_scores: dict,
        use_strong: bool = False,
        deterministic_rules_only: bool = False,
        level: int | None = None,
    ) -> dict:
        """
        Assess all citizens. Returns {citizen_id: assessment_dict}.
        Uses rule scores to prioritise and optionally skip obvious cases.
        """
        print(f"\n[Assessment] Assessing {len(profiles)} citizens...")

        if deterministic_rules_only or not self.llm_enabled:
            print("[Assessment] Locked deterministic mode: rules-only (LLM bypassed)")
            all_assessments = {}
            for cid in sorted(profiles.keys(), key=lambda c: rule_scores.get(c, 0), reverse=True):
                rule_s = rule_scores.get(cid, 0)
                all_assessments[cid] = {
                    "citizen_id": cid,
                    "llm_risk_score": None,
                    "rule_score": rule_s,
                    "blended_score": round(rule_s, 4),
                    "decision": 1 if rule_s >= self.threshold else 0,
                    "reasoning": "Locked deterministic rule decision",
                }
            return all_assessments

        all_assessments = {}
        llm_candidates = []

        effective_safe_cutoff = self.rule_safe_cutoff
        effective_flag_cutoff = self.rule_flag_cutoff
        if level is not None and level >= 2:
            effective_safe_cutoff = float(os.getenv("RULE_SAFE_CUTOFF_L2PLUS", "0.08"))
            effective_flag_cutoff = float(os.getenv("RULE_FLAG_CUTOFF_L2PLUS", str(self.rule_flag_cutoff)))

        sorted_citizens = sorted(profiles.keys(),
                                 key=lambda c: rule_scores.get(c, 0), reverse=True)

        for cid in sorted_citizens:
            rule_s = rule_scores.get(cid, 0)
            if rule_s >= effective_flag_cutoff:
                all_assessments[cid] = {
                    "citizen_id": cid,
                    "llm_risk_score": None,
                    "rule_score": rule_s,
                    "blended_score": round(rule_s, 4),
                    "decision": 1,
                    "reasoning": "High-confidence rule signal",
                }
                print(f"  {cid}: blended={rule_s:.3f} ⚠️  FLAGGED | High-confidence rule signal")
            elif rule_s <= effective_safe_cutoff:
                all_assessments[cid] = {
                    "citizen_id": cid,
                    "llm_risk_score": None,
                    "rule_score": rule_s,
                    "blended_score": round(rule_s, 4),
                    "decision": 0,
                    "reasoning": "High-confidence stable rule signal",
                }
                print(f"  {cid}: blended={rule_s:.3f} ✓  safe | High-confidence stable rule signal")
            else:
                llm_candidates.append(cid)

        if not llm_candidates:
            return all_assessments

        batches = [llm_candidates[i:i+self.batch_size]
                   for i in range(0, len(llm_candidates), self.batch_size)]

        for i, batch in enumerate(batches):
            print(f"[Assessment] LLM batch {i+1}/{len(batches)} "
                  f"({'strong' if use_strong else 'fast'})...")
            texts = [self._compact_profile_text(profiles[cid]) for cid in batch]
            assessments = self._call_llm(texts, use_strong=use_strong)

            for item in assessments:
                cid = self._normalize_citizen_id(item.get("citizen_id"))
                if cid and cid in profiles:
                    llm_score = float(item.get("risk_score", 0) or 0)
                    llm_score = min(max(llm_score, 0.0), 1.0)
                    rule_s = rule_scores.get(cid, 0)
                    blended = 0.70 * llm_score + 0.30 * rule_s

                    profile = profiles.get(cid, {})
                    no_escalation = (
                        not profile.get("has_specialist")
                        and not profile.get("has_followup")
                        and profile.get("unique_event_types", 0) <= 3
                    )
                    if rule_s < 0.20 and llm_score > 0.70 and no_escalation:
                        blended = min(blended, 0.35)

                    if level is not None and level >= 2:
                        low_rule_guard = float(os.getenv("L2PLUS_LOW_RULE_FP_GUARD", "0.12"))
                        llm_spike_guard = float(os.getenv("L2PLUS_LLM_SPIKE_GUARD", "0.55"))
                        blended_cap = float(os.getenv("L2PLUS_BLENDED_CAP", "0.42"))
                        if rule_s <= low_rule_guard and llm_score >= llm_spike_guard and no_escalation:
                            blended = min(blended, blended_cap)
                        if rule_s < 0.18 and llm_score < 0.65 and blended >= self.threshold:
                            blended = min(blended, self.threshold - 0.01)

                    all_assessments[cid] = {
                        "citizen_id": cid,
                        "llm_risk_score": llm_score,
                        "rule_score": rule_s,
                        "blended_score": round(blended, 4),
                        "decision": item.get("decision", 0),
                        "reasoning": item.get("reasoning", ""),
                    }
                    status = "⚠️  FLAGGED" if blended >= self.threshold else "✓  safe"
                    print(f"  {cid}: blended={blended:.3f} {status} | {item.get('reasoning','')[:80]}")

            for cid in batch:
                if cid not in all_assessments:
                    rule_s = rule_scores.get(cid, 0)
                    all_assessments[cid] = {
                        "citizen_id": cid,
                        "llm_risk_score": None,
                        "rule_score": rule_s,
                        "blended_score": round(rule_s, 4),
                        "decision": 1 if rule_s >= self.threshold else 0,
                        "reasoning": "Rule-based fallback (LLM did not return assessment)",
                    }

        return all_assessments

    def make_final_decisions(self, assessments: dict) -> tuple[list[str], list[str]]:
        """
        Apply threshold to blended scores.
        Returns (flagged_ids, safe_ids).
        """
        flagged, safe = [], []
        for cid, a in assessments.items():
            if a["blended_score"] >= self.threshold:
                flagged.append(cid)
            else:
                safe.append(cid)
        return sorted(flagged), sorted(safe)
