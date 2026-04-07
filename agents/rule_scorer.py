"""
agents/rule_scorer.py
Zero-token heuristic pre-scorer for wellbeing risk.
Produces a float 0.0-1.0 per citizen based on computed features.
Calibrated against the sandbox data patterns.
"""

from utils.data_loader import build_citizen_profiles


def rule_score(profile: dict) -> float:
    """
    Compute a heuristic risk score for a citizen profile.
    Returns 0.0 (very healthy) to 1.0 (high risk).

    Key insight from sandbox data analysis:
    - WNACROYX (Craig Connor) is the clear positive case:
        PA trend: -26.3, Sleep trend: -13.3, Env trend: +32.0
        Has specialist + follow-up events, escalating event types
    - Others are stable with small fluctuations
    """
    score = 0.0

    # --- Physical Activity decline (most important signal) ---
    pa_trend = profile.get("pa_trend", 0)
    if pa_trend <= -20:
        score += 0.40
    elif pa_trend <= -10:
        score += 0.25
    elif pa_trend <= -5:
        score += 0.12

    # Recent PA absolute low
    pa_recent = profile.get("pa_recent", 50)
    if pa_recent < 25:
        score += 0.20
    elif pa_recent < 35:
        score += 0.10

    # --- Sleep Quality decline ---
    sleep_trend = profile.get("sleep_trend", 0)
    if sleep_trend <= -10:
        score += 0.20
    elif sleep_trend <= -5:
        score += 0.10

    sleep_recent = profile.get("sleep_recent", 50)
    if sleep_recent < 30:
        score += 0.15
    elif sleep_recent < 40:
        score += 0.08

    # --- Environmental Exposure rising (bad) ---
    env_trend = profile.get("env_trend", 0)
    if env_trend >= 30:
        score += 0.20
    elif env_trend >= 15:
        score += 0.10
    elif env_trend >= 5:
        score += 0.04

    env_recent = profile.get("env_recent", 30)
    if env_recent >= 80:
        score += 0.20
    elif env_recent >= 65:
        score += 0.10

    # --- Medical escalation events (very strong signal) ---
    if profile.get("has_specialist"):
        score += 0.20 * min(profile.get("specialist_count", 1), 3) / 3
    if profile.get("has_followup"):
        score += 0.15 * min(profile.get("followup_count", 1), 3) / 3

    # More unique event types = more clinical attention needed
    unique_types = profile.get("unique_event_types", 3)
    if unique_types >= 5:
        score += 0.15
    elif unique_types >= 4:
        score += 0.08

    # --- Volatility in PA (instability) ---
    pa_vol = profile.get("pa_volatility", 0)
    if pa_vol > 8:
        score += 0.05

    # --- Compound-signal bonus (improves L2/L3 recall without over-flagging) ---
    decline_signals = 0
    if pa_trend <= -8:
        decline_signals += 1
    if sleep_trend <= -6:
        decline_signals += 1
    if env_trend >= 12:
        decline_signals += 1
    if decline_signals >= 2:
        score += 0.08

    # --- Age factor (elderly amplify risk slightly) ---
    age = profile.get("age") or 0
    if age >= 80:
        score += 0.05  # not dominant — stable elderly still score low

    # --- Mobility change signal (reduced mobility may signal decline) ---
    geo_spread = profile.get("geo_spread", 0)
    job = (profile.get("job") or "").lower()
    # A driver with low geo spread = mobility collapse
    if "driver" in job or "delivery" in job:
        if geo_spread < 5:
            score += 0.10

    return min(score, 1.0)


def apply_rules_to_profiles(profiles: dict) -> dict:
    """
    Returns {citizen_id: rule_score} for all citizens.
    """
    scores = {}
    for cid, profile in profiles.items():
        s = rule_score(profile)
        scores[cid] = s
        print(f"  [RuleScorer] {cid}: {s:.3f}")
    return scores
