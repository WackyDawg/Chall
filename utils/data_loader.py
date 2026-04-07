"""
utils/data_loader.py
Loads the MirrorLife dataset structure:
  - status.csv         (health monitoring events per citizen)
  - users.json         (citizen demographics)
  - locations.json     (GPS pings)
  - personas.md        (narrative citizen profiles — rich LLM context)
"""

import json
import zipfile
import csv
import re
from pathlib import Path
from collections import defaultdict
import statistics
import math


def load_level_data(zip_path: str) -> dict:
    """
    Unzip and load all dataset files. Returns a structured dict.
    """
    zip_path = Path(zip_path)
    extract_dir = zip_path.parent / zip_path.stem
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Recursively find all files, skip __MACOSX
    all_files = [f for f in extract_dir.rglob("*")
                 if f.is_file() and "__MACOSX" not in str(f)]

    data = {
        "status": [],
        "users": {},
        "locations": [],
        "personas": {},
        "raw_files": {},
    }

    for fpath in sorted(all_files):
        name = fpath.name.lower()
        print(f"[DataLoader] Reading: {fpath.name}")

        if "status" in name and name.endswith(".csv"):
            with open(fpath, newline="", encoding="utf-8") as f:
                data["status"] = list(csv.DictReader(f))

        elif "users" in name and name.endswith(".json"):
            with open(fpath, encoding="utf-8") as f:
                raw = json.load(f)
            data["users"] = {u["user_id"]: u for u in raw}

        elif "locations" in name and name.endswith(".json"):
            with open(fpath, encoding="utf-8") as f:
                data["locations"] = json.load(f)

        elif "personas" in name and name.endswith(".md"):
            data["personas"] = parse_personas_md(fpath.read_text(encoding="utf-8"))

        else:
            # Store anything extra as raw text/json for the LLM
            try:
                with open(fpath, encoding="utf-8") as f:
                    data["raw_files"][fpath.name] = f.read()
            except Exception:
                pass

    print(f"[DataLoader] Loaded: {len(data['status'])} status events, "
          f"{len(data['users'])} users, {len(data['locations'])} location pings, "
          f"{len(data['personas'])} personas")

    return data


def parse_personas_md(text: str) -> dict:
    """Parse personas.md into {citizen_id: persona_text} dict."""
    personas = {}
    # Split on ## headers that contain an ID (8-char uppercase)
    blocks = re.split(r'\n## ', text)
    for block in blocks:
        match = re.match(r'([A-Z]{8})', block.strip())
        if match:
            cid = match.group(1)
            personas[cid] = block.strip()
    return personas


def build_citizen_profiles(data: dict) -> dict:
    """
    Aggregate all data sources into one rich profile per citizen.
    Returns {citizen_id: CitizenProfile dict}
    """
    profiles = {}

    # Group status events by citizen
    citizen_events = defaultdict(list)
    for row in data["status"]:
        citizen_events[row["CitizenID"]].append(row)

    # Group location pings by citizen
    citizen_locs = defaultdict(list)
    for loc in data["locations"]:
        uid = loc.get("user_id") or loc.get("BioTag")
        if uid:
            citizen_locs[uid].append(loc)

    all_citizens = set(citizen_events.keys()) | set(data["users"].keys())

    for cid in all_citizens:
        user = data["users"].get(cid, {})
        events = sorted(citizen_events[cid], key=lambda x: x.get("Timestamp", ""))
        locs = citizen_locs.get(cid, [])
        persona = data["personas"].get(cid, "")

        # Compute derived features
        profile = _compute_features(cid, user, events, locs, persona)
        profiles[cid] = profile

    return profiles


def _compute_features(cid: str, user: dict, events: list, locs: list, persona: str) -> dict:
    """Compute numerical and qualitative features for one citizen."""
    age = 2026 - int(user.get("birth_year", 2000)) if user.get("birth_year") else None

    # --- Status-based features ---
    pa = [float(e["PhysicalActivityIndex"]) for e in events if e.get("PhysicalActivityIndex")]
    sleep = [float(e["SleepQualityIndex"]) for e in events if e.get("SleepQualityIndex")]
    env = [float(e["EnvironmentalExposureLevel"]) for e in events if e.get("EnvironmentalExposureLevel")]

    def trend(vals):
        if len(vals) < 4:
            return 0.0
        early = statistics.mean(vals[:len(vals)//2])
        late = statistics.mean(vals[len(vals)//2:])
        return late - early

    def recent_avg(vals, n=3):
        return statistics.mean(vals[-n:]) if len(vals) >= n else (statistics.mean(vals) if vals else 0)

    def volatility(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    event_types = [e.get("EventType", "") for e in events]
    unique_event_types = set(event_types)
    # Escalation indicator: specialist or follow-up events suggest health concern
    has_specialist = any("specialist" in t.lower() for t in event_types)
    has_followup = any("follow-up" in t.lower() for t in event_types)
    specialist_count = sum(1 for t in event_types if "specialist" in t.lower())
    followup_count = sum(1 for t in event_types if "follow-up" in t.lower())

    # Recent (last 3) vs early (first 3) comparisons
    pa_trend = trend(pa)
    sleep_trend = trend(sleep)
    env_trend = trend(env)
    pa_recent = recent_avg(pa)
    sleep_recent = recent_avg(sleep)
    env_recent = recent_avg(env)

    # Minimum values (rock-bottom signals)
    pa_min = min(pa) if pa else 0
    sleep_min = min(sleep) if sleep else 0

    # --- Location-based features ---
    n_locs = len(locs)
    if locs:
        lat_vals = [l["lat"] for l in locs]
        lng_vals = [l["lng"] for l in locs]
        try:
            lat_std = statistics.stdev(lat_vals) if len(lat_vals) > 1 else 0.0
            lng_std = statistics.stdev(lng_vals) if len(lng_vals) > 1 else 0.0
        except Exception:
            lat_std = lng_std = 0.0
        geo_spread = math.sqrt(lat_std**2 + lng_std**2)
        unique_cities = len(set(l.get("city", "") for l in locs))
    else:
        geo_spread = 0.0
        unique_cities = 0

    return {
        "citizen_id": cid,
        "age": age,
        "job": user.get("job", "Unknown"),
        "city": user.get("residence", {}).get("city", "Unknown"),
        "persona_text": persona,

        # Trend signals (negative = declining)
        "pa_trend": round(pa_trend, 2),
        "sleep_trend": round(sleep_trend, 2),
        "env_trend": round(env_trend, 2),  # rising = more exposure = worse

        # Recent absolute values
        "pa_recent": round(pa_recent, 2),
        "sleep_recent": round(sleep_recent, 2),
        "env_recent": round(env_recent, 2),

        # Worst readings
        "pa_min": pa_min,
        "sleep_min": sleep_min,
        "env_max": max(env) if env else 0,

        # Volatility
        "pa_volatility": round(volatility(pa), 2),
        "sleep_volatility": round(volatility(sleep), 2),

        # Event escalation signals
        "has_specialist": has_specialist,
        "has_followup": has_followup,
        "specialist_count": specialist_count,
        "followup_count": followup_count,
        "unique_event_types": len(unique_event_types),
        "total_events": len(events),

        # Mobility
        "geo_spread": round(geo_spread, 4),
        "unique_cities": unique_cities,
        "location_pings": n_locs,

        # Raw series for LLM context
        "pa_series": pa,
        "sleep_series": sleep,
        "env_series": env,
        "event_types": event_types,
    }


def profile_to_text(profile: dict) -> str:
    """Convert a citizen profile to a concise text block for LLM analysis."""
    lines = [
        f"CitizenID: {profile['citizen_id']}",
        f"Age: {profile['age']} | Job: {profile['job']} | City: {profile['city']}",
        f"Physical Activity — recent avg: {profile['pa_recent']}, trend: {profile['pa_trend']:+.1f}, min: {profile['pa_min']}, volatility: {profile['pa_volatility']}",
        f"Sleep Quality     — recent avg: {profile['sleep_recent']}, trend: {profile['sleep_trend']:+.1f}, min: {profile['sleep_min']}, volatility: {profile['sleep_volatility']}",
        f"Environmental Exp — recent avg: {profile['env_recent']}, trend: {profile['env_trend']:+.1f}, max: {profile['env_max']}",
        f"Event escalation: specialist_visits={profile['specialist_count']}, followups={profile['followup_count']}, unique_event_types={profile['unique_event_types']}",
        f"Mobility: geo_spread={profile['geo_spread']}, unique_cities={profile['unique_cities']}, location_pings={profile['location_pings']}",
        f"PA series:    {profile['pa_series']}",
        f"Sleep series: {profile['sleep_series']}",
        f"Env series:   {profile['env_series']}",
        f"Event types:  {profile['event_types']}",
    ]
    if profile.get("persona_text"):
        lines.append(f"\nPersona background:\n{profile['persona_text'][:600]}")
    return "\n".join(lines)