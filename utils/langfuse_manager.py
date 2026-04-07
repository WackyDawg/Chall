"""
utils/langfuse_manager.py
All Langfuse session tracking. Required by the challenge for scoring/validation.
"""

import os
import re
import ulid
from langfuse import Langfuse, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

_client: Langfuse | None = None

def get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        )
    return _client

def generate_session_id() -> str:
    """Unique session ID in required format: TEAMNAME-ULID"""
    team_raw = os.getenv("TEAM_NAME", "mirroreye")
    team = re.sub(r"[^A-Za-z0-9]", "", team_raw)
    if not team:
        team = "mirroreye"
    return f"{team}-{ulid.new().str}"

def get_callback_handler() -> CallbackHandler:
    return CallbackHandler()

def update_session(session_id: str):
    client = get_client()
    if hasattr(client, "update_current_trace"):
        client.update_current_trace(session_id=session_id)
        return
    if hasattr(client, "update_current_observation"):
        client.update_current_observation(session_id=session_id)

@observe()
def _session_marker():
    return "ok"


def ensure_session_trace(session_id: str):
    with propagate_attributes(session_id=session_id):
        _session_marker()


def flush():
    get_client().flush()
    print("[Langfuse] ✓ Traces flushed")

def get_session_cost(session_id: str) -> float:
    client = get_client()
    try:
        traces, page = [], 1
        while True:
            resp = client.api.trace.list(session_id=session_id, limit=100, page=page)
            if not resp.data:
                break
            traces.extend(resp.data)
            if len(resp.data) < 100:
                break
            page += 1
        total = 0.0
        for trace in traces:
            detail = client.api.trace.get(trace.id)
            if detail and hasattr(detail, "observations"):
                for obs in detail.observations:
                    if getattr(obs, "type", None) == "GENERATION":
                        cost = getattr(obs, "calculated_total_cost", None)
                        if cost:
                            total += cost
        return total
    except Exception as e:
        print(f"[Langfuse] Cost query failed: {e}")
        return 0.0

def print_cost_summary(session_id: str):
    cost = get_session_cost(session_id)
    print(f"\n[Langfuse] Session: {session_id}")
    print(f"[Langfuse] Estimated spend: ${cost:.4f}\n")