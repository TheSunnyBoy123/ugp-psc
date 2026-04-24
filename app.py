"""
app.py — Streamlit UI for Perovskite Solar Cell Research Chatbot

Tabs: Chat | Data Explorer | Domain Knowledge
Router: intent → disambiguation → plan → execute → explain
"""

import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Ensure chatbot/ is on sys.path ──
_CHATBOT_DIR = str(Path(__file__).resolve().parent)
if _CHATBOT_DIR not in sys.path:
    sys.path.insert(0, _CHATBOT_DIR)

load_dotenv(os.path.join(_CHATBOT_DIR, ".env"))

from data_engine import DataEngine
from intent_classifier import classify_intent
from disambiguation import check_disambiguation
import query_planner
import query_executor
from design_pipeline import run_design_pipeline
from materials_project_client import MaterialsProjectClient
from regression_engine import RegressionEngine

# Try to import Local LLM client (optional)
_LOCAL_LLM_AVAILABLE = False
try:
    from local_llm_client import LocalLLMClient
    _LOCAL_LLM_AVAILABLE = True
except ImportError:
    pass

# Try to import SSH LLM client (optional)
_SSH_LLM_AVAILABLE = False
_SSH_CLIENT_INSTANCE = None
try:
    from ssh_llm_client import SSHLLMClient
    _SSH_LLM_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PSC Research Agent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(_CHATBOT_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Logging setup ───────────────────────────────────────────────────────────
# Emit detailed, coloured-style structured logs to the terminal so that every
# step of the pipeline is visible when running `streamlit run app.py`.

class _PipelineFormatter(logging.Formatter):
    """Custom formatter: colour-coded level tag + timestamp + message."""

    LEVEL_COLOURS = {
        logging.DEBUG:    "\033[96m",   # Cyan
        logging.INFO:     "\033[92m",   # Green
        logging.WARNING:  "\033[93m",   # Yellow
        logging.ERROR:    "\033[91m",   # Red
        logging.CRITICAL: "\033[95m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        level_tag = f"{colour}{self.BOLD}[{record.levelname:8s}]{self.RESET}"
        ts = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
        module = f"\033[90m{record.module}\033[0m"  # dim grey
        msg = record.getMessage()
        formatted = f"{ts} {level_tag} {module}: {msg}"
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def _configure_logger() -> logging.Logger:
    _logger = logging.getLogger("psc_agent")
    if _logger.handlers:          # avoid duplicate handlers on Streamlit reruns
        return _logger
    _logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_PipelineFormatter())
    _logger.addHandler(handler)
    _logger.propagate = False     # don't bubble to root logger
    return _logger


logger = _configure_logger()


# ═══════════════════════════════════════════════════════════════
# SSH TUNNEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def _get_ssh_client():
    """Get or create the SSH client instance."""
    global _SSH_CLIENT_INSTANCE
    if _SSH_CLIENT_INSTANCE is None and _SSH_LLM_AVAILABLE:
        _SSH_CLIENT_INSTANCE = SSHLLMClient(
            ssh_host="gpu02.cc.iitk.ac.in",
            ssh_user="sunrajp23",
            local_port=8000,
            remote_port=8000,
        )
    return _SSH_CLIENT_INSTANCE


def connect_ssh_tunnel(password: str = None) -> tuple[bool, str]:
    """Establish SSH tunnel connection with optional password.
    
    Returns (success, message)
    """
    client = _get_ssh_client()
    if client is None:
        return False, "SSH client not available"
    
    import subprocess
    
    ssh_cmd = [
        "ssh",
        "-N",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-L", f"{client.local_port}:localhost:{client.remote_port}",
        f"{client.ssh_user}@{client.ssh_host}",
    ]
    
    try:
        if password:
            proc = subprocess.Popen(
                ["sshpass", "-p", password] + ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        
        client._proc = proc
        
        for i in range(30):
            time.sleep(1)
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                return False, f"SSH connection failed: {stderr}"
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{client.local_port}/v1/models", timeout=2)
                client._connected = True
                return True, "SSH tunnel connected successfully"
            except:
                continue
        
        client._connected = True
        return True, "SSH tunnel established"
        
    except FileNotFoundError as e:
        if "sshpass" in str(e):
            return False, "sshpass not installed. Run: brew install sshpass"
        return False, f"SSH command not found: {e}"
    except Exception as e:
        return False, f"Connection error: {e}"


def disconnect_ssh_tunnel():
    """Close SSH tunnel connection."""
    global _SSH_CLIENT_INSTANCE
    if _SSH_CLIENT_INSTANCE is not None:
        if hasattr(_SSH_CLIENT_INSTANCE, "_proc"):
            _SSH_CLIENT_INSTANCE._proc.terminate()
        _SSH_CLIENT_INSTANCE._connected = False
        _SSH_CLIENT_INSTANCE = None


# ═══════════════════════════════════════════════════════════════
# PROCESS STEPS TRACKING
# ═══════════════════════════════════════════════════════════════

class ProcessSteps:
    """Track and display processing steps in the UI."""
    
    def __init__(self):
        self.steps = []
        self.container = None
    
    def add(self, step: str, status: str = "pending"):
        """Add a step with status: pending, running, done, error"""
        emoji = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}.get(status, "•")
        self.steps.append({"step": step, "status": status, "emoji": emoji})
    
    def update(self, index: int, status: str):
        """Update status of a specific step"""
        if 0 <= index < len(self.steps):
            self.steps[index]["status"] = status
            self.steps[index]["emoji"] = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}.get(status, "•")
    
    def render(self, container):
        """Render steps in a Streamlit container."""
        for s in self.steps:
            color = {
                "pending": "#888888",
                "running": "#3498db",
                "done": "#2ecc71",
                "error": "#e74c3c"
            }.get(s["status"], "#888888")
            container.markdown(
                f'<div style="padding:4px 0;color:{color}">{s["emoji"]} {s["step"]}</div>',
                unsafe_allow_html=True
            )
    
    def clear(self):
        """Clear all steps."""
        self.steps = []


# Global process steps instance
_process_steps = ProcessSteps()


# ═══════════════════════════════════════════════════════════════
# INITIALIZATION (cached)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def init_engine():
    logger.info("━" * 60)
    logger.info("🚀  INITIALIZING DATA ENGINE")
    logger.info("━" * 60)
    engine = DataEngine(
        perovskite_path=os.path.join(_CHATBOT_DIR, "data", "perovskite_db.csv"),
        matbench_dir=os.path.join(_CHATBOT_DIR, "data"),
    )
    for ds_name, ds_df in engine.datasets.items():
        logger.info("  📂  dataset=%-30s  rows=%d  cols=%d",
                    ds_name, len(ds_df), len(ds_df.columns))
    logger.info("✅  Data engine ready — %d dataset(s) loaded", len(engine.datasets))
    logger.info("━" * 60)
    return engine


@st.cache_resource
def _load_api_keys():
    """Load API keys from keys.md file."""
    keys_path = os.path.expanduser("~/keys.md")
    keys = []
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            for line in f:
                k = line.strip()
                if k and k.startswith("AIza"):
                    keys.append(k)
        logger.info("🔑  Loaded %d API key(s) from %s", len(keys), keys_path)
    # Fallback to env var
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    if env_key and env_key not in keys:
        keys.insert(0, env_key)
        logger.info("🔑  Using GOOGLE_API_KEY from environment")
    if not keys:
        raise RuntimeError("No API keys found. Add keys to ~/keys.md or set GOOGLE_API_KEY")
    logger.info("🔑  Total API keys available: %d", len(keys))
    return keys


class LLMLoader:
    """Unified LLM loader that routes to either local vLLM or Gemini API."""

    def __init__(
        self,
        use_local: bool = False,
        local_port: int = 8000,
        local_model: str = "qwen-coder-3.5",
        temperature: float = 0.1,
    ):
        self._use_local = use_local
        self._temperature = temperature
        logger.info(
            "[LLM]  Initialised  use_local=%s  port=%d  model=%s  temp=%.1f",
            use_local, local_port, local_model, temperature
        )

    def invoke(self, prompt: str, temperature: float = None) -> str:
        """Invoke the appropriate LLM (local or Gemini)."""
        temp = temperature if temperature is not None else self._temperature

        if self._use_local:
            if not _LOCAL_LLM_AVAILABLE:
                raise RuntimeError(
                    "Local LLM client not available. "
                    "Ensure local_llm_client.py exists and contains LocalLLMClient class."
                )
            return _LLM_LOCAL.invoke(prompt, temperature=temp)

        # Default to Gemini via _LLM_GEMINI
        return _LLM_GEMINI.invoke(prompt, temperature=temp)

    def __getattr__(self, name):
        # Delegate attributes to the active LLM client
        if self._use_local and hasattr(_LLM_LOCAL, name):
            return getattr(_LLM_LOCAL, name)
        if not self._use_local and hasattr(_LLM_GEMINI, name):
            return getattr(_LLM_GEMINI, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Cached Gemini LLM loader (keys-loaded via _load_api_keys at runtime)
@st.cache_resource(show_status=False)
def _make_gemini_loader():
    keys = _load_api_keys()
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=keys[0],
        temperature=0.1,
    )
    logger.info("[GEMINI]  Loaded gemini-2.5-flash with key #1")
    return llm


# Cached local LLM client (no API keys needed)
_local_llm: Optional[LocalLLMClient] = None


def _make_local_loader() -> LocalLLMClient:
    """Create local LLM client (http://localhost:8000/v1/chat/completions)."""
    global _local_llm
    if _local_llm is None:
        _local_llm = LocalLLMClient(port=8000, model_name="qwen-coder-3.5", timeout=120)
    logger.info("[LOCAL]  Local vLLM client connected to localhost:8000")
    return _local_llm


# Global instances (created/stale on toggle)
_LLM_GEMINI = None
_LLM_LOCAL = None


def get_llm(use_local: bool = False):
    """Get the appropriate LLM client based on use_local flag."""
    global _LLM_GEMINI, _LLM_LOCAL

    if use_local:
        if _LLM_LOCAL is None:
            _LLM_LOCAL = _make_local_loader()
        return _LLM_LOCAL
    else:
        if _LLM_GEMINI is None:
            _LLM_GEMINI = _make_gemini_loader()
        return _LLM_GEMINI


class RotatingLLM:
    """LLM wrapper that auto-rotates between multiple API keys on quota errors."""

    def __init__(self, keys: list, model: str = "gemini-2.5-flash", temperature: float = 0.1):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._keys = keys
        self._model = model
        self._temperature = temperature
        self._key_idx = 0
        self._banned = set()
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=keys[0],
            temperature=temperature,
        )
        logger.info("[GEMINI]  RotatingLLM initialised with %d key(s)", len(keys))

    def _make_llm(self, key: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self._model,
            google_api_key=key,
            temperature=self._temperature,
        )

    def _is_quota_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        quota_signals = [
            "429", "resource exhausted", "quota", "rate limit",
            "free tier", "limit exceeded", "resourceexhausted",
            "too many requests", "billing", "exceeded your current",
            "api_key_invalid", "api key expired", "api key invalid",
            "expired", "renew the api key",
        ]
        return any(s in msg for s in quota_signals)

    def _rotate_key(self) -> bool:
        self._banned.add(self._key_idx)
        for _ in range(len(self._keys)):
            self._key_idx = (self._key_idx + 1) % len(self._keys)
            if self._key_idx not in self._banned:
                self._llm = self._make_llm(self._keys[self._key_idx])
                logger.warning("Rotated to API key #%d", self._key_idx + 1)
                return True
        logger.error("All API keys exhausted!")
        return False

    def invoke(self, *args, **kwargs):
        last_error = None
        attempts = 0
        while attempts < len(self._keys):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as e:
                if self._is_quota_error(e):
                    logger.warning(
                        "Key #%d quota/rate error (attempt %d/%d): %s",
                        self._key_idx + 1, attempts + 1, len(self._keys), str(e)[:120],
                    )
                    if not self._rotate_key():
                        raise RuntimeError("All API keys exhausted — no free tier remaining") from e
                    attempts += 1
                    last_error = e
                else:
                    logger.error("LLM non-quota error: %s", str(e)[:200])
                    raise
        raise last_error or RuntimeError("All API keys exhausted")

    def __getattr__(self, name):
        return getattr(self._llm, name)


@st.cache_resource
def init_llm(use_local: bool = False):
    """Initialize LLM with optional local vLLM routing."""
    if use_local:
        return get_llm(use_local=True)
    keys = _load_api_keys()
    return RotatingLLM(keys, model="gemini-2.5-flash", temperature=0.1)


@st.cache_resource
def init_mp_client():
    return MaterialsProjectClient()


@st.cache_resource
def init_regression_engine():
    return RegressionEngine()


engine = init_engine()
llm = init_llm()  # Default (Gemini) - cached instance
mp_client = init_mp_client()
regression_engine = init_regression_engine()

_global_llm_local = None
_global_llm_gemini = None


def get_current_llm(use_local: bool = None):
    """
    Get current LLM instance based on session state.
    If use_local is None, reads from st.session_state; otherwise uses the flag directly.
    """
    global _global_llm_local, _global_llm_gemini

    if use_local is None:
        # Read from session state (Streamlit context)
        use_local = st.session_state.get("local_llm", False)

    if use_local:
        if _global_llm_local is None and _LOCAL_LLM_AVAILABLE:
            _global_llm_local = _make_local_loader()
        return _global_llm_local
    else:
        if _global_llm_gemini is None:
            _global_llm_gemini = RotatingLLM(_load_api_keys(), model="gemini-2.5-flash", temperature=0.1)
        return _global_llm_gemini


def extract_llm_text(response) -> str:
    """Safely extract text from LangChain LLM response."""
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        first = content[0] if content else ""
        return first.get("text", str(first)) if isinstance(first, dict) else str(first)
    return str(content)


def choose_table(query: str) -> str:
    """Resolve query to target table name."""
    q = query.lower()
    for name in engine.datasets:
        if name.lower() in q:
            return name
    # Default to perovskite_db for PSC-related queries
    if "perovskite_db" in engine.datasets:
        return "perovskite_db"
    return list(engine.datasets.keys())[0] if engine.datasets else ""





def log(event: str, level: str = "info", **kwargs):
    """Structured log helper — wraps the module logger."""
    extras = "  ".join(f"{k}={v!r}" for k, v in kwargs.items())
    msg = f"[{event}]  {extras}" if extras else f"[{event}]"
    getattr(logger, level)(msg)


# ═══════════════════════════════════════════════════════════════
# DISAMBIGUATION STATE HELPERS
# ═══════════════════════════════════════════════════════════════

def _parse_disambiguation_choice(raw: str, n_options: int):
    """
    Parse a user reply that should be a disambiguation choice.

    Accepts:
      - a bare number:  "2"
      - ordinal words:  "first", "second", "third"
      - "option N" / "choice N"

    Returns 0-based index if valid, else None.
    """
    text = raw.strip().lower()
    # Bare digit
    if text.isdigit():
        idx = int(text) - 1
        return idx if 0 <= idx < n_options else None
    # Ordinal words
    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
    if text in ordinals:
        idx = ordinals[text]
        return idx if idx < n_options else None
    # "option 2" / "choice 3"
    m = re.match(r'(?:option|choice)\s+(\d+)', text)
    if m:
        idx = int(m.group(1)) - 1
        return idx if 0 <= idx < n_options else None
    return None


# ═══════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════

def route(query: str) -> dict:
    """
    Main router: intent → plan → execute → explain.

    Returns dict with keys: tool, result, query_plan, error
    """
    t0 = time.perf_counter()
    logger.info("")
    logger.info("═" * 70)
    logger.info("🔍  ROUTER  query=%r", query)
    logger.info("═" * 70)

    # ── Step -1: Stateful disambiguation resolution ──
    pending = st.session_state.get("pending_disambiguation")
    if pending:
        options = pending.get("options", [])
        choice_idx = _parse_disambiguation_choice(query, len(options))
        if choice_idx is not None:
            chosen = options[choice_idx]
            action = chosen.get("action", {})
            st.session_state.pending_disambiguation = None  # clear state

            rewrite = action.get("rewrite")
            table = action.get("table")

            if rewrite:
                logger.info(
                    "[DISAMBIG]  resolved choice=%d  rewrite=%r",
                    choice_idx + 1, rewrite,
                )
                return route(rewrite)  # recurse with rewritten query
            elif table:
                logger.info(
                    "[DISAMBIG]  resolved choice=%d  table=%r  — re-routing original query",
                    choice_idx + 1, table,
                )
                # Replace table mentions in original query or just re-route
                original_query = pending.get("original_query", query)
                return route(original_query)
        else:
            # The reply doesn't look like a choice — treat as a new query
            # and clear the pending state so it doesn't linger.
            logger.info(
                "[DISAMBIG]  input %r is not a valid choice; treating as new query",
                query,
            )
            st.session_state.pending_disambiguation = None

    # ── Step 0: Intent classification ──
    logger.debug("[INTENT]  classifying intent …")
    intent = classify_intent(query, get_current_llm())
    logger.info("[INTENT]  intent=%r", intent)

    # ── DOMAIN_ONLY → LLM answers directly ──
    if intent == "DOMAIN_ONLY":
        logger.info("[ROUTER]  → domain handler")
        return _handle_domain(query)

    # ── MATERIAL_LOOKUP → MP property retrieval ──
    if intent == "MATERIAL_LOOKUP":
        logger.info("[ROUTER]  → material lookup handler")
        return _handle_material_lookup(query)

    # ── MULTI_STEP → complex query needing DB + MP + LLM ──
    if intent == "MULTI_STEP":
        logger.info("[ROUTER]  → multi-step handler")
        return _handle_multi_step(query)

    # ── PROPERTY_PREDICT → regression pipeline ──
    if intent == "PROPERTY_PREDICT":
        logger.info("[ROUTER]  → property prediction handler")
        return _handle_property_predict(query)

    # ── DESIGN → autonomous pipeline ──
    if intent == "DESIGN":
        logger.info("[ROUTER]  → design pipeline")
        return _handle_design(query)

    # ── Everything else → Unified Planner → Execute ──
    table_name = choose_table(query)
    logger.info("[ROUTER]  resolved table=%r", table_name)
    if table_name not in engine.datasets:
        logger.error("[ROUTER]  table %r not found in datasets", table_name)
        return {"error": f"Table '{table_name}' not found"}

    # Disambiguation check
    logger.debug("[DISAMBIG]  checking disambiguation …")
    disambig = check_disambiguation(query, list(engine.datasets.keys()))
    if disambig and disambig.get("needs_user_choice"):
        logger.info("[DISAMBIG]  needs user choice — returning options")
        return {"tool": "disambiguation", "result": disambig}
    logger.debug("[DISAMBIG]  ok — no disambiguation needed")

    # Generate unified plan (includes validation + repair)
    logger.info("[PLANNING]  generating query plan  table=%r", table_name)
    ontology = engine.get_column_ontology(table_name, top_values=6)
    logger.debug("[PLANNING]  ontology columns=%d", len(ontology) if isinstance(ontology, dict) else 0)

    try:
        plan_t0 = time.perf_counter()
        plan_result = query_planner.generate_unified_plan(
            query=query, llm=llm, engine=engine,
            table_name=table_name, ontology_data=ontology,
        )
        plan_elapsed = (time.perf_counter() - plan_t0) * 1000
    except Exception as e:
        logger.error("[PLAN_ERROR]  %s", str(e))
        logger.debug("[PLAN_ERROR]  traceback:\n%s", traceback.format_exc())
        return {"error": f"Query planning failed: {e}"}

    plan = plan_result["plan"]
    repaired = plan_result.get("repaired", False)
    relevant_cols = plan_result.get("relevant_columns", [])
    logger.info(
        "[PLAN_OK]  operation=%r  relevant_cols=%d  repaired=%s  plan_ms=%.1f",
        plan.get("operation"), len(relevant_cols), repaired, plan_elapsed,
    )
    logger.debug("[PLAN_OK]  full plan:\n%s", json.dumps(plan, indent=2))

    # Execute
    logger.info("[EXECUTE]  running query …")
    exec_t0 = time.perf_counter()
    exec_result = query_executor.execute(
        plan=plan, engine=engine, table_name=table_name,
        enforce_provenance=True,
    )
    exec_elapsed = (time.perf_counter() - exec_t0) * 1000
    logger.debug("[EXECUTE]  executor returned in %.1f ms", exec_elapsed)

    if exec_result.get("status") in ("error", "execution_error"):
        logger.error("[EXECUTE]  execution error: %s", exec_result.get("error", "?"))
        return {"error": f"Execution failed: {exec_result.get('error', '?')}"}

    if exec_result.get("status") == "validation_error":
        errors = exec_result.get("errors", [])
        suggestions = exec_result.get("suggestions", [])
        logger.warning("[EXECUTE]  validation errors: %s", errors)
        msg = "Validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        if suggestions:
            msg += "\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)
        return {"error": msg}

    dt_ms = exec_result.get("execution_time_ms", 0)
    logger.info(
        "[EXEC_OK]  engine_ms=%.1f  rows_in=%d  rows_filtered=%d  result_rows=%d  warnings=%d",
        dt_ms,
        exec_result.get("rows_before", 0),
        exec_result.get("rows_after_filters", 0),
        exec_result.get("result_rows", 0),
        len(exec_result.get("warnings", [])),
    )
    if exec_result.get("warnings"):
        for w in exec_result["warnings"]:
            logger.warning("[EXEC_WARN]  %s", w)

    run_result = {
        "result_type": exec_result.get("result_type", "table"),
        "result": exec_result.get("result", []),
        "operation": exec_result.get("operation", ""),
        "rows_before": exec_result.get("rows_before", 0),
        "rows_after_filters": exec_result.get("rows_after_filters", 0),
        "result_rows": exec_result.get("result_rows", 0),
        "execution_time_ms": dt_ms,
        "warnings": exec_result.get("warnings", []),
    }

    # Auto-detect if interpretation is needed
    q_lower = query.lower()
    needs_interpretation = any(kw in q_lower for kw in [
        "why", "explain", "compare", "interpret", "what does",
        "significance", "implication",
    ]) or intent == "DATA_PLUS_EXPLANATION"

    total_ms = (time.perf_counter() - t0) * 1000
    logger.debug("[ROUTER]  needs_interpretation=%s", needs_interpretation)

    if needs_interpretation:
        logger.info("[INTERPRET]  generating LLM interpretation …")
        interp_t0 = time.perf_counter()
        preview = exec_result.get("result", [])
        if isinstance(preview, list):
            preview = preview[:15]

        interp_prompt = f"""You are a perovskite solar cell expert. Interpret the data below.

Rules:
- Cite specific numbers from the data
- Do NOT invent values not present in the result
- Mention sample sizes/uncertainty when relevant
- Keep it concise (3-5 sentences)

User query: {query}
Query plan: {json.dumps(plan, indent=2)}
Matching rows: {exec_result.get('rows_after_filters', '?')}

Data:
{json.dumps(preview, indent=2, default=str)}"""

        llm_current = get_current_llm()
        interpretation = extract_llm_text(llm_current.invoke(interp_prompt))
        interp_ms = (time.perf_counter() - interp_t0) * 1000
        logger.info("[INTERPRET]  done  interp_ms=%.1f  total_ms=%.1f", interp_ms, total_ms)
        final = {"data_result": run_result, "interpretation": interpretation}
        return {"tool": "hybrid", "result": final, "query_plan": plan}

    logger.info("[ROUTER]  ✅  data_query complete  total_ms=%.1f", total_ms)
    return {"tool": "data_query", "result": run_result, "query_plan": plan}


def _handle_domain(query: str) -> dict:
    """Domain question — LLM answers, auto-augmented with DB + MP data when relevant."""
    logger.info("[DOMAIN]  answering domain query …")
    t0 = time.perf_counter()

    # Try to extract a chemical formula for auto-augmentation
    formula = _extract_formula(query)
    augmentation_sections = []

    if formula:
        logger.info("[DOMAIN]  formula=%r detected — auto-augmenting with data", formula)

        # ── Augment 1: Materials Project properties ──
        try:
            mp_props = mp_client.get_material_properties(formula)
            if mp_props and "error" not in mp_props and not mp_props.get("needs_user_choice"):
                mp_text = mp_client.format_properties_text(mp_props)
                augmentation_sections.append(
                    f"Materials Project computed data for {formula}:\n{mp_text}"
                )
                logger.debug("[DOMAIN]  MP augmentation: %d chars", len(mp_text))
            elif mp_props and mp_props.get("needs_user_choice"):
                # Multiple polymorphs — include summary
                n = mp_props.get("n_phases", "?")
                opts = mp_props.get("options", [])
                lines = [f"Materials Project: {n} polymorphs found for {formula}:"]
                for opt in opts[:5]:
                    bg = opt.get("band_gap")
                    stable = "stable" if opt.get("is_stable") else "unstable"
                    cs = opt.get("crystal_system", "?")
                    lines.append(f"  - {opt.get('mp_id', '?')}: {cs}, Eg={bg:.3f} eV, {stable}")
                augmentation_sections.append("\n".join(lines))
        except Exception as e:
            logger.debug("[DOMAIN]  MP augmentation failed: %s", e)

        # ── Augment 2: Perovskite DB usage stats ──
        try:
            db_context = _get_db_context_for_formula(formula)
            if db_context:
                augmentation_sections.append(db_context)
                logger.debug("[DOMAIN]  DB augmentation: %d chars", len(db_context))
        except Exception as e:
            logger.debug("[DOMAIN]  DB augmentation failed: %s", e)

    prompt = f"""You are an expert in perovskite solar cells and materials science.
Answer clearly and concisely. Focus on scientific accuracy.
When data is provided below, ALWAYS cite specific numbers from it.
Do NOT invent values — only use what is given.

Question: {query}"""

    if augmentation_sections:
        prompt += "\n\n" + "\n\n".join(augmentation_sections)
        prompt += "\n\nUse the data above to give a data-grounded answer (3-5 sentences). Cite specific values."
    else:
        prompt += "\n\nProvide a brief, informative answer (2-4 sentences)."

    try:
        llm_current = get_current_llm()
        answer = extract_llm_text(llm_current.invoke(prompt))
        logger.info("[DOMAIN]  ✅  llm_ms=%.1f  answer_len=%d  augmented=%s",
                    (time.perf_counter() - t0) * 1000, len(answer), bool(augmentation_sections))
        return {"tool": "domain", "result": answer}
    except Exception as e:
        logger.error("[DOMAIN]  error: %s", str(e))
        return {"error": f"Domain answer error: {e}"}


def _handle_design(query: str) -> dict:
    """Design pipeline — propose, predict, validate."""
    logger.info("[DESIGN]  starting autonomous design pipeline …")
    t0 = time.perf_counter()
    try:
        result = run_design_pipeline(
            query, engine=engine, llm=llm, mp_client=mp_client,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        n = result.get("n_candidates", 0)
        logger.info("[DESIGN]  ✅  candidates=%d  elapsed_ms=%.1f", n, elapsed)
        return {"tool": "design", "result": result}
    except Exception as e:
        logger.error("[DESIGN]  pipeline error: %s", str(e))
        logger.debug("[DESIGN]  traceback:\n%s", traceback.format_exc())
        return {"error": f"Design pipeline error: {e}"}


def _get_db_context_for_formula(formula: str) -> Optional[str]:
    """Search perovskite_db for devices using this material and return a summary."""
    df = engine.datasets.get("perovskite_db")
    if df is None:
        return None

    formula_lower = formula.lower()
    # Search across all stack/sequence columns
    stack_cols = [c for c in df.columns if "stack_sequence" in c.lower()]
    mask = pd.Series(False, index=df.index)
    matched_col = None
    for col in stack_cols:
        col_mask = df[col].astype(str).str.lower().str.contains(formula_lower, na=False)
        if col_mask.any() and not matched_col:
            matched_col = col
        mask = mask | col_mask

    count = int(mask.sum())
    if count == 0:
        return None

    subset = df[mask]
    lines = [f"Perovskite database: {count} devices use {formula}"]

    # Determine which layer role this material plays
    role_map = {
        "ETL_stack_sequence": "ETL (electron transport layer)",
        "HTL_stack_sequence": "HTL (hole transport layer)",
        "Perovskite_composition_short_form": "perovskite absorber",
        "Substrate_stack_sequence": "substrate",
        "Backcontact_stack_sequence": "back contact",
    }
    if matched_col and matched_col in role_map:
        lines.append(f"  Primary role: {role_map[matched_col]}")

    # PCE stats
    if "JV_default_PCE" in subset.columns:
        pce = subset["JV_default_PCE"].dropna()
        if len(pce) > 0:
            lines.append(f"  PCE stats: mean={pce.mean():.2f}%, max={pce.max():.2f}%, "
                         f"median={pce.median():.2f}%, n={len(pce)}")

    # Common architectures
    if "Cell_architecture" in subset.columns:
        arch_counts = subset["Cell_architecture"].dropna().value_counts().head(3)
        if len(arch_counts) > 0:
            arch_str = ", ".join(f"{v}:{c}" for v, c in arch_counts.items())
            lines.append(f"  Architectures: {arch_str}")

    return "\n".join(lines)


def _handle_material_lookup(query: str) -> dict:
    """Material property lookup via Materials Project API + DB cross-reference."""
    logger.info("[MAT_LOOKUP]  starting material lookup …")
    t0 = time.perf_counter()

    # Check if this is a disambiguation follow-up (user chose a material_id)
    mp_id_match = re.search(r'(mp-\d+)', query)
    if mp_id_match:
        material_id = mp_id_match.group(1)
        logger.info("[MAT_LOOKUP]  fetching by material_id=%r", material_id)
        props = mp_client.get_properties_by_id(material_id)
        if "error" not in props:
            props_text = mp_client.format_properties_text(props)
            prompt = f"""You are a materials science expert. The user asked: "{query}"

Here is the data from the Materials Project API:
{props_text}

Provide a clear, informative answer that:
1. States the key property values from the data above
2. Briefly explains what these values mean physically
3. Mentions the Materials Project as the data source

Keep it concise (3–5 sentences)."""
            try:
                llm_current = get_current_llm()
                answer = extract_llm_text(llm_current.invoke(prompt))
            except Exception as e:
                logger.error("[MAT_LOOKUP]  LLM error: %s", e)
                answer = props_text
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("[MAT_LOOKUP]  ✅  material_id=%r  elapsed_ms=%.1f", material_id, elapsed)
            return {
                "tool": "material_lookup",
                "result": {
                    "formula": props.get("formula", material_id),
                    "properties": props,
                    "answer": answer,
                    "properties_text": props_text,
                },
            }

    # Extract chemical formula from query
    formula = _extract_formula(query)
    if not formula:
        logger.warning("[MAT_LOOKUP]  no formula found in query")
        formula = _llm_extract_formula(query)

    if not formula:
        return {"error": "Could not identify a material formula in your query. "
                          "Try specifying it directly, e.g. 'properties of TiO2'."}

    logger.info("[MAT_LOOKUP]  extracted formula=%r", formula)

    # Determine which properties to fetch based on query
    q_lower = query.lower()
    if any(kw in q_lower for kw in ["electronic", "band gap", "bandgap", "metal", "conductor"]):
        props = mp_client.get_electronic_structure(formula)
    elif any(kw in q_lower for kw in ["thermodynamic", "formation energy", "stability", "decompos",
                                        "hull", "ehull", "e_hull", "decomposition"]):
        props = mp_client.get_thermodynamic_data(formula)
    elif any(kw in q_lower for kw in ["structure", "crystal", "space group", "lattice"]):
        props = mp_client.get_structure_info(formula)
    else:
        props = mp_client.get_material_properties(formula)

    # Handle disambiguation: multiple polymorphs found
    if props.get("needs_user_choice"):
        logger.info("[MAT_LOOKUP]  multiple polymorphs found — returning disambiguation")
        props_text = mp_client.format_properties_text(props)
        options = []
        for opt in props.get("options", []):
            mp_id = opt.get("mp_id", "?")
            label = opt.get("label", "?")
            bg = opt.get("band_gap")
            bg_str = f", Eg={bg:.3f} eV" if bg is not None else ""
            stable_tag = " ✅ stable" if opt.get("is_stable") else ""
            options.append({
                "label": f"{label}{bg_str}{stable_tag}",
                "description": f"Material ID: {mp_id}",
                "action": {"rewrite": f"properties of {mp_id}"},
            })
        disambig_result = {
            "needs_user_choice": True,
            "reason": f"Multiple phases found for **{formula}** ({props.get('n_phases', '?')} entries). Please select one:",
            "options": options,
            "original_query": query,
        }
        return {"tool": "disambiguation", "result": disambig_result}

    if "error" in props:
        logger.warning("[MAT_LOOKUP]  MP returned error: %s", props["error"])
        search_result = mp_client.search_materials(formula)
        if "error" not in search_result:
            props = search_result
        else:
            return {"error": props["error"]}

    props_text = mp_client.format_properties_text(props)

    # ── Cross-reference with perovskite DB ──
    db_context = _get_db_context_for_formula(formula)
    if db_context:
        logger.info("[MAT_LOOKUP]  DB cross-ref found for %r", formula)

    prompt = f"""You are a materials science expert. The user asked: "{query}"

Here is the data from the Materials Project API:
{props_text}"""

    if db_context:
        prompt += f"""\n\nExperimental data from the perovskite solar cell database:
{db_context}"""

    prompt += """\n\nProvide a clear, informative answer that:
1. States the key property values from the Materials Project data above
2. Briefly explains what these values mean physically
3. Mentions the Materials Project as the data source"""

    if db_context:
        prompt += "\n4. Also mentions how this material is used in real devices (from the database)"

    prompt += "\n\nKeep it concise (3–5 sentences)."

    try:
        llm_current = get_current_llm()
        answer = extract_llm_text(llm_current.invoke(prompt))
    except Exception as e:
        logger.error("[MAT_LOOKUP]  LLM error: %s", e)
        answer = props_text

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[MAT_LOOKUP]  ✅  formula=%r  db_cross_ref=%s  elapsed_ms=%.1f",
                formula, bool(db_context), elapsed)

    return {
        "tool": "material_lookup",
        "result": {
            "formula": formula,
            "properties": props,
            "answer": answer,
            "properties_text": props_text,
            "db_context": db_context,
        },
    }


def _handle_multi_step(query: str) -> dict:
    """Complex query handler: DB search + MP API + LLM synthesis."""
    logger.info("[MULTI_STEP]  starting multi-step pipeline …")
    t0 = time.perf_counter()
    steps_log = []  # accumulate evidence from each step

    # ── Step 1: Use LLM to decompose the query ──
    decompose_prompt = f"""You are a research assistant for perovskite solar cells.

The user asked: "{query}"

You have access to:
1. A database of 43,000+ perovskite solar cell device records with columns for:
   - ETL_stack_sequence, HTL_stack_sequence, Perovskite_composition_short_form
   - Cell_architecture, Substrate_stack_sequence, Backcontact_stack_sequence
   - JV_default_PCE, JV_default_Voc, JV_default_Jsc, JV_default_FF
   - Ref_DOI_number, Ref_journal, etc.
2. Materials Project API for computed properties (band gap, formation energy, stability, density)

Analyze the query and respond in JSON:
{{
  "material": "the chemical formula mentioned (e.g. TiO2)",
  "layer_role": "which layer role this material serves (ETL/HTL/perovskite/substrate/backcontact or null)",
  "task_type": "substitute|compare|property_lookup|general",
  "layer_column": "the DB column to search (e.g. ETL_stack_sequence) or null"
}}

Respond with ONLY the JSON."""

    try:
        llm_current = get_current_llm()
        decomp_raw = extract_llm_text(llm_current.invoke(decompose_prompt))
        json_match = re.search(r'\{[^{}]*\}', decomp_raw)
        decomp = json.loads(json_match.group()) if json_match else {}
    except Exception as e:
        logger.warning("[MULTI_STEP]  decompose failed: %s — using defaults", e)
        decomp = {}

    material = decomp.get("material") or _extract_formula(query) or ""
    layer_col = decomp.get("layer_column") or ""
    task_type = decomp.get("task_type", "general")
    layer_role = decomp.get("layer_role", "")

    logger.info("[MULTI_STEP]  decomposed: material=%r  layer_col=%r  task=%r  role=%r",
                material, layer_col, task_type, layer_role)
    steps_log.append(f"Query analysis: material={material}, role={layer_role}, task={task_type}")

    df = engine.datasets.get("perovskite_db")
    if df is None:
        return {"error": "Perovskite database not loaded"}

    # ── Step 2: Find the material in the DB and get usage context ──
    db_context = None
    if material:
        db_context = _get_db_context_for_formula(material)
        if db_context:
            steps_log.append(db_context)

    # ── Step 3: For substitute/compare queries — find alternatives ──
    alternatives_data = []
    if task_type in ("substitute", "compare") and layer_col and layer_col in df.columns:
        logger.info("[MULTI_STEP]  searching for alternatives in %r", layer_col)

        # Get unique materials in this layer column, with PCE stats
        col_series = df[layer_col].dropna().astype(str)

        # Tokenize pipe-separated stacks to get individual materials
        all_materials = []
        for val in col_series:
            parts = [p.strip() for p in val.split("|")]
            all_materials.extend(parts)

        from collections import Counter
        mat_counts = Counter(all_materials)
        # Remove empty/common substrates
        for skip in ["", "none", "nan", "0"]:
            mat_counts.pop(skip, None)

        # Get top alternatives (excluding the query material)
        top_alternatives = []
        for mat_name, count in mat_counts.most_common(20):
            if material.lower() in mat_name.lower():
                continue  # skip the query material itself
            if count < 5:
                continue  # skip very rare materials

            # Get PCE stats for this material
            mask = col_series.str.contains(re.escape(mat_name), case=False, na=False)
            subset = df[mask]
            pce = subset["JV_default_PCE"].dropna() if "JV_default_PCE" in subset.columns else pd.Series(dtype=float)

            alt_info = {
                "material": mat_name,
                "device_count": count,
                "pce_mean": round(float(pce.mean()), 2) if len(pce) > 0 else None,
                "pce_max": round(float(pce.max()), 2) if len(pce) > 0 else None,
                "pce_median": round(float(pce.median()), 2) if len(pce) > 0 else None,
            }
            top_alternatives.append(alt_info)
            if len(top_alternatives) >= 8:
                break

        # Sort by device_count (popularity) then by pce_mean
        top_alternatives.sort(key=lambda x: (x.get("pce_mean") or 0), reverse=True)
        top_alternatives = top_alternatives[:6]

        if top_alternatives:
            steps_log.append(f"Database alternatives for {layer_role or layer_col}:")
            for alt in top_alternatives:
                pce_str = f"mean PCE={alt['pce_mean']}%" if alt['pce_mean'] else "no PCE data"
                steps_log.append(f"  - {alt['material']}: {alt['device_count']} devices, "
                                 f"{pce_str}, max PCE={alt.get('pce_max', '?')}%")
            alternatives_data = top_alternatives
        logger.info("[MULTI_STEP]  found %d alternatives", len(top_alternatives))

    # ── Step 4: Fetch MP properties for top alternatives ──
    mp_data = {}
    materials_to_lookup = [material] if material else []
    if alternatives_data:
        materials_to_lookup += [a["material"] for a in alternatives_data[:4]]

    for mat in materials_to_lookup:
        try:
            props = mp_client.get_material_properties(mat)
            if props and "error" not in props and not props.get("needs_user_choice"):
                mp_data[mat] = {
                    "band_gap": props.get("band_gap"),
                    "formation_energy": props.get("formation_energy_per_atom"),
                    "energy_above_hull": props.get("energy_above_hull"),
                    "is_stable": props.get("is_stable"),
                    "density": props.get("density"),
                    "crystal_system": props.get("crystal_system"),
                }
                logger.debug("[MULTI_STEP]  MP data for %r: bg=%.3f",
                             mat, props.get("band_gap", 0))
            elif props and props.get("needs_user_choice"):
                # Take the most stable polymorph's data
                opts = props.get("options", [])
                stable = [o for o in opts if o.get("is_stable")]
                chosen = stable[0] if stable else (opts[0] if opts else None)
                if chosen:
                    mp_data[mat] = {
                        "band_gap": chosen.get("band_gap"),
                        "is_stable": chosen.get("is_stable"),
                        "crystal_system": chosen.get("crystal_system"),
                        "note": f"{len(opts)} polymorphs exist, using most stable",
                    }
        except Exception as e:
            logger.debug("[MULTI_STEP]  MP lookup failed for %r: %s", mat, e)

    if mp_data:
        steps_log.append("Materials Project computed properties:")
        for mat, data in mp_data.items():
            parts = []
            if data.get("band_gap") is not None:
                parts.append(f"Eg={data['band_gap']:.3f} eV")
            if data.get("energy_above_hull") is not None:
                parts.append(f"E_hull={data['energy_above_hull']:.4f} eV/atom")
            if data.get("is_stable") is not None:
                parts.append("stable" if data["is_stable"] else "unstable")
            if data.get("density") is not None:
                parts.append(f"ρ={data['density']:.2f} g/cm³")
            steps_log.append(f"  - {mat}: {', '.join(parts)}")

    # ── Step 5: LLM synthesis ──
    logger.info("[MULTI_STEP]  synthesising final answer (%d evidence lines) …", len(steps_log))
    evidence_text = "\n".join(steps_log)

    synthesis_prompt = f"""You are an expert perovskite solar cell researcher.

The user asked: "{query}"

Here is all the data I gathered from our database of 43,000+ real device records
and the Materials Project computed properties API:

{evidence_text}

Provide a comprehensive, data-grounded answer that:
1. Directly answers the user's question
2. Cites SPECIFIC numbers from the data above (device counts, PCE values, band gaps, etc.)
3. Explains the scientific reasoning (e.g., why certain alternatives might work better)
4. If suggesting alternatives, rank them and explain why
5. Mentions both experimental (database) and theoretical (Materials Project) evidence

Do NOT invent any data. Only cite values present above.
Be thorough but concise (5-8 sentences). Use markdown formatting."""

    try:
        llm_current = get_current_llm()
        answer = extract_llm_text(llm_current.invoke(synthesis_prompt))
    except Exception as e:
        logger.error("[MULTI_STEP]  synthesis LLM error: %s", e)
        answer = f"## Data Gathered\n\n" + "\n".join(f"- {s}" for s in steps_log)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[MULTI_STEP]  ✅  steps=%d  alternatives=%d  mp_lookups=%d  elapsed_ms=%.1f",
                len(steps_log), len(alternatives_data), len(mp_data), elapsed)

    return {
        "tool": "multi_step",
        "result": {
            "answer": answer,
            "steps": steps_log,
            "alternatives": alternatives_data,
            "mp_data": mp_data,
            "material": material,
            "task_type": task_type,
        },
    }


def _handle_property_predict(query: str) -> dict:
    """Regression-based property prediction."""
    logger.info("[PREDICT]  starting property prediction …")
    t0 = time.perf_counter()

    # Use LLM to extract prediction parameters
    extract_prompt = f"""You are a materials science data analyst. The user wants to predict a property.

User query: "{query}"

The available target properties for prediction include:
- JV_default_PCE (power conversion efficiency, %)
- JV_default_Voc (open-circuit voltage, V)
- JV_default_Jsc (short-circuit current, mA/cm²)
- JV_default_FF (fill factor)
- Stability_PCE_T80 (time to 80% of initial PCE, hours)

The available feature columns include:
- HTL_stack_sequence (hole transport layer material)
- ETL_stack_sequence (electron transport layer material)
- Perovskite_composition_short_form (perovskite composition)
- Cell_architecture (n-i-p or p-i-n)
- Substrate_stack_sequence (substrate)
- Backcontact_stack_sequence (back contact)
- Perovskite_band_gap (band gap in eV)

Extract from the query:
1. target: which property to predict (one of the target properties above)
2. features: which feature columns to use (list, pick relevant ones)
3. constraints: any fixed values the user specified

IMPORTANT: NEVER include JV_default_Voc, JV_default_Jsc, or JV_default_FF as features
when predicting JV_default_PCE, because PCE = Voc × Jsc × FF (data leakage).

Respond in JSON only:
{{
  "target": "column_name",
  "features": ["col1", "col2", ...],
  "constraints": {{"col_name": "value", ...}}
}}"""

    try:
        llm_current = get_current_llm()
        extraction = extract_llm_text(llm_current.invoke(extract_prompt))
        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', extraction)
        if json_match:
            params = json.loads(json_match.group())
        else:
            params = json.loads(extraction)
    except Exception as e:
        logger.error("[PREDICT]  LLM extraction failed: %s", e)
        # Default prediction
        params = {
            "target": "JV_default_PCE",
            "features": ["HTL_stack_sequence", "ETL_stack_sequence",
                         "Perovskite_composition_short_form", "Cell_architecture"],
            "constraints": {},
        }

    target = params.get("target", "JV_default_PCE")
    features = params.get("features", ["HTL_stack_sequence", "ETL_stack_sequence"])
    constraints = params.get("constraints", {})

    logger.info("[PREDICT]  target=%r  features=%s  constraints=%s",
                target, features, constraints)

    # Get the dataset
    df = engine.datasets.get("perovskite_db")
    if df is None:
        return {"error": "Perovskite database not loaded"}

    # Run regression
    result = regression_engine.predict_property(
        target_col=target,
        feature_cols=features,
        constraints=constraints,
        df=df,
    )

    if "error" in result:
        logger.error("[PREDICT]  regression error: %s", result["error"])
        return {"error": f"Prediction failed: {result['error']}"}

    # Augment with LLM interpretation
    interp_prompt = f"""You are a perovskite solar cell expert. Interpret this prediction:

User query: "{query}"

Prediction results:
- Target: {result['target']}
- Predicted value: {result['predicted_value']}
- 95% prediction interval: {result['prediction_interval_95']}
- Model R² score: {result['r2_score']}
- Training samples: {result['n_train']}
- Constraints used: {result['constraints']}
- Top features: {result['top_features']}

Provide a concise interpretation (3-5 sentences) that:
1. States the predicted value with units
2. Comments on the prediction reliability based on R² and the prediction interval
3. Mentions which features matter most"""

    try:
        llm_current = get_current_llm()
        interpretation = extract_llm_text(llm_current.invoke(interp_prompt))
    except Exception as e:
        logger.error("[PREDICT]  LLM interpretation error: %s", e)
        interpretation = f"Predicted {target}: {result['predicted_value']}"

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("[PREDICT]  ✅  predicted=%s  r2=%.4f  elapsed_ms=%.1f",
                result['predicted_value'], result['r2_score'], elapsed)

    return {
        "tool": "property_predict",
        "result": {
            "prediction": result,
            "interpretation": interpretation,
        },
    }


def _extract_formula(query: str) -> Optional[str]:
    """Extract a chemical formula from the query using regex."""
    # Common material formulas in PSC domain
    known = [
        "MAPbI3", "FAPbI3", "CsPbI3", "MAPbBr3", "CsPbBr3",
        "TiO2", "SnO2", "ZnO", "NiO", "NiOx", "Cu2O",
        "PCBM", "C60", "BCP", "PTAA",
        "Spiro-MeOTAD", "PEDOT:PSS",
    ]
    q = query
    for mat in known:
        if mat.lower() in q.lower():
            return mat

    # General chemical formula regex
    # Matches: TiO2, MAPbI3, Cs0.05FA0.85MA0.10PbI3, Si, GaAs
    pattern = re.compile(
        r'\b([A-Z][a-z]?(?:\d*\.?\d+)?(?:[A-Z][a-z]?(?:\d*\.?\d+)?){0,8})\b'
    )
    for m in pattern.finditer(query):
        candidate = m.group(0)
        if len(candidate) >= 2:
            # Must have at least one element-like pattern (uppercase + optional lowercase + digits)
            if re.match(r'^[A-Z][a-z]?\d*', candidate):
                # Exclude common words
                if candidate.lower() not in {
                    "the", "and", "for", "with", "from", "this", "what",
                    "are", "how", "does", "can", "not", "its", "has",
                }:
                    return candidate
    return None


def _llm_extract_formula(query: str) -> Optional[str]:
    """Use LLM to extract a chemical formula from the query."""
    prompt = f"""Extract the chemical formula or material name from this query.
If there is no specific material mentioned, return NONE.

Query: "{query}"

Reply with ONLY the chemical formula (e.g. TiO2, MAPbI3, Si) or NONE."""
    try:
        llm_current = get_current_llm()
        result = extract_llm_text(llm_current.invoke(prompt)).strip()
        if result.upper() == "NONE" or len(result) > 30:
            return None
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div class="hero-header">
    <h1>⚛️ PSC Agent</h1>
    <p class="hero-sub">
        Perovskite Solar Cells
        <span class="hero-sep">·</span>Data-Grounded
        <span class="hero-sep">·</span>
        <span class="llm-badge" id="llm-badge">Gemini</span>
    </p>
</div>
<script>
// Update LLM badge when toggle changes
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('local_llm');
    const badge = document.getElementById('llm-badge');
    if (toggle && badge) {
        toggle.addEventListener('change', function() {
            badge.textContent = this.checked ? 'qwen-coder-3.5' : 'Gemini';
            badge.style.color = this.checked ? '#ff6b6b' : 'inherit';
        });
    }
});
</script>
""", unsafe_allow_html=True)

    # Dataset stats
    pce_stats = engine.get_summary("perovskite_db").get("pce_stats", {})
    ds_count = len(engine.datasets)
    total_rows = sum(len(df) for df in engine.datasets.values())

    st.markdown(f"""
<div class="metric-row">
    <div class="metric-card teal"><div class="label">Datasets</div><div class="value">{ds_count}</div></div>
    <div class="metric-card blue"><div class="label">Records</div><div class="value">{total_rows:,}</div></div>
</div>
""", unsafe_allow_html=True)

    if pce_stats:
        st.markdown(f"""
<div class="metric-row">
    <div class="metric-card amber"><div class="label">Avg PCE</div><div class="value">{pce_stats.get('mean', 0):.1f}%</div></div>
    <div class="metric-card rose"><div class="label">Max PCE</div><div class="value">{pce_stats.get('max', 0):.1f}%</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── SSH Tunnel Management ─────────────────────────────────────────────────
    if _SSH_LLM_AVAILABLE:
        st.markdown("##### 🔌 SSH GPU Connection")
        
        ssh_connected = st.session_state.get("ssh_connected", False)
        
        if not ssh_connected:
            with st.expander("🌐 Connect to GPU Server", expanded=True):
                st.caption("Connect to gpu02.cc.iitk.ac.in to use local Qwen model")
                
                ssh_password = st.text_input(
                    "Password (for SSH key passphrase if needed)",
                    type="password",
                    key="ssh_password",
                    help="Enter your IITK password if prompted"
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔗 Connect", key="ssh_connect", use_container_width=True):
                        with st.status("Connecting to GPU server...") as status:
                            st.write("Establishing SSH tunnel...")
                            success, msg = connect_ssh_tunnel(ssh_password if ssh_password else None)
                            if success:
                                st.session_state.ssh_connected = True
                                st.session_state.ssh_password = ssh_password
                                status.update(state="complete", label="✅ Connected")
                                st.rerun()
                            else:
                                status.update(state="error", label="❌ Connection Failed")
                                st.error(msg)
                with col2:
                    st.caption("Or use SSH key if configured")
        else:
            st.success("🟢 SSH Tunnel Active")
            st.caption("Connected to gpu02.cc.iitk.ac.in:8000")
            
            if st.button("🔴 Disconnect", key="ssh_disconnect", use_container_width=True):
                disconnect_ssh_tunnel()
                st.session_state.ssh_connected = False
                st.session_state.ssh_password = ""
                st.rerun()
    else:
        st.caption("🔌 SSH client not available")

    st.markdown("---")

    # ── Local LLM Toggle ─────────────────────────────────────────────────────
    if _LOCAL_LLM_AVAILABLE:
        ssh_connected = st.session_state.get("ssh_connected", False)
        
        if ssh_connected:
            use_local = st.session_state.get("local_llm", False)
            use_local = st.toggle(
                "🖥️ Use Local Qwen Model (via SSH tunnel)",
                value=use_local,
                key="local_llm",
                help="Route LLM inference to GPU server via SSH tunnel"
            )
            if use_local:
                st.success("Using Qwen model on GPU server")
            else:
                st.caption("Using Google Gemini API")
        else:
            st.info("🔗 Connect to GPU server first to use local Qwen model")
            
            use_local = st.session_state.get("local_llm", False)
            use_local = st.toggle(
                "🖥️ Use Local Qwen Model (localhost:8000)",
                value=use_local,
                key="local_llm",
                help="Route LLM inference to local vLLM server"
            )
            
            if use_local:
                st.caption("Warning: Ensure vLLM is running locally")
    else:
        st.caption("Local LLM client not available — using Google Gemini API")

    st.markdown("---")
    st.markdown("##### 🚀 Quick Queries")

    quick_queries = [
        ("📊", "What is the average PCE?"),
        ("🏆", "Top 10 highest PCE devices"),
        ("🔬", "Properties of TiO2"),
        ("📈", "Predict PCE for Spiro-MeOTAD HTL"),
        ("🏗️", "Design a high-efficiency p-i-n device"),
    ]
    for icon, q in quick_queries:
        if st.button(f"{icon}  {q}", key=f"quick_{q}", width="stretch"):
            st.session_state["_pending_query"] = q


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize use_local_llm flag (defaults to False)
if "local_llm" not in st.session_state:
    st.session_state.local_llm = False


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab_chat, tab_explore, tab_knowledge = st.tabs(["💬 Chat", "📊 Data Explorer", "📚 Domain Knowledge"])


# ═══════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════

_BADGE_HTML = {
    "domain":          '<span class="badge badge-domain">🧠 DOMAIN</span>',
    "data":            '<span class="badge badge-data">📊 DATA</span>',
    "hybrid":          '<span class="badge badge-hybrid">🔀 HYBRID</span>',
    "design":          '<span class="badge badge-design">🏗️ DESIGN</span>',
    "material_lookup": '<span class="badge badge-hybrid">🔬 MATERIAL</span>',
    "multi_step":      '<span class="badge badge-hybrid">🔗 MULTI-STEP</span>',
    "property_predict":'<span class="badge badge-design">📈 PREDICT</span>',
    "disambig":        '<span class="badge badge-domain">🤔 CHOOSE</span>',
    "error":           '<span class="badge badge-error">❌ ERROR</span>',
}

def _render_badge(badge: str):
    """Render a colored badge above the response."""
    html = _BADGE_HTML.get(badge, "")
    if html:
        st.markdown(html, unsafe_allow_html=True)


def _render_query_plan(plan: dict):
    """Render a query plan inside an expander."""
    with st.expander("🔍 Query Plan", expanded=False):
        st.markdown(
            f'<div class="query-plan-block"><pre>{json.dumps(plan, indent=2)}</pre></div>',
            unsafe_allow_html=True,
        )


def _render_stats(data: dict, n_filtered="?"):
    """Render statistics result as a styled card."""
    items = ""
    for k, v in data.items():
        items += f'<div class="stats-item"><span class="stats-key">{k}</span><span class="stats-val">{v}</span></div>'
    st.markdown(f"""
<div class="stats-card">
    <div class="stats-header">Statistics · {n_filtered} matching rows</div>
    {items}
</div>
""", unsafe_allow_html=True)


def _render_data_result(result_dict: dict):
    """Render a data query result (table, scalar, stats)."""
    data = result_dict.get("result", [])
    rtype = result_dict.get("result_type", "table")
    n_filtered = result_dict.get("rows_after_filters", "?")
    n_shown = result_dict.get("result_rows", "?")
    exec_ms = result_dict.get("execution_time_ms", "?")
    op_name = result_dict.get("operation", "?")

    if rtype == "scalar" and isinstance(data, dict):
        _render_stats(data, n_filtered)
    elif rtype == "stats" and isinstance(data, dict):
        _render_stats(data, n_filtered)
    elif rtype == "table" and isinstance(data, list) and data:
        st.caption(f"**{op_name}** — {n_filtered} matching rows, showing {n_shown} ({exec_ms}ms)")
        try:
            st.dataframe(pd.DataFrame(data), width="stretch")
        except Exception:
            st.json(data[:20])
    else:
        st.text(str(data))

    # Warnings
    warnings = result_dict.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w, icon="⚠️")


def _render_design_result(dp: dict):
    """Render a design pipeline result."""
    design = dp.get("design", {})
    pred = dp.get("prediction", {})
    val = dp.get("validation", {})

    st.markdown("### 🏗️ Design → Predict → Validate")

    # Stack table
    stack = design.get("stack", {})
    arch = design.get("architecture", "")
    st.markdown(f"**Architecture:** `{arch}`")

    stack_rows = []
    for layer, mat in stack.items():
        if mat and layer != "architecture":
            stack_rows.append({"Layer": layer.title(), "Material": mat})
    if stack_rows:
        st.dataframe(pd.DataFrame(stack_rows), width="stretch", hide_index=True)

    rationale = design.get("rationale", "")
    if rationale:
        st.caption(f"_{rationale}_")

    # Performance
    perf = pred.get("predicted_performance", {})
    if perf:
        n_sim = pred.get("n_similar_devices", "?")
        st.markdown(f"### 📈 Predicted Performance (n={n_sim})")
        perf_rows = []
        for metric, stats in perf.items():
            iqr = stats.get("IQR", [None, None])
            iqr_s = f"{iqr[0]}–{iqr[1]}" if iqr and iqr[0] is not None else "?"
            perf_rows.append({
                "Metric": metric,
                "Median": stats.get("median", "?"),
                "IQR": iqr_s,
                "Range": f"{stats.get('min', '?')}–{stats.get('max', '?')}",
            })
        st.dataframe(pd.DataFrame(perf_rows), width="stretch", hide_index=True)

    # Confidence
    conf = val.get("confidence", "?")
    emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "❓")
    st.markdown(f"**Confidence:** {emoji} **{conf}**")
    for w in val.get("warnings", []):
        st.warning(w, icon="⚠️")


def _render_material_lookup(result: dict):
    """Render a material property lookup result."""
    formula = result.get("formula", "?")
    props = result.get("properties", {})
    answer = result.get("answer", "")

    st.markdown(answer)

    # Show raw properties in expander
    if props and not props.get("error"):
        with st.expander(f"🔬 Raw Properties — {formula}", expanded=False):
            props_text = result.get("properties_text", "")
            if props_text:
                st.markdown(props_text)
            else:
                st.json(props)


def _render_prediction_result(result: dict):
    """Render a property prediction result."""
    pred = result.get("prediction", {})
    interpretation = result.get("interpretation", "")

    st.markdown(interpretation)

    if pred:
        with st.expander("📈 Prediction Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                val = pred.get('predicted_value', '?')
                st.metric("Predicted Value", f"{val}")
            with col2:
                ci = pred.get('prediction_interval_95', ('?', '?'))
                st.metric("95% PI", f"{ci[0]} – {ci[1]}")
            with col3:
                r2 = pred.get('r2_score', '?')
                st.metric("Model R²", f"{r2}")

            st.caption(f"Target: `{pred.get('target', '?')}` · "
                       f"Training samples: {pred.get('n_train', '?')} · "
                       f"Features: {pred.get('n_features', '?')}")

            # Feature importances
            top_feats = pred.get("top_features", [])
            if top_feats:
                st.markdown("**Top Feature Importances:**")
                feat_df = pd.DataFrame(top_feats)
                st.dataframe(feat_df, width="stretch", hide_index=True)

            # Constraints
            constraints = pred.get("constraints", {})
            if constraints:
                st.markdown("**Constraints used:**")
                for k, v in constraints.items():
                    st.markdown(f"- `{k}` = {v}")


def _render_multi_step(result: dict):
    """Render a multi-step research result."""
    answer = result.get("answer", "")
    st.markdown(answer)

    steps = result.get("steps", [])
    alternatives = result.get("alternatives", [])
    mp_data = result.get("mp_data", {})

    # Show alternatives table if available
    if alternatives:
        with st.expander("📊 Alternatives Data (from 43k+ device database)", expanded=True):
            alt_df = pd.DataFrame(alternatives)
            # Rename columns for readability
            rename = {
                "material": "Material",
                "device_count": "Devices",
                "pce_mean": "Avg PCE (%)",
                "pce_max": "Max PCE (%)",
                "pce_median": "Median PCE (%)",
            }
            alt_df = alt_df.rename(columns={k: v for k, v in rename.items() if k in alt_df.columns})
            st.dataframe(alt_df, width="stretch", hide_index=True)

    # Show MP properties if available
    if mp_data:
        with st.expander("🔬 Materials Project Properties", expanded=False):
            mp_rows = []
            for mat, data in mp_data.items():
                row = {"Material": mat}
                if data.get("band_gap") is not None:
                    row["Band Gap (eV)"] = round(data["band_gap"], 3)
                if data.get("energy_above_hull") is not None:
                    row["E_hull (eV/atom)"] = round(data["energy_above_hull"], 4)
                if data.get("is_stable") is not None:
                    row["Stable"] = "✅" if data["is_stable"] else "❌"
                if data.get("density") is not None:
                    row["Density (g/cm³)"] = round(data["density"], 2)
                if data.get("crystal_system"):
                    row["Crystal System"] = data["crystal_system"]
                mp_rows.append(row)
            if mp_rows:
                st.dataframe(pd.DataFrame(mp_rows), width="stretch", hide_index=True)

    # Show evidence trail
    if steps:
        with st.expander("🔍 Evidence Trail", expanded=False):
            for step in steps:
                st.text(step)


def _render_message(msg: dict):
    """Render a single saved message, including data/design/etc."""
    if msg.get("badge"):
        _render_badge(msg["badge"])

    if msg.get("content"):
        st.markdown(msg["content"])

    # Render attached data result inline
    if "data_result" in msg:
        _render_data_result(msg["data_result"])

    # Render design result inline
    if "design_result" in msg:
        _render_design_result(msg["design_result"])

    # Render material lookup inline
    if "material_lookup_result" in msg:
        _render_material_lookup(msg["material_lookup_result"])

    # Render prediction result inline
    if "prediction_result" in msg:
        _render_prediction_result(msg["prediction_result"])

    # Render multi-step result inline
    if "multi_step_result" in msg:
        _render_multi_step(msg["multi_step_result"])

    # Query plan
    if "query_plan" in msg:
        _render_query_plan(msg["query_plan"])


# ═══════════════════════════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════════════════════════

with tab_chat:

    # ── Welcome screen (only when no messages) ──
    if not st.session_state.messages:
        st.markdown("""
<div class="welcome-container">
    <span class="welcome-icon">⚛️</span>
    <div class="welcome-title">What can I help you with?</div>
    <p class="welcome-sub">Ask about perovskite solar cells, explore data, or design new devices.</p>
</div>

<div class="cap-grid">
    <div class="cap-card">
        <span class="cap-icon">📊</span>
        <div class="cap-title">Data Queries</div>
        <div class="cap-desc">Filter, aggregate, and analyze 43k+ perovskite device records</div>
    </div>
    <div class="cap-card">
        <span class="cap-icon">🔬</span>
        <div class="cap-title">Material Properties</div>
        <div class="cap-desc">Look up band gaps, density, crystal structure from Materials Project</div>
    </div>
    <div class="cap-card">
        <span class="cap-icon">📈</span>
        <div class="cap-title">Property Prediction</div>
        <div class="cap-desc">Predict PCE, Voc, Jsc using ML regression on real device data</div>
    </div>
    <div class="cap-card">
        <span class="cap-icon">🏗️</span>
        <div class="cap-title">Device Design</div>
        <div class="cap-desc">Propose stacks and predict performance from empirical data</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Render chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _render_message(msg)

    # ── Handle disambiguation button callbacks ──
    if "_disambig_choice" in st.session_state:
        chosen = st.session_state.pop("_disambig_choice")
        # Inject the rewritten query directly
        st.session_state["_pending_query"] = chosen

    # ── Chat input (always pinned at bottom by Streamlit) ──
    pending = st.session_state.pop("_pending_query", None)
    user_input = st.chat_input("Ask about perovskite solar cells...")
    prompt = pending or user_input

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and render assistant response
        with st.chat_message("assistant"):
            # Create process steps container
            steps_container = st.container()
            
            # Initialize steps for this query
            steps = [
                ("🔍 Analyzing query intent", "pending"),
                ("📊 Generating query plan", "pending"),
                ("🔎 Retrieving data", "pending"),
                ("🧠 Processing results", "pending"),
            ]
            
            steps_placeholder = steps_container.empty()
            for emoji, status in steps:
                steps_placeholder.markdown(
                    f'<div style="padding:2px 0;color:#888888">⏳ {emoji}</div>',
                    unsafe_allow_html=True
                )
            
            # Update step status
            def update_step(idx, status, emoji_override=None):
                emoji_map = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}
                emoji = emoji_override or emoji_map.get(status, "•")
                steps[idx] = (steps[idx][0], status, emoji)
                display = ""
                for e, s, em in steps:
                    color = {"pending": "#888888", "running": "#3498db", "done": "#2ecc71", "error": "#e74c3c"}.get(s, "#888888")
                    display += f'<div style="padding:2px 0;color:{color}">{em} {e.split(" ", 1)[1]}</div>'
                steps_placeholder.markdown(display, unsafe_allow_html=True)
            
            try:
                update_step(0, "running")
                with st.spinner("Analyzing query..."):
                    result = route(prompt)
                update_step(0, "done")
                
                update_step(1, "done", "✅")
                update_step(2, "done", "✅")
                update_step(3, "done", "✅")
                
            except Exception as e:
                update_step(0, "error")
                st.error(f"Error: {e}")
                result = {"error": str(e)}

            # Clear steps after a delay
            time.sleep(0.5)
            steps_placeholder.empty()

            tool = result.get("tool", "")
            query_plan = result.get("query_plan")
            badge = "data"
            saved_msg = {"role": "assistant", "content": "", "badge": badge}

            if "error" in result:
                # ── Error ──
                badge = "error"
                _render_badge(badge)
                err_text = f"⚠️ {result['error']}"
                st.markdown(err_text)
                saved_msg.update(badge=badge, content=err_text)

            elif tool == "domain":
                # ── Domain-only answer ──
                badge = "domain"
                _render_badge(badge)
                answer = result.get("result", "")
                st.markdown(answer)
                saved_msg.update(badge=badge, content=answer)

            elif tool == "disambiguation":
                # ── Disambiguation: render clickable buttons ──
                badge = "disambig"
                _render_badge(badge)
                d = result.get("result", {})
                reason = d.get("reason", "Multiple interpretations possible.")
                st.markdown(f"**{reason}**")
                saved_msg.update(badge=badge, content=f"🤔 **{reason}**")

                options = d.get("options", [])
                for i, opt in enumerate(options):
                    label = opt.get("label", f"Option {i+1}")
                    desc = opt.get("description", "")
                    action = opt.get("action", {})
                    rewrite = action.get("rewrite", label)

                    col_btn, col_desc = st.columns([1, 2])
                    with col_btn:
                        if st.button(f"▶ {label}", key=f"disambig_{i}_{hash(prompt)}", width="stretch"):
                            st.session_state["_disambig_choice"] = rewrite
                            st.rerun()
                    with col_desc:
                        st.caption(desc)

            elif tool == "material_lookup":
                # ── Material property lookup ──
                badge = "material_lookup"
                _render_badge(badge)
                ml = result.get("result", {})
                _render_material_lookup(ml)
                saved_msg.update(badge=badge, content=ml.get("answer", ""), material_lookup_result=ml)

            elif tool == "multi_step":
                # ── Multi-step research result ──
                badge = "multi_step"
                _render_badge(badge)
                ms = result.get("result", {})
                _render_multi_step(ms)
                saved_msg.update(badge=badge, content=ms.get("answer", ""), multi_step_result=ms)

            elif tool == "property_predict":
                # ── Property prediction ──
                badge = "property_predict"
                _render_badge(badge)
                pr = result.get("result", {})
                _render_prediction_result(pr)
                saved_msg.update(badge=badge, content=pr.get("interpretation", ""), prediction_result=pr)

            elif tool == "design":
                # ── Design pipeline result ──
                badge = "design"
                _render_badge(badge)
                dp = result.get("result", {})
                _render_design_result(dp)
                saved_msg.update(badge=badge, content="", design_result=dp)

            elif tool == "hybrid":
                # ── Data + interpretation ──
                badge = "hybrid"
                _render_badge(badge)
                h = result.get("result", {})
                interpretation = h.get("interpretation", "")
                st.markdown(interpretation)

                dr = h.get("data_result", {})
                if isinstance(dr, dict) and dr.get("result_type") == "table":
                    data = dr.get("result", [])
                    if isinstance(data, list) and data:
                        with st.expander("📊 View Data", expanded=False):
                            try:
                                st.dataframe(pd.DataFrame(data[:50]), width="stretch")
                            except Exception:
                                st.json(data[:20])

                saved_msg.update(badge=badge, content=interpretation, data_result=dr)

            elif tool == "data_query":
                # ── Pure data result ──
                badge = "data"
                _render_badge(badge)
                dr = result.get("result", {})
                _render_data_result(dr)
                saved_msg.update(badge=badge, content="", data_result=dr)

            else:
                text = str(result.get("result", ""))
                st.markdown(text)
                saved_msg.update(content=text)

            # Show query plan
            if query_plan:
                _render_query_plan(query_plan)
                saved_msg["query_plan"] = query_plan

            saved_msg["badge"] = badge
            st.session_state.messages.append(saved_msg)


# ═══════════════════════════════════════════════════════════════
# DATA EXPLORER TAB
# ═══════════════════════════════════════════════════════════════

with tab_explore:
    st.markdown("""
    <div class="section-head">
        <span class="section-icon">📊</span>
        <span class="section-title">Dataset Explorer</span>
    </div>
    """, unsafe_allow_html=True)

    ds_names = list(engine.datasets.keys())
    if not ds_names:
        st.warning("No datasets loaded.")
        st.stop()

    selected = st.selectbox(
        "Select Dataset",
        range(len(ds_names)),
        format_func=lambda i: f"{'☀️' if ds_names[i] == 'perovskite_db' else '🔬'} {ds_names[i]}",
        label_visibility="collapsed",
    )
    ds_name = ds_names[selected]
    df = engine.datasets[ds_name]

    st.markdown(f"""
<div class="metric-row">
    <div class="metric-card teal"><div class="label">Rows</div><div class="value">{len(df):,}</div></div>
    <div class="metric-card blue"><div class="label">Columns</div><div class="value">{len(df.columns)}</div></div>
    <div class="metric-card amber"><div class="label">Numeric</div><div class="value">{len(df.select_dtypes(include='number').columns)}</div></div>
    <div class="metric-card rose"><div class="label">Non-Null %</div><div class="value">{df.notna().mean().mean()*100:.0f}%</div></div>
</div>
""", unsafe_allow_html=True)

    st.dataframe(df.head(100), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv_data, f"{ds_name}.csv", "text/csv")
    with col2:
        with st.expander("📋 Column Details"):
            num_c = df.select_dtypes(include="number").columns.tolist()
            str_c = df.select_dtypes(include="object").columns.tolist()
            if num_c:
                st.markdown(f"**Numeric ({len(num_c)}):** {', '.join(num_c[:20])}")
            if str_c:
                st.markdown(f"**Text ({len(str_c)}):** {', '.join(str_c[:20])}")

    if len(df.select_dtypes(include="number").columns) > 0:
        with st.expander("📈 Quick Statistics"):
            st.dataframe(df.describe(), width="stretch")


# ═══════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE TAB
# ═══════════════════════════════════════════════════════════════

with tab_knowledge:
    st.markdown("""
    <div class="section-head">
        <span class="section-icon">📚</span>
        <span class="section-title">Materials Science Knowledge Base</span>
        <span class="section-subtitle">Domain context for intelligent query understanding</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div class="knowledge-card">
    <h4>☀️ Perovskite Solar Cells</h4>
    <p>The <strong>perovskite_db</strong> contains experimental data on perovskite solar cell devices from published literature.</p>
    <ul>
        <li><strong>Architecture</strong> — n-i-p (regular) vs p-i-n (inverted)</li>
        <li><strong>Stack Layers</strong> — Substrate → ETL → Perovskite → HTL → Back Contact</li>
        <li><strong>Performance</strong> — PCE (%), Voc (V), Jsc (mA/cm²), FF</li>
        <li><strong>Common ETLs</strong> — TiO₂, SnO₂, PCBM, C60</li>
        <li><strong>Common HTLs</strong> — Spiro-OMeTAD, PTAA, PEDOT:PSS, NiO</li>
        <li><strong>Substrates</strong> — ITO, FTO (glass-based)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🔬 Physics Bounds & Sanity Checks")
    st.markdown("""
The system applies physics-based sanity checks to flag unrealistic values:

| Metric | Valid Range | Warning |
|--------|-----------|---------| 
| PCE | 0–30% | > 30% likely error |
| Voc | 0–1.3 V | > 1.3V may be tandem |
| Jsc | 0–30 mA/cm² | > 30 unusually high |
| FF | 0.2–0.9 | Outside range = failure |
""")

    if mp_client.online:
        st.markdown("### 🌐 Materials Project (Online)")
        st.markdown("The system can fetch DFT-computed properties (band gap, formation energy, stability) from the Materials Project API.")
    else:
        st.markdown("### 🌐 Materials Project (Offline)")
        st.markdown("Set `MP_API_KEY` in `.env` to enable live Materials Project lookups.")
