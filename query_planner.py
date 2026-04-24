"""
query_planner.py — Unified LLM-Driven Structured Query Plan Generator

Pipeline:
  1. Semantic column retrieval (sentence-transformer dense embeddings, top-15)
  2. Synonym resolution (user terms → canonical DB values)
  3. LLM generates a single JSON query plan (data + schema queries)
  4. Validation → repair loop (retry once if invalid)
  5. Post-processing (synonym injection, provenance columns, not_null)
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("psc_agent")

from ontology import (
    resolve_synonym,
    extract_materials_from_query,
    get_filter_columns,
    normalize_material_pattern,
    PROVENANCE_COLUMNS,
    PERFORMANCE_COLUMNS,
    DEVICE_CONTEXT_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════
# NOTE: Keyword-based TERM_TO_COLUMNS mapping was REMOVED.
# Column retrieval now uses pure cosine similarity on precomputed
# sentence-transformer embeddings (built from rich metadata text).
# See retrieve_relevant_columns() and data/column_embeddings.npz
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# SEMANTIC COLUMN RETRIEVAL (Dense Embeddings + keyword fallback)
# ═══════════════════════════════════════════════════════════════

import os
import numpy as np

_SBERT_AVAILABLE = False
_SBERT_MODEL = None
_COL_EMBEDDINGS_CACHE: Dict[str, Any] = {}   # key = cache_key → embeddings array

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed — falling back to keyword-only column retrieval")

_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Path to the precomputed embeddings file produced by generate_column_embeddings.py
_MODULE_DIR = os.path.dirname(__file__)
_PRECOMPUTED_EMBEDDINGS_CANDIDATES = [
    os.path.join(_MODULE_DIR, "data", "column_embeddings.npz"),
    os.path.join(os.path.dirname(_MODULE_DIR), "data", "column_embeddings.npz"),
]

# In-memory store for the precomputed .npz contents (loaded once)
_PRECOMPUTED: Dict[str, Any] = {}  # keys: 'embeddings', 'columns'


def _load_precomputed_embeddings() -> bool:
    """
    Load column embeddings from the precomputed .npz file into _PRECOMPUTED.
    Returns True on success, False if the file is missing or unreadable.
    """
    global _PRECOMPUTED
    if _PRECOMPUTED:
        return True  # already loaded
    embeddings_path = next(
        (path for path in _PRECOMPUTED_EMBEDDINGS_CANDIDATES if os.path.isfile(path)),
        None,
    )
    if not embeddings_path:
        logger.warning(
            "[SEMANTIC]  precomputed embeddings not found. Checked: %s. "
            "Run generate_column_embeddings.py to create it.",
            _PRECOMPUTED_EMBEDDINGS_CANDIDATES,
        )
        return False
    try:
        data = np.load(embeddings_path, allow_pickle=False)
        _PRECOMPUTED["embeddings"] = data["embeddings"]   # float32 (N, dim), unit-normed
        _PRECOMPUTED["columns"]    = data["columns"].tolist()  # list[str], length N
        logger.info(
            "[SEMANTIC]  loaded precomputed embeddings: %d columns, dim=%d  (source: %s)",
            len(_PRECOMPUTED["columns"]),
            _PRECOMPUTED["embeddings"].shape[1],
            embeddings_path,
        )
        return True
    except Exception as exc:
        logger.warning("[SEMANTIC]  failed to load precomputed embeddings: %s", exc)
        _PRECOMPUTED.clear()
        return False


def _get_sbert_model():
    """Lazily load and cache the sentence-transformer model."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        logger.info("[SEMANTIC]  loading sentence-transformer model '%s' …", _SBERT_MODEL_NAME)
        _SBERT_MODEL = SentenceTransformer(_SBERT_MODEL_NAME)
        logger.info("[SEMANTIC]  model loaded successfully")
    return _SBERT_MODEL


def _get_column_embeddings(all_columns: List[str]):
    """
    Return dense embeddings for all_columns.

    Strategy (in priority order):
    1. If data/column_embeddings.npz exists AND its column list is a superset
       of all_columns, slice and return the precomputed vectors directly.
       These were encoded from rich text (name + description + keywords + group)
       so they carry much more signal than the plain snake_case fallback.
    2. Check the in-process runtime cache (covers live-encoded batches).
    3. Fall back to live encoding via sentence-transformers (original behaviour).
    """
    # ── 1. Try precomputed file ───────────────────────────────────
    if _load_precomputed_embeddings():
        pre_cols = _PRECOMPUTED["columns"]      # ordered list from .npz
        pre_emb  = _PRECOMPUTED["embeddings"]   # float32 (N, dim)

        # Build index: column_name → row index in precomputed array
        pre_index = {c: i for i, c in enumerate(pre_cols)}

        # Check if every requested column has a precomputed embedding
        if all(c in pre_index for c in all_columns):
            indices   = [pre_index[c] for c in all_columns]
            embeddings = pre_emb[indices]          # sub-array in requested order
            logger.debug(
                "[SEMANTIC]  using precomputed embeddings for %d/%d columns",
                len(all_columns), len(pre_cols),
            )
            return embeddings
        else:
            missing = [c for c in all_columns if c not in pre_index]
            logger.warning(
                "[SEMANTIC]  %d columns not in precomputed file — falling back to live encode: %s",
                len(missing), missing[:10],
            )

    # ── 2. Runtime cache (live-encoded results) ───────────────────
    cache_key = str(hash(tuple(all_columns)))
    if cache_key in _COL_EMBEDDINGS_CACHE:
        return _COL_EMBEDDINGS_CACHE[cache_key]

    # ── 3. Live encode via sentence-transformers ──────────────────
    logger.info(
        "[SEMANTIC]  live-encoding embeddings for %d columns (no precomputed file or column mismatch)",
        len(all_columns),
    )
    model = _get_sbert_model()
    col_descriptions = [col.replace("_", " ").lower() for col in all_columns]
    embeddings = model.encode(
        col_descriptions,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    _COL_EMBEDDINGS_CACHE[cache_key] = embeddings
    logger.info("[SEMANTIC]  live-encoded & cached embeddings for %d columns", len(all_columns))
    return embeddings


def retrieve_relevant_columns(
    query: str,
    all_columns: List[str],
    max_columns: int = 10,
) -> List[str]:
    """
    Retrieve the top-N most relevant columns for a user query using
    **pure cosine similarity** between the query embedding and
    precomputed column embeddings (built from rich metadata text:
    column name + description + keywords + group).

    No keyword matching or hard-coded column lists — the embeddings
    carry all the semantic signal.

    Falls back to returning the first `max_columns` columns only if
    sentence-transformers is unavailable.
    """
    if not _SBERT_AVAILABLE or not all_columns:
        logger.warning("[SEMANTIC]  sentence-transformers not available — returning first %d columns", max_columns)
        return all_columns[:max_columns]

    try:
        model = _get_sbert_model()
        col_embeddings = _get_column_embeddings(all_columns)

        # Encode the user query (always live — single sentence, negligible cost)
        query_embedding = model.encode(
            [query.lower()],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        similarities = (query_embedding @ col_embeddings.T).flatten()

        # Take top-N by similarity score
        top_indices = similarities.argsort()[::-1][:max_columns]
        top_cols = [all_columns[i] for i in top_indices if similarities[i] > 0.05]

        logger.info(
            "[SEMANTIC]  top %d columns by cosine similarity:",
            len(top_cols),
        )
        for rank, i in enumerate(top_indices[:max_columns], 1):
            logger.info(
                "[SEMANTIC]    #%d  %-45s  sim=%.3f",
                rank, all_columns[i], float(similarities[i]),
            )

        return top_cols

    except Exception as e:
        logger.warning("[SEMANTIC]  embedding retrieval failed: %s — returning first %d columns", str(e), max_columns)
        return all_columns[:max_columns]


def build_column_context(
    columns: List[str],
    ontology_data: Optional[Dict] = None,
) -> str:
    """Builds compact column context for the LLM prompt."""
    if not ontology_data or "columns" not in ontology_data:
        return "\n".join(f"  - {c}" for c in columns)

    lines = []
    col_data = ontology_data.get("columns", {})
    for col in columns:
        info = col_data.get(col, {})
        group = info.get("group", "?")
        dtype = info.get("dtype", "?")
        pct = round(info.get("non_null_ratio", 0) * 100, 1)
        options = info.get("options", [])
        if options:
            vals = ", ".join(str(o.get("value", "")) for o in options[:5])
            lines.append(f"  - {col} | {group} | {dtype} | {pct}% | e.g. {vals}")
        else:
            lines.append(f"  - {col} | {group} | {dtype} | {pct}%")
    return "\n".join(lines)


def build_synonym_context(query: str) -> str:
    """Resolves synonyms and builds context for the LLM."""
    materials = extract_materials_from_query(query)
    if not materials:
        return ""
    lines = ["Resolved synonyms:"]
    for user_term, canonical in materials:
        cols = get_filter_columns(user_term)
        col_str = ", ".join(cols) if cols else "Cell_stack_sequence"
        lines.append(f'  "{user_term}" → "{canonical}" (search in: {col_str})')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# UNIFIED LLM PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════

PLANNER_PROMPT = """You are a query planner for a scientific database of perovskite solar cell devices.
The table "perovskite_db" has {row_count} device records with {col_count} columns.

YOUR TASK: Convert the user's question into a JSON query plan. Return ONLY valid JSON.

CRITICAL RULES:
- Return ONLY a single JSON object — no prose, no markdown, no explanation.
- Use ONLY columns listed below.
- "filters" MUST be a JSON array, e.g. "filters": [...]
- Use "contains" for stack/sequence columns (pipe-separated values like "ITO | PEDOT:PSS | ...").
- Use "equals" or "contains" for Cell_architecture (values: "nip", "pin").

THINK STEP-BY-STEP before choosing the operation:
1. What entity is the user asking about? (devices, papers/DOIs, materials, statistics, columns/schema)
2. Are they asking to LIST items, COUNT them, COMPARE groups, or get STATISTICS?
3. Does the query imply grouping (by paper, by material, by architecture)?
4. Is this a SCHEMA question about the database itself (columns, fields, metadata)?

OPERATION SELECTION GUIDE:
- "What columns exist / what fields / schema / column names..." → list_columns
- "Describe columns / column details / what does column X contain..." → describe_columns
- "What is the average/mean/median X..." → global_aggregate
- "Which papers / DOIs..." → groupby_aggregate with group_by="Ref_DOI_number"
- "Top N devices / best devices..." → top_k (k=10 default unless specified)
- "How many / count..." → count
- "Compare X vs Y / difference between..." → groupby_aggregate
- "Distribution of / statistics for..." → distribution (single column stats)
- "What materials / unique values..." → unique
- "List / show / find devices where..." → filter (limit=20 default)
- "Correlation between X and Y..." → correlation
- "Random / sample..." → sample

OPERATIONS:
  list_columns        — return column names (optionally filtered by "pattern")
  describe_columns    — return column metadata (dtype, non-null%, top values)
  global_aggregate    — compute aggregation without grouping (mean, median, etc.)
  filter              — return matching rows (default limit=20)
  count               — count matching rows
  top_k               — top K rows sorted by a column (default k=10)
  groupby_aggregate   — group by a column and aggregate metrics
  unique              — list distinct values (tokenize=true for pipe-separated stacks)
  distribution        — compute statistics for a numeric column
  correlation         — compute correlation between two numeric columns
  sample              — return random sample rows

FILTER OPERATORS:
  equals, contains, gt, gte, lt, lte, in, not_null, is_null, regex

  not_null — exclude rows where the column is NaN/empty (no value needed)
  is_null  — include only rows where the column is NaN/empty (no value needed)
  regex    — match using Python regex pattern

IMPORTANT RULES:
  - When grouping by Ref_DOI_number, ALWAYS add {{"column": "Ref_DOI_number", "op": "not_null"}} to filters
  - For material matching in stack columns, use "contains" (the system auto-handles variants)

FILTER FORMAT (must always be an array):
  "filters": [
    {{"column": "X", "op": "contains", "value": "Y"}},
    {{"column": ["X", "Y"], "op": "contains", "value": "Z"}},
    {{"any_of": [{{"column": "X", "op": "equals", "value": "A"}}, ...]}}
  ]

JSON SCHEMA:
{{
  "operation": "list_columns|describe_columns|global_aggregate|filter|count|top_k|groupby_aggregate|unique|distribution|correlation|sample",
  "table": "perovskite_db",
  "filters": [<array of filter conditions>],
  "group_by": "column_name",
  "aggregation": {{"column": "func"}},
  "sort_by": "column_name",
  "ascending": false,
  "k": 10,
  "limit": 20,
  "select_columns": ["col1", "col2"],
  "column": "column_name",
  "column_x": "col1",
  "column_y": "col2",
  "tokenize": false,
  "top_k": 50,
  "pattern": "keyword"
}}

Only include keys relevant to the chosen operation.

FEW-SHOT EXAMPLES:

Q: "what columns exist in the perovskite database?"
A: {{
  "operation": "list_columns",
  "table": "perovskite_db"
}}

Q: "describe the PCE and Voc columns"
A: {{
  "operation": "describe_columns",
  "table": "perovskite_db",
  "select_columns": ["JV_default_PCE", "JV_default_Voc"]
}}

Q: "what is the average PCE?"
A: {{
  "operation": "global_aggregate",
  "table": "perovskite_db",
  "filters": [],
  "aggregation": {{"JV_default_PCE": "mean"}}
}}

Q: "what columns are related to stability?"
A: {{
  "operation": "list_columns",
  "table": "perovskite_db",
  "pattern": "stability"
}}

Q: "Which DOI papers have the best PCE when using PEDOT:PSS?"
A: {{
  "operation": "groupby_aggregate",
  "table": "perovskite_db",
  "filters": [
    {{"column": "HTL_stack_sequence", "op": "contains", "value": "PEDOT:PSS"}},
    {{"column": "Ref_DOI_number", "op": "not_null"}}
  ],
  "group_by": "Ref_DOI_number",
  "aggregation": {{"JV_default_PCE": "max"}},
  "sort_by": "JV_default_PCE",
  "ascending": false,
  "k": 10,
  "select_columns": ["Ref_DOI_number", "JV_default_PCE"]
}}

Q: "Top 10 highest PCE devices"
A: {{
  "operation": "top_k",
  "table": "perovskite_db",
  "filters": [],
  "sort_by": "JV_default_PCE",
  "ascending": false,
  "k": 10,
  "select_columns": ["Ref_DOI_number", "JV_default_PCE", "Cell_architecture", "HTL_stack_sequence", "ETL_stack_sequence", "Perovskite_composition_short_form"]
}}

Q: "Compare n-i-p vs p-i-n average PCE"
A: {{
  "operation": "groupby_aggregate",
  "table": "perovskite_db",
  "filters": [],
  "group_by": "Cell_architecture",
  "aggregation": {{"JV_default_PCE": "mean", "JV_default_Voc": "mean", "JV_default_FF": "mean"}}
}}

Q: "How many devices use SnO2 as ETL?"
A: {{
  "operation": "count",
  "table": "perovskite_db",
  "filters": [{{"column": "ETL_stack_sequence", "op": "contains", "value": "SnO2"}}]
}}

{synonym_context}

RELEVANT COLUMNS (from {col_count} total):
{column_context}

USER QUESTION:
{query}

Return ONLY the JSON query plan:"""


# ═══════════════════════════════════════════════════════════════
# REPAIR PROMPT (used when validation fails)
# ═══════════════════════════════════════════════════════════════

REPAIR_PROMPT = """Your previous query plan had validation errors.
Fix the errors below and return ONLY the corrected JSON query plan.

ORIGINAL PLAN:
{plan_json}

VALIDATION ERRORS:
{errors}

{suggestions}

AVAILABLE COLUMNS:
{available_columns}

Return ONLY the corrected JSON query plan:"""


# ═══════════════════════════════════════════════════════════════
# JSON EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _extract_json_object(text: str) -> Optional[str]:
    """
    Extracts the outermost JSON object from text using brace counting.
    More reliable than regex for nested JSON.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ═══════════════════════════════════════════════════════════════
# PLAN GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_plan(
    query: str,
    llm,
    engine=None,
    table_name: str = "perovskite_db",
    ontology_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Generates a structured query plan from a natural language query.

    Returns:
        {plan, table_name, relevant_columns, generation_time_ms}
    """
    t0 = time.perf_counter()

    # Get columns
    if engine and table_name in engine.datasets:
        all_columns = list(engine.datasets[table_name].columns)
        row_count = len(engine.datasets[table_name])
    else:
        all_columns = []
        row_count = 0

    logger.info("[PLANNER]  step 1/5 — retrieving relevant columns  total_cols=%d", len(all_columns))

    # Step 1: Retrieve relevant columns (pure cosine similarity on precomputed embeddings)
    relevant_columns = retrieve_relevant_columns(
        query, all_columns, max_columns=10
    )
    logger.info("[PLANNER]  step 1/5 done  relevant_cols=%d: %s",
                len(relevant_columns), relevant_columns)

    logger.debug("[PLANNER]  step 2/5 — building column context + synonym resolution")
    # Step 2: Build context
    column_context = build_column_context(relevant_columns, ontology_data)
    synonym_context = build_synonym_context(query)
    if synonym_context:
        logger.info("[PLANNER]  synonyms resolved:\n%s", synonym_context)

    logger.debug("[PLANNER]  step 3/5 — building LLM prompt  rows=%d  cols=%d",
                 row_count, len(all_columns))
    # Step 3: Build prompt
    prompt = PLANNER_PROMPT.format(
        row_count=row_count,
        col_count=len(all_columns),
        column_context=column_context,
        synonym_context=synonym_context,
        query=query,
    )

    logger.info("[PLANNER]  step 4/5 — calling LLM to generate query plan …")
    # Step 4: Call LLM
    response = llm.invoke(prompt)

    # Robust content extraction for all Gemini response formats
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # Handle list-of-blocks (Gemini multi-part responses)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    content = str(content).strip()
    logger.debug("[PLANNER]  LLM raw output (first 300 chars): %s", content[:300])

    logger.debug("[PLANNER]  step 5/5 — extracting JSON from LLM response")
    # Step 5: Extract JSON — find outermost { ... }
    json_str = _extract_json_object(content)
    if not json_str:
        logger.error("[PLANNER]  could not find a JSON object in LLM response")
        raise ValueError(f"Planner did not produce valid JSON. Raw: {content[:300]}")

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("[PLANNER]  JSON parse error: %s", str(e))
        raise ValueError(f"Invalid JSON from planner: {e}. Raw: {json_str[:300]}")

    # Ensure plan is a dict (not a string or list)
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Planner returned a JSON string, not object: {plan[:200]}")
    if not isinstance(plan, dict):
        raise ValueError(f"Planner returned {type(plan).__name__}, not dict: {str(plan)[:200]}")

    logger.info("[PLANNER]  plan extracted: operation=%r  filters=%d",
                plan.get("operation"), len(plan.get("filters", []) or []))

    logger.debug("[PLANNER]  post-processing plan (synonym injection, provenance, defaults)")
    # Step 6: Post-process
    plan = _post_process(plan, query, all_columns)
    logger.debug("[PLANNER]  post-process done: select_cols=%s  group_by=%r  sort_by=%r",
                 plan.get("select_columns"), plan.get("group_by"), plan.get("sort_by"))

    dt_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("[PLANNER]  ✅  plan generation done in %.1f ms", dt_ms)

    return {
        "plan": plan,
        "table_name": table_name,
        "relevant_columns": relevant_columns,
        "generation_time_ms": dt_ms,
    }


# ═══════════════════════════════════════════════════════════════
# UNIFIED PLAN GENERATION (with validation + repair)
# ═══════════════════════════════════════════════════════════════

def generate_unified_plan(
    query: str,
    llm,
    engine=None,
    table_name: str = "perovskite_db",
    ontology_data: Optional[Dict] = None,
    max_repair_attempts: int = 1,
) -> Dict[str, Any]:
    """
    Generates a plan, validates it, and repairs if invalid (up to max_repair_attempts).

    Returns:
        {plan, table_name, relevant_columns, generation_time_ms, repaired}
    """
    from query_executor import validate_plan, ALLOWED_OPERATIONS

    logger.info("[UNIFIED_PLAN]  generating initial plan …")
    result = generate_plan(query, llm, engine, table_name, ontology_data)
    plan = result["plan"]

    # Schema operations don't need column validation
    if plan.get("operation") in ("list_columns", "describe_columns"):
        logger.info("[UNIFIED_PLAN]  schema operation — skipping column validation")
        result["repaired"] = False
        return result

    # Validate against actual columns
    if engine and table_name in engine.datasets:
        columns = list(engine.datasets[table_name].columns)
    else:
        columns = []

    logger.info("[UNIFIED_PLAN]  validating plan against %d columns …", len(columns))
    validation = validate_plan(plan, columns)
    if validation.get("valid"):
        logger.info("[UNIFIED_PLAN]  ✅  plan is valid — no repair needed")
        result["repaired"] = False
        return result

    # Repair loop
    errors = validation.get("errors", [])
    suggestions = validation.get("suggestions", [])
    logger.warning("[UNIFIED_PLAN]  plan invalid! errors: %s", errors)

    for attempt in range(max_repair_attempts):
        logger.info("[UNIFIED_PLAN]  repair attempt %d/%d …", attempt + 1, max_repair_attempts)
        repair_prompt = REPAIR_PROMPT.format(
            plan_json=json.dumps(plan, indent=2),
            errors="\n".join(f"- {e}" for e in errors),
            suggestions="\n".join(f"- {s}" for s in suggestions) if suggestions else "(none)",
            available_columns=", ".join(columns[:50]),
        )

        try:
            logger.info("[UNIFIED_PLAN]  calling LLM for plan repair …")
            response = llm.invoke(repair_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                parts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
                content = "\n".join(parts)
            content = str(content).strip()
            logger.debug("[UNIFIED_PLAN]  repair LLM response (first 200): %s", content[:200])

            json_str = _extract_json_object(content)
            if json_str:
                repaired_plan = json.loads(json_str)
                if isinstance(repaired_plan, dict):
                    repaired_plan = _post_process(repaired_plan, query, columns)
                    re_val = validate_plan(repaired_plan, columns)
                    if re_val.get("valid"):
                        logger.info("[UNIFIED_PLAN]  ✅  repaired after attempt %d", attempt + 1)
                        result["plan"] = repaired_plan
                        result["repaired"] = True
                        return result
                    else:
                        errors = re_val.get("errors", [])
                        suggestions = re_val.get("suggestions", [])
                        logger.warning("[UNIFIED_PLAN]  repair attempt %d still invalid: %s",
                                       attempt + 1, errors)
                        plan = repaired_plan
        except Exception as e:
            logger.error("[UNIFIED_PLAN]  repair error on attempt %d: %s", attempt + 1, str(e))

    # Return best-effort plan even if still invalid
    logger.warning("[UNIFIED_PLAN]  ⚠️  all repair attempts exhausted — returning best-effort plan")
    result["repaired"] = False
    return result


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════

def _post_process(plan: Dict, query: str, all_columns: List[str]) -> Dict:
    """
    Post-processes the plan:
      - Normalize filters (dict → list)
      - Resolve synonym values in filters
      - Smart defaults for k, limit, select_columns
      - Inject provenance columns for device queries
      - Auto-inject not_null for DOI-grouped queries
      - Validate column references
    """
    col_set = set(all_columns) if all_columns else set()

    # Normalize filters from dict to list
    filters = plan.get("filters")
    if isinstance(filters, dict):
        plan["filters"] = [filters]
    elif not isinstance(filters, list):
        plan["filters"] = []

    # Resolve filter values and auto-upgrade to regex for known materials
    for f in plan.get("filters", []):
        if not isinstance(f, dict):
            continue
        if "any_of" in f:
            for cond in f["any_of"]:
                if isinstance(cond, dict):
                    _resolve_filter(cond)
        else:
            _resolve_filter(f)

    op = plan.get("operation", "")
    gb = plan.get("group_by", "")

    # ── Auto-inject not_null for DOI-grouped queries ──
    if gb == "Ref_DOI_number":
        existing_filters = plan.get("filters", [])
        has_doi_not_null = any(
            isinstance(f, dict)
            and f.get("column") == "Ref_DOI_number"
            and f.get("op") == "not_null"
            for f in existing_filters
        )
        if not has_doi_not_null:
            existing_filters.append({"column": "Ref_DOI_number", "op": "not_null"})
            plan["filters"] = existing_filters

    # Smart defaults for top_k
    if op == "top_k":
        if plan.get("k", 0) < 5:
            plan["k"] = 10
        # Enrich select_columns if too sparse
        sel = plan.get("select_columns", [])
        if not isinstance(sel, list) or len(sel) < 3:
            plan["select_columns"] = [
                "Ref_DOI_number", "JV_default_PCE", "JV_default_Voc",
                "JV_default_Jsc", "JV_default_FF", "Cell_architecture",
                "HTL_stack_sequence", "ETL_stack_sequence",
                "Perovskite_composition_short_form",
            ]

    # Smart defaults for filter
    if op == "filter":
        if not plan.get("limit") or plan["limit"] < 5:
            plan["limit"] = 20

    # Ensure provenance columns for device-level queries
    if op in ("filter", "top_k") and "select_columns" in plan:
        sel = plan["select_columns"]
        if isinstance(sel, list):
            for prov in ["Ref_DOI_number"]:
                if prov not in sel and prov in col_set:
                    sel.insert(0, prov)

    # Validate column references — remove invalid ones silently
    if col_set:
        if "select_columns" in plan and isinstance(plan["select_columns"], list):
            plan["select_columns"] = [c for c in plan["select_columns"] if c in col_set]
        if "group_by" in plan and plan["group_by"] not in col_set:
            plan["group_by"] = None
        if "sort_by" in plan and plan["sort_by"] not in col_set:
            plan["sort_by"] = None

    return plan


def _resolve_filter(cond: Dict) -> None:
    """Resolves synonym in a single filter condition."""
    val = cond.get("value")
    if not isinstance(val, str):
        return

    resolved = resolve_synonym(val)
    if resolved != val:
        cond["value"] = resolved

    # Normalize architecture values
    col = cond.get("column", "")
    if isinstance(col, str) and col == "Cell_architecture":
        v = val.lower().replace("–", "-").replace("—", "-").replace(" ", "")
        if v in {"p-i-n", "pin", "inverted"}:
            cond["value"] = "pin"
            cond["op"] = "contains"
        elif v in {"n-i-p", "nip", "regular"}:
            cond["value"] = "nip"
            cond["op"] = "contains"
