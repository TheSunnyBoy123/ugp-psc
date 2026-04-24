"""
intent_classifier.py — 8-Class Intent Classifier (LLM-first for formula queries)

Classes:
  DATA_QUERY               — needs DB query, no explanation
  DATA_PLUS_EXPLANATION    — needs DB query + LLM interpretation
  DOMAIN_ONLY             — LLM answers directly (no data)
  DESIGN                  — device stack design request
  MATERIAL_LOOKUP         — material property lookup via Materials Project
  PROPERTY_PREDICT        — regression-based property prediction
  MULTI_STEP              — complex query needing DB + MP API + LLM synthesis
  DISAMBIGUATION_REQUIRED — query too ambiguous

Strategy:
  - DESIGN keywords always win (fast path)
  - If query contains a chemical formula → LLM-first classification (smartest routing)
  - Otherwise → keyword rules for fast, deterministic classification
  - Final fallback → LLM classification
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("psc_agent")


# ═══════════════════════════════════════════════════════════════
# KEYWORD SETS
# ═══════════════════════════════════════════════════════════════

DATA_KEYWORDS = {
    "pce", "voc", "jsc", "ff", "efficiency", "fill factor",
    "device", "devices", "cell", "cells", "papers", "paper",
    "doi", "database", "dataset", "db", "rows", "table",
    "filter", "show me", "find", "search", "list", "get",
    "average", "mean", "median", "max", "min", "std",
    "count", "number of", "how many",
    "distribution", "histogram", "plot", "graph", "chart",
    "highest", "lowest", "best", "worst", "top",
    "above", "below", "greater than", "less than",
    "stack", "substrate", "electrode", "backcontact",
    "compare", "comparison", "versus", "vs",
    # Schema / metadata
    "columns", "column names", "column", "schema", "fields",
    "describe", "metadata",
}

DOMAIN_TRIGGERS = {
    "what is", "what does", "what are",
    "define", "definition of",
    "role of", "function of", "purpose of",
    "why", "how does", "explain", "describe",
    "difference between",
    "advantages of", "disadvantages of",
    "typical stack", "common stack",
}

DESIGN_KEYWORDS = {
    "design", "propose", "suggest a stack",
    "build a device", "create a device",
    "predict performance", "estimate pce",
    "new device", "novel stack", "optimal stack",
}

MATERIAL_LOOKUP_KEYWORDS = {
    "properties of", "band gap of", "bandgap of",
    "formation energy of", "density of",
    "crystal structure of", "space group of",
    "is it stable", "thermodynamic stability",
    "look up", "lookup", "materials project",
    "mp data", "electronic structure",
    "what is the band gap", "what is the density",
    "what is the formation energy",
    # Crystal / structure
    "crystal shape", "crystal system", "crystal type",
    "lattice", "lattice parameter", "unit cell",
    "space group", "point group",
    # Hull energy / stability
    "hull gap", "hull energy", "energy above hull", "e above hull",
    "ehull", "e_hull", "decomposition energy",
    "is it stable", "stability of", "thermodynamic stability",
    "polymorph", "polymorphs", "crystal phase", "phase of",
    # Weight / density
    "weight of", "molecular weight", "molar mass",
}

SUBSTITUTE_KEYWORDS = {
    "substitute", "alternative", "replace", "replacement",
    "instead of", "comparable", "similar to", "swap",
    "equivalent", "analogous", "competing material",
    "what else", "other materials", "other chemicals",
    "can i use", "switch from", "swap out",
}

PROPERTY_PREDICT_KEYWORDS = {
    "predict", "regression", "estimate",
    "forecast", "what would", "if i use",
    "expected value", "property prediction",
    "predicted", "regress", "train model",
    "what pce", "what efficiency",
}


# ═══════════════════════════════════════════════════════════════
# CHEMICAL FORMULA / MATERIAL ID PATTERNS
# ═══════════════════════════════════════════════════════════════

# Matches Materials Project IDs like mp-390, mp-1234, mp 390, MP-390
_MP_ID_RE = re.compile(r'\bmp[\s-]+(\d+)\b', re.IGNORECASE)

# Matches chemical formulas like TiO2, MAPbI3, Cs0.05FA0.85, SnO2
# Requires at least one uppercase letter followed by optional lowercase + digits
_FORMULA_RE = re.compile(
    r'\b([A-Z][a-z]?(?:\d*\.?\d+)?(?:[A-Z][a-z]?(?:\d*\.?\d+)?){0,8})\b'
)

# Well-known PSC material names (not always caught by formula regex)
_KNOWN_MATERIALS = {
    "spiro-meotad", "spiro", "pedot:pss", "pedot", "pcbm", "ptaa",
    "p3ht", "c60", "bcp", "bathocuproine",
}


def _has_chemical_formula(q: str) -> bool:
    """Check if query contains a likely chemical formula."""
    q_lower = q.lower()
    # Check known material names first
    for mat in _KNOWN_MATERIALS:
        if mat in q_lower:
            return True
    for m in _FORMULA_RE.finditer(q):
        candidate = m.group(0)
        # Must have at least 2 chars and contain an uppercase letter
        if len(candidate) >= 2 and any(c.isupper() for c in candidate):
            # Exclude common English words that match
            if candidate.lower() not in {
                "the", "and", "for", "with", "from", "this", "that",
                "use", "can", "how", "does", "are", "not", "has",
                "its", "was", "but", "also", "then", "than",
                "pce", "etl", "htl", "led", "ito", "fto",
                "some", "which", "what", "have", "will",
            }:
                return True
    return False


def _extract_formulas(q: str) -> list:
    """Extract all chemical formula candidates from query."""
    results = []
    q_lower = q.lower()
    for mat in _KNOWN_MATERIALS:
        if mat in q_lower:
            results.append(mat)
    for m in _FORMULA_RE.finditer(q):
        candidate = m.group(0)
        if len(candidate) >= 2 and any(c.isupper() for c in candidate):
            if candidate.lower() not in {
                "the", "and", "for", "with", "from", "this", "that",
                "use", "can", "how", "does", "are", "not", "has",
                "its", "was", "but", "also", "then", "than",
                "pce", "etl", "htl", "led", "ito", "fto",
                "some", "which", "what", "have", "will",
            }:
                results.append(candidate)
    return results


# ═══════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def _has_keyword(q: str, keywords: set) -> bool:
    """Word-boundary keyword match to avoid substring false positives."""
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', q):
            return True
    return False


def classify_intent(query: str, llm=None) -> str:
    """
    Classifies query intent.

    Strategy:
      1. DESIGN keywords always win (fast path, no ambiguity)
      2. If chemical formula detected → LLM-first (smartest routing)
      3. Otherwise → keyword rules for fast, deterministic classification
      4. Final fallback → LLM classification
    """
    q = query.lower().strip()
    logger.debug("[INTENT]  starting classification  query=%r", query)

    # ── 0. Materials Project ID → instant MATERIAL_LOOKUP ──
    mp_id_match = _MP_ID_RE.search(query)
    if mp_id_match:
        logger.info("[INTENT]  MP ID detected (mp-%s) → MATERIAL_LOOKUP", mp_id_match.group(1))
        return "MATERIAL_LOOKUP"

    has_formula = _has_chemical_formula(query)
    has_design_kw = any(kw in q for kw in DESIGN_KEYWORDS)
    has_substitute_kw = _has_keyword(q, SUBSTITUTE_KEYWORDS)

    # Split data keywords into "strong" (always mean data) and "weak"
    strong_data = {
        "pce", "voc", "jsc", "ff", "efficiency", "fill factor",
        "doi", "database", "dataset", "db", "rows", "table",
        "filter", "show me", "find", "search", "list", "get",
        "average", "mean", "median", "std",
        "count", "number of", "how many",
        "distribution", "histogram", "plot", "graph", "chart",
        "highest", "lowest", "best", "worst", "top",
        "above", "below", "greater than", "less than",
        "compare", "comparison", "versus", "vs",
        "columns", "column names", "column", "schema", "fields",
        "describe", "metadata",
    }
    weak_data = {
        "device", "devices", "cell", "cells", "papers", "paper",
        "stack", "substrate", "electrode", "backcontact",
        "max", "min",
    }

    has_strong_data = _has_keyword(q, strong_data)
    has_weak_data = _has_keyword(q, weak_data)
    has_domain_trigger = any(kw in q for kw in DOMAIN_TRIGGERS)
    has_material_lookup = any(kw in q for kw in MATERIAL_LOOKUP_KEYWORDS)
    has_property_predict = any(kw in q for kw in PROPERTY_PREDICT_KEYWORDS)

    logger.debug(
        "[INTENT]  flags: formula=%s  design=%s  substitute=%s  "
        "material_lookup=%s  property_predict=%s  "
        "domain_trigger=%s  strong_data=%s  weak_data=%s",
        has_formula, has_design_kw, has_substitute_kw,
        has_material_lookup, has_property_predict,
        has_domain_trigger, has_strong_data, has_weak_data,
    )

    # ── 1. DESIGN always wins (unambiguous) ──
    if has_design_kw:
        logger.info("[INTENT]  rule hit → DESIGN (design keyword matched)")
        return "DESIGN"

    # ── 2. Chemical formula detected → smart routing ──
    if has_formula:
        formulas = _extract_formulas(query)
        logger.info("[INTENT]  formula detected: %s — using smart routing", formulas)

        # 2a. Substitute/alternative + formula → MULTI_STEP
        if has_substitute_kw:
            logger.info("[INTENT]  → MULTI_STEP (formula + substitute keyword)")
            return "MULTI_STEP"

        # 2b. Strong data keywords + formula → data query (hybrid)
        if has_strong_data:
            logger.info("[INTENT]  → DATA_PLUS_EXPLANATION (formula + strong data keyword)")
            return "DATA_PLUS_EXPLANATION"

        # 2c. Use LLM for everything else with a formula
        #     (catches hull gap, property questions, comparisons, etc.)
        if llm:
            logger.info("[INTENT]  formula present, no clear rule → LLM classifier")
            return _llm_classify(query, llm)

        # 2d. No LLM available — fall back to heuristic
        if has_material_lookup:
            logger.info("[INTENT]  → MATERIAL_LOOKUP (formula + MP keyword, no LLM)")
            return "MATERIAL_LOOKUP"
        logger.info("[INTENT]  → MATERIAL_LOOKUP (formula present, default, no LLM)")
        return "MATERIAL_LOOKUP"

    # ── 3. No formula — keyword rules (fast path) ──

    # 3a. Material lookup keywords (without formula, still useful)
    if has_material_lookup and not has_strong_data:
        logger.info("[INTENT]  rule hit → MATERIAL_LOOKUP (MP keyword matched)")
        return "MATERIAL_LOOKUP"

    # 3b. Substitute keywords without formula → MULTI_STEP
    if has_substitute_kw:
        logger.info("[INTENT]  rule hit → MULTI_STEP (substitute keyword, no formula)")
        return "MULTI_STEP"

    # 3c. Property prediction
    if has_property_predict:
        logger.info("[INTENT]  rule hit → PROPERTY_PREDICT (prediction keyword matched)")
        return "PROPERTY_PREDICT"

    # 3d. Domain trigger + no strong data → domain answer
    if has_domain_trigger and not has_strong_data:
        logger.info("[INTENT]  rule hit → DOMAIN_ONLY (domain trigger, no data keyword)")
        return "DOMAIN_ONLY"

    # 3e. Domain trigger + strong data → hybrid
    if has_domain_trigger and has_strong_data:
        logger.info("[INTENT]  rule hit → DATA_PLUS_EXPLANATION (domain trigger + data keyword)")
        return "DATA_PLUS_EXPLANATION"

    # 3f. Any data keyword → data query
    if has_strong_data or has_weak_data:
        logger.info("[INTENT]  rule hit → DATA_QUERY (data keyword matched)")
        return "DATA_QUERY"

    # ── 4. LLM fallback ──
    if llm:
        logger.info("[INTENT]  no rule matched → falling back to LLM classifier")
        return _llm_classify(query, llm)

    logger.info("[INTENT]  no rule matched, no LLM → defaulting to DOMAIN_ONLY")
    return "DOMAIN_ONLY"


def _llm_classify(query: str, llm) -> str:
    """LLM-based classification with full tool awareness."""
    logger.info("[INTENT/LLM]  calling LLM for intent classification …")
    prompt = f"""You are a query router for a perovskite solar cell research assistant.

The system has these capabilities:
1. DATA_QUERY — Query a database of 43,000+ real perovskite solar cell device records
   (performance metrics, materials, architectures, papers). Use when the answer is IN the database.
2. DATA_PLUS_EXPLANATION — Same as DATA_QUERY but also adds scientific interpretation.
3. MATERIAL_LOOKUP — Look up a specific material's computed properties (band gap, formation energy,
   energy above hull, crystal structure, density, stability) from the Materials Project API.
   Use for: "what is the hull gap for X", "is X stable", "band gap of X", any specific
   property question about a chemical formula.
4. MULTI_STEP — Complex queries requiring MULTIPLE data sources. Use when the answer needs:
   - Searching the database AND looking up materials properties
   - Finding alternatives/substitutes (need DB for what's used + MP for properties)
   - Comparing materials across both experimental data and computed properties
   - "What can I substitute X with" / "alternatives to X" / "compare X vs Y"
5. DESIGN — User wants to design/propose a new device stack.
6. PROPERTY_PREDICT — Predict a property value using ML regression on the database.
7. DOMAIN_ONLY — General scientific knowledge question that doesn't need any data lookup.
   Use ONLY when the answer is purely conceptual/theoretical with no specific material or data.
8. DISAMBIGUATION_REQUIRED — Query is too vague to determine intent.

CRITICAL: If the query mentions a specific chemical formula AND asks about its properties,
stability, hull energy, band gap, etc. → MATERIAL_LOOKUP (not DOMAIN_ONLY).
If the query asks about substitutes, alternatives, or comparisons → MULTI_STEP.
DOMAIN_ONLY should be a LAST RESORT — prefer data-grounded answers whenever possible.

Query: "{query}"

Reply with ONLY the category name, nothing else."""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            first = content[0] if content else ""
            content = first.get("text", str(first)) if isinstance(first, dict) else str(first)
        text = str(content).strip().upper().replace(" ", "_")
        logger.debug("[INTENT/LLM]  raw response=%r  parsed=%r", str(content).strip(), text)

        valid = {
            "DATA_QUERY", "DATA_PLUS_EXPLANATION", "DOMAIN_ONLY",
            "DESIGN", "MATERIAL_LOOKUP", "PROPERTY_PREDICT",
            "MULTI_STEP", "DISAMBIGUATION_REQUIRED",
        }
        if text in valid:
            logger.info("[INTENT/LLM]  LLM classified → %s", text)
            return text
        logger.warning("[INTENT/LLM]  LLM returned unrecognised class %r — defaulting to MATERIAL_LOOKUP", text)
        # Default to MATERIAL_LOOKUP instead of DOMAIN_ONLY when we know there's a formula
        return "MATERIAL_LOOKUP"
    except Exception as e:
        logger.error("[INTENT/LLM]  LLM call failed: %s — defaulting to MATERIAL_LOOKUP", str(e))

    return "MATERIAL_LOOKUP"
