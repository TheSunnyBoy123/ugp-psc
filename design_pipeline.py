"""
design_pipeline.py — Autonomous Device Stack Design & Performance Prediction

Pipeline:
  1. Parse user request → normalized stack {substrate, ETL, perovskite, HTL, backcontact}
  2. Find similar devices in DB via weighted similarity scoring
  3. Compute performance statistics (median, IQR, range)
  4. Estimate confidence (LOW/MEDIUM/HIGH based on sample size)
  5. Physics validation (flag unrealistic combinations)
"""

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ontology import PHYSICS_BOUNDS, resolve_synonym


# ═══════════════════════════════════════════════════════════════
# SIMILARITY WEIGHTS
# ═══════════════════════════════════════════════════════════════

LAYER_WEIGHTS = {
    "Cell_architecture":  3,
    "HTL_stack_sequence":  3,
    "ETL_stack_sequence":  2,
    "Substrate_stack_sequence": 1,
    "Backcontact_stack_sequence": 1,
    "Perovskite_composition_short_form": 2,
}

PERFORMANCE_METRICS = [
    "JV_default_PCE", "JV_default_Voc",
    "JV_default_Jsc", "JV_default_FF",
]


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_design_pipeline(
    query: str,
    engine,
    llm,
    mp_client=None,
    n_similar: int = 100,
) -> Dict[str, Any]:
    """
    Full design → predict → validate pipeline.

    Returns:
        {design, prediction, validation, n_candidates}
    """
    # Step 1: Parse user request into a stack
    stack = _parse_design_request(query, llm)

    # Step 2: Find similar devices
    if "perovskite_db" not in engine.datasets:
        return {"error": "perovskite_db not loaded"}

    df = engine.datasets["perovskite_db"]
    similar = _find_similar_devices(df, stack, n_similar)

    # Step 3: Compute performance statistics
    prediction = _compute_performance(similar)

    # Step 4: Confidence estimation
    n = prediction.get("n_similar", 0)
    if n < 30:
        confidence = "LOW"
    elif n <= 200:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    # Step 5: Physics validation
    warnings = _validate_design(stack, prediction)

    return {
        "design": {
            "stack": stack,
            "architecture": stack.get("architecture", ""),
            "rationale": f"Based on {n} similar devices in the database",
        },
        "prediction": {
            "predicted_performance": prediction.get("metrics", {}),
            "n_similar_devices": n,
        },
        "validation": {
            "confidence": confidence,
            "warnings": warnings,
        },
        "n_candidates": n,
    }


# ═══════════════════════════════════════════════════════════════
# STACK PARSING
# ═══════════════════════════════════════════════════════════════

def _parse_design_request(query: str, llm) -> Dict[str, str]:
    """Uses LLM to extract a structured stack from the user's request."""
    prompt = f"""Extract a perovskite solar cell device stack from this request.
Return ONLY a JSON object with these keys (use null if not specified):

{{
  "substrate": "e.g. ITO, FTO",
  "etl": "e.g. SnO2, TiO2, PCBM",
  "perovskite": "e.g. MAPbI3, FAPbI3",
  "htl": "e.g. Spiro-MeOTAD, PTAA, PEDOT:PSS",
  "backcontact": "e.g. Au, Ag, Carbon",
  "architecture": "nip or pin"
}}

User request: "{query}"

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            first = content[0] if content else ""
            content = first.get("text", str(first)) if isinstance(first, dict) else str(first)
        content = str(content).strip()

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            stack = json.loads(match.group())
            # Resolve synonyms
            for key in ["substrate", "etl", "perovskite", "htl", "backcontact"]:
                if stack.get(key):
                    stack[key] = resolve_synonym(stack[key])
            return stack
    except Exception:
        pass

    # Fallback: extract from keywords
    return _fallback_parse(query)


def _fallback_parse(query: str) -> Dict[str, str]:
    """Simple keyword-based stack extraction."""
    q = query.lower()
    stack = {
        "substrate": None, "etl": None, "perovskite": None,
        "htl": None, "backcontact": None, "architecture": None,
    }
    kw_map = {
        "ito": ("substrate", "ITO"), "fto": ("substrate", "FTO"),
        "sno2": ("etl", "SnO2"), "tio2": ("etl", "TiO2"),
        "pcbm": ("etl", "PCBM"), "c60": ("etl", "C60"),
        "spiro": ("htl", "Spiro-MeOTAD"), "ptaa": ("htl", "PTAA"),
        "pedot": ("htl", "PEDOT:PSS"), "nio": ("htl", "NiO"),
        "au": ("backcontact", "Au"), "ag": ("backcontact", "Ag"),
        "carbon": ("backcontact", "Carbon"),
        "n-i-p": ("architecture", "nip"), "nip": ("architecture", "nip"),
        "p-i-n": ("architecture", "pin"), "pin": ("architecture", "pin"),
    }
    for kw, (layer, val) in kw_map.items():
        if kw in q:
            stack[layer] = val
    return stack


# ═══════════════════════════════════════════════════════════════
# SIMILARITY SEARCH
# ═══════════════════════════════════════════════════════════════

def _find_similar_devices(
    df: pd.DataFrame,
    stack: Dict[str, str],
    n: int = 100,
) -> pd.DataFrame:
    """Finds the most similar devices using weighted layer matching."""
    col_map = {
        "architecture":  "Cell_architecture",
        "htl":           "HTL_stack_sequence",
        "etl":           "ETL_stack_sequence",
        "substrate":     "Substrate_stack_sequence",
        "backcontact":   "Backcontact_stack_sequence",
        "perovskite":    "Perovskite_composition_short_form",
    }

    scores = pd.Series(0.0, index=df.index)

    for layer, target in stack.items():
        if not target:
            continue
        col = col_map.get(layer)
        if not col or col not in df.columns:
            continue
        weight = LAYER_WEIGHTS.get(col, 1)

        if layer == "architecture":
            # Architecture: exact match
            mask = df[col].astype(str).str.contains(target, case=False, na=False)
        else:
            # Stack columns: contains match
            mask = df[col].astype(str).str.contains(
                re.escape(target), case=False, na=False
            )
        scores += mask.astype(float) * weight

    # Filter to devices with some match
    matched = df[scores > 0].copy()
    matched["_sim_score"] = scores[scores > 0]
    matched = matched.sort_values("_sim_score", ascending=False).head(n)
    return matched


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE PREDICTION
# ═══════════════════════════════════════════════════════════════

def _compute_performance(similar: pd.DataFrame) -> Dict[str, Any]:
    """Computes performance statistics from similar devices."""
    metrics = {}
    for col in PERFORMANCE_METRICS:
        if col not in similar.columns:
            continue
        vals = pd.to_numeric(similar[col], errors="coerce").dropna()
        if vals.empty:
            continue
        short = col.replace("JV_default_", "")
        metrics[short] = {
            "median": round(float(vals.median()), 2),
            "mean": round(float(vals.mean()), 2),
            "IQR": [
                round(float(vals.quantile(0.25)), 2),
                round(float(vals.quantile(0.75)), 2),
            ],
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "count": int(len(vals)),
        }
    return {"metrics": metrics, "n_similar": len(similar)}


# ═══════════════════════════════════════════════════════════════
# PHYSICS VALIDATION
# ═══════════════════════════════════════════════════════════════

def _validate_design(stack: Dict, prediction: Dict) -> List[str]:
    """Validates design against physics constraints."""
    warnings = []
    metrics = prediction.get("metrics", {})

    # PCE sanity
    pce = metrics.get("PCE", {})
    if pce.get("median", 0) > 26:
        warnings.append("⚠️ Predicted median PCE > 26% — unusually high for single-junction PSC")

    # Material-specific caps
    htl = (stack.get("htl") or "").upper()
    if "PEDOT" in htl and pce.get("median", 0) > 23:
        warnings.append("⚠️ PEDOT:PSS devices rarely exceed ~23% PCE experimentally")

    # Voc check
    voc = metrics.get("Voc", {})
    if voc.get("median", 0) > 1.25:
        warnings.append("⚠️ Voc > 1.25V — consider whether this is a tandem device")

    # FF check
    ff = metrics.get("FF", {})
    if ff.get("median", 0) < 0.4:
        warnings.append("⚠️ FF < 0.4 — may indicate systematic device failure in similar stacks")

    # Low sample size
    n = prediction.get("n_similar", 0)
    if n < 10:
        warnings.append(f"⚠️ Very few similar devices found (n={n}). Prediction unreliable.")

    return warnings
