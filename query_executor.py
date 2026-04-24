"""
query_executor.py — Pandas-Based Query Plan Executor

Translates JSON query plans into Pandas operations.

Provides:
  - validate_plan()  — pre-execution schema validation
  - execute()        — run plan against DataEngine
  - Physics sanity checks on results
  - Provenance column enforcement
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ontology import PHYSICS_BOUNDS, PROVENANCE_COLUMNS, normalize_material_pattern

logger = logging.getLogger("psc_agent")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALLOWED_OPERATIONS = {
    "filter", "count", "top_k", "groupby_aggregate",
    "unique", "distribution", "correlation", "sample",
    "list_columns", "describe_columns", "global_aggregate",
}

ALLOWED_FILTER_OPS = {"equals", "contains", "regex", "gt", "gte", "lt", "lte", "in", "not_null", "is_null"}

ALLOWED_AGGS = {"mean", "median", "min", "max", "count", "std", "sum"}

MAX_RESULT_ROWS = 200


# ═══════════════════════════════════════════════════════════════
# FILTER NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def _normalize_filters(plan: Dict[str, Any]) -> None:
    """
    Normalizes the 'filters' field in-place.
    LLMs sometimes produce:
      - filters as a dict {"any_of": [...]} instead of [{"any_of": [...]}]
      - filters as a single dict {"column":..., "op":..., "value":...}
    This normalizes to always be a list of dicts.
    """
    filters = plan.get("filters")
    if filters is None:
        plan["filters"] = []
        return
    if isinstance(filters, dict):
        # Single filter or bare any_of — wrap in list
        plan["filters"] = [filters]
        return
    if isinstance(filters, list):
        # Remove any non-dict entries
        plan["filters"] = [f for f in filters if isinstance(f, dict)]
        return
    plan["filters"] = []


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_plan(plan: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    """
    Validates a query plan before execution.

    Returns:
        {"valid": True} or {"valid": False, "errors": [...], "suggestions": [...]}
    """
    errors = []
    suggestions = []
    col_set = set(columns)

    # Check operation
    op = plan.get("operation", "")
    if op not in ALLOWED_OPERATIONS:
        errors.append(f"Unknown operation '{op}'. Allowed: {sorted(ALLOWED_OPERATIONS)}")

    # Normalize filters first
    _normalize_filters(plan)

    # Check filters
    for i, f in enumerate(plan.get("filters", [])):
        if not isinstance(f, dict):
            continue
        if "any_of" in f:
            any_of = f["any_of"]
            if not isinstance(any_of, list):
                continue
            for j, cond in enumerate(any_of):
                if isinstance(cond, dict):
                    _validate_condition(cond, col_set, f"filters[{i}].any_of[{j}]", errors, suggestions)
        else:
            _validate_condition(f, col_set, f"filters[{i}]", errors, suggestions)

    # Check select_columns
    for c in plan.get("select_columns", []) or []:
        if c not in col_set:
            errors.append(f"select_columns: unknown column '{c}'")
            _suggest_column(c, columns, suggestions)

    # Check group_by
    gb = plan.get("group_by")
    if gb and gb not in col_set:
        errors.append(f"group_by: unknown column '{gb}'")
        _suggest_column(gb, columns, suggestions)

    # Check sort_by
    sb = plan.get("sort_by")
    if sb and sb not in col_set:
        errors.append(f"sort_by: unknown column '{sb}'")
        _suggest_column(sb, columns, suggestions)

    # Check column (for unique/distribution)
    col = plan.get("column")
    if col and col not in col_set:
        errors.append(f"column: unknown column '{col}'")
        _suggest_column(col, columns, suggestions)

    # Check aggregation columns
    agg = plan.get("aggregation", {})
    if isinstance(agg, dict):
        for c in agg:
            if c not in col_set:
                errors.append(f"aggregation: unknown column '{c}'")
                _suggest_column(c, columns, suggestions)

    if errors:
        return {"valid": False, "errors": errors, "suggestions": suggestions}
    return {"valid": True}


def _validate_condition(
    cond: Dict, col_set: set, label: str,
    errors: List[str], suggestions: List[str],
) -> None:
    col = cond.get("column")
    op = cond.get("op", "")

    if isinstance(col, list):
        for c in col:
            if c not in col_set:
                errors.append(f"{label}: unknown column '{c}'")
    elif col and col not in col_set:
        errors.append(f"{label}: unknown column '{col}'")

    if op not in ALLOWED_FILTER_OPS:
        errors.append(f"{label}: unknown operator '{op}'")


def _suggest_column(target: str, columns: List[str], suggestions: List[str]) -> None:
    """Fuzzy-match column suggestion."""
    t = target.lower()
    matches = [c for c in columns if t in c.lower() or c.lower() in t]
    if matches:
        suggestions.append(f"Did you mean: {', '.join(matches[:3])}?")


# ═══════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════

def execute(
    plan: Dict[str, Any],
    engine,
    table_name: str = "perovskite_db",
    enforce_provenance: bool = True,
) -> Dict[str, Any]:
    """
    Executes a validated query plan against the DataEngine.

    Returns:
        {status, result, result_type, rows_before, rows_after_filters,
         result_rows, execution_time_ms, warnings, query_plan}
    """
    t0 = time.perf_counter()

    op = plan.get("operation", "filter")
    logger.info("[EXECUTOR]  operation=%r  table=%r", op, table_name)

    # ── Schema operations (don't need table data or filters) ──
    if op == "list_columns":
        tbl = plan.get("table", table_name)
        if tbl not in engine.datasets:
            logger.error("[EXECUTOR]  table %r not found", tbl)
            return {"status": "error", "error": f"Table '{tbl}' not found"}
        all_cols = list(engine.datasets[tbl].columns)
        # Optional keyword filter
        pattern = plan.get("pattern", "")
        if pattern:
            all_cols = [c for c in all_cols if pattern.lower() in c.lower()]
        logger.info("[EXECUTOR]  listing columns  total=%d  pattern=%r  matched=%d",
                    len(engine.datasets[tbl].columns), pattern, len(all_cols))
        result = [{"column_name": c, "index": i} for i, c in enumerate(all_cols)]
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "ok", "result": result, "result_type": "table",
            "operation": op, "rows_before": len(engine.datasets[tbl]),
            "rows_after_filters": len(engine.datasets[tbl]),
            "result_rows": len(result), "execution_time_ms": dt_ms,
            "warnings": [], "query_plan": plan,
        }

    if op == "describe_columns":
        tbl = plan.get("table", table_name)
        if tbl not in engine.datasets:
            logger.error("[EXECUTOR]  table %r not found", tbl)
            return {"status": "error", "error": f"Table '{tbl}' not found"}
        df = engine.datasets[tbl]
        sel = plan.get("select_columns", [])
        target_cols = [c for c in sel if c in df.columns] if sel else list(df.columns)[:30]
        logger.info("[EXECUTOR]  describing %d columns", len(target_cols))
        result = []
        for col in target_cols:
            s = df[col]
            non_null = int(s.notna().sum())
            info = {
                "column_name": col,
                "dtype": str(s.dtype),
                "non_null_count": non_null,
                "non_null_pct": round(non_null / max(len(df), 1) * 100, 1),
                "unique_count": int(s.nunique(dropna=True)),
            }
            # Add top values for low-cardinality columns
            if info["unique_count"] <= 120:
                vc = s.dropna().astype(str).value_counts().head(5)
                info["top_values"] = ", ".join(f"{idx} ({cnt})" for idx, cnt in vc.items())
            # Add stats for numeric columns
            if "float" in str(s.dtype) or "int" in str(s.dtype):
                vals = pd.to_numeric(s, errors="coerce").dropna()
                if not vals.empty:
                    info["mean"] = round(float(vals.mean()), 4)
                    info["min"] = round(float(vals.min()), 4)
                    info["max"] = round(float(vals.max()), 4)
            result.append(info)
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "ok", "result": result, "result_type": "table",
            "operation": op, "rows_before": len(df),
            "rows_after_filters": len(df),
            "result_rows": len(result), "execution_time_ms": dt_ms,
            "warnings": [], "query_plan": plan,
        }

    # ── Data operations (need table + filters) ──
    if table_name not in engine.datasets:
        logger.error("[EXECUTOR]  table %r not found", table_name)
        return {"status": "error", "error": f"Table '{table_name}' not found"}

    df = engine.datasets[table_name]
    columns = list(df.columns)
    rows_before = len(df)

    logger.info("[EXECUTOR]  validating plan against schema …")
    # Validate
    validation = validate_plan(plan, columns)
    if not validation.get("valid"):
        logger.warning("[EXECUTOR]  plan validation failed: %s", validation.get("errors", []))
        return {
            "status": "validation_error",
            "errors": validation.get("errors", []),
            "suggestions": validation.get("suggestions", []),
        }
    logger.debug("[EXECUTOR]  plan validation passed")

    # Apply filters
    filters = plan.get("filters", []) or []
    logger.info("[EXECUTOR]  applying %d filter(s) to %d rows …", len(filters), rows_before)
    for i, f in enumerate(filters):
        if isinstance(f, dict) and "any_of" in f:
            logger.debug("[EXECUTOR]  filter[%d]: any_of (%d conditions)", i, len(f["any_of"]))
        elif isinstance(f, dict):
            logger.debug("[EXECUTOR]  filter[%d]: col=%r  op=%r  val=%r",
                         i, f.get("column"), f.get("op"), str(f.get("value", ""))[:60])
    filtered = _apply_filters(df, filters)
    rows_after = len(filtered)
    logger.info("[EXECUTOR]  after filters: %d → %d rows  (dropped %d)",
                rows_before, rows_after, rows_before - rows_after)

    # Enforce provenance on device-level ops
    if enforce_provenance and op in ("filter", "top_k", "sample"):
        _inject_provenance(plan, columns)

    logger.info("[EXECUTOR]  running operation=%r …", op)
    try:
        result_data, result_type, result_rows = _execute_operation(
            filtered, plan, op, columns
        )
        logger.info("[EXECUTOR]  operation done  result_type=%r  result_rows=%d",
                    result_type, result_rows)
    except Exception as e:
        logger.error("[EXECUTOR]  operation raised exception: %s", str(e))
        return {
            "status": "execution_error",
            "error": str(e),
            "rows_before": rows_before,
            "rows_after_filters": rows_after,
        }

    # Physics sanity check on result
    warnings = _physics_check(result_data, result_type)
    if warnings:
        logger.warning("[EXECUTOR]  physics check warnings: %s", warnings)
    else:
        logger.debug("[EXECUTOR]  physics check: ok")

    dt_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("[EXECUTOR]  ✅  execution complete in %.1f ms", dt_ms)

    return {
        "status": "ok",
        "result": result_data,
        "result_type": result_type,
        "operation": op,
        "rows_before": rows_before,
        "rows_after_filters": rows_after,
        "result_rows": result_rows,
        "execution_time_ms": dt_ms,
        "warnings": warnings,
        "query_plan": plan,
    }


# ═══════════════════════════════════════════════════════════════
# FILTER APPLICATION
# ═══════════════════════════════════════════════════════════════

def _apply_filters(df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    out = df
    for f in filters:
        if not isinstance(f, dict):
            continue
        if "any_of" in f:
            any_of = f["any_of"]
            if not isinstance(any_of, list):
                continue
            combined = pd.Series(False, index=out.index)
            for cond in any_of:
                if isinstance(cond, dict):
                    combined |= _mask(out, cond)
            out = out[combined.fillna(False)]
        else:
            out = out[_mask(out, f).fillna(False)]
    return out


def _mask(df: pd.DataFrame, cond: Dict) -> pd.Series:
    col = cond.get("column")
    op = cond.get("op", "contains")
    val = cond.get("value")

    # Handle not_null / is_null (no value needed)
    if op == "not_null":
        cols = col if isinstance(col, list) else [col]
        combined = pd.Series(True, index=df.index)
        for c in cols:
            if c not in df.columns:
                continue
            combined &= df[c].notna() & (df[c].astype(str).str.strip() != "")
        return combined
    if op == "is_null":
        cols = col if isinstance(col, list) else [col]
        combined = pd.Series(False, index=df.index)
        for c in cols:
            if c not in df.columns:
                continue
            combined |= df[c].isna() | (df[c].astype(str).str.strip() == "")
        return combined

    # Auto-upgrade 'contains' → 'regex' when a material regex pattern exists
    if op == "contains" and val is not None:
        pattern = normalize_material_pattern(str(val))
        if pattern is not None:
            op = "regex"
            val = pattern

    # Multi-column OR
    cols = col if isinstance(col, list) else [col]

    if len(cols) > 1 and op in {"equals", "contains", "in", "regex"}:
        combined = pd.Series(False, index=df.index)
        for c in cols:
            if c not in df.columns:
                continue
            combined |= _single_mask(df[c], op, val)
        return combined

    c = cols[0]
    if c not in df.columns:
        return pd.Series(False, index=df.index)
    return _single_mask(df[c], op, val)


def _single_mask(series: pd.Series, op: str, val: Any) -> pd.Series:
    if op == "equals":
        return series.astype(str).str.lower() == str(val).lower()
    if op == "contains":
        return series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
    if op == "regex":
        try:
            return series.astype(str).str.contains(str(val), case=False, na=False, regex=True)
        except re.error:
            # Fall back to literal if regex is invalid
            return series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
    if op in ("gt", "gte", "lt", "lte"):
        num = pd.to_numeric(series, errors="coerce")
        try:
            v = float(val)
        except (TypeError, ValueError):
            return pd.Series(False, index=series.index)
        if op == "gt":  return num > v
        if op == "gte": return num >= v
        if op == "lt":  return num < v
        if op == "lte": return num <= v
    if op == "in":
        values = [str(x).lower() for x in (val if isinstance(val, list) else [val])]
        return series.astype(str).str.lower().isin(values)
    return pd.Series(False, index=series.index)


# ═══════════════════════════════════════════════════════════════
# OPERATION DISPATCH
# ═══════════════════════════════════════════════════════════════

def _execute_operation(
    df: pd.DataFrame, plan: Dict, op: str, columns: List[str],
) -> tuple:
    """Returns (result_data, result_type, result_rows)."""

    if op == "count":
        return {"count": len(df)}, "scalar", 1

    if op == "filter":
        limit = min(int(plan.get("limit", 20)), MAX_RESULT_ROWS)
        sel = plan.get("select_columns")
        out = df
        if isinstance(sel, list) and sel:
            valid = [c for c in sel if c in df.columns]
            if valid:
                out = df[valid]
        return out.head(limit).to_dict(orient="records"), "table", min(limit, len(out))

    if op == "top_k":
        k = min(int(plan.get("k", 10)), MAX_RESULT_ROWS)
        sort_col = plan.get("sort_by") or plan.get("column")
        asc = bool(plan.get("ascending", False))
        if sort_col and sort_col in df.columns:
            df = df.copy()
            df["_sort"] = pd.to_numeric(df[sort_col], errors="coerce")
            df = df.dropna(subset=["_sort"]).sort_values("_sort", ascending=asc).drop(columns=["_sort"])
        sel = plan.get("select_columns")
        if isinstance(sel, list) and sel:
            valid = [c for c in sel if c in df.columns]
            if valid:
                df = df[valid]
        out = df.head(k)
        return out.to_dict(orient="records"), "table", len(out)

    if op == "unique":
        col = plan.get("column")
        if not col or col not in df.columns:
            return {"error": f"Unknown column '{col}'"}, "error", 0
        tokenize = bool(plan.get("tokenize", False))
        top_k = min(int(plan.get("top_k", 50)), 500)
        if tokenize:
            counts: Dict[str, Dict] = {}
            for val in df[col].dropna().astype(str):
                for part in re.split(r"[|/,;]", val):
                    tok = part.strip()
                    if not tok:
                        continue
                    key = tok.lower()
                    counts[key] = counts.get(key, {"value": tok, "count": 0})
                    counts[key]["count"] += 1
            rows = sorted(counts.values(), key=lambda x: x["count"], reverse=True)[:top_k]
            return rows, "table", len(rows)
        vc = df[col].astype(str).value_counts(dropna=True).head(top_k)
        rows = [{"value": idx, "count": int(cnt)} for idx, cnt in vc.items()]
        return rows, "table", len(rows)

    if op == "groupby_aggregate":
        gb_col = plan.get("group_by")
        agg_map = plan.get("aggregation", {})
        if not gb_col or gb_col not in df.columns:
            return {"error": f"Invalid group_by column '{gb_col}'"}, "error", 0

        result_rows = []
        for key, gdf in df.groupby(gb_col):
            row = {gb_col: key}
            for metric, func in agg_map.items():
                if metric not in gdf.columns:
                    continue
                vals = pd.to_numeric(gdf[metric], errors="coerce").dropna()
                if vals.empty:
                    continue
                fn = func if isinstance(func, str) else str(func)
                if fn in ("mean", "median", "min", "max", "std", "sum", "count"):
                    row[f"{metric}_{fn}"] = round(float(getattr(vals, fn)()), 4)
            result_rows.append(row)

        # Sort by first aggregation column if sort_by specified
        sort_by = plan.get("sort_by")
        asc = bool(plan.get("ascending", False))
        if sort_by:
            matching_keys = [k for k in result_rows[0] if sort_by in k] if result_rows else []
            if matching_keys:
                result_rows.sort(key=lambda r: r.get(matching_keys[0], 0), reverse=not asc)

        k = min(int(plan.get("k", 50)), MAX_RESULT_ROWS)
        result_rows = result_rows[:k]
        return result_rows, "table", len(result_rows)

    if op == "distribution":
        col = plan.get("column")
        if not col or col not in df.columns:
            return {"error": f"Unknown column '{col}'"}, "error", 0
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return {"error": f"No numeric data in '{col}'"}, "error", 0
        stats = {
            "column": col,
            "count": int(len(vals)),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "q25": round(float(vals.quantile(0.25)), 4),
            "q75": round(float(vals.quantile(0.75)), 4),
        }
        return stats, "stats", 1

    if op == "correlation":
        col_x = plan.get("column_x") or plan.get("column")
        col_y = plan.get("column_y")
        if not col_x or not col_y:
            return {"error": "correlation requires column_x and column_y"}, "error", 0
        if col_x not in df.columns or col_y not in df.columns:
            return {"error": f"Unknown columns: {col_x}, {col_y}"}, "error", 0
        x = pd.to_numeric(df[col_x], errors="coerce")
        y = pd.to_numeric(df[col_y], errors="coerce")
        valid = df[[col_x, col_y]].assign(x=x, y=y).dropna(subset=["x", "y"])
        if len(valid) < 3:
            return {"error": "Too few data points for correlation"}, "error", 0
        corr = float(valid["x"].corr(valid["y"]))
        return {
            "column_x": col_x, "column_y": col_y,
            "correlation": round(corr, 4),
            "n": len(valid),
        }, "stats", 1

    if op == "sample":
        n = min(int(plan.get("limit", 20)), MAX_RESULT_ROWS)
        out = df.sample(min(n, len(df))) if len(df) > 0 else df.head(0)
        return out.to_dict(orient="records"), "table", len(out)

    if op == "global_aggregate":
        agg_map = plan.get("aggregation", {})
        if not agg_map:
            return {"error": "global_aggregate requires 'aggregation'"}, "error", 0
        result = {}
        for metric, func in agg_map.items():
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            fn = func if isinstance(func, str) else str(func)
            if fn in ("mean", "median", "min", "max", "std", "sum", "count"):
                result[f"{metric}_{fn}"] = round(float(getattr(vals, fn)()), 4)
        if not result:
            return {"error": "No valid numeric columns for aggregation"}, "error", 0
        return result, "stats", 1

    return {"error": f"Unsupported operation '{op}'"}, "error", 0


# ═══════════════════════════════════════════════════════════════
# PROVENANCE & PHYSICS
# ═══════════════════════════════════════════════════════════════

def _inject_provenance(plan: Dict, columns: List[str]) -> None:
    """Ensures provenance columns in select_columns."""
    sel = plan.get("select_columns")
    if not isinstance(sel, list):
        return
    for prov in PROVENANCE_COLUMNS:
        if prov in columns and prov not in sel:
            sel.insert(0, prov)


def _physics_check(result: Any, result_type: str) -> List[str]:
    """Flags unrealistic values in results."""
    warnings = []
    if result_type != "table" or not isinstance(result, list):
        return warnings

    for row in result[:50]:  # check first 50 rows
        if not isinstance(row, dict):
            continue
        for col, bounds in PHYSICS_BOUNDS.items():
            val = row.get(col)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v > bounds["max"] or v < bounds["min"]:
                warnings.append(f"⚠️ {col}={v} {bounds['unit']}: {bounds['warn']}")
                break  # one warning per row max

    return list(set(warnings))[:5]  # deduplicate, cap at 5


# ═══════════════════════════════════════════════════════════════
# RESULT FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_result_markdown(result: Dict[str, Any], max_rows: int = 10) -> str:
    """Formats a query result as a markdown table (for fallback display)."""
    data = result.get("result", [])
    rtype = result.get("result_type", "")

    if rtype == "scalar":
        if isinstance(data, dict):
            return "\n".join(f"**{k}**: {v}" for k, v in data.items())
        return str(data)

    if rtype == "stats":
        if isinstance(data, dict):
            return "\n".join(f"- **{k}**: {v}" for k, v in data.items())
        return str(data)

    if rtype == "table" and isinstance(data, list) and data:
        try:
            df = pd.DataFrame(data[:max_rows])
            return df.to_markdown(index=False)
        except Exception:
            return str(data[:max_rows])

    return str(data)
