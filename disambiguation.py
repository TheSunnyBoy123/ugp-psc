"""
disambiguation.py — Multi-Interpretation Detection & Structured Choices

Returns structured options when a query is ambiguous:
  - Dataset-level: query could target multiple loaded datasets
  - Query-level: vague comparisons or missing context
"""

from typing import Dict, List, Optional

from ontology import extract_materials_from_query


# ═══════════════════════════════════════════════════════════════
# DATASET-LEVEL DISAMBIGUATION
# ═══════════════════════════════════════════════════════════════

def check_disambiguation(
    query: str,
    dataset_names: List[str],
) -> Optional[Dict]:
    """
    Checks if query needs disambiguation.
    Returns None if unambiguous, or a dict with structured choices.
    """
    # 1. Dataset-level: does the query match multiple datasets?
    q = query.lower()
    matches = [name for name in dataset_names if name.lower() in q]
    if len(matches) > 1:
        return {
            "needs_user_choice": True,
            "reason": "Your query matches multiple datasets.",
            "options": [
                {
                    "label": f"Query {name}",
                    "description": f"Search the {name} dataset",
                    "action": {"table": name},
                }
                for name in matches
            ],
        }

    # 2. Query-level: vague comparisons
    vague = _check_vague_comparison(query)
    if vague:
        return vague

    return None


# ═══════════════════════════════════════════════════════════════
# QUERY-LEVEL DISAMBIGUATION
# ═══════════════════════════════════════════════════════════════

def _check_vague_comparison(query: str) -> Optional[Dict]:
    """Detects ambiguous comparison queries."""
    q = query.lower().replace("–", "-").replace("—", "-")

    materials = extract_materials_from_query(query)
    if not materials:
        return None

    has_compare = any(kw in q for kw in ["compare", "versus", "vs"])
    if not has_compare:
        return None

    # If comparison is already specific, no disambiguation needed
    specific = any(kw in q for kw in [
        "efficiency", "pce", "voc", "jsc", "ff",
        "architecture", "n-i-p", "p-i-n",
        "substrate", "ito", "fto",
        "stability", "best", "top", "worst",
    ])
    if specific:
        return None

    canonical = materials[0][1]
    return {
        "needs_user_choice": True,
        "reason": f"What aspect of {canonical} devices would you like to compare?",
        "options": [
            {
                "label": f"Compare {canonical} vs other materials (efficiency)",
                "description": f"Find efficiency differences between {canonical} and alternatives",
                "action": {"rewrite": f"Compare devices using {canonical} vs alternatives by PCE"},
            },
            {
                "label": f"Compare {canonical} across architectures (n-i-p vs p-i-n)",
                "description": f"How does {canonical} perform in inverted vs regular?",
                "action": {"rewrite": f"Compare n-i-p vs p-i-n for {canonical} devices"},
            },
            {
                "label": f"Find best {canonical} papers",
                "description": f"Which publications report the highest PCE with {canonical}?",
                "action": {"rewrite": f"Top 10 highest PCE papers using {canonical}"},
            },
        ],
    }
