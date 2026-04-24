"""
test_improvements.py — Verify four chatbot robustness improvements:

1. Materials Project polymorph disambiguation helper
2. PCE feature leakage exclusion
3. K-fold target encoding (structural check)
4. Prediction interval key naming
"""

import sys
import os
import types
import warnings

# Ensure chatbot dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅  {name}")
    else:
        FAILED += 1
        msg = f"  ❌  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ═══════════════════════════════════════════════════════════════
# TEST 1: _select_or_disambiguate helper
# ═══════════════════════════════════════════════════════════════
print("\n═══ TEST 1: Polymorph disambiguation helper ═══")

from materials_project_client import MaterialsProjectClient

client = MaterialsProjectClient(api_key=None)  # offline mode

# Mock docs as simple namespace objects
def _make_doc(mp_id, formula, crystal, bg, stable):
    d = types.SimpleNamespace()
    d.material_id = mp_id
    d.formula_pretty = formula
    d.band_gap = bg
    d.is_stable = stable
    d.symmetry = {"crystal_system": crystal, "symbol": "P1"}
    return d

# Single doc → should return (doc, None)
docs_single = [_make_doc("mp-1", "TiO2", "Tetragonal", 3.0, True)]
doc, disambig = client._select_or_disambiguate(docs_single, "TiO2")
check("Single doc returns doc directly", doc is not None and disambig is None)

# Two docs, one stable → should return (stable_doc, None)
docs_one_stable = [
    _make_doc("mp-1", "TiO2", "Tetragonal", 3.0, True),
    _make_doc("mp-2", "TiO2", "Monoclinic", 3.5, False),
]
doc, disambig = client._select_or_disambiguate(docs_one_stable, "TiO2")
check("One stable doc returns it", doc is not None and disambig is None)
check("Correct stable doc selected", doc.material_id == "mp-1")

# Three docs, none or multiple stable → should return (None, disambiguation)
docs_multi = [
    _make_doc("mp-1", "TiO2", "Tetragonal", 3.0, False),
    _make_doc("mp-2", "TiO2", "Orthorhombic", 3.3, False),
    _make_doc("mp-3", "TiO2", "Monoclinic", 3.5, False),
]
doc, disambig = client._select_or_disambiguate(docs_multi, "TiO2")
check("Multiple non-stable docs returns disambiguation", doc is None and disambig is not None)
check("Disambiguation has needs_user_choice", disambig.get("needs_user_choice") is True)
check("Disambiguation has all 3 options", len(disambig.get("options", [])) == 3)
check("Each option has mp_id", all("mp_id" in o for o in disambig["options"]))
check("Each option has crystal_system", all("crystal_system" in o for o in disambig["options"]))


# ═══════════════════════════════════════════════════════════════
# TEST 2: PCE feature leakage exclusion
# ═══════════════════════════════════════════════════════════════
print("\n═══ TEST 2: PCE feature leakage exclusion ═══")

import numpy as np
import pandas as pd
from regression_engine import RegressionEngine, PCE_LEAKAGE_COLUMNS

re_engine = RegressionEngine()

# Create a small synthetic dataset
np.random.seed(42)
n = 200
df_test = pd.DataFrame({
    "JV_default_PCE": np.random.uniform(5, 25, n),
    "JV_default_Voc": np.random.uniform(0.8, 1.2, n),
    "JV_default_Jsc": np.random.uniform(15, 25, n),
    "JV_default_FF": np.random.uniform(0.5, 0.85, n),
    "HTL_stack_sequence": np.random.choice(["Spiro", "PTAA", "NiO"], n),
    "ETL_stack_sequence": np.random.choice(["TiO2", "SnO2", "PCBM"], n),
    "Perovskite_band_gap": np.random.uniform(1.4, 1.7, n),
})

check("PCE_LEAKAGE_COLUMNS defined",
      PCE_LEAKAGE_COLUMNS == {"JV_default_Voc", "JV_default_Jsc", "JV_default_FF"})

# Predict PCE including leakage features — should be auto-removed
result = re_engine.predict_property(
    target_col="JV_default_PCE",
    feature_cols=["JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
                  "HTL_stack_sequence", "ETL_stack_sequence", "Perovskite_band_gap"],
    constraints={},
    df=df_test,
)

check("PCE prediction succeeds even with leakage cols provided",
      "error" not in result, result.get("error", ""))

if "error" not in result:
    # The leakage features should NOT appear in top_features
    top_feat_names = {f["feature"] for f in result.get("top_features", [])}
    leaked_in_result = top_feat_names & PCE_LEAKAGE_COLUMNS
    check("Leakage columns not in top features",
          len(leaked_in_result) == 0,
          f"found: {leaked_in_result}")

# All leakage only → should error
result_all_leak = re_engine.predict_property(
    target_col="JV_default_PCE",
    feature_cols=["JV_default_Voc", "JV_default_Jsc", "JV_default_FF"],
    constraints={},
    df=df_test,
)
check("All-leakage features returns error",
      "error" in result_all_leak,
      result_all_leak.get("error", ""))


# ═══════════════════════════════════════════════════════════════
# TEST 3: Prediction interval key naming
# ═══════════════════════════════════════════════════════════════
print("\n═══ TEST 3: Prediction interval naming ═══")

result_pi = re_engine.predict_property(
    target_col="JV_default_PCE",
    feature_cols=["HTL_stack_sequence", "ETL_stack_sequence", "Perovskite_band_gap"],
    constraints={},
    df=df_test,
)

check("Result has prediction_interval_95 key",
      "prediction_interval_95" in result_pi,
      f"keys: {list(result_pi.keys())}")

check("Result does NOT have confidence_interval_95 key",
      "confidence_interval_95" not in result_pi)

if "prediction_interval_95" in result_pi:
    pi = result_pi["prediction_interval_95"]
    check("Prediction interval is a tuple/list of 2",
          len(pi) == 2 and pi[0] <= pi[1],
          f"pi={pi}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: K-fold target encoding structure
# ═══════════════════════════════════════════════════════════════
print("\n═══ TEST 4: K-fold target encoding ═══")

# Verify that the _train_model includes K-fold encoding path
import inspect
source = inspect.getsource(re_engine._train_model)
check("_train_model uses KFold", "KFold" in source)
check("_train_model uses OOF encoding", "X_oof" in source)
check("_train_model fits per-fold encoder", "enc_fold" in source)
check("_train_model fits final encoder on full data", "encoder.fit(X[cat_cols], y)" in source)


# ═══════════════════════════════════════════════════════════════
# TEST 5: get_properties_by_id exists
# ═══════════════════════════════════════════════════════════════
print("\n═══ TEST 5: get_properties_by_id method ═══")

check("get_properties_by_id method exists",
      hasattr(client, "get_properties_by_id") and callable(client.get_properties_by_id))

# Offline mode should return error
result_by_id = client.get_properties_by_id("mp-2657")
check("Offline mode returns error dict",
      isinstance(result_by_id, dict) and "error" in result_by_id)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═' * 50}")
print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'═' * 50}\n")

sys.exit(0 if FAILED == 0 else 1)
