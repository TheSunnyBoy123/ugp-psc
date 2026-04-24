"""
regression_engine.py — On-Demand Regression for Property Prediction

Provides:
  - RegressionEngine: trains XGBoost regressors on-demand from DataFrames
  - Supports mixed numeric + categorical features via K-fold TargetEncoder
  - Returns predictions with prediction intervals and feature importances
  - Auto-excludes leakage columns (Voc/Jsc/FF) when predicting PCE
  - Caches trained models in memory for fast subsequent queries
"""

import hashlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("psc_agent")

# Columns that must NEVER be used as features when predicting PCE,
# because PCE ≈ Voc × Jsc × FF — including them is data leakage.
PCE_LEAKAGE_COLUMNS = {"JV_default_Voc", "JV_default_Jsc", "JV_default_FF"}

class RegressionEngine:
    """On-demand XGBoost regression with TargetEncoder for categoricals."""

    def __init__(self):
        self._model_cache: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────

    def predict_property(
        self,
        target_col: str,
        feature_cols: List[str],
        constraints: Optional[Dict[str, str]] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Train (or retrieve cached) XGBoost model and predict.

        Args:
            target_col:   Column to predict (e.g. 'JV_default_PCE')
            feature_cols: Columns to use as features
            constraints:  Fixed feature values for prediction (e.g. {'HTL_stack_sequence': 'Spiro-MeOTAD'})
            df:           Source DataFrame

        Returns:
            {predicted_value, confidence_interval_95, r2_score,
             n_train, n_features, top_features, model_key}
        """
        if df is None:
            return {"error": "No DataFrame provided"}

        constraints = constraints or {}

        # Validate columns exist
        missing_target = target_col not in df.columns
        if missing_target:
            return {"error": f"Target column '{target_col}' not found in dataset"}

        valid_features = [c for c in feature_cols if c in df.columns]
        if not valid_features:
            return {"error": f"No valid feature columns found. Tried: {feature_cols}"}

        # ── Guard against feature leakage when predicting PCE ──
        if "PCE" in target_col.upper():
            leaked = [c for c in valid_features if c in PCE_LEAKAGE_COLUMNS]
            if leaked:
                logger.warning(
                    "[REGRESSION]  removing leakage features for PCE target: %s",
                    leaked,
                )
                valid_features = [c for c in valid_features if c not in PCE_LEAKAGE_COLUMNS]
                if not valid_features:
                    return {"error": "All requested features are leakage columns for PCE "
                                     "(Voc, Jsc, FF). Provide non-leakage features."}

        # Build cache key
        cache_key = self._cache_key(target_col, valid_features)

        # Get or train model
        model_info = self._model_cache.get(cache_key)
        if model_info is None:
            logger.info("[REGRESSION]  training new model  target=%r  features=%d",
                        target_col, len(valid_features))
            model_info = self._train_model(df, target_col, valid_features)
            if "error" in model_info:
                return model_info
            self._model_cache[cache_key] = model_info
            logger.info(
                "[REGRESSION]  model trained  r2=%.4f  n_train=%d",
                model_info["r2_score"], model_info["n_train"],
            )
        else:
            logger.info("[REGRESSION]  using cached model  key=%s", cache_key[:12])

        # Build prediction input
        pred_row = self._build_prediction_row(
            df, valid_features, constraints, model_info,
        )
        if "error" in pred_row:
            return pred_row

        # Predict
        try:
            model = model_info["model"]
            encoder = model_info["encoder"]
            cat_cols = model_info["cat_cols"]
            num_cols = model_info["num_cols"]

            X_pred = pred_row[valid_features].copy()

            # Encode categoricals
            if cat_cols:
                for col in cat_cols:
                    if col in X_pred.columns:
                        X_pred[col] = X_pred[col].astype(str)
                X_pred[cat_cols] = encoder.transform(X_pred[cat_cols])

            predicted = float(model.predict(X_pred)[0])

            # Prediction interval from training residuals
            # NOTE: these are prediction intervals, not confidence intervals
            # for the mean — they quantify expected spread around new predictions.
            residual_std = model_info.get("residual_std", 0)
            pi_95 = (
                round(predicted - 1.96 * residual_std, 4),
                round(predicted + 1.96 * residual_std, 4),
            )

            # Feature importances
            importances = model.feature_importances_
            feat_names = valid_features
            top_feats = sorted(
                zip(feat_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            return {
                "predicted_value": round(predicted, 4),
                "prediction_interval_95": pi_95,
                "r2_score": round(model_info["r2_score"], 4),
                "n_train": model_info["n_train"],
                "n_features": len(valid_features),
                "target": target_col,
                "constraints": constraints,
                "top_features": [
                    {"feature": f, "importance": round(float(imp), 4)}
                    for f, imp in top_feats
                ],
                "model_key": cache_key[:12],
            }
        except Exception as e:
            logger.error("[REGRESSION]  prediction failed: %s", e)
            return {"error": f"Prediction failed: {e}"}

    def get_available_targets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """List columns suitable as regression targets (numeric with >100 non-null)."""
        targets = []
        for col in df.columns:
            if df[col].dtype in ("float64", "float32", "int64", "int32"):
                non_null = int(df[col].notna().sum())
                if non_null >= 100:
                    targets.append({
                        "column": col,
                        "non_null": non_null,
                        "mean": round(float(df[col].mean()), 4),
                        "std": round(float(df[col].std()), 4),
                    })
        return sorted(targets, key=lambda x: x["non_null"], reverse=True)

    # ──────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────

    def _train_model(
        self, df: pd.DataFrame, target: str, features: List[str],
    ) -> Dict[str, Any]:
        """Train XGBoost with TargetEncoder for categorical features."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return {"error": "XGBoost not installed (pip install xgboost)"}

        try:
            from category_encoders import TargetEncoder
        except ImportError:
            TargetEncoder = None

        from sklearn.model_selection import cross_val_score

        # Prepare data
        subset = df[features + [target]].dropna(subset=[target])
        if len(subset) < 50:
            return {"error": f"Too few rows with non-null '{target}' ({len(subset)} < 50)"}

        X = subset[features].copy()
        y = pd.to_numeric(subset[target], errors="coerce")
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < 50:
            return {"error": f"Too few numeric rows for '{target}' ({len(y)} < 50)"}

        # Identify categorical vs numeric columns
        cat_cols = []
        num_cols = []
        for col in features:
            if col not in X.columns:
                continue
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                cat_cols.append(col)
                X[col] = X[col].astype(str).fillna("unknown")
            else:
                num_cols.append(col)
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)

        # Encode categoricals with K-fold target encoding to avoid data leakage
        encoder = None
        if cat_cols and TargetEncoder is not None:
            from sklearn.model_selection import KFold as _KFold

            n_splits = min(5, len(X) // 10)  # at least 10 samples per fold
            if n_splits < 2:
                n_splits = 2

            kf = _KFold(n_splits=n_splits, shuffle=True, random_state=42)
            X_oof = X.copy()
            X_oof[cat_cols] = np.nan  # placeholder for OOF values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_idx, val_idx in kf.split(X):
                    enc_fold = TargetEncoder(cols=cat_cols, return_df=True)
                    enc_fold.fit(X.iloc[train_idx][cat_cols], y.iloc[train_idx])
                    X_oof.iloc[
                        val_idx,
                        X_oof.columns.get_indexer(cat_cols),
                    ] = enc_fold.transform(X.iloc[val_idx][cat_cols]).values

            # Fit a final encoder on full data for future transform() calls
            encoder = TargetEncoder(cols=cat_cols, return_df=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                encoder.fit(X[cat_cols], y)

            # Use the OOF-encoded values for training
            X[cat_cols] = X_oof[cat_cols]
            logger.info("[REGRESSION]  K-fold target encoding applied  k=%d", n_splits)
        elif cat_cols:
            # Fallback: label encode
            for col in cat_cols:
                X[col] = X[col].astype("category").cat.codes

        # Train XGBoost
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

            # Cross-validation R²
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(y) // 10), scoring="r2")
            r2 = float(np.mean(cv_scores))

        # Residual std for prediction intervals
        y_pred_train = model.predict(X)
        residuals = y.values - y_pred_train
        residual_std = float(np.std(residuals))

        return {
            "model": model,
            "encoder": encoder,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "r2_score": r2,
            "n_train": len(y),
            "residual_std": residual_std,
            "feature_medians": {c: float(X[c].median()) for c in X.columns},
        }

    def _build_prediction_row(
        self,
        df: pd.DataFrame,
        features: List[str],
        constraints: Dict[str, str],
        model_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build a single-row DataFrame for prediction using constraints + medians."""
        row = {}
        medians = model_info.get("feature_medians", {})
        cat_cols = model_info.get("cat_cols", [])

        for col in features:
            if col in constraints:
                row[col] = constraints[col]
            elif col in cat_cols:
                # Use modal value from training data
                if col in df.columns:
                    mode_vals = df[col].dropna().mode()
                    row[col] = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "unknown"
                else:
                    row[col] = "unknown"
            else:
                # Numeric: use median from training
                row[col] = medians.get(col, 0)

        return pd.DataFrame([row])

    def _cache_key(self, target: str, features: List[str]) -> str:
        """Deterministic cache key from target + sorted features."""
        sig = f"{target}:{','.join(sorted(features))}"
        return hashlib.md5(sig.encode()).hexdigest()
