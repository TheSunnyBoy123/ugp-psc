"""
data_engine.py — Core Data Loader & Column Ontology Generator

Provides:
  - Load perovskite_db.csv (~43k rows, 410 columns) into Pandas
  - Load matbench JSON datasets
  - Generate column ontology: {column → group, dtype, top_values, non_null_pct}
  - Precompute dataset summaries
  - Semantic column suggestion for queries
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# COLUMN GROUP PREFIXES
# ═══════════════════════════════════════════════════════════════

_PREFIX_GROUPS = {
    "Ref_":          "reference_metadata",
    "Cell_":         "device_architecture",
    "Substrate_":    "substrate",
    "ETL_":          "etl_layer",
    "Perovskite_":   "perovskite_layer",
    "HTL_":          "htl_layer",
    "Backcontact_":  "backcontact_layer",
    "Add_lay_":      "additional_layers",
    "JV_":           "jv_performance",
    "Stability_":    "stability",
    "Module_":       "module",
    "Encapsulation_":"encapsulation",
    "Outdoor_":      "outdoor_testing",
}


class DataEngine:
    """Central data store — loads CSVs, builds ontology, serves queries."""

    def __init__(
        self,
        perovskite_path: str = "data/perovskite_db.csv",
        matbench_dir: str = "data",
        extra_dataset_dirs: Optional[List[str]] = None,
        literature_dir: Optional[str] = None,
    ):
        self.perovskite_path = perovskite_path
        self.matbench_dir = matbench_dir
        self.extra_dataset_dirs = extra_dataset_dirs or []
        self.literature_dir = literature_dir
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_summaries: Dict[str, dict] = {}
        self.dataset_catalog: Dict[str, dict] = {}
        self._ontology_cache: Dict[str, dict] = {}

        self._load_perovskite()
        self._load_matbench()
        self._load_extra_datasets()
        self._load_literature_index()
        self._compute_summaries()

    # ───────────────── LOADERS ─────────────────

    def _load_perovskite(self) -> None:
        path = self._resolve(self.perovskite_path)
        if not os.path.exists(path):
            print(f"⚠️  Perovskite CSV not found: {path}")
            return
        df = pd.read_csv(path, low_memory=False)
        self.datasets["perovskite_db"] = df
        self.dataset_catalog["perovskite_db"] = {
            "category": "psc",
            "source_path": path,
            "description": "Experimental perovskite solar cell device database",
        }
        print(f"✅ Loaded perovskite_db: {len(df):,} rows × {len(df.columns)} cols")

    def _load_matbench(self) -> None:
        data_dir = self._resolve(self.matbench_dir)
        if not os.path.isdir(data_dir):
            return
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith("matbench_") or not fn.endswith(".json"):
                continue
            name = fn.replace(".json", "")
            try:
                with open(os.path.join(data_dir, fn)) as f:
                    raw = json.load(f)
                df = self._normalize_matbench(name, raw)
                self.datasets[name] = df
                self.dataset_catalog[name] = {
                    "category": "matbench",
                    "source_path": os.path.join(data_dir, fn),
                    "description": "Matbench benchmark dataset",
                }
                print(f"  Loaded {name}: {len(df):,} rows")
            except Exception as e:
                print(f"  ⚠️  Failed to load {fn}: {e}")

    def _load_extra_datasets(self) -> None:
        mapping = {
            "materials_names.csv": {
                "name": "feature_selection_2_materials",
                "description": "Unique materials list extracted for feature selection",
            },
            "materials_names_frequency.csv": {
                "name": "feature_selection_2_material_frequencies",
                "description": "Material frequencies and extracted property columns",
            },
            "plain_pdb.csv": {
                "name": "feature_selection_2_plain_pdb",
                "description": "Compact PSC dataset with key stack columns and PCE",
            },
            "feature_pdb.csv": {
                "name": "feature_selection_2_feature_pdb",
                "description": "Feature-tokenized PSC stack dataset",
            },
            "feature_engineered_first_match.csv": {
                "name": "feature_selection_2_feature_engineered_first_match",
                "description": "Feature-engineered dataset using first-match material properties",
            },
            "feature_engineered_full.csv": {
                "name": "feature_selection_2_feature_engineered_full",
                "description": "Feature-engineered dataset with full extracted properties",
            },
        }

        for base_dir in self.extra_dataset_dirs:
            resolved_dir = self._resolve(base_dir)
            if not os.path.isdir(resolved_dir):
                continue

            for filename, meta in mapping.items():
                path = os.path.join(resolved_dir, filename)
                if not os.path.exists(path):
                    continue

                try:
                    df = pd.read_csv(path, low_memory=False)
                    dataset_name = meta["name"]
                    self.datasets[dataset_name] = df
                    self.dataset_catalog[dataset_name] = {
                        "category": "feature_selection_2",
                        "source_path": path,
                        "description": meta["description"],
                    }
                    print(
                        f"  Loaded {dataset_name}: {len(df):,} rows × {len(df.columns)} cols"
                    )
                except Exception as e:
                    print(f"  ⚠️  Failed to load {path}: {e}")

    def _load_literature_index(self) -> None:
        if not self.literature_dir:
            return

        root = Path(self._resolve(self.literature_dir))
        if not root.exists():
            return

        records = []
        for path in sorted(root.rglob("*.pdf")):
            rel_path = path.relative_to(root)
            parts = rel_path.parts
            material = ""
            collection = parts[0] if parts else ""
            if len(parts) >= 2 and parts[0] == "data":
                material = parts[1]
            elif len(parts) >= 2:
                material = parts[-2]

            records.append(
                {
                    "relative_path": str(rel_path),
                    "filename": path.name,
                    "title": path.stem,
                    "collection": collection,
                    "material": material,
                    "parent_dir": path.parent.name,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                }
            )

        if records:
            df = pd.DataFrame(records)
            self.datasets["lit_pdf_index"] = df
            self.dataset_catalog["lit_pdf_index"] = {
                "category": "literature",
                "source_path": str(root),
                "description": "Index of PDFs available under Lit/",
            }
            print(f"  Indexed lit_pdf_index: {len(df):,} PDFs")

    def _normalize_matbench(self, name: str, data: Any) -> pd.DataFrame:
        if isinstance(data, dict):
            if {"index", "columns", "data"}.issubset(data.keys()):
                return pd.DataFrame(data["data"], columns=data["columns"], index=data["index"])
            if "data" in data:
                payload = data["data"]
                idx = data.get("index")
                if isinstance(payload, list):
                    if name == "matbench_perovskites" and payload and isinstance(payload[0], list):
                        cols = ["structure", "target"] + [f"extra_{i}" for i in range(2, len(payload[0]))]
                        df = pd.DataFrame(payload, columns=cols[:len(payload[0])])
                    else:
                        df = pd.DataFrame(payload)
                elif isinstance(payload, dict):
                    df = pd.DataFrame(payload)
                else:
                    df = pd.DataFrame(payload)
                if idx is not None and len(idx) == len(df):
                    df.index = idx
                return df
            return pd.DataFrame(data)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame(data)

    def _resolve(self, path: str) -> str:
        """Resolve path relative to this file's directory."""
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    # ───────────────── SUMMARIES ─────────────────

    def _compute_summaries(self) -> None:
        for name, df in self.datasets.items():
            num_cols = df.select_dtypes(include="number").columns.tolist()
            txt_cols = df.select_dtypes(include="object").columns.tolist()
            summary: Dict[str, Any] = {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": num_cols[:20],
                "text_columns": txt_cols[:20],
            }
            if name == "perovskite_db" and "JV_default_PCE" in df.columns:
                pce = pd.to_numeric(df["JV_default_PCE"], errors="coerce").dropna()
                summary["pce_stats"] = {
                    "mean": round(float(pce.mean()), 2),
                    "median": round(float(pce.median()), 2),
                    "max": round(float(pce.max()), 2),
                    "min": round(float(pce.min()), 2),
                    "count": int(len(pce)),
                }
            meta = self.dataset_catalog.get(name, {})
            if meta:
                summary["category"] = meta.get("category")
                summary["description"] = meta.get("description")
                summary["source_path"] = meta.get("source_path")
            self.dataset_summaries[name] = summary
        print(f"📊 Summaries computed for {len(self.dataset_summaries)} datasets")

    def get_summary(self, table: str) -> dict:
        return self.dataset_summaries.get(table, {})

    # ───────────────── COLUMN ONTOLOGY ─────────────────

    def get_column_ontology(
        self,
        table: str = "perovskite_db",
        top_values: int = 8,
        max_unique: int = 120,
    ) -> Dict[str, Any]:
        """Builds semantic column ontology for LLM grounding."""
        cache_key = f"{table}:{top_values}:{max_unique}"
        if cache_key in self._ontology_cache:
            return self._ontology_cache[cache_key]

        if table not in self.datasets:
            return {"error": f"Table '{table}' not found"}

        df = self.datasets[table]
        columns: Dict[str, dict] = {}
        for col in df.columns:
            s = df[col]
            non_null = int(s.notna().sum())
            nunique = int(s.nunique(dropna=True))
            info: Dict[str, Any] = {
                "group": self._column_group(col),
                "dtype": str(s.dtype),
                "non_null_count": non_null,
                "non_null_ratio": round(non_null / max(len(df), 1), 4),
                "unique_count": nunique,
                "options": [],
            }
            # Enumerate top values for low-cardinality columns
            if 0 < nunique <= max_unique:
                vc = s.dropna().astype(str).value_counts().head(top_values)
                info["options"] = [
                    {"value": idx, "count": int(cnt)} for idx, cnt in vc.items()
                ]
            elif "stack_sequence" in col.lower():
                info["options"] = self._tokenize_top(s, top_values)
            columns[col] = info

        ontology = {
            "table_name": table,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
        }
        self._ontology_cache[cache_key] = ontology
        return ontology

    def _column_group(self, col: str) -> str:
        for prefix, group in _PREFIX_GROUPS.items():
            if col.startswith(prefix):
                return group
        return "other"

    def _tokenize_top(self, series: pd.Series, n: int = 8) -> List[dict]:
        counts: Dict[str, Dict] = {}
        for val in series.dropna().astype(str):
            for part in re.split(r"[|/,;]", val):
                tok = part.strip()
                if not tok:
                    continue
                key = tok.lower()
                counts[key] = counts.get(key, {"value": tok, "count": 0})
                counts[key]["count"] += 1
        ranked = sorted(counts.values(), key=lambda x: x["count"], reverse=True)[:n]
        return [{"value": r["value"], "count": r["count"]} for r in ranked]

    # ───────────────── SCHEMA ACCESS ─────────────────

    def get_columns(self, table: str = "perovskite_db") -> List[str]:
        if table not in self.datasets:
            return []
        return list(self.datasets[table].columns)

    def inspect_schema(self, table: str) -> Dict[str, Any]:
        if table not in self.datasets:
            return {"error": f"Table '{table}' not found"}
        df = self.datasets[table]
        col_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "float" in dtype or "int" in dtype:
                col_types[col] = "numeric"
            elif "datetime" in dtype:
                col_types[col] = "datetime"
            else:
                col_types[col] = "string"
        return {
            "table_name": table,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": col_types,
        }

    # ───────────────── COLUMN SUGGESTION ─────────────────

    def suggest_columns(
        self,
        query: str,
        table: str = "perovskite_db",
        max_cols: int = 15,
    ) -> List[str]:
        """Keyword-based column suggestion for wide tables."""
        from ontology import (
            PERFORMANCE_COLUMNS, PROVENANCE_COLUMNS, DEVICE_CONTEXT_COLUMNS,
        )
        if table not in self.datasets:
            return []
        all_cols = self.datasets[table].columns.tolist()
        if len(all_cols) <= max_cols:
            return all_cols

        q = query.lower()
        selected: List[str] = []

        # Always include core columns
        for c in PERFORMANCE_COLUMNS + PROVENANCE_COLUMNS + DEVICE_CONTEXT_COLUMNS:
            if c in all_cols and c not in selected:
                selected.append(c)

        # Keyword → column mapping
        _kw_map = {
            "pce": ["JV_default_PCE"], "efficiency": ["JV_default_PCE"],
            "voc": ["JV_default_Voc"], "jsc": ["JV_default_Jsc"],
            "ff": ["JV_default_FF"], "fill factor": ["JV_default_FF"],
            "substrate": ["Substrate_stack_sequence"],
            "etl": ["ETL_stack_sequence", "ETL_thickness_list"],
            "htl": ["HTL_stack_sequence", "HTL_thickness_list"],
            "backcontact": ["Backcontact_stack_sequence"],
            "architecture": ["Cell_architecture"],
            "composition": ["Perovskite_composition_short_form"],
            "band gap": ["Perovskite_band_gap"], "bandgap": ["Perovskite_band_gap"],
            "stability": ["Stability_PCE_T80", "Stability_PCE_T95"],
            "doi": ["Ref_DOI_number"], "journal": ["Ref_journal"],
            "date": ["Ref_publication_date"], "year": ["Ref_publication_date"],
            "ito": ["Substrate_stack_sequence"], "fto": ["Substrate_stack_sequence"],
        }
        for kw, cols in _kw_map.items():
            if kw in q:
                for c in cols:
                    if c in all_cols and c not in selected:
                        selected.append(c)

        # Columns explicitly mentioned by name
        for col in all_cols:
            if col.lower() in q and col not in selected:
                selected.append(col)

        return selected[:max_cols]
