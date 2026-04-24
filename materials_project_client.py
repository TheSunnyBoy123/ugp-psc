"""
materials_project_client.py — Materials Project API Client

Provides:
  - get_material_properties(formula) → band gap, formation energy, density, stability
  - search_materials(query, fields) → general-purpose multi-property search
  - get_electronic_structure(formula) → band gap type, is_metal
  - get_thermodynamic_data(formula) → formation energy, e_above_hull, decomposition info
  - get_structure_info(formula) → crystal system, space group, lattice params
  - format_properties_text(props) → rich text summary for LLM augmentation
  - Disk cache via diskcache to avoid repeated API calls
  - Offline mode when MP_API_KEY is not set
  - Structured disambiguation for multiple phases
"""

import logging
import os
from typing import Any, Dict, List, Optional

try:
    import diskcache
except ImportError:
    diskcache = None

logger = logging.getLogger("psc_agent")


# ═══════════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════════

class MaterialsProjectClient:
    """Cached Materials Project API client with offline fallback."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = ".mp_cache"):
        self.api_key = api_key or os.environ.get("MP_API_KEY", "")
        self.online = bool(self.api_key)
        self._mpr = None

        # Setup disk cache
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_dir)
        if diskcache:
            self.cache = diskcache.Cache(cache_path)
        else:
            self.cache = {}

        if self.online:
            try:
                from mp_api.client import MPRester
                self._mpr = MPRester(self.api_key)
                logger.info("✅ Materials Project: online mode")
            except Exception as e:
                logger.warning("⚠️  Materials Project API init failed: %s", e)
                self.online = False
        else:
            logger.info("ℹ️  Materials Project: offline mode (set MP_API_KEY for live API)")

    # ──────────────────────────────────────────────────────────
    # SHARED: polymorph disambiguation helper
    # ──────────────────────────────────────────────────────────

    def _select_or_disambiguate(
        self, docs, formula: str, extra_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Given a list of MP docs for a formula, either:
          - return the single doc (if only one)
          - return the sole stable doc (if exactly one is stable)
          - return a needs_user_choice dict listing all polymorphs

        Returns (doc, None) if a single doc was selected, or
                (None, disambiguation_dict) if user must choose.
        """
        if len(docs) == 1:
            return docs[0], None

        stable = [d for d in docs if getattr(d, "is_stable", False)]
        if len(stable) == 1:
            return stable[0], None

        # Multiple phases → build disambiguation options
        options = []
        for d in docs[:8]:  # cap at 8 options
            sym = getattr(d, "symmetry", {})
            crystal = sym.get("crystal_system", "?") if isinstance(sym, dict) else "?"
            space = sym.get("symbol", "") if isinstance(sym, dict) else ""
            opt = {
                "label": f"{getattr(d, 'formula_pretty', formula)} ({crystal}{', ' + space if space else ''})",
                "mp_id": str(getattr(d, "material_id", "?")),
                "band_gap": getattr(d, "band_gap", None),
                "is_stable": getattr(d, "is_stable", False),
                "crystal_system": crystal,
            }
            # Include any extra fields the caller cares about
            if extra_fields:
                for fld in extra_fields:
                    opt[fld] = getattr(d, fld, None)
            options.append(opt)

        disambiguation = {
            "needs_user_choice": True,
            "formula": formula,
            "n_phases": len(docs),
            "options": options,
        }
        return None, disambiguation

    # ──────────────────────────────────────────────────────────
    # CORE: get_material_properties (original)
    # ──────────────────────────────────────────────────────────

    def get_material_properties(self, formula: str) -> Dict[str, Any]:
        """
        Fetches computed properties for a material formula.

        Returns dict with: band_gap, formation_energy, density, is_stable,
                           energy_above_hull, crystal_system, space_group
        Or disambiguation dict if multiple phases exist.
        """
        cache_key = f"props:{formula}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        try:
            docs = self._mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull",
                    "density", "symmetry", "is_stable",
                ],
            )
        except Exception as e:
            logger.error("[MP]  API error for '%s': %s", formula, e)
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No data found for '{formula}'"}

        # Polymorph disambiguation
        doc, disambig = self._select_or_disambiguate(docs, formula)
        if disambig is not None:
            self._cache_set(cache_key, disambig)
            return disambig

        sym = getattr(doc, "symmetry", {})
        result = {
            "formula": getattr(doc, "formula_pretty", formula),
            "material_id": str(getattr(doc, "material_id", "?")),
            "band_gap_eV": getattr(doc, "band_gap", None),
            "formation_energy_eV_atom": getattr(doc, "formation_energy_per_atom", None),
            "energy_above_hull_eV": getattr(doc, "energy_above_hull", None),
            "density_g_cm3": getattr(doc, "density", None),
            "is_stable": getattr(doc, "is_stable", False),
            "crystal_system": sym.get("crystal_system") if isinstance(sym, dict) else None,
            "space_group": sym.get("symbol") if isinstance(sym, dict) else None,
        }
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # EXTENDED: search_materials
    # ──────────────────────────────────────────────────────────

    def search_materials(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        General-purpose search by formula or elements.

        Returns a dict with 'results' list — each entry containing the
        requested fields plus material_id and formula.
        """
        cache_key = f"search:{query}:{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        default_fields = [
            "material_id", "formula_pretty", "band_gap",
            "formation_energy_per_atom", "energy_above_hull",
            "density", "symmetry", "is_stable",
        ]
        search_fields = fields or default_fields

        try:
            # Try as formula first
            docs = self._mpr.materials.summary.search(
                formula=query,
                fields=search_fields,
                num_chunks=1,
            )
            if not docs:
                # Try as elements (e.g., "Si" or "Ti,O")
                elements = [e.strip() for e in query.split(",")]
                docs = self._mpr.materials.summary.search(
                    elements=elements,
                    fields=search_fields,
                    num_chunks=1,
                )
        except Exception as e:
            logger.error("[MP]  search error for '%s': %s", query, e)
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No materials found for '{query}'"}

        results = []
        for doc in docs[:limit]:
            entry = {}
            for field in search_fields:
                val = getattr(doc, field, None)
                if field == "symmetry" and isinstance(val, dict):
                    entry["crystal_system"] = val.get("crystal_system")
                    entry["space_group"] = val.get("symbol")
                elif field == "material_id":
                    entry[field] = str(val)
                else:
                    entry[field] = val
            results.append(entry)

        result = {"query": query, "n_results": len(results), "results": results}
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # EXTENDED: get_electronic_structure
    # ──────────────────────────────────────────────────────────

    def get_electronic_structure(self, formula: str) -> Dict[str, Any]:
        """Fetch electronic structure: band gap, type, is_gap_direct, is_metal."""
        cache_key = f"electronic:{formula}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        try:
            docs = self._mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id", "formula_pretty", "band_gap",
                    "is_stable", "is_metal", "is_gap_direct",
                ],
            )
        except Exception as e:
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No data found for '{formula}'"}

        # Polymorph disambiguation
        doc, disambig = self._select_or_disambiguate(docs, formula)
        if disambig is not None:
            self._cache_set(cache_key, disambig)
            return disambig

        result = {
            "formula": getattr(doc, "formula_pretty", formula),
            "material_id": str(getattr(doc, "material_id", "?")),
            "band_gap_eV": getattr(doc, "band_gap", None),
            "is_metal": getattr(doc, "is_metal", None),
            "is_gap_direct": getattr(doc, "is_gap_direct", None),
            "is_stable": getattr(doc, "is_stable", False),
        }
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # EXTENDED: get_thermodynamic_data
    # ──────────────────────────────────────────────────────────

    def get_thermodynamic_data(self, formula: str) -> Dict[str, Any]:
        """Fetch thermodynamic data: formation energy, e_above_hull, stability."""
        cache_key = f"thermo:{formula}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        try:
            docs = self._mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id", "formula_pretty",
                    "formation_energy_per_atom", "energy_above_hull",
                    "is_stable", "decomposition_products",
                ],
            )
        except Exception as e:
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No data found for '{formula}'"}

        # Polymorph disambiguation
        doc, disambig = self._select_or_disambiguate(docs, formula)
        if disambig is not None:
            self._cache_set(cache_key, disambig)
            return disambig

        decomp = getattr(doc, "decomposition_products", None)
        decomp_str = None
        if decomp:
            try:
                if isinstance(decomp, list):
                    decomp_str = ", ".join(str(d) for d in decomp[:5])
                else:
                    decomp_str = str(decomp)
            except Exception:
                decomp_str = None

        result = {
            "formula": getattr(doc, "formula_pretty", formula),
            "material_id": str(getattr(doc, "material_id", "?")),
            "formation_energy_eV_atom": getattr(doc, "formation_energy_per_atom", None),
            "energy_above_hull_eV": getattr(doc, "energy_above_hull", None),
            "is_stable": getattr(doc, "is_stable", False),
            "decomposition_products": decomp_str,
        }
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # EXTENDED: get_structure_info
    # ──────────────────────────────────────────────────────────

    def get_structure_info(self, formula: str) -> Dict[str, Any]:
        """Fetch structural info: crystal system, space group, lattice params, density."""
        cache_key = f"structure:{formula}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        try:
            docs = self._mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id", "formula_pretty", "symmetry",
                    "density", "volume", "nsites",
                ],
            )
        except Exception as e:
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No data found for '{formula}'"}

        # Polymorph disambiguation
        doc, disambig = self._select_or_disambiguate(docs, formula)
        if disambig is not None:
            self._cache_set(cache_key, disambig)
            return disambig

        sym = getattr(doc, "symmetry", {})
        result = {
            "formula": getattr(doc, "formula_pretty", formula),
            "material_id": str(getattr(doc, "material_id", "?")),
            "crystal_system": sym.get("crystal_system") if isinstance(sym, dict) else None,
            "space_group": sym.get("symbol") if isinstance(sym, dict) else None,
            "point_group": sym.get("point_group") if isinstance(sym, dict) else None,
            "density_g_cm3": getattr(doc, "density", None),
            "volume_A3": getattr(doc, "volume", None),
            "n_sites": getattr(doc, "nsites", None),
        }
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # FORMAT: get_bandgap_context (original)
    # ──────────────────────────────────────────────────────────

    def get_bandgap_context(self, formula: str) -> str:
        """Returns a text summary of band gap data for LLM augmentation."""
        props = self.get_material_properties(formula)
        if "error" in props:
            return f"No Materials Project data for {formula}."
        if props.get("needs_user_choice"):
            n = props.get("n_phases", "?")
            return f"Multiple phases found for {formula} ({n} entries). Specify the phase."

        bg = props.get("band_gap_eV")
        fe = props.get("formation_energy_eV_atom")
        parts = [f"Materials Project data for {props.get('formula', formula)}:"]
        if bg is not None:
            parts.append(f"  Band gap: {bg:.3f} eV")
        if fe is not None:
            parts.append(f"  Formation energy: {fe:.4f} eV/atom")
        if props.get("is_stable"):
            parts.append("  Thermodynamically stable")
        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # FORMAT: format_properties_text (new)
    # ──────────────────────────────────────────────────────────

    def format_properties_text(self, props: Dict[str, Any]) -> str:
        """
        Format a material properties dict into rich text for LLM context.
        Handles both single-material and search results.
        """
        if "error" in props:
            return f"⚠️ {props['error']}"

        if props.get("needs_user_choice"):
            lines = [f"Multiple phases found for **{props.get('formula', '?')}** "
                     f"({props.get('n_phases', '?')} entries):"]
            for opt in props.get("options", []):
                stable_tag = " ✅ stable" if opt.get("is_stable") else ""
                bg = opt.get("band_gap")
                bg_str = f", Eg={bg:.3f} eV" if bg is not None else ""
                lines.append(f"  - {opt['label']} ({opt['mp_id']}{bg_str}{stable_tag})")
            return "\n".join(lines)

        if "results" in props:
            # Multi-result from search_materials
            lines = [f"**Materials Project results for '{props.get('query', '?')}'** "
                     f"({props.get('n_results', 0)} results):"]
            for r in props.get("results", []):
                parts = [f"**{r.get('formula_pretty', '?')}** ({r.get('material_id', '?')})"]
                if r.get("band_gap") is not None:
                    parts.append(f"Eg={r['band_gap']:.3f} eV")
                if r.get("formation_energy_per_atom") is not None:
                    parts.append(f"Ef={r['formation_energy_per_atom']:.4f} eV/atom")
                if r.get("density") is not None:
                    parts.append(f"ρ={r['density']:.2f} g/cm³")
                if r.get("is_stable"):
                    parts.append("✅ stable")
                lines.append("  - " + " | ".join(parts))
            return "\n".join(lines)

        # Single-material result
        f = props.get("formula", "?")
        lines = [f"**{f}** (Materials Project {props.get('material_id', '?')}):"]

        _field_map = [
            ("band_gap_eV", "Band gap", "eV", ".3f"),
            ("formation_energy_eV_atom", "Formation energy", "eV/atom", ".4f"),
            ("energy_above_hull_eV", "Energy above hull", "eV/atom", ".4f"),
            ("density_g_cm3", "Density", "g/cm³", ".2f"),
            ("volume_A3", "Volume", "ų", ".2f"),
            ("n_sites", "Sites in unit cell", "", "d"),
        ]
        for key, label, unit, fmt in _field_map:
            val = props.get(key)
            if val is not None:
                lines.append(f"  - **{label}**: {val:{fmt}} {unit}".rstrip())

        # Boolean / string fields
        if props.get("is_stable") is not None:
            lines.append(f"  - **Stable**: {'Yes ✅' if props['is_stable'] else 'No ❌'}")
        if props.get("is_metal") is not None:
            lines.append(f"  - **Metallic**: {'Yes' if props['is_metal'] else 'No'}")
        if props.get("is_gap_direct") is not None:
            lines.append(f"  - **Direct gap**: {'Yes' if props['is_gap_direct'] else 'No (indirect)'}")
        for key in ("crystal_system", "space_group", "point_group"):
            val = props.get(key)
            if val:
                lines.append(f"  - **{key.replace('_', ' ').title()}**: {val}")
        if props.get("decomposition_products"):
            lines.append(f"  - **Decomposition products**: {props['decomposition_products']}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # LOOKUP BY MATERIAL_ID  (after disambiguation)
    # ──────────────────────────────────────────────────────────

    def get_properties_by_id(self, material_id: str) -> Dict[str, Any]:
        """Fetch properties for a specific material_id (e.g. 'mp-2657')."""
        cache_key = f"by_id:{material_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.online or not self._mpr:
            return {"error": "Materials Project not available (offline mode)"}

        try:
            docs = self._mpr.materials.summary.search(
                material_ids=[material_id],
                fields=[
                    "material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "energy_above_hull",
                    "density", "symmetry", "is_stable",
                    "is_metal", "is_gap_direct", "volume", "nsites",
                ],
            )
        except Exception as e:
            return {"error": f"API error: {e}"}

        if not docs:
            return {"error": f"No data for material_id '{material_id}'"}

        doc = docs[0]
        sym = getattr(doc, "symmetry", {})
        result = {
            "formula": getattr(doc, "formula_pretty", "?"),
            "material_id": str(getattr(doc, "material_id", material_id)),
            "band_gap_eV": getattr(doc, "band_gap", None),
            "formation_energy_eV_atom": getattr(doc, "formation_energy_per_atom", None),
            "energy_above_hull_eV": getattr(doc, "energy_above_hull", None),
            "density_g_cm3": getattr(doc, "density", None),
            "is_stable": getattr(doc, "is_stable", False),
            "is_metal": getattr(doc, "is_metal", None),
            "is_gap_direct": getattr(doc, "is_gap_direct", None),
            "crystal_system": sym.get("crystal_system") if isinstance(sym, dict) else None,
            "space_group": sym.get("symbol") if isinstance(sym, dict) else None,
            "volume_A3": getattr(doc, "volume", None),
            "n_sites": getattr(doc, "nsites", None),
        }
        self._cache_set(cache_key, result)
        return result

    # ──────────────────────────────────────────────────────────
    # CACHE HELPERS
    # ──────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[Any]:
        if isinstance(self.cache, dict):
            return self.cache.get(key)
        try:
            return self.cache.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        if isinstance(self.cache, dict):
            self.cache[key] = value
        else:
            try:
                self.cache.set(key, value, expire=86400 * 7)  # 7 day TTL
            except Exception:
                pass
