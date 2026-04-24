"""
ontology.py — Material Synonym Resolution, Column Mapping & Domain Ontology

Provides:
  - SYNONYMS: user-typed terms → canonical database values
  - MATERIAL_COLUMN_MAP: canonical material → columns to search
  - MATERIAL_REGEX_PATTERNS: canonical material → regex for variant matching
  - DOMAIN_ONTOLOGY: comprehensive domain-specific term catalog
  - resolve_synonym(), extract_materials_from_query(), get_filter_columns()
  - normalize_material_pattern(): get regex for a material's DB variants
  - Constants: PROVENANCE_COLUMNS, PERFORMANCE_COLUMNS, DEVICE_CONTEXT_COLUMNS
"""

import re
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# MATERIAL REGEX PATTERNS: canonical name → regex for DB variants
# Handles spelling variants like PEDOT:PSS / PEDOT-PSS / PEDOT PSS
# ═══════════════════════════════════════════════════════════════

MATERIAL_REGEX_PATTERNS: Dict[str, str] = {
    # ── HTL Materials ──
    "PEDOT:PSS":      r"PEDOT[:\-\s]?PSS",
    "Spiro-MeOTAD":   r"[Ss]piro[\-\s]?[Mm]e[Oo][Tt][Aa][Dd]",
    "PTAA":           r"PTAA",
    "NiO":            r"NiO[x]?",
    "NiOx":           r"NiO[x]",
    "P3HT":           r"P3HT",
    "Cu2O":           r"Cu2O",
    "CuSCN":          r"CuSCN",
    "poly-TPD":       r"[Pp]oly[\-\s]?TPD",
    "CZTS":           r"CZTS",
    "CuI":            r"CuI",
    "V2O5":           r"V2O5",
    "MoO3":           r"MoO[x3]",
    "WO3":            r"WO[x3]",
    "TAPC":           r"TAPC",
    "NPB":            r"NPB",
    "CuPc":           r"CuPc",
    "SAM":            r"\bSAM\b",
    "Me-4PACz":       r"Me[\-\s]?4PACz",
    "2PACz":          r"2PACz",

    # ── ETL Materials ──
    "TiO2":           r"TiO2?",
    "SnO2":           r"Sn[\-\s]?O2",
    "PCBM":           r"PC[6B]?[\d]*BM|PCBM",
    "C60":            r"C60|C₆₀",
    "ZnO":            r"ZnO",
    "BCP":            r"BCP",
    "ICBA":           r"ICBA",
    "AZO":            r"AZO",
    "Nb2O5":          r"Nb2O5",
    "In2O3":          r"In2O3",
    "CdS":            r"CdS",
    "LiF":            r"LiF",
    "PFN":            r"PFN[\-\s]?Br|PFN",
    "PEIE":           r"PEIE",

    # ── Substrates / Electrodes ──
    "ITO":            r"ITO",
    "FTO":            r"FTO",
    "AZO":            r"AZO",
    "Glass":          r"[Gg]lass",
    "PET":            r"PET",
    "PEN":            r"PEN",
    "Ti":             r"\bTi\b",
    "Steel":          r"[Ss]teel",

    # ── Back Contacts ──
    "Au":             r"\bAu\b",
    "Ag":             r"\bAg\b",
    "Carbon":         r"[Cc]arbon|\bC\b",
    "Al":             r"\bAl\b",
    "Cu":             r"\bCu\b",
    "MoOx/Ag":        r"MoO[x3][/|\s]?Ag",
    "IZO":            r"IZO",

    # ── Perovskite Compositions ──
    "MAPbI3":         r"MA[Pp]b[Ii]3|CH3NH3PbI3",
    "FAPbI3":         r"FA[Pp]b[Ii]3|HC\(NH2\)2PbI3",
    "CsPbI3":         r"Cs[Pp]b[Ii]3",
    "MAPbBr3":        r"MA[Pp]b[Bb]r3|CH3NH3PbBr3",
    "CsPbBr3":        r"Cs[Pp]b[Bb]r3",
    "FACsPb":         r"FA.*Cs.*Pb|Cs.*FA.*Pb",
    "Sn-Pb":          r"Sn[\-\s]?Pb|Pb[\-\s]?Sn",
    "MASnI3":         r"MA[Ss]n[Ii]3",
}


# ═══════════════════════════════════════════════════════════════
# DOMAIN ONTOLOGY: Comprehensive catalog of PSC domain terms
# ═══════════════════════════════════════════════════════════════

DOMAIN_ONTOLOGY: Dict[str, Dict[str, List[str]]] = {
    "htl_materials": {
        "description": "Hole Transport Layer materials",
        "terms": [
            "PEDOT:PSS", "Spiro-MeOTAD", "PTAA", "NiO", "NiOx", "P3HT",
            "Cu2O", "CuSCN", "CuI", "poly-TPD", "CZTS", "V2O5", "MoO3",
            "WO3", "TAPC", "NPB", "CuPc", "SAM", "Me-4PACz", "2PACz",
            "PTPD", "EH44", "Rubrene", "TIPS-Pentacene",
        ],
    },
    "etl_materials": {
        "description": "Electron Transport Layer materials",
        "terms": [
            "TiO2", "SnO2", "PCBM", "C60", "ZnO", "BCP", "ICBA", "AZO",
            "Nb2O5", "In2O3", "CdS", "LiF", "PFN", "PFN-Br", "PEIE",
            "BaSnO3", "SrTiO3", "WO3", "CeO2", "MgO",
        ],
    },
    "substrates_electrodes": {
        "description": "Substrate and transparent electrode materials",
        "terms": [
            "ITO", "FTO", "AZO", "Glass", "PET", "PEN", "Ti", "Steel",
            "Willow Glass", "Sapphire", "Quartz", "IZO",
        ],
    },
    "back_contacts": {
        "description": "Back contact / counter electrode materials",
        "terms": [
            "Au", "Ag", "Carbon", "Al", "Cu", "MoOx/Ag", "IZO",
            "Cr/Au", "Ti/Au", "Graphene", "CNT",
        ],
    },
    "architectures": {
        "description": "Device architecture types",
        "terms": [
            "n-i-p", "p-i-n", "nip", "pin", "regular", "inverted",
            "mesoscopic", "planar", "tandem", "flexible",
        ],
    },
    "perovskite_compositions": {
        "description": "Perovskite ABX3 compositions and cations/anions",
        "terms": [
            "MAPbI3", "FAPbI3", "CsPbI3", "MAPbBr3", "CsPbBr3",
            "FA0.85MA0.15", "Cs0.05FA0.85MA0.10", "triple cation",
            "double cation", "mixed halide", "Sn-Pb", "MASnI3",
            "2D perovskite", "Ruddlesden-Popper", "Dion-Jacobson",
        ],
    },
    "deposition_methods": {
        "description": "Fabrication and deposition techniques",
        "terms": [
            "spin coating", "slot-die", "blade coating", "spray coating",
            "inkjet printing", "screen printing", "evaporation",
            "co-evaporation", "vapor deposition", "CVD", "ALD",
            "sputtering", "electrodeposition", "dip coating",
            "one-step", "two-step", "sequential deposition",
            "anti-solvent", "vacuum flash", "hot casting",
        ],
    },
    "solvents_additives": {
        "description": "Common solvents and additives in PSC fabrication",
        "terms": [
            "DMF", "DMSO", "GBL", "NMP", "chlorobenzene", "toluene",
            "diethyl ether", "isopropanol", "ethanol", "acetonitrile",
            "MACl", "PbCl2", "NH4Cl", "KI", "RbI", "TBABF4",
            "Li-TFSI", "tBP", "FK209", "DOPA",
        ],
    },
    "dopants_passivators": {
        "description": "Dopants, passivation agents, and interface modifiers",
        "terms": [
            "Li-TFSI", "tBP", "FK209", "DOPA", "PEAI", "BAI",
            "EDAI2", "GABr", "OAI", "PMMA", "PEG",
            "KI", "RbCl", "CsI", "GuaSCN", "thiourea",
        ],
    },
    "performance_metrics": {
        "description": "JV and stability performance parameters",
        "terms": [
            "PCE", "Voc", "Jsc", "FF", "fill factor",
            "EQE", "IPCE", "band gap", "Eg",
            "T80", "T95", "stability", "lifetime",
            "hysteresis index", "HI", "MPP tracking",
            "stabilized PCE", "champion device",
        ],
    },
    "characterization_methods": {
        "description": "Measurement and characterization techniques",
        "terms": [
            "XRD", "SEM", "TEM", "AFM", "PL", "TRPL",
            "UPS", "XPS", "EIS", "KPFM", "GIWAXS", "GISAXS",
            "UV-Vis", "FTIR", "Raman", "TGA", "DSC",
            "dark JV", "illuminated JV", "C-V",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# SYNONYM TABLE: user-typed term → canonical DB value
# ═══════════════════════════════════════════════════════════════


SYNONYMS: Dict[str, str] = {
    # HTL materials
    "spiro":            "Spiro-MeOTAD",
    "spiro-meotad":     "Spiro-MeOTAD",
    "spiro-ometad":     "Spiro-MeOTAD",
    "ptaa":             "PTAA",
    "pedot":            "PEDOT:PSS",
    "pedot:pss":        "PEDOT:PSS",
    "pedot pss":        "PEDOT:PSS",
    "pedot-pss":        "PEDOT:PSS",
    "pedot/pss":        "PEDOT:PSS",
    "nio":              "NiO",
    "niox":             "NiOx",
    "nickel oxide":     "NiOx",
    "p3ht":             "P3HT",
    "cuprous":          "Cu2O",
    "cuprous oxide":    "Cu2O",
    "cu2o":             "Cu2O",
    "cusc":             "CuSCN",
    "cuscn":            "CuSCN",
    "copper thiocyanate":"CuSCN",
    "cui":              "CuI",
    "copper iodide":    "CuI",
    "poly-tpd":         "poly-TPD",
    "polytpd":          "poly-TPD",
    "me-4pacz":         "Me-4PACz",
    "4pacz":            "Me-4PACz",
    "2pacz":            "2PACz",
    "sam":              "SAM",
    "moo3":             "MoO3",
    "moox":             "MoO3",
    "v2o5":             "V2O5",
    "wo3":              "WO3",

    # ETL materials
    "tio2":             "TiO2",
    "titania":          "TiO2",
    "sno2":             "SnO2",
    "tin oxide":        "SnO2",
    "pcbm":             "PCBM",
    "pc61bm":           "PCBM",
    "pc71bm":           "PCBM",
    "c60":              "C60",
    "fullerene":        "C60",
    "buckminsterfullerene": "C60",
    "zno":              "ZnO",
    "zinc oxide":       "ZnO",
    "bcp":              "BCP",
    "bathocuproine":    "BCP",
    "icba":             "ICBA",
    "lif":              "LiF",
    "lithium fluoride": "LiF",
    "pfn":              "PFN",
    "pfn-br":           "PFN-Br",
    "peie":             "PEIE",
    "nb2o5":            "Nb2O5",

    # Substrate / electrode
    "ito":              "ITO",
    "indium tin oxide": "ITO",
    "fto":              "FTO",
    "fluorine tin oxide": "FTO",
    "pet":              "PET",
    "pen":              "PEN",
    "glass":            "Glass",

    # Back contact
    "gold":             "Au",
    "au":               "Au",
    "silver":           "Ag",
    "ag":               "Ag",
    "carbon":           "Carbon",
    "aluminum":         "Al",
    "al":               "Al",
    "copper":           "Cu",
    "cu":               "Cu",

    # Architecture
    "n-i-p":            "nip",
    "nip":              "nip",
    "regular":          "nip",
    "p-i-n":            "pin",
    "pin":              "pin",
    "inverted":         "pin",
    "mesoscopic":       "Mesoscopic",
    "planar":           "Planar",

    # Perovskite compositions
    "mapi":             "MAPbI3",
    "mapbi3":           "MAPbI3",
    "ch3nh3pbi3":       "MAPbI3",
    "methylammonium lead iodide": "MAPbI3",
    "fapi":             "FAPbI3",
    "fapbi3":           "FAPbI3",
    "formamidinium lead iodide": "FAPbI3",
    "cspbi3":           "CsPbI3",
    "cesium lead iodide": "CsPbI3",
    "mapbbr3":          "MAPbBr3",
    "cspbbr3":          "CsPbBr3",
    "triple cation":    "Cs0.05FA0.85MA0.10",
    "double cation":    "FAMA",

    # Deposition methods
    "spin coat":        "Spin-coating",
    "spin coating":     "Spin-coating",
    "blade coat":       "Blade-coating",
    "blade coating":    "Blade-coating",
    "slot die":         "Slot-die",
    "slot-die":         "Slot-die",
    "evaporation":      "Evaporation",
    "co-evaporation":   "Co-evaporation",
    "sputtering":       "Sputtering",
    "ald":              "ALD",
    "cvd":              "CVD",
}

# ═══════════════════════════════════════════════════════════════
# MATERIAL → COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════

MATERIAL_COLUMN_MAP: Dict[str, List[str]] = {
    # HTL
    "Spiro-MeOTAD": ["HTL_stack_sequence", "Cell_stack_sequence"],
    "PTAA":         ["HTL_stack_sequence", "Cell_stack_sequence"],
    "PEDOT:PSS":    ["HTL_stack_sequence", "Cell_stack_sequence"],
    "NiO":          ["HTL_stack_sequence", "Cell_stack_sequence"],
    "NiOx":         ["HTL_stack_sequence", "Cell_stack_sequence"],
    "P3HT":         ["HTL_stack_sequence", "Cell_stack_sequence"],
    "Cu2O":         ["HTL_stack_sequence", "Cell_stack_sequence"],
    "CuSCN":        ["HTL_stack_sequence", "Cell_stack_sequence"],

    # ETL
    "TiO2":         ["ETL_stack_sequence", "Cell_stack_sequence"],
    "SnO2":         ["ETL_stack_sequence", "Cell_stack_sequence"],
    "PCBM":         ["ETL_stack_sequence", "Cell_stack_sequence"],
    "C60":          ["ETL_stack_sequence", "Cell_stack_sequence"],
    "ZnO":          ["ETL_stack_sequence", "Cell_stack_sequence"],
    "BCP":          ["ETL_stack_sequence", "Cell_stack_sequence"],

    # Substrate / electrode
    "ITO":          ["Substrate_stack_sequence", "Cell_stack_sequence"],
    "FTO":          ["Substrate_stack_sequence", "Cell_stack_sequence"],
    "PET":          ["Substrate_stack_sequence"],
    "PEN":          ["Substrate_stack_sequence"],
    "Glass":        ["Substrate_stack_sequence"],

    # Back contact
    "Au":           ["Backcontact_stack_sequence", "Cell_stack_sequence"],
    "Ag":           ["Backcontact_stack_sequence", "Cell_stack_sequence"],
    "Carbon":       ["Backcontact_stack_sequence", "Cell_stack_sequence"],
    "Al":           ["Backcontact_stack_sequence", "Cell_stack_sequence"],
    "Cu":           ["Backcontact_stack_sequence", "Cell_stack_sequence"],

    # Architecture
    "nip":          ["Cell_architecture"],
    "pin":          ["Cell_architecture"],
    "Mesoscopic":   ["Cell_architecture"],
    "Planar":       ["Cell_architecture"],

    # Perovskite compositions
    "MAPbI3":       ["Perovskite_composition_short_form", "Perovskite_composition_long_form"],
    "FAPbI3":       ["Perovskite_composition_short_form", "Perovskite_composition_long_form"],
    "CsPbI3":       ["Perovskite_composition_short_form", "Perovskite_composition_long_form"],
    "MAPbBr3":      ["Perovskite_composition_short_form", "Perovskite_composition_long_form"],
}

# ═══════════════════════════════════════════════════════════════
# COLUMN CONSTANTS
# ═══════════════════════════════════════════════════════════════

PROVENANCE_COLUMNS = ["Ref_ID", "Ref_DOI_number"]

PERFORMANCE_COLUMNS = [
    "JV_default_PCE", "JV_default_Voc",
    "JV_default_Jsc", "JV_default_FF",
]

DEVICE_CONTEXT_COLUMNS = [
    "Cell_stack_sequence", "Cell_architecture",
    "Substrate_stack_sequence", "ETL_stack_sequence",
    "Perovskite_composition_short_form",
    "HTL_stack_sequence", "Backcontact_stack_sequence",
]

# Physics sanity bounds
PHYSICS_BOUNDS = {
    "JV_default_PCE":  {"min": 0, "max": 30,  "unit": "%",     "warn": "PCE > 30% is likely an error"},
    "JV_default_Voc":  {"min": 0, "max": 1.3, "unit": "V",     "warn": "Voc > 1.3V may indicate tandem or anomaly"},
    "JV_default_Jsc":  {"min": 0, "max": 30,  "unit": "mA/cm²","warn": "Jsc > 30 mA/cm² is unusually high"},
    "JV_default_FF":   {"min": 0.2, "max": 0.9, "unit": "",    "warn": "FF outside 0.2–0.9 likely indicates failure"},
}


# ═══════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def resolve_synonym(term: str) -> str:
    """Resolves a user-typed term to its canonical database form."""
    key = term.strip().lower().replace("–", "-").replace("—", "-")
    return SYNONYMS.get(key, term)


def extract_materials_from_query(query: str) -> List[Tuple[str, str]]:
    """
    Extracts (user_term, canonical_form) pairs from a query string.
    Returns list of recognized materials/terms sorted longest-match-first.
    """
    q = query.lower().replace("–", "-").replace("—", "-")
    found = []
    seen = set()

    # Sort by length descending so "pedot:pss" matches before "pedot"
    for key in sorted(SYNONYMS.keys(), key=len, reverse=True):
        if key in q:
            canonical = SYNONYMS[key]
            if canonical not in seen:
                found.append((key, canonical))
                seen.add(canonical)
    return found


def get_filter_columns(material_term: str) -> List[str]:
    """Returns the database columns to search for a given material term."""
    canonical = resolve_synonym(material_term)
    return MATERIAL_COLUMN_MAP.get(canonical, [])


def get_architecture_filter(value: str) -> str:
    """Normalizes architecture value for filtering."""
    v = value.strip().lower().replace("–", "-").replace("—", "-").replace(" ", "")
    if v in {"p-i-n", "pin", "inverted"}:
        return "pin"
    if v in {"n-i-p", "nip", "regular"}:
        return "nip"
    return value


def normalize_material_pattern(material: str) -> Optional[str]:
    """
    Returns a regex pattern for matching a material in the database,
    accounting for common spelling variants (e.g., PEDOT:PSS vs PEDOT-PSS).

    Returns None if no pattern is defined (falls back to exact match).
    """
    # Check direct canonical lookup
    if material in MATERIAL_REGEX_PATTERNS:
        return MATERIAL_REGEX_PATTERNS[material]

    # Check if it's a synonym first, then look up canonical
    canonical = resolve_synonym(material)
    if canonical in MATERIAL_REGEX_PATTERNS:
        return MATERIAL_REGEX_PATTERNS[canonical]

    return None


def get_domain_ontology_summary() -> str:
    """
    Returns a human-readable summary of the domain ontology
    for LLM context grounding.
    """
    lines = ["Domain Ontology for Perovskite Solar Cells:"]
    for category, data in DOMAIN_ONTOLOGY.items():
        desc = data["description"]
        terms = ", ".join(data["terms"][:10])
        extra = len(data["terms"]) - 10
        suffix = f" (+{extra} more)" if extra > 0 else ""
        lines.append(f"  {desc}: {terms}{suffix}")
    return "\n".join(lines)
