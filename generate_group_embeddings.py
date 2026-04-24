"""
generate_group_embeddings.py
────────────────────────────
Pre-computes dense sentence-transformer embeddings for each column GROUP
in the perovskite database and writes them to data/group_embeddings.npz.

Each group is represented by a rich text document that combines:
  • The group name (humanised)
  • All column names in the group (humanised)
  • All unique keywords from columns in the group
  • All unique descriptions from columns in the group

This enables a two-stage retrieval:
  1. Find top relevant GROUPS via cosine similarity (16 groups, no noise)
  2. Return all columns from matched groups → LLM picks which to use

Usage:
    python generate_group_embeddings.py
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sentence-transformer embeddings for column groups."
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join(os.path.dirname(__file__), "data", "perovskite_db_column_metadata.csv"),
        help="Path to perovskite_db_column_metadata.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "group_embeddings.npz"),
        help="Output .npz file path",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    return parser.parse_args()


def build_group_text(group_name: str, group_df: pd.DataFrame) -> str:
    """
    Build a rich text document for a column group by combining:
    - Human-readable group name
    - All column names (humanised)
    - All descriptions
    - All unique keywords
    """
    parts = []

    # 1. Group name
    readable_group = group_name.replace("_", " ").strip()
    parts.append(readable_group)

    # 2. All column names (humanised, deduplicated)
    col_names = group_df["column_name"].dropna().unique()
    readable_cols = [c.replace("_", " ").strip() for c in col_names]
    parts.append(". ".join(readable_cols[:15]))  # limit to avoid too-long texts

    # 3. All unique descriptions
    descs = group_df["Description"].dropna().unique()
    # Take a representative sample of descriptions
    parts.append(". ".join(str(d) for d in descs[:10]))

    # 4. All unique keywords (merged from all columns in group)
    all_keywords = set()
    for kw_raw in group_df["Keywords"].dropna():
        kw_str = str(kw_raw).strip().strip('"').strip("'")
        if kw_str.lower() not in ("nan", "none", ""):
            for kw in kw_str.split(","):
                kw = kw.strip().lower()
                if kw:
                    all_keywords.add(kw)

    if all_keywords:
        parts.append(" ".join(sorted(all_keywords)))

    return ". ".join(filter(None, parts))


def main():
    args = parse_args()

    # 1. Load metadata
    print(f"[1/4]  Loading metadata from: {args.metadata}")
    if not os.path.isfile(args.metadata):
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.metadata)
    df = df.dropna(subset=["column_name"])
    df["column_name"] = df["column_name"].astype(str).str.strip()
    df = df[df["column_name"] != ""]

    print(f"       Found {len(df)} columns in {df['group'].nunique()} groups.")

    # 2. Build group texts
    print("[2/4]  Building text documents for each group …")
    groups = sorted(df["group"].dropna().unique())
    group_texts = []
    group_names = []
    group_columns = {}  # group_name -> list of column names

    for g in groups:
        g_df = df[df["group"] == g]
        text = build_group_text(g, g_df)
        group_texts.append(text)
        group_names.append(g)
        group_columns[g] = g_df["column_name"].tolist()
        print(f"       {g:25s}  ({len(g_df):3d} cols)  text_len={len(text)}")

    # 3. Encode
    print(f"\n[3/4]  Loading SentenceTransformer model '{args.model}' …")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(args.model)
    print(f"       Encoding {len(group_texts)} group documents …")
    t0 = time.perf_counter()
    embeddings = model.encode(
        group_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"       Done in {elapsed:.1f}s.  Shape: {embeddings.shape}")

    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[4/4]  Saving to: {args.output}")

    # Save group_columns as JSON string since npz doesn't support dicts
    import json
    group_cols_json = json.dumps(group_columns)

    np.savez(
        args.output,
        embeddings=embeddings.astype(np.float32),
        groups=np.array(group_names, dtype="U256"),
        texts=np.array(group_texts, dtype="U4096"),
        group_columns_json=np.array([group_cols_json], dtype="U65536"),
    )

    size_kb = os.path.getsize(args.output) / 1024
    print(f"       Saved  {size_kb:.1f} KB")
    print("\n✅  Done!")


if __name__ == "__main__":
    main()
