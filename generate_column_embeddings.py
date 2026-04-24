import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sentence-transformer embeddings for perovskite DB columns."
    )
    parser.add_argument(
        "--metadata",
        default=os.path.join(os.path.dirname(__file__), "data", "perovskite_db_column_metadata.csv"),
        help="Path to perovskite_db_column_metadata.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "column_embeddings.npz"),
        help="Output .npz file path",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# TEXT BUILDER
# ─────────────────────────────────────────────────────────────────

def build_column_text(row: pd.Series) -> str:
    """
    Construct a rich natural-language document for a single column.

    Combines the column name (humanised), description, keywords, and group
    so that the embedding captures multiple angles of meaning.
    """
    parts = []

    # 1. Human-readable column name
    col_name: str = str(row.get("column_name", "")).strip()
    if col_name:
        readable = col_name.replace("_", " ").strip()
        parts.append(readable)

    # 2. Description
    description: str = str(row.get("Description", "")).strip()
    if description and description.lower() not in ("nan", "none", ""):
        parts.append(description)

    # 3. Keywords  (stored as comma-separated string, possibly quoted)
    keywords_raw: str = str(row.get("Keywords", "")).strip().strip('"').strip("'")
    if keywords_raw and keywords_raw.lower() not in ("nan", "none", ""):
        # Normalise — replace commas+spaces with spaces so they become
        # natural continuation of the text
        kw_text = keywords_raw.replace(",", " ").replace("  ", " ").strip()
        parts.append(kw_text)

    # 4. Group tag  (e.g. "Perovskite", "JV", "HTL", "Stability" …)
    group: str = str(row.get("group", "")).strip()
    if group and group.lower() not in ("nan", "none", ""):
        parts.append(group)

    return ". ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load metadata ──────────────────────────────────────────
    print(f"[1/4]  Loading metadata from: {args.metadata}")
    if not os.path.isfile(args.metadata):
        print(f"ERROR: metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.metadata)

    # Drop any completely empty trailing rows
    df = df.dropna(subset=["column_name"])
    df["column_name"] = df["column_name"].astype(str).str.strip()
    df = df[df["column_name"] != ""]

    print(f"       Found {len(df)} columns in metadata.")

    # ── 2. Build text documents ───────────────────────────────────
    print("[2/4]  Building text documents for each column …")
    texts = [build_column_text(row) for _, row in df.iterrows()]
    column_names = df["column_name"].tolist()

    # Quick sanity print
    print(f"       Example (column[0]):  {column_names[0]!r}")
    print(f"       → text: {texts[0][:120]!r}")
    print()

    # ── 3. Load model & encode ────────────────────────────────────
    print(f"[3/4]  Loading SentenceTransformer model '{args.model}' …")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers is not installed.\n"
            "       Run:  pip install sentence-transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    model = SentenceTransformer(args.model)
    print(f"       Model loaded.  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    print(f"       Encoding {len(texts)} column documents  (batch_size={args.batch_size}) …")
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm → cosine sim = dot product
    )
    elapsed = time.perf_counter() - t0
    print(f"       Done in {elapsed:.1f}s.  Shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # ── 4. Save .npz ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[4/4]  Saving embeddings to: {args.output}")

    np.savez(
        args.output,
        embeddings=embeddings.astype(np.float32),
        columns=np.array(column_names, dtype="U256"),   # unicode strings
        texts=np.array(texts, dtype="U2048"),
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"       Saved  {size_mb:.2f} MB")
    print()
    print("✅  Done!  You can now load the embeddings with:")
    print(f"       data = np.load('{args.output}', allow_pickle=False)")
    print( "       embeddings = data['embeddings']   # shape (N, dim)")
    print( "       columns    = data['columns']      # column names")


if __name__ == "__main__":
    main()
