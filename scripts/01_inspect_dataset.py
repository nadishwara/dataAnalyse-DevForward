"""Beginner-friendly, read-only profile of Senti4SD's downloaded training CSV.

Run from the project root:
    python scripts/01_inspect_dataset.py

The original CSV is NEVER overwritten.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save chart without opening a graphics window.
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Datasets" / "Original" / "train3098itemPOLARITY.csv"
REPORTS = ROOT / "Reports"


def guess_delimiter(path: Path) -> str:
    """Guess CSV separator; falls back to comma if uncertain."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            snippet = stream.read(16_384)
        return csv.Sniffer().sniff(snippet, delimiters=",;\t|").delimiter
    except (OSError, UnicodeError, csv.Error):
        return ","


def read_csv_safely(path: Path):
    """Try common separators/encodings and prefer recognisable columns."""
    separators = list(dict.fromkeys([guess_delimiter(path), ",", ";", "\t", "|"]))
    candidates = []
    errors = []
    for encoding in ("utf-8-sig", "cp1252"):
        for separator in separators:
            try:
                frame = pd.read_csv(path, sep=separator, encoding=encoding, dtype=str)
                # Favour structured, plausible files with known text/label columns.
                cols = {str(col).strip().lower() for col in frame.columns}
                known = sum(
                    name in cols
                    for name in ("text", "polarity", "sentiment", "label", "final")
                )
                score = (len(frame.columns) > 1, known, -abs(len(frame.columns) - 3))
                candidates.append((score, frame, separator, encoding))
            except (UnicodeError, pd.errors.ParserError, ValueError, OSError) as exc:
                errors.append(f"{encoding}, sep={separator!r}: {exc}")
        if candidates:
            break
    if not candidates:
        raise ValueError(
            "Could not parse the CSV file. Please upload it for inspection.\n"
            + "\n".join(errors[:4])
        )
    _, frame, separator, encoding = max(candidates, key=lambda item: item[0])
    return frame, separator, encoding


def find_column(columns, candidates):
    lookup = {str(column).strip().lower(): column for column in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def main() -> int:
    print("=" * 68)
    print("DEVFORWARD / SENTI4SD - STEP 1: DATASET INSPECTION")
    print("=" * 68)
    if not SOURCE.is_file():
        print(f"ERROR: CSV not found at:\n{SOURCE}")
        print(
            "Copy the downloaded file into Datasets/Original/ and keep its name unchanged."
        )
        return 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        df, separator, encoding = read_csv_safely(SOURCE)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    text_col = find_column(
        df.columns,
        [
            "text",
            "body",
            "content",
            "comment",
            "message",
            "post",
            "sentence",
            "document",
        ],
    )
    label_col = find_column(
        df.columns,
        ["polarity", "sentiment", "final", "label", "class", "sentiment_label"],
    )

    missing = df.isna().sum()
    # Missing or blank strings are recorded separately; neither is silently removed.
    blank = {
        name: int(df[name].dropna().astype(str).str.strip().eq("").sum())
        for name in df.columns
    }
    duplicates = int(df.duplicated().sum())

    lines = [
        "DevForward - Senti4SD dataset inspection (descriptive audit; NOT IP model evaluation)",
        f"Source filename: {SOURCE.name}",
        "Original repository: https://github.com/collab-uniba/Senti4SD",
        "Citation: Calefato et al. (2018), doi:10.1007/s10664-017-9546-9",
        f"Parsed separator: {repr(separator)} | Encoding: {encoding}",
        f"Number of data rows: {len(df):,}",
        f"Number of columns: {len(df.columns)}",
        f"Column names: {list(df.columns)}",
        f"Exact duplicate rows: {duplicates:,}",
        "Missing values per column:",
    ]
    lines.extend(f"  {col}: {int(missing[col]):,}" for col in df.columns)
    lines.append("Blank-string values per column:")
    lines.extend(f"  {col}: {blank[col]:,}" for col in df.columns)

    if text_col:
        words = df[text_col].dropna().astype(str).str.split().str.len()
        lines += [
            f"Detected text column: {text_col}",
            (
                f"Average words per non-missing text: {words.mean():.1f}"
                if len(words)
                else "No non-missing text rows"
            ),
        ]
    else:
        lines.append(
            "Text column NOT automatically recognised; inspect columns manually."
        )

    if label_col:
        labels = (
            df[label_col]
            .fillna("<MISSING>")
            .astype(str)
            .str.strip()
            .replace("", "<EMPTY>")
        )
        counts = labels.value_counts(dropna=False)
        lines.append(f"Detected label column: {label_col}")
        lines.append("Label distribution:")
        lines.extend(f"  {name}: {int(count):,}" for name, count in counts.items())
        table = counts.rename_axis("original_label").reset_index(name="count")
        table.to_csv(
            REPORTS / "label_distribution.csv", index=False, encoding="utf-8-sig"
        )
        plt.figure(figsize=(8, 4.5))
        plt.bar(table["original_label"].astype(str), table["count"])
        plt.title("Senti4SD - Original label distribution")
        plt.xlabel("Original dataset label (NOT an IP-related label)")
        plt.ylabel("Number of records")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(REPORTS / "label_distribution.png", dpi=160)
        plt.close()
    else:
        lines.append(
            "Label column NOT automatically recognised; inspect columns manually."
        )

    lines += [
        "",
        "IMPORTANT RESEARCH WARNING:",
        "Original sentiment labels are not validated impostor-phenomenon labels.",
        "Do NOT train an IP-related detector using those labels as ground truth.",
        "Do NOT change the original file. Inspect text relevance and plan separate human annotation.",
        "Respect the source permissions and privacy when sharing samples.",
    ]
    report_text = "\n".join(lines) + "\n"
    (REPORTS / "dataset_summary.txt").write_text(report_text, encoding="utf-8")
    # First 20 rows for local manual inspection; original file stays untouched.
    df.head(20).to_csv(
        REPORTS / "sample_first_20.csv", index=False, encoding="utf-8-sig"
    )

    print(report_text)
    print("First three records (truncated for console readability):")
    print(df.head(3).to_string(index=False, max_colwidth=65))
    print("\nSaved under Reports/:")
    print("  dataset_summary.txt")
    print("  sample_first_20.csv")
    if label_col:
        print("  label_distribution.csv")
        print("  label_distribution.png")
    print("\nSUCCESS: Original data was read, not changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
