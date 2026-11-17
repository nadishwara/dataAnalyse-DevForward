"""Merge Label Studio human annotations with Ollama pilot suggestions
to produce the Gold-Standard Pilot Dataset and compute inter-annotator agreement.

Run from project root:
    python scripts/03_merge_pilot_annotations.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUGGESTIONS_CSV = (
    BASE_DIR / "Datasets" / "Annotations" / "senti4sd_pilot_150_llm_v2_suggestions.csv"
)
LS_EXPORT_CSV = (
    BASE_DIR / "Datasets" / "Annotations" / "senti4sd_pilot_150_ls_export.csv"
)
GOLD_OUTPUT_CSV = (
    BASE_DIR / "Datasets" / "Annotations" / "senti4sd_pilot_150_gold_standard.csv"
)


def get_field(row, *candidates):
    """Retrieve the first non-empty matching column value (handles 'col' and 'data.col')."""
    for name in candidates:
        if name in row and row[name] is not None:
            val = str(row[name]).strip()
            if val:
                return val
    return ""


def parse_ls_evidence_span(raw_span):
    """Extract clean text substring from Label Studio's JSON span export."""
    if not raw_span:
        return ""
    raw_span = raw_span.strip()
    if raw_span.startswith("[") or raw_span.startswith("{"):
        try:
            parsed = json.loads(raw_span)
            if isinstance(parsed, list):
                texts = [
                    item.get("text", "").strip()
                    for item in parsed
                    if isinstance(item, dict) and item.get("text")
                ]
                return " | ".join(texts)
            if isinstance(parsed, dict):
                return parsed.get("text", "").strip()
        except json.JSONDecodeError:
            pass
    return raw_span


def compute_cohens_kappa(y_true, y_pred):
    """Compute Cohen's Kappa (k) between human ground truth and LLM predictions."""
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    p_o = agree / n
    p_e = sum((y_true.count(lbl) / n) * (y_pred.count(lbl) / n) for lbl in labels)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main():
    if not SUGGESTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Original suggestions file not found: {SUGGESTIONS_CSV}"
        )
    if not LS_EXPORT_CSV.exists():
        raise FileNotFoundError(f"Label Studio export file not found: {LS_EXPORT_CSV}")

    # 1. Load Label Studio exported human annotations indexed by Source_ID
    ls_by_id = {}
    with LS_EXPORT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = get_field(row, "Source_ID", "data.Source_ID")
            if sid:
                ls_by_id[sid] = row

    # 2. Merge into original suggestions structure
    merged_rows = []
    matched_count = 0
    missing_ids = []

    llm_patterns = []
    human_patterns = []
    llm_screenings = []
    human_screenings = []

    with SUGGESTIONS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        for row in reader:
            sid = row["Source_ID"].strip()
            ls_row = ls_by_id.get(sid)

            if ls_row:
                h_screen = get_field(ls_row, "Human_Final_Screening_Decision")
                h_pattern = get_field(ls_row, "Human_Final_IP_Pattern")
                h_cat = get_field(ls_row, "Human_Final_Category")
                h_span_raw = get_field(ls_row, "Human_Final_Evidence_Span")
                h_span = parse_ls_evidence_span(h_span_raw)

                if h_screen and h_pattern:
                    # Normalize category when Absent
                    if h_pattern == "Absent" and not h_cat:
                        h_cat = "None"
                    elif h_pattern == "Uncertain" and not h_cat:
                        h_cat = "Uncertain"

                    # Generate concise audit note comparing LLM vs Human
                    llm_p = row.get("Potential_IP_Pattern", "").strip()
                    if llm_p == h_pattern:
                        note = f"Human verified via Label Studio; agrees with LLM ({h_screen}/{h_pattern})."
                    else:
                        note = f"Human override via Label Studio; changed from LLM ({llm_p}) to ({h_pattern})."

                    row["Human_Review_Status"] = "COMPLETED"
                    row["Human_Final_Screening_Decision"] = h_screen
                    row["Human_Final_IP_Pattern"] = h_pattern
                    row["Human_Final_Category"] = h_cat
                    row["Human_Final_Evidence_Span"] = h_span
                    row["Human_Review_Notes"] = note

                    matched_count += 1
                    llm_patterns.append(llm_p)
                    human_patterns.append(h_pattern)
                    llm_screenings.append(row.get("Screening_Decision", "").strip())
                    human_screenings.append(h_screen)
                else:
                    missing_ids.append(sid)
            else:
                missing_ids.append(sid)

            merged_rows.append(row)

    # 3. Write Gold-Standard CSV (original files remain untouched)
    with GOLD_OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    # 4. Print Research Audit Report
    print("=" * 65)
    print("DEVFORWARD PILOT GOLD-STANDARD MERGE & AUDIT REPORT")
    print("=" * 65)
    print(f"Total Pilot Records       : {len(merged_rows)}")
    print(f"Human Review COMPLETED    : {matched_count}")
    print(f"Unannotated / Missing     : {len(missing_ids)}")
    if missing_ids:
        print(f"Missing Source_IDs        : {', '.join(missing_ids[:10])}")
    print(f"Saved Gold-Standard File  : {GOLD_OUTPUT_CSV}")
    print("-" * 65)

    if matched_count > 0:
        pat_counts = Counter(human_patterns)
        cat_counts = Counter(
            r["Human_Final_Category"]
            for r in merged_rows
            if r["Human_Review_Status"] == "COMPLETED"
        )

        agree_pat = (
            sum(1 for a, b in zip(human_patterns, llm_patterns) if a == b)
            / matched_count
            * 100
        )
        kappa_pat = compute_cohens_kappa(human_patterns, llm_patterns)

        agree_scr = (
            sum(1 for a, b in zip(human_screenings, llm_screenings) if a == b)
            / matched_count
            * 100
        )
        kappa_scr = compute_cohens_kappa(human_screenings, llm_screenings)

        print("1. Gold-Standard IP Pattern Distribution:")
        for k, v in pat_counts.items():
            print(f"   - {k:<15}: {v} ({v/matched_count*100:.1f}%)")

        print("\n2. Gold-Standard Category Distribution:")
        for k, v in cat_counts.items():
            print(f"   - {k:<24}: {v}")

        print("\n3. Human vs. Ollama (qwen2.5) Agreement Metrics (RQ2):")
        print(
            f"   - IP Pattern Agreement  : {agree_pat:.2f}% | Cohen's Kappa (k) = {kappa_pat:.4f}"
        )
        print(
            f"   - Screening Agreement   : {agree_scr:.2f}% | Cohen's Kappa (k) = {kappa_scr:.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
