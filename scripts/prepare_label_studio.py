import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = (
    BASE_DIR / "Datasets" / "Annotations" / "senti4sd_pilot_150_llm_v2_suggestions.csv"
)
OUTPUT_JSON = BASE_DIR / "Datasets" / "Annotations" / "senti4sd_pilot_150_ls.json"

tasks = []

with INPUT_CSV.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        screening = row.get("Screening_Decision") or "Not_relevant"
        pattern = row.get("Potential_IP_Pattern") or "Absent"
        category = row.get("Pattern_Category") or (
            "None" if pattern == "Absent" else "Uncertain"
        )
        evidence = row.get("Evidence_Span", "")
        text = row.get("Developer_Text", "")

        result = [
            {
                "from_name": "Human_Final_Screening_Decision",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [screening]},
            },
            {
                "from_name": "Human_Final_IP_Pattern",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [pattern]},
            },
            {
                "from_name": "Human_Final_Category",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [category]},
            },
        ]

        # Highlight evidence span automatically if present
        if (
            evidence
            and evidence in text
            and category
            in [
                "Fear_of_Incompetence",
                "Luck_Attribution",
                "Achievement_Discounting",
                "Other_or_Mixed",
            ]
        ):
            start = text.find(evidence)
            result.append(
                {
                    "from_name": "Human_Final_Evidence_Span",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": start + len(evidence),
                        "text": evidence,
                        "labels": [category],
                    },
                }
            )

        tasks.append(
            {
                "data": row,
                "predictions": [
                    {
                        "model_version": row.get("LLM_Model", "qwen2.5:latest"),
                        "result": result,
                    }
                ],
            }
        )

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Success! Created {OUTPUT_JSON} with {len(tasks)} pre-filled tasks.")
