"""Select locally reviewed examples and create deterministic dataset splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from selection import allocate_sizes, build_splits, calibration_summary, candidate_score, select_balanced


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--calibration-judgments", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    records = load_jsonl(args.candidates)
    judgments = {value["id"]: value for value in load_jsonl(args.judgments)} if args.judgments else {}
    calibration_values = load_jsonl(args.calibration_judgments) if args.calibration_judgments else []
    calibration = calibration_summary(calibration_values)
    eligible = [record for record in records if candidate_score(record) is not None]
    selected = select_balanced(eligible, min(args.target, len(eligible)), max_language_fraction=0.50)
    sizes = allocate_sizes(len(selected))
    splits = build_splits(selected, sizes, seed=args.seed)
    output = {name: [record["example"] for record in values] for name, values in splits.items()}
    root = args.project_root.resolve()
    write_json(root / "data" / "processed" / "dataset.json", {
        "train": output["train"],
        "validation": output["validation"],
    })
    write_json(root / "data" / "test" / "test.json", {"test": output["test"]})
    report = {
        "method": "deterministic structural filtering with heuristic group-isolated splitting",
        "correctness_guarantee": False,
        "warning": "These examples are curated candidates, not proof that every answer is correct.",
        "judged": len(judgments),
        "local_judge_used_for_selection": False,
        "local_judge_calibration": calibration,
        "calibration_file_sha256": hashlib.sha256(args.calibration_judgments.read_bytes()).hexdigest() if args.calibration_judgments else None,
        "structurally_eligible_recognized_language": len(eligible),
        "selected": len(selected),
        "target": args.target,
        "shortage": max(0, args.target - len(selected)),
        "split_sizes": sizes,
        "seed": args.seed,
        "language_counts": dict(sorted(Counter(record["language"] for record in selected).items())),
        "source_indices": {name: [record["source_index"] for record in values] for name, values in splits.items()},
    }
    write_json(root / "results" / "curation-selected" / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
