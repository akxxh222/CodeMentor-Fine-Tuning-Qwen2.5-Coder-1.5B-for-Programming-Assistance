"""Build a conservative, locally validated training subset.

Validation is deliberately syntax-focused. Dataset programs are compiled or
parsed but never executed, and no accepted row is described as semantically
proven merely because it compiles.
"""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile

try:
    from .curation import language_of
except ImportError:
    from curation import language_of


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TOOLS = {
    "javascript": "node",
    "java": "javac",
    "c#": "dotnet",
    "bash": "bash",
    "go": "gofmt",
}
PLACEHOLDER = re.compile(
    r"(?i)\b(?:TODO|FIXME|your code here|implementation goes here|not implemented)\b"
)
ACTION = re.compile(
    r"(?i)\b(?:write|create|implement|develop|design|build|generate|code|program|"
    r"function|method|class|script|query|statement)\b"
)


def prompt_fingerprint(example):
    normalized = {
        "instruction": " ".join(example.get("instruction", "").split()).casefold(),
        "input": "\n".join(line.rstrip() for line in example.get("input", "").strip().splitlines()),
    }
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_languages():
    available = {"python"}
    for language, executable in SUPPORTED_TOOLS.items():
        if shutil.which(executable):
            available.add(language)
    return available


def extract_code(output, language):
    blocks = re.findall(r"```(?:[A-Za-z0-9_+#.-]+)?\s*\n(.*?)```", output, re.S)
    if len(blocks) == 1:
        return blocks[0].strip()
    return output.strip()


def _balanced(text, opening, closing):
    depth = 0
    quote = None
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {'"', "'"}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if quote:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _run(command, code, suffix, filename=None):
    with tempfile.TemporaryDirectory(prefix="codementor-verify-") as directory:
        path = Path(directory) / (filename or f"answer{suffix}")
        path.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [part.format(path=str(path), directory=directory) for part in command],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return completed.returncode == 0


def syntax_valid(language, code):
    try:
        if language == "python":
            compile(ast.parse(code), "<dataset>", "exec")
            return True
        if language == "javascript":
            return _run(["node", "--check", "{path}"], code, ".js")
        if language == "bash":
            return _run(["bash", "-n", "{path}"], code, ".sh")
        if language == "go":
            return _run(["gofmt", "-e", "{path}"], code, ".go")
        if language == "java":
            match = re.search(r"\b(?:public\s+)?class\s+(\w+)", code)
            if not match:
                return False
            return _run(
                ["javac", "-proc:none", "{path}"],
                code,
                ".java",
                filename=f"{match.group(1)}.java",
            )
        if language == "c#":
            with tempfile.TemporaryDirectory(prefix="codementor-verify-") as directory:
                root = Path(directory)
                (root / "Program.cs").write_text(code, encoding="utf-8")
                (root / "Verify.csproj").write_text(
                    '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
                    '<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework>'
                    '<ImplicitUsings>enable</ImplicitUsings><Nullable>disable</Nullable>'
                    "</PropertyGroup></Project>",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    ["dotnet", "build", "--nologo", "--verbosity", "quiet"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired, SyntaxError, ValueError):
        return False
    return False


def validate_example(example, available_languages=None):
    available = discover_languages() if available_languages is None else available_languages
    reasons = []
    if not isinstance(example, dict) or any(
        not isinstance(example.get(key), str) for key in ("instruction", "input", "output")
    ):
        return {"accepted": False, "language": "unknown", "reasons": ["invalid_schema"]}

    instruction = example["instruction"].strip()
    output = example["output"].strip()
    language = language_of(example)
    if not instruction or not output:
        reasons.append("empty_required_field")
    if language not in available:
        reasons.append("validator_unavailable")
    if not ACTION.search(instruction):
        reasons.append("not_code_generation")
    if len(output) < 20 or len(output) > 4000:
        reasons.append("output_length")
    if output.count("```") % 2 or not all(
        _balanced(output, left, right) for left, right in (("(", ")"), ("[", "]"), ("{", "}"))
    ):
        reasons.append("incomplete_output")
    if PLACEHOLDER.search(output):
        reasons.append("placeholder")

    code = extract_code(output, language)
    if language in available and output and not syntax_valid(language, code):
        reasons.append("syntax_failed")
    return {
        "accepted": not reasons,
        "language": language,
        "reasons": reasons,
    }


def build_verified_dataset(
    train,
    validation,
    test,
    limit=1000,
    seed=3407,
    available_languages=None,
):
    held_out = {prompt_fingerprint(row) for row in [*validation, *test]}
    rejected = Counter()
    candidates = []
    seen = set()
    for row in train:
        prompt_id = prompt_fingerprint(row)
        if prompt_id in held_out:
            rejected["held_out_prompt"] += 1
            continue
        if prompt_id in seen:
            rejected["duplicate_prompt"] += 1
            continue
        seen.add(prompt_id)
        verdict = validate_example(row, available_languages=available_languages)
        if not verdict["accepted"]:
            rejected.update(verdict["reasons"])
            continue
        candidates.append((row, verdict["language"], prompt_id))

    rng = random.Random(seed)
    candidates.sort(key=lambda item: item[2])
    rng.shuffle(candidates)
    languages = {item[1] for item in candidates}
    selected = []
    counts = Counter()
    maximum_per_language = limit if len(languages) < 2 else max(1, limit // 2)
    for row, language, _ in candidates:
        if len(selected) >= limit:
            break
        if counts[language] >= maximum_per_language:
            continue
        selected.append(row)
        counts[language] += 1
    selected.sort(key=prompt_fingerprint)

    report = {
        "method": "conservative local syntax/compile validation; semantic correctness not guaranteed",
        "source_train": len(train),
        "accepted_candidates": len(candidates),
        "selected": len(selected),
        "limit": limit,
        "seed": seed,
        "available_languages": sorted(
            discover_languages() if available_languages is None else available_languages
        ),
        "language_counts": dict(sorted(counts.items())),
        "rejection_reasons": dict(sorted(rejected.items())),
    }
    return selected, report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/dataset.json")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/test/test.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/verified_dataset.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/verified-curation/report.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))["test"]
    available = discover_languages()
    selected, report = build_verified_dataset(
        source["train"], source["validation"], test, args.limit, args.seed, available
    )
    verified_validation, validation_report = build_verified_dataset(
        source["validation"], [], test, min(100, len(source["validation"])), args.seed, available
    )
    report["validation_selected"] = len(verified_validation)
    report["validation_rejection_reasons"] = validation_report["rejection_reasons"]
    write_json(args.output, {"train": selected, "validation": verified_validation})
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
