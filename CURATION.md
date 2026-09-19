# CodeMentor dataset curation status

This file identifies the dataset artifacts used by the current CodeMentor V2 adapter. The project has two separate curation stages; the 7,000-example artifact is an intermediate development split, not the dataset passed directly to V2 training.

## Current source of truth

| Artifact | Contents | Current role |
|---|---:|---|
| `data/raw/codealpaca.json` | 20,022 examples | Local snapshot of the source dataset |
| `data/processed/dataset.json` | 6,000 train + 500 validation | General curated development split |
| `data/test/test.json` | 500 test | General held-out test partition |
| `data/processed/verified_dataset.json` | 1,000 train + 100 validation | **Actual dataset used to train V2** |
| `results/verified-curation/report.json` | V2 filter report | Authoritative V2 selection counts |
| `results/adapter-v2-manifest.json` | Dataset/model hashes and settings | V2 provenance |

The active adapter is `models/adapter`. `src/train.py` defaults to `data/processed/verified_dataset.json` and writes new runs to `models/adapter_candidate`, preventing accidental replacement of the active adapter.

## Stage 1: general 7,000-example split

The original local audit processes all 20,022 CodeAlpaca rows through `src/curation_report.py`, `src/curation.py`, `src/selection.py`, and `src/select_splits.py`.

It checks schema, required content, text integrity, Qwen chat length, programming relevance, placeholders, code-fence balance, Python parsing, narrow factual contradictions, exact duplicates, conflicting answers, reused answers, and lexical similarity groups. Eligible recognized-language records are ranked deterministically and capped so no language exceeds 50% of the selected set.

The selected 7,000 records are divided with seed `3407`:

- 6,000 training records;
- 500 validation records; and
- 500 test records.

Detected duplicate/similarity groups stay within one partition. This is heuristic lexical isolation, not proof that semantic paraphrases never cross splits.

The cached local Qwen judge reviewed 13,120 candidates but failed calibration with four false passes across seven controls. Its judgments were retained as audit evidence and were **not** used for selection.

### Stage 1 filter reference

| Category | Applied check | Purpose |
|---|---|---|
| Schema | Requires string `instruction`, `input`, and `output` fields. | Removes malformed records. |
| Required content | Requires non-empty instruction and output. | Removes unusable prompt/answer pairs. |
| Text integrity | Rejects replacement characters and disallowed control characters. | Removes damaged text. |
| Token length | Measures the complete Qwen chat-formatted conversation and rejects records over 512 tokens. | Keeps examples within the training context. |
| Programming relevance | Flags unknown-language records without programming terminology. | Removes likely unrelated tasks. |
| Completeness | Detects unbalanced code fences and placeholder text such as `TODO` and `FIXME`. | Removes visibly unfinished answers. |
| Python syntax | Parses and compiles Python answers and flags unresolved bare names. | Removes malformed or obviously incomplete Python. |
| Narrow factual checks | Independently checks supported prime-factor, binary-conversion, random-range, Java-debugging, and LCS-interface patterns. | Rejects known contradictions without pretending to be a general semantic judge. |
| Exact duplicates | Fingerprints the normalized complete row. | Removes repeated records. |
| Prompt conflicts | Detects identical prompts with different answers. | Avoids contradictory supervision. |
| Reused answers | Detects identical outputs across records. | Reduces repeated supervision. |
| Near duplicates | Groups prompts with lexical Jaccard similarity of at least `0.85`. | Keeps detected related prompts inside one split. |

Stage 1 recognizes 20 programming-language labels, but only Python receives a general parser/compiler check at this stage. The other language labels are heuristic and are used for selection balance.

## Stage 2: V2 training subset

`src/verified_curation.py` starts from the 6,000-record Stage 1 training partition. It uses the original validation and test prompts only to prevent leakage; it never selects test examples for training.

The V2 pass:

1. validates the `instruction`, `input`, and `output` schema;
2. normalizes and hashes instruction/input prompts;
3. rejects training prompts that overlap validation or test prompts;
4. removes duplicate training prompts;
5. requires a code-generation action in the instruction;
6. rejects empty, placeholder, unusually short/long, or incomplete output;
7. checks balanced parentheses, brackets, braces, and code fences;
8. parses Python and runs available local syntax/compile tools for JavaScript, Java, Bash, Go, and C#;
9. never executes dataset programs; and
10. selects deterministically with seed `3407` and a 50% per-language cap.

Recorded V2 selection results:

| Measure | Count |
|---|---:|
| Stage 1 training records considered | 6,000 |
| V2 candidates passing mechanical checks | 3,240 |
| V2 training records selected | 1,000 |
| V2 validation records selected | 100 |

V2 training-language distribution:

| Language | Examples |
|---|---:|
| Python | 500 |
| JavaScript | 388 |
| Java | 111 |
| Go | 1 |

Although Bash and C# validators were available locally, no examples in those languages entered the final 1,000 after all filters and balancing. C, C++, SQL, CSS, and other unsupported languages were excluded from the V2 subset.

### Stage 2 filter reference

| Category | Applied check | Rejection reason |
|---|---|---|
| Schema | Requires string `instruction`, `input`, and `output`. | `invalid_schema` |
| Required content | Requires non-empty instruction and output. | `empty_required_field` |
| Leakage | Excludes normalized prompt hashes found in validation or test. | `held_out_prompt` |
| Duplicate prompt | Keeps one normalized instruction/input prompt. | `duplicate_prompt` |
| Task type | Requires code-generation action wording. | `not_code_generation` |
| Output length | Requires 20 through 4,000 output characters. | `output_length` |
| Completeness | Requires balanced fences, parentheses, brackets, and braces. | `incomplete_output` |
| Placeholder | Rejects unfinished markers such as `TODO`, `FIXME`, and `not implemented`. | `placeholder` |
| Validator availability | Requires a configured validator for the detected language. | `validator_unavailable` |
| Syntax/compilation | Parses or compiles extracted code without executing it. | `syntax_failed` |

Language-specific Stage 2 checks:

| Language | Validator |
|---|---|
| Python | `ast.parse()` and Python compilation |
| JavaScript | `node --check` |
| Java | `javac -proc:none` |
| Bash | `bash -n` |
| Go | `gofmt -e` |
| C# | Temporary .NET 8 project with `dotnet build` |

Passing candidates are sorted by prompt fingerprint, shuffled with seed `3407`, limited to at most 50% from one language, and sorted again for stable output. Validator discovery depends on locally installed executables, so rebuilding in a different environment can change the eligible pool.

## Correctness boundary

Neither stage certifies semantic correctness for every answer. Parsing or compilation can detect malformed code but cannot establish algorithmic behavior, edge-case handling, dependency availability, explanation accuracy, or compliance with the instruction.

Independent review found semantically incorrect targets inside the 1,000-example V2 set despite their passing mechanical checks. The dataset must therefore be described as **syntax/compile-screened**, not correctness-verified.

The 11 questions used to compare baseline and V2 were also used in the V2 promotion decision. They are development regression prompts, not an unbiased final test. See `results/finetuned_vs_baseline.md` for the exact scores and limitations.

## Reproducing the datasets

Run the original cached audit with explicit local Arrow and tokenizer paths:

```powershell
python src/data_preparation.py --audit --arrow 'C:\path\to\code-alpaca-20k-train.arrow' --tokenizer 'C:\path\to\tokenizer\snapshot' --output-dir results/curation-new
```

After the 6,000/500/500 files exist, rebuild the V2 subset:

```powershell
python src/verified_curation.py
```

Outputs should be written to new locations when preserving previous reports. Exact hashes for the promoted V2 artifacts are stored in `results/adapter-v2-manifest.json`.
