# Dataset curation: CodeAlpaca source to the V2 training subset

This document records both deterministic local curation stages used by the current project. Stage 1 turns the 20,022 CodeAlpaca source examples into a general 7,000-record development dataset. Stage 2 selects the stricter 1,000-training/100-validation subset actually used for the active V2 adapter. Neither stage guarantees semantic correctness.

The Stage 1 files are:

- `data/processed/dataset.json` — 6,000 training and 500 validation examples.
- `data/test/test.json` — 500 held-out test examples.
- `results/curation-postreview/` — audit outputs for the original 20,022 rows.
- `results/curation-selected/report.json` — selection, language, provenance, and split metadata for the final 7,000 rows.

The current V2 files are:

- `data/processed/verified_dataset.json` — **1,000 V2 training and 100 V2 validation examples**.
- `results/verified-curation/report.json` — V2 validator availability, selection, language, and rejection counts.
- `results/adapter-v2-manifest.json` — hashes tying the promoted adapter to its dataset and configuration.

## Complete pipeline overview

```text
CodeAlpaca Arrow cache (20,022)
    |
    +-- src/curation_report.py : local audit orchestration
    |       +-- src/curation.py : row checks and duplicate grouping
    |
    +-- 17,269 structurally clean candidates
            |
            +-- src/selection.py : eligibility and balanced ranking
            +-- src/select_splits.py : deterministic selection and split
                    |
                    +-- 12,883 eligible, recognized-language candidates
                    +-- 7,000 selected candidates
                            |
                            +-- 6,000 train / 500 validation / 500 test
                                    |
                                    +-- src/verified_curation.py
                                            +-- prompt leakage exclusion
                                            +-- local syntax/compile checks
                                            +-- deterministic balanced selection
                                                    |
                                                    +-- 1,000 V2 train / 100 V2 validation
```

The Stage 1 counts are from `results/curation-postreview/report.json` and `results/curation-selected/report.json`. The V2 counts are from `results/verified-curation/report.json` and the generated dataset itself.

## Stage 1: source loading and audit controls

| Category | What is checked | File and named block | Result when triggered |
| --- | --- | --- | --- |
| Offline source loading | Uses one exact cached Arrow dataset and one exact local tokenizer snapshot; sets Hugging Face and datasets offline flags. | `src/curation_report.py` — `main()` | Prevents a new download or a changing remote source from affecting the audit. |
| Reproducibility metadata | Records source, tokenizer, code hashes, library versions, token distribution, and reason counts. | `src/curation_report.py` — `sha256_file()` and the `report` dictionary in `main()` | Makes the audit traceable and repeatable. |
| Chat-format token length | Counts tokens after applying the Qwen chat template to the complete user prompt and assistant answer. Maximum is 512. | `src/curation.py` — `conversation_tokens()`; `src/curation_report.py` — `--max-tokens`; `src/curation.py` — `inspect_row()` | Rejects `token_limit` or `token_count_unavailable`. |

## Stage 1: per-example quality filters

All filters below are applied by `src/curation.py` in `inspect_row()`. A row with an applicable reason is not eligible for final selection because `src/selection.py` `candidate_score()` requires an empty `reasons` list.

| Filtering category | Specific checks and reason labels | Why it is used |
| --- | --- | --- |
| Data schema and completeness | Requires dictionary rows containing string `instruction`, `input`, and `output` fields: `invalid_schema`. Requires non-empty `instruction` and `output`: `empty_instruction`, `empty_output`. | Removes corrupt or unusable training pairs. |
| Text integrity | Detects Unicode replacement characters: `replacement_character`; detects disallowed control characters: `control_character`. | Removes damaged text and hidden control data. |
| Length | Rejects chat-formatted records over 512 Qwen tokens: `token_limit`. | Keeps examples within the intended context/training budget. |
| Programming relevance | Marks unknown-language records without programming-related terms: `programming_relevance_uncertain`. | Avoids unrelated general-domain tasks. This is conservative and heuristic, not a classifier. |
| Code completeness | Detects unmatched triple-backtick fences: `unbalanced_code_fence`. Detects template text such as `TODO`, `FIXME`, `your code here`, or `implementation goes here`: `possible_placeholder`. | Avoids incomplete code and unfilled templates. |
| Language identification | Identifies JavaScript, TypeScript, Python, Java, C++, C#, SQL, HTML, CSS, Ruby, Rust, Go, Swift, Kotlin, PHP, Bash, Scala, Perl, R, and C from the instruction or a small set of output patterns. | Enables language-specific checks and multilingual selection. Implemented by `language_of()`. Labels are heuristic. |
| Python syntax | Parses and compiles Python output or fenced Python blocks using Python AST. | Marks `python_syntax_or_fragment` if parsing fails and `python_compile_warning` when a `SyntaxWarning` is emitted. |
| Python name resolution | Collects defined and loaded names from the answer and, where parseable, the supplied input. Flags undefined bare names as `python_unresolved_name`. | Catches common incomplete Python examples that reference undeclared variables/modules. Implemented by `_python_names()` and the Python branch of `inspect_row()`. |
| Python task interface | For the specifically recognized “longest common subsequence ... two strings” task, checks that a Python solution does not require more than two non-default positional arguments. | Flags `lcs_interface_mismatch` to catch a known wrong public interface. This is intentionally narrow rather than a general semantic verifier. |
| Exact prime-factor facts | For an exact whole-question / whole-answer format, independently computes the largest prime factor. | Flags `incorrect_prime_factor`; otherwise records `verified_exact_prime_factor_answer`. |
| JavaScript random-range fact | Detects the exact case where a task requires a random number between 0 and 1 inclusively but the answer is only `Math.random()`. | Flags `random_endpoint_mismatch`, because `Math.random()` excludes 1. |
| Java debug explanation | Detects a known Java assignment-operator error pattern when the explanation incorrectly identifies the operator. | Flags `incorrect_debug_explanation`. |
| Exact binary conversion | For the exact binary-to-decimal task format, converts the provided binary input independently. | Flags `incorrect_binary_conversion`; records `verified_exact_binary_answer` if correct; marks non-numeric explanations `binary_explanation_requires_review`. |

## Stage 1: duplicate, conflict, and similarity controls

`src/curation.py` `audit_duplicates(rows, records, threshold=0.85)` runs after all per-row checks.

| Filtering category | Specific check / reason label | Why it is used |
| --- | --- | --- |
| Exact duplicate record | SHA-256 fingerprint over the complete normalized JSON row: `duplicate_row`. | Removes identical examples. Implemented by `fingerprint()`. |
| Same prompt, different answer | Groups identical instruction-plus-input prompts and detects distinct outputs: `conflicting_answers`. | Avoids training on mutually contradictory answers. |
| Reused answer | Groups identical stripped outputs: `shared_answer`. | Avoids repeatedly training on the same response. |
| Near-duplicate prompt group | Builds deterministic connected groups for prompts with lexical Jaccard similarity of at least 0.85: `similarity_group_requires_review`. | Reduces obvious prompt duplication and supplies a group ID for split isolation. It can miss paraphrases and code clones, so it is not a proof of no leakage. |

## Stage 1: eligibility and 7,000-item selection

`src/selection.py` `candidate_score(record)` applies the final eligibility gate. A selected record must have all of the following:

- audit `status == "review"` and no audit reasons;
- a recognized, non-`unknown`, non-`mixed` language;
- for Python, `syntax == "python_parse_pass"`;
- a complete Qwen chat length from 32 through 512 tokens;
- non-empty string instruction and output fields.

The function then assigns a deterministic ranking score based on useful length, supplied input, output length, task/action wording, code-like output, and whether the language appears in the instruction. These are ranking signals only; they are not correctness claims.

`src/selection.py` `select_balanced()` takes the highest-ranked candidates while applying a 50% maximum per-language cap. This preserves multilingual coverage and prevents one language, especially Python, from dominating the 7,000 selected rows. The result contained 20 recognized languages; the exact language counts are in `results/curation-selected/report.json`.

| Stage | Count |
| --- | ---: |
| Original CodeAlpaca rows audited | 20,022 |
| Structurally clean candidates with no audit reason | 17,269 |
| Eligible recognized-language candidates | 12,883 |
| Final selected candidates | 7,000 |

## Stage 1: split isolation and output files

`src/selection.py` `allocate_sizes()` produces 6,000 training, 500 validation, and 500 test rows when at least 7,000 candidates are available.

`src/selection.py` `build_splits(records, sizes, seed=3407)` groups records by the duplicate/similarity group created in `audit_duplicates()`, shuffles reproducibly with seed 3407, and assigns whole groups to one split only. This is a **heuristic group-isolated split**: it prevents the detected groups from crossing train, validation, and test, but cannot prove that undetected semantic paraphrases never cross splits.

`src/select_splits.py` `main()` writes the clean training format to:

- `data/processed/dataset.json`: `train` and `validation` arrays.
- `data/test/test.json`: `test` array.

For provenance without adding source identifiers to model training records, the original indices for every final row are stored in `results/curation-selected/report.json` under `source_indices`.

## Stage 1: local Qwen review, evaluated but not used as a filter

`src/local_review.py` can load a cached local Qwen causal-language model using `load_local_model()` and score the next-token probabilities for `PASS` versus `FAIL`. Its review prompt is created by `src/selection.py` `build_judge_prompt()`.

The local judge was calibrated with seven known controls by `src/selection.py` `calibration_summary()`. Its results were 3 correct, 4 false passes, and 0 false fails. Because calibration failed, `src/select_splits.py` explicitly sets `local_judge_used_for_selection` to `false`. Therefore, no model verdict was used to claim correctness or choose the final 7,000 rows.

## Stage 2: V2 syntax/compile-screened selection

`src/verified_curation.py` is the source of truth for the dataset used by `src/train.py`. It consumes only the Stage 1 training partition for training selection. Validation and test records contribute prompt fingerprints for leakage prevention but are never added to V2 training.

### V2 filters

| Category | Check | Result when triggered |
|---|---|---|
| Schema | Requires string `instruction`, `input`, and `output` fields. | `invalid_schema` |
| Required content | Requires non-empty instruction and output. | `empty_required_field` |
| Leakage | Normalizes and SHA-256 hashes instruction/input; excludes hashes found in Stage 1 validation or test. | `held_out_prompt` |
| Duplicate prompt | Keeps one normalized instruction/input prompt in the training candidate pool. | `duplicate_prompt` |
| Task type | Requires code-generation action wording such as write, create, implement, function, class, script, query, or program. | `not_code_generation` |
| Output length | Requires 20 through 4,000 output characters. | `output_length` |
| Completeness | Requires balanced code fences, parentheses, brackets, and braces. | `incomplete_output` |
| Placeholder | Rejects `TODO`, `FIXME`, `your code here`, `implementation goes here`, and `not implemented`. | `placeholder` |
| Validator availability | Requires a configured local parser/compiler for the detected language. | `validator_unavailable` |
| Syntax/compilation | Parses or compiles extracted code without running the program. | `syntax_failed` |

### V2 language validators

| Language | Local check |
|---|---|
| Python | `ast.parse()` followed by Python compilation |
| JavaScript | `node --check` |
| Java | `javac -proc:none` |
| Bash | `bash -n` |
| Go | `gofmt -e` |
| C# | Temporary .NET 8 project and `dotnet build` |

Validators are discovered from locally installed executables, so rerunning the selector in a different environment may change the available candidate pool. Dataset programs are not executed.

### V2 deterministic selection

Passing training candidates are sorted by prompt fingerprint, shuffled with seed `3407`, and selected up to a limit of 1,000. When multiple languages are available, no language may occupy more than 50% of the selected set. The final records are sorted again by prompt fingerprint for stable output.

| V2 measure | Count |
|---|---:|
| Stage 1 training records considered | 6,000 |
| Candidates passing V2 mechanical checks | 3,240 |
| V2 training records selected | 1,000 |
| V2 validation records selected | 100 |

| V2 training language | Count |
|---|---:|
| Python | 500 |
| JavaScript | 388 |
| Java | 111 |
| Go | 1 |

The recorded environment discovered Python, JavaScript, Java, Bash, Go, and C# validators. Bash and C# nevertheless contributed no examples to the final 1,000 after all filters and selection constraints. Languages without an available V2 validator, including the C/C++/SQL/CSS examples in this run, were excluded from V2 training.

## Important limitations

- Stage 1 applies a static parser/compiler only to Python; its other languages remain structurally screened candidates.
- Stage 2 applies the language validators listed above, but parse/compile success does not prove runtime behavior, dependencies, edge cases, task compliance, or explanation accuracy.
- The exact factual checks cover only deliberately narrow task formats.
- Similarity grouping is lexical and heuristic.
- No source dataset code is executed; the pipeline remains local and does not require paid services.
- Independent review found semantically incorrect answers in the V2 dataset despite their passing mechanical checks. Do not call it correctness-certified.
- The 11-question V2 comparison was used for promotion and is a development regression sample, not an unbiased final benchmark.

For authoritative machine-readable counts and limitations, see `results/curation-postreview/report.json`, `results/curation-selected/report.json`, `results/verified-curation/report.json`, and `results/adapter-v2-manifest.json`.
