# Historical Adapter V2 Implementation Plan

> **Status:** Implemented and promoted with documented deviations. The active V2 uses 1,000 training and 100 validation examples from `data/processed/verified_dataset.json` and is stored at `models/adapter`. The 11-prompt promotion comparison is development regression evidence, not a held-out final benchmark. Mechanical validation did not certify semantic correctness. This file preserves the pre-implementation plan; current facts are in `README.md`, `CURATION.md`, and `results/finetuned_vs_baseline.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and evaluate a new CodeMentor LoRA adapter trained on at most 1,000 conservatively verified examples, without changing the existing UI or inference behavior.

**Architecture:** A standalone curation module filters only the existing 6,000-example training partition through deterministic quality rules and locally available language validators. Training consumes a conversational prompt/completion dataset so loss applies only to assistant completions, writes to `models/adapter_candidate`, and leaves `models/adapter` untouched until evaluation proves improvement.

**Tech Stack:** Python 3.12, pytest, TRL SFTTrainer, Unsloth QLoRA, local Python/Node/Java/.NET/Bash/Go validation tools.

**Spec:** User-approved design in the 2026-09-19 conversation: correctness over dataset size, no test leakage, preserve current implementation and adapter, promote only after held-out improvement.

## Global Constraints

- Never select examples from `data/processed/dataset.json` validation or `data/test/test.json` test.
- Maximum candidate training size is 1,000; fewer examples are acceptable.
- Do not execute untrusted dataset programs; syntax/compile checks only.
- Exclude languages when their validator is unavailable.
- Preserve `models/adapter` and write the new model to `models/adapter_candidate`.
- Keep Flask UI and inference behavior unchanged until the candidate passes evaluation.

## Review Focus

- Train/test leakage: candidate fingerprints must not intersect validation or test fingerprints.
- Validator absence: affected languages must be rejected, never silently accepted.
- Incomplete programs: unbalanced or visibly truncated answers must be rejected.
- Prompt/completion masking: training must use completion-only loss rather than all-token language modeling.
- Promotion safety: candidate artifacts must not overwrite the current adapter during training.

---

### Task 1: Conservative verified-dataset builder

**Files:**
- Create: `src/verified_curation.py`
- Create: `tests/test_verified_curation.py`
- Create when run: `data/processed/verified_dataset.json`
- Create when run: `results/verified-curation/report.json`

**Interfaces:**
- Consumes: current `train` records from `data/processed/dataset.json`; validation/test records only for leakage exclusion.
- Produces: `build_verified_dataset(train, validation, test, limit, seed) -> (records, report)` and CLI-generated dataset/report artifacts.

- [ ] Write tests for deterministic selection, leakage exclusion, unavailable-validator rejection, incomplete-output rejection, Python syntax checking, and the 1,000-record cap.
- [ ] Run `python -m pytest tests/test_verified_curation.py -q` and confirm failures because the module does not exist.
- [ ] Implement the smallest conservative filter and deterministic balanced selector that satisfies the tests.
- [ ] Run the focused tests and then `python -m pytest -q`.
- [ ] Generate the verified dataset and inspect its size, language distribution, and rejection reasons.

### Task 2: Completion-only candidate training configuration

**Files:**
- Modify: `src/train.py`
- Create: `tests/test_training_config.py`
- Create when run: `models/adapter_candidate/`

**Interfaces:**
- Consumes: `data/processed/verified_dataset.json` containing `train` and `validation` lists.
- Produces: prompt/completion records and a candidate adapter under a CLI-selected output directory.

- [ ] Write tests proving prompt and completion roles are separated, defaults point to candidate artifacts, the learning rate is reduced, and completion-only loss is enabled.
- [ ] Run the focused test and confirm it fails against the current all-token configuration.
- [ ] Add CLI/configuration helpers while preserving the existing model and LoRA architecture.
- [ ] Run focused and full suites.
- [ ] Stop the Flask process only after all CPU-side checks pass, then train the candidate adapter.

### Task 3: Held-out evaluation and promotion decision

**Files:**
- Create: `results/finetuned_vs_baseline.md`
- Preserve: `models/adapter/`

**Interfaces:**
- Consumes: baseline model, current adapter, candidate adapter, unchanged held-out prompts.
- Produces: verbatim outputs, correctness/code-validity scores, and a promote/reject decision.

- [ ] Verify the candidate adapter files and training state.
- [ ] Generate deterministic responses from baseline, current adapter, and candidate on the same independent sessions.
- [ ] Score outputs without repairing them and record evidence in the report.
- [ ] Promote only if candidate correctness and code-validity exceed both the baseline and current adapter; otherwise retain the current adapter.
- [ ] Run the full automated suite and final artifact checks.
