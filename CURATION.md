# Local dataset quality audit

The existing `python src/data_preparation.py` summary remains available.
The new `--audit` mode uses exact local Arrow and tokenizer snapshot paths:

```powershell
python src/data_preparation.py --audit --arrow 'C:\path\to\code-alpaca-20k-train.arrow' --tokenizer 'C:\path\to\tokenizer\snapshot' --output-dir results/curation
python -m unittest discover -s tests -v
```

To include the real-tokenizer regression test, set
`$env:CODEMENTOR_TEST_TOKENIZER = 'C:\path\to\tokenizer\snapshot'` before
running the tests. Without this variable that one integration test is skipped.
The test verifies that long conversations exceed 512 tokens and prevents
counting dictionary fields instead of token IDs across Transformers versions.

The strengthened completed run is in `results/curation-postreview/`. The earlier
`results/curation-verified/` run remains as pre-review evidence, and the
`results/curation-final/` run is explicitly marked invalid due to a token-count
bug and must not be used.

## Selected dataset

The final deterministic selection contains 7,000 unique source examples:

- `data/processed/dataset.json`: 6,000 training and 500 validation examples.
- `data/test/test.json`: 500 test examples.
- `results/curation-selected/report.json`: counts, language distribution,
  calibration result, seed, and exact source indices for every split.

Selection requires an empty post-review audit-reason list, a recognized
programming language, no more than 512 complete chat tokens, and successful
Python static compilation when the language is Python. The Python audit also
flags unresolved bare names. Narrow validators reject the known incorrect
prime-factor, binary explanation, Java debugging, and LCS interface patterns.
Rows are ranked by deterministic structural quality signals and capped so one
language cannot exceed half of the selection. Seed 3407 assigns each detected
review group to one split. This is heuristic group isolation: lexical
similarity can miss paraphrases and code clones.

The cached Qwen 0.5B model reviewed 13,120 candidates. Its seven-control
calibration produced four false passes, so its scores are saved for audit and
are not used for selection. The selected files are provisional, locally
curated candidates, not independently approved answers. Establishing that
stronger status requires task-specific tests or human review of every row.

The audit requires the already-installed `datasets` and `transformers` packages.
It sets Hugging Face offline flags and uses `local_files_only=True`. No paid
services, model downloads, GPU, or execution of dataset code are involved.
Output directories must be new so previous reports are preserved.

## Interpretation

- `reject`: invalid schema, missing required content, damaged characters,
  excess tokens, exact duplicate, or a narrowly established factual error.
  A rejection such as excess tokens is eligibility-related, not proof that the
  answer is wrong.
- `review`: semantic correctness remains unverified. Flags may additionally
  identify syntax problems/fragments, placeholders, conflicting answers,
  shared answers, or similar prompts.
- `candidates_for_review.jsonl`: review rows with no detected flags. These are
  candidates, not approved training examples.
- `accepted.json`: only complete answers independently verified by the narrow
  binary-conversion or largest-prime-factor validators, passing all other checks.
  Syntax or length alone cannot promote a row. These math tasks do not establish
  programming-language coverage or usefulness for a coding mentor.
- `audit.jsonl`: every original row, stable content hash, source index, token
  count, heuristic language, review group, and all applicable reasons.
- `report.json`: aggregate counts, token percentiles, source/tokenizer/code
  hashes, package versions, and limitations.

The 512-token limit counts the entire user/assistant conversation using the
cached Qwen chat template, including its default system text and role markers.
No content is truncated or rewritten. Future training must use the same format
or repeat the token audit with its final format.

Only Python has a static parser/compiler check. Passing it does not verify
runtime behavior or fulfill the instruction. Other languages stay eligible for
review but their syntax is not asserted valid. Lexical token-set Jaccard
similarity at 0.85 and identical outputs create connected review groups. This
can overgroup unrelated tasks and miss paraphrases; it is not semantic
deduplication. A reviewer must resolve conflicts before any future split.

For a correctness-certified release, the next stage needs independently checked
expected behavior/test cases or human review of each instruction, input,
answer, and explanation. A small local language model cannot certify
correctness. Keep only independently approved examples even if that produces
fewer than 7,000.
