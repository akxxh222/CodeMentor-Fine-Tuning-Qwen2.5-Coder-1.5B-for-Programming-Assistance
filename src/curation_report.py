"""Offline entry point for reproducible dataset auditing."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from curation import FIELDS, inspect_row, audit_duplicates, conversation_tokens


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrow', required=True, type=Path, help='Exact cached Arrow file; never downloads.')
    parser.add_argument('--tokenizer', required=True, type=Path, help='Exact local tokenizer snapshot directory.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'curation')
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new directory to preserve previous runs.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from datasets import Dataset
    from transformers import AutoTokenizer
    dataset = Dataset.from_file(str(args.arrow.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), local_files_only=True)
    rows = list(dataset)
    records = []
    print(f'Auditing {len(rows)} rows locally...', flush=True)
    for i, row in enumerate(rows):
        count = None
        if isinstance(row, dict) and all(isinstance(row.get(k), str) for k in FIELDS):
            count = conversation_tokens(row, tokenizer)
        record = inspect_row(row, count, args.max_tokens)
        record['source_index'] = i
        records.append(record)
        if (i + 1) % 5000 == 0:
            print(f'Tokenized and checked {i + 1} rows', flush=True)
    print('Checking duplicate, conflicting, and lexically similar prompts...', flush=True)
    audit_duplicates(rows, records)
    statuses = Counter(r['status'] for r in records)
    reasons = Counter(reason for r in records for reason in r['reasons'])
    counts = sorted(r['tokens'] for r in records if r['tokens'] is not None)
    quantiles = {str(p): counts[round((len(counts) - 1) * p / 100)] if counts else None for p in (0, 25, 50, 75, 90, 95, 99, 100)}
    candidate_ids = [i for i, r in enumerate(records) if r['status'] == 'review' and not r['reasons']]
    accepted_ids = [i for i, r in enumerate(records) if r['status'] == 'accepted']
    report = {
        'source': str(args.arrow.resolve()), 'source_sha256': sha256_file(args.arrow),
        'tokenizer': str(args.tokenizer.resolve()),
        'tokenizer_files_sha256': {p.name: sha256_file(p) for p in sorted(args.tokenizer.iterdir()) if p.suffix in ('.json', '.txt') and p.name != 'config.json'},
        'versions': {name: importlib.metadata.version(name) for name in ('datasets', 'transformers', 'tokenizers')},
        'code_sha256': {p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name('curation.py'))},
        'total': len(rows), 'statuses': dict(statuses), 'reason_counts': dict(sorted(reasons.items())),
        'languages_all_heuristic': dict(sorted(Counter(r['language'] for r in records).items())),
        'languages_candidates_heuristic': dict(sorted(Counter(records[i]['language'] for i in candidate_ids).items())),
        'syntax': dict(Counter(r['syntax'] for r in records)), 'token_percentiles': quantiles,
        'max_tokens': args.max_tokens, 'similarity_jaccard_threshold': 0.85,
        'structural_candidates_requiring_semantic_review': len(candidate_ids),
        'accepted': len(accepted_ids), 'target': 7000, 'shortage': max(0, 7000 - len(accepted_ids)),
        'split_created': False,
        'limitations': [
            'Structural candidates are NOT correctness-verified training examples.',
            'Only exact supported numeric binary conversions and whole prime-factor answers have independent semantic validators; other answers require review.',
            'Only Python receives static compile checks; other language syntax remains unchecked.',
            'Compile success does not establish behavior, dependencies, task compliance, or explanation accuracy.',
            'Factual rejection rules cover narrow prime-factor and binary-conversion question formats and a direct Math.random endpoint mismatch.',
            'Lexical similarity can miss paraphrases and can group distinct tasks; groups require review.',
            'Language labels are heuristics and are not a validated language classifier.',
            'No dataset code is executed and no model judge is used.',
            'The 512-token limit applies to complete user+assistant chat formatting, including tokenizer-added tokens.',
        ],
    }
    # Build all results before writing. Exclusive creation prevents overwrites.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        with (args.output_dir / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    write_json('report.json', report)
    write_json('accepted.json', [rows[i] for i in accepted_ids])
    with (args.output_dir / 'audit.jsonl').open('x', encoding='utf-8') as handle:
        for row, record in zip(rows, records):
            handle.write(json.dumps(dict(**record, example=row), ensure_ascii=False) + '\n')
    with (args.output_dir / 'candidates_for_review.jsonl').open('x', encoding='utf-8') as handle:
        for i in candidate_ids:
            handle.write(json.dumps(dict(**records[i], example=rows[i]), ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
