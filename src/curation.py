"""Conservative local audit. A structural pass is NOT semantic approval.

Dataset code is never executed. Language identification and similarity are
heuristics; unsupported languages and ambiguous snippets require review.
"""
import ast
import builtins
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import unicodedata
import warnings

FIELDS = ('instruction', 'input', 'output')
LANGUAGES = ('javascript', 'typescript', 'python', 'java', 'c++', 'c#',
             'sql', 'html', 'css', 'ruby', 'rust', 'go', 'swift', 'kotlin',
             'php', 'bash', 'scala', 'perl', 'r', 'c')


def _python_names(tree):
    defined, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded).add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            defined.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            defined.update(alias.asname or alias.name for alias in node.names if alias.name != '*')
    return defined, loaded


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def conversation_tokens(row, tokenizer):
    user = row['instruction'] + ('\n\n' + row['input'] if row['input'].strip() else '')
    messages = [dict(role='user', content=user), dict(role='assistant', content=row['output'])]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def language_of(row):
    text = row['instruction'].lower()
    found = [lang for lang in LANGUAGES if re.search(r'(?<!\w)' + re.escape(lang) + r'(?!\w)', text)]
    # Avoid matching C within explicitly named C++/C#.
    if 'c++' in found or 'c#' in found:
        found = [lang for lang in found if lang != 'c']
    if found:
        return found[0] if len(found) == 1 else 'mixed'
    output = row['output']
    for pattern, lang in [(r'(?m)^\s*(def \w+\(|from \w+ import |import (os|random)\b)', 'python'),
                          (r'\bMath\.random\(|\bconsole\.log\(', 'javascript'),
                          (r'(?i)\b(CREATE TABLE|SELECT .+ FROM|INSERT INTO)\b', 'sql'),
                          (r'(?i)<(?:html|!doctype html)\b', 'html')]:
        if re.search(pattern, output):
            return lang
    return 'unknown'


def inspect_row(row, token_count, max_tokens=512):
    result = dict(id=fingerprint(row), status='review', reasons=[], language='unknown',
                  tokens=token_count, syntax='not_checked', semantic='unverified')
    reasons = result['reasons']
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in FIELDS):
        result.update(status='reject', reasons=['invalid_schema'])
        return result
    for key in ('instruction', 'output'):
        if not row[key].strip():
            reasons.append('empty_' + key)
    text = '\n'.join(row[k] for k in FIELDS)
    if '\ufffd' in text:
        reasons.append('replacement_character')
    if any(unicodedata.category(c) == 'Cc' and c not in '\n\r\t' for c in text):
        reasons.append('control_character')
    if token_count is None:
        reasons.append('token_count_unavailable')
    elif token_count > max_tokens:
        reasons.append('token_limit')
    if reasons:
        result['status'] = 'reject'
    result['language'] = language_of(row)
    out = row['output'].strip()
    if result['language'] == 'unknown' and not re.search(
            r'(?i)\b(code|program|programming|function|class|algorithm|array|list|tuple|dictionary|'
            r'string|loop|recursion|sort|stack|queue|tree|graph|debug|exception|api|json|database|prime|'
            r'regex|regular expression|binary|compiler|variable|integer|vector)\b', text):
        reasons.append('programming_relevance_uncertain')
    if out.count('```') % 2:
        reasons.append('unbalanced_code_fence')
    if re.search(r'(?i)\b(TODO|FIXME|your code here|implementation goes here)\b', out):
        reasons.append('possible_placeholder')
    if result['language'] == 'python':
        blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', out, re.S)
        candidates = blocks or [out]
        try:
            trees = []
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', SyntaxWarning)
                for code in candidates:
                    tree = ast.parse(code)
                    compile(tree, '<dataset>', 'exec')
                    trees.append(tree)
            if captured:
                reasons.append('python_compile_warning')
            result['syntax'] = 'python_parse_pass'
            defined, loaded = set(), set()
            for tree in trees:
                local_defined, local_loaded = _python_names(tree)
                defined.update(local_defined)
                loaded.update(local_loaded)
            try:
                input_tree = ast.parse(row['input'])
                input_defined, _ = _python_names(input_tree)
                defined.update(input_defined)
            except (SyntaxError, ValueError):
                pass
            unresolved = loaded - defined - set(dir(builtins)) - {'__name__'}
            if unresolved:
                reasons.append('python_unresolved_name')
            if re.search(r'(?i)longest common subsequence.*two strings', row['instruction']):
                functions = [node for tree in trees for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if functions:
                    required = len(functions[0].args.posonlyargs) + len(functions[0].args.args) - len(functions[0].args.defaults)
                    if required > 2:
                        reasons.append('lcs_interface_mismatch')
        except (SyntaxError, ValueError, RecursionError):
            result['syntax'] = 'python_parse_failed_or_fragment'
            reasons.append('python_syntax_or_fragment')
    # Narrow whole-question recognition avoids judging unrelated numbers/code.
    match = re.fullmatch(r'What is the largest prime factor of (?:the number )?(\d{1,7})\?', row['instruction'].strip(), re.I)
    answer = re.fullmatch(r'The largest prime factor of (\d+) is (\d+)\.', out, re.I)
    if match and answer and not row['input'].strip() and match[1] == answer[1] and int(match[1]) >= 2:
        n, factor = int(match[1]), 2
        largest = 1
        while factor * factor <= n:
            while n % factor == 0:
                largest, n = factor, n // factor
            factor += 1
        expected = max(largest, n)
        if int(answer[2]) != expected:
            reasons.append('incorrect_prime_factor')
            result.update(status='reject', semantic='contradiction_found')
        else:
            result['semantic'] = 'verified_exact_prime_factor_answer'
    if (re.search(r'(?i)random (?:number|float).*between 0 and 1.*inclusiv', row['instruction'])
            and not row['input'].strip()
            and re.fullmatch(r'Math\.random\(\);?(?:\s*//[^\n]*)?', out)):
        reasons.append('random_endpoint_mismatch')
        result.update(status='reject', semantic='contradiction_found')
    if (re.search(r'(?i)debug this Java code', row['instruction'])
            and re.search(r'int\s+\w+\s*=\s*\w+\s+\w+\s*;', row['input'])
            and re.search(r'(?i)operator (?:being used )?was\s+["\']?\w+["\']?\s+instead of', out)):
        reasons.append('incorrect_debug_explanation')
    if re.fullmatch(r'Convert the following binary number to a decimal number\.', row['instruction'].strip(), re.I):
        binary = re.fullmatch(r'Binary Number:\s*([01]{1,64})', row['input'].strip(), re.I)
        if binary:
            if out.isdecimal():
                if int(out) != int(binary[1], 2):
                    reasons.append('incorrect_binary_conversion')
                    result.update(status='reject', semantic='contradiction_found')
                else:
                    result['semantic'] = 'verified_exact_binary_answer'
            else:
                reasons.append('binary_explanation_requires_review')
    if result['semantic'].startswith('verified_') and not reasons:
        result['status'] = 'accepted'
    return result


def audit_duplicates(rows, records, threshold=0.85):
    """Exact conflicts plus lexical Jaccard similarity on prompt token sets.

    Frequency-ordered prefix postings retrieve Jaccard candidate pairs; this is deterministic
    and does not claim to detect semantic paraphrases. Components are review
    groups and must stay together in any future split.
    """
    prompts, answers, seen = defaultdict(list), defaultdict(list), {}
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    sets, postings = [], defaultdict(set)
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in FIELDS):
            sets.append(set())
        else:
            prompt = '\n'.join(row[k].strip() for k in ('instruction', 'input'))
            sets.append(set(re.findall(r'\w+|[^\w\s]', unicodedata.normalize('NFKC', prompt).lower())))
    frequency = Counter(token for tokens in sets for token in tokens)
    for i, (row, record) in enumerate(zip(rows, records)):
        if 'invalid_schema' in record['reasons']:
            continue
        if record['id'] in seen:
            record['reasons'].append('duplicate_row')
            record['status'] = 'reject'
            union(i, seen[record['id']])
        else:
            seen[record['id']] = i
        prompt = '\n'.join(row[k].strip() for k in ('instruction', 'input'))
        prompts[prompt].append(i)
        answers[row['output'].strip()].append(i)
        tokens = sets[i]
        # Jaccard >= t implies intersection >= ceil(t * len(each set)).
        # With one global ordering, such sets must share a prefix token.
        ordered = sorted(tokens, key=lambda token: (frequency[token], token))
        prefix = ordered[:len(tokens) - math.ceil(threshold * len(tokens)) + 1]
        candidates = set()
        for token in prefix:
            candidates.update(postings[token])
        for j in sorted(candidates):
            other = sets[j]
            if min(len(tokens), len(other)) < threshold * max(len(tokens), len(other)):
                continue
            if len(tokens & other) / len(tokens | other) >= threshold:
                union(i, j)
        for token in prefix:
            postings[token].add(i)
    for indices in prompts.values():
        if len({rows[i]['output'].strip() for i in indices}) > 1:
            for i in indices:
                records[i]['reasons'].append('conflicting_answers')
    for indices in answers.values():
        if len(indices) > 1:
            for i in indices:
                records[i]['reasons'].append('shared_answer')
                union(indices[0], i)
    sizes = defaultdict(int)
    for i in range(len(rows)):
        sizes[root(i)] += 1
    for i, record in enumerate(records):
        record['group'] = root(i)
        if sizes[root(i)] > 1:
            record['reasons'].append('similarity_group_requires_review')
        record['reasons'] = sorted(set(record['reasons']))
        if record['status'] == 'accepted' and record['reasons']:
            record['status'] = 'review'
