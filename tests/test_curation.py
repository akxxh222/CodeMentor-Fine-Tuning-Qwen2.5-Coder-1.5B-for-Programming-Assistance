import unittest
import random
from src.curation import inspect_row, audit_duplicates, fingerprint


def row(instruction='Write a Python function to double n.', output='def double(n):\n    return n * 2', input=''):
    return dict(instruction=instruction, input=input, output=output)


class CurationTests(unittest.TestCase):
    def test_valid_syntax_does_not_establish_correctness(self):
        result = inspect_row(row(output='def double(n):\n    return n * 3'), 40)
        self.assertEqual(result['status'], 'review')
        self.assertEqual(result['syntax'], 'python_parse_pass')

    def test_empty_input_allowed_but_empty_output_rejected(self):
        self.assertEqual(inspect_row(row(), 40)['status'], 'review')
        self.assertIn('empty_output', inspect_row(row(output='  '), 40)['reasons'])

    def test_bad_schema_is_rejected_without_crashing(self):
        self.assertEqual(inspect_row({'instruction': None}, None)['status'], 'reject')

    def test_token_limit_and_encoding(self):
        self.assertIn('token_limit', inspect_row(row(), 513)['reasons'])
        self.assertNotIn('token_limit', inspect_row(row(), 512)['reasons'])
        self.assertIn('replacement_character', inspect_row(row(output='bad\ufffd'), 40)['reasons'])

    def test_python_fragment_is_not_called_incorrect(self):
        result = inspect_row(row(output='    return n * 2'), 40)
        self.assertEqual(result['status'], 'review')
        self.assertIn('python_syntax_or_fragment', result['reasons'])

    def test_multilanguage_recognition(self):
        for language in ['Java', 'JavaScript', 'C++', 'C#', 'SQL', 'Ruby', 'Go', 'Rust']:
            self.assertEqual(inspect_row(row(instruction=f'Write {language} code.'), 40)['language'], language.lower())

    def test_incorrect_largest_prime_factor(self):
        example = row('What is the largest prime factor of the number 885?', 'The largest prime factor of 885 is 5.')
        self.assertIn('incorrect_prime_factor', inspect_row(example, 40)['reasons'])

    def test_factual_rule_does_not_apply_to_code_generation(self):
        example = row('Write code to find the largest prime factor of 885.', 'def factor(n):\n    return 5')
        self.assertNotIn('incorrect_prime_factor', inspect_row(example, 40)['reasons'])

    def test_exact_prime_answer_can_be_verified(self):
        example = row('What is the largest prime factor of the number 885?', 'The largest prime factor of 885 is 59.')
        self.assertEqual(inspect_row(example, 40)['status'], 'accepted')

    def test_exact_binary_answer_can_be_verified_but_overlimit_cannot(self):
        example = row('Convert the following binary number to a decimal number.', '19', 'Binary Number: 10011')
        self.assertEqual(inspect_row(example, 40)['status'], 'accepted')
        self.assertEqual(inspect_row(example, 513)['status'], 'reject')

    def test_conflict_revokes_semantic_acceptance(self):
        good = row('What is the largest prime factor of the number 885?', 'The largest prime factor of 885 is 59.')
        bad = dict(good, output='The largest prime factor of 885 is 5.')
        records = [inspect_row(r, 40) for r in (good, bad)]
        audit_duplicates([good, bad], records)
        self.assertEqual(records[0]['status'], 'review')

    def test_conflicting_answers_flag_both_rows(self):
        rows = [row(), row(output='def double(n):\n    return 3 * n')]
        records = [inspect_row(r, 40) for r in rows]
        audit_duplicates(rows, records)
        self.assertTrue(all('conflicting_answers' in r['reasons'] for r in records))

    def test_duplicate_and_near_duplicate_groups(self):
        rows = [row(), row(), row(instruction='Write a Python function to double n!')]
        records = [inspect_row(r, 40) for r in rows]
        audit_duplicates(rows, records)
        self.assertIn('duplicate_row', records[1]['reasons'])
        self.assertEqual(records[0]['group'], records[2]['group'])

    def test_content_id_preserves_code_case(self):
        self.assertNotEqual(fingerprint(row(output='x = 1')), fingerprint(row(output='X = 1')))

    def test_python_compile_warning_requires_review(self):
        result = inspect_row(row(output='x = 1\nif x is 1:\n    print(x)'), 40)
        self.assertIn('python_compile_warning', result['reasons'])

    def test_binary_answer_with_wrong_explanation_is_flagged(self):
        example = row('Convert the following binary number to a decimal number.',
                      '19 (10011 = 2^4 + 2^3 + 2^1 = 16 + 8 + 1 = 25)',
                      'Binary Number: 10011')
        self.assertIn('binary_explanation_requires_review', inspect_row(example, 40)['reasons'])

    def test_unclear_programming_relevance_is_flagged(self):
        result = inspect_row(row('Name the capital of France.', 'Paris.'), 30)
        self.assertIn('programming_relevance_uncertain', result['reasons'])

    def test_python_unresolved_global_is_flagged_but_input_variable_is_allowed(self):
        missing = row('Find phrase frequency in Python.', 'def frequency(text):\n    return nltk.FreqDist(text.split())')
        self.assertIn('python_unresolved_name', inspect_row(missing, 40)['reasons'])
        supplied = row('Convert data to JSON in Python.', 'import json\nresult = json.dumps(data)', 'data = {"x": 1}')
        self.assertNotIn('python_unresolved_name', inspect_row(supplied, 40)['reasons'])

    def test_lcs_answer_with_hidden_required_length_arguments_is_flagged(self):
        example = row('Using python, implement a method to find the longest common subsequence in two strings',
                      'def lcs(str1, str2, n, m):\n    return 0',
                      'str1 = "AGGTAB"\nstr2 = "GXTXAYB"')
        self.assertIn('lcs_interface_mismatch', inspect_row(example, 80)['reasons'])

    def test_incorrect_java_debug_explanation_is_flagged(self):
        example = row('Debug this Java code and explain what was wrong.',
                      'The operator being used was "b" instead of "+" and returned the wrong value.',
                      'public static int add(int a, int b){ int c = a b; return c; }')
        self.assertIn('incorrect_debug_explanation', inspect_row(example, 80)['reasons'])

    def test_similarity_index_matches_bruteforce_components(self):
        rng = random.Random(17)
        sets = [set(rng.sample(list('abcdefghijklmnopqrst'), rng.randint(8, 20))) for _ in range(80)]
        rows = [row(instruction=' '.join(sorted(s)), output=f'answer {i}') for i, s in enumerate(sets)]
        records = [inspect_row(r, 40) for r in rows]
        audit_duplicates(rows, records)
        groups = [{i} for i in range(len(sets))]
        for i in range(len(sets)):
            for j in range(i):
                if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= 0.85:
                    merged = groups[i] | groups[j]
                    for member in merged:
                        groups[member] = merged
        for i in range(len(sets)):
            actual = {j for j in range(len(sets)) if records[i]['group'] == records[j]['group']}
            self.assertEqual(actual, groups[i])


if __name__ == '__main__':
    unittest.main()
