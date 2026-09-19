import unittest

from src.verified_curation import (
    build_verified_dataset,
    prompt_fingerprint,
    validate_example,
)


def example(instruction="Write a Python function to double n.", output=None, input=""):
    return {
        "instruction": instruction,
        "input": input,
        "output": output or "def double(n):\n    return n * 2",
    }


class VerifiedCurationTests(unittest.TestCase):
    def test_python_requires_valid_complete_syntax(self):
        valid = validate_example(example(), available_languages={"python"})
        invalid = validate_example(
            example(output="def double(n):\n    return (") ,
            available_languages={"python"},
        )

        self.assertTrue(valid["accepted"])
        self.assertFalse(invalid["accepted"])
        self.assertIn("syntax_failed", invalid["reasons"])

    def test_unavailable_language_is_rejected(self):
        result = validate_example(
            example("Write a Java program.", "public class Main {}"),
            available_languages={"python"},
        )

        self.assertFalse(result["accepted"])
        self.assertIn("validator_unavailable", result["reasons"])

    def test_explicit_empty_validator_set_disables_all_languages(self):
        result = validate_example(example(), available_languages=set())

        self.assertFalse(result["accepted"])
        self.assertIn("validator_unavailable", result["reasons"])

    def test_incomplete_or_placeholder_output_is_rejected(self):
        incomplete = validate_example(
            example(output="def double(n):\n    return n * 2\n```"),
            available_languages={"python"},
        )
        placeholder = validate_example(
            example(output="def double(n):\n    # TODO: implement\n    pass"),
            available_languages={"python"},
        )

        self.assertIn("incomplete_output", incomplete["reasons"])
        self.assertIn("placeholder", placeholder["reasons"])

    def test_prompt_fingerprint_ignores_output(self):
        first = example(output="def double(n): return n * 2")
        second = example(output="def double(n): return n + n")

        self.assertEqual(prompt_fingerprint(first), prompt_fingerprint(second))

    def test_selection_excludes_validation_and_test_prompts(self):
        leaked = example("Write a Python function to add one.", "def add_one(n): return n + 1")
        safe = example("Write a Python function to double n.")

        selected, report = build_verified_dataset(
            [leaked, safe],
            [dict(leaked)],
            [],
            limit=10,
            seed=7,
            available_languages={"python"},
        )

        self.assertEqual(selected, [safe])
        self.assertEqual(report["rejection_reasons"]["held_out_prompt"], 1)

    def test_selection_is_deterministic_and_respects_limit(self):
        rows = [
            example(f"Write a Python function number {index}.", f"def value_{index}():\n    return {index}")
            for index in range(8)
        ]

        first, _ = build_verified_dataset(
            rows, [], [], limit=3, seed=42, available_languages={"python"}
        )
        second, _ = build_verified_dataset(
            list(reversed(rows)), [], [], limit=3, seed=42, available_languages={"python"}
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
