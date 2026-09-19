from datasets import load_dataset
from collections import Counter
import re

DATASET_NAME = "flwrlabs/code-alpaca-20k"


def load_data():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nDataset:")
    print(dataset)

    return dataset


def analyze_data(dataset):
    data = dataset["train"]

    print("\n========== BASIC STATISTICS ==========")
    print("Total examples:", len(data))
    print("Columns:", data.column_names)

    # --------------------------------------------------
    # Empty fields
    # --------------------------------------------------

    empty_inputs = sum(
        1 for example in data
        if not example["input"].strip()
    )

    empty_outputs = sum(
        1 for example in data
        if not example["output"].strip()
    )

    print("\nEmpty inputs:", empty_inputs)
    print("Non-empty inputs:", len(data) - empty_inputs)

    print("Empty outputs:", empty_outputs)
    print("Non-empty outputs:", len(data) - empty_outputs)

    # --------------------------------------------------
    # Length statistics
    # --------------------------------------------------

    instruction_lengths = [
        len(example["instruction"])
        for example in data
    ]

    input_lengths = [
        len(example["input"])
        for example in data
    ]

    output_lengths = [
        len(example["output"])
        for example in data
    ]

    total_lengths = [
        len(example["instruction"])
        + len(example["input"])
        + len(example["output"])
        for example in data
    ]

    print("\n========== LENGTH STATISTICS ==========")

    print("\nInstruction length:")
    print("  Min:", min(instruction_lengths))
    print("  Max:", max(instruction_lengths))
    print("  Average:", round(sum(instruction_lengths) / len(instruction_lengths), 2))

    print("\nInput length:")
    print("  Min:", min(input_lengths))
    print("  Max:", max(input_lengths))
    print("  Average:", round(sum(input_lengths) / len(input_lengths), 2))

    print("\nOutput length:")
    print("  Min:", min(output_lengths))
    print("  Max:", max(output_lengths))
    print("  Average:", round(sum(output_lengths) / len(output_lengths), 2))

    print("\nTotal example length:")
    print("  Min:", min(total_lengths))
    print("  Max:", max(total_lengths))
    print("  Average:", round(sum(total_lengths) / len(total_lengths), 2))

    # --------------------------------------------------
    # Length buckets
    # --------------------------------------------------

    print("\n========== LENGTH DISTRIBUTION ==========")

    buckets = [
        ("<= 500 chars", 0, 500),
        ("501-1000 chars", 501, 1000),
        ("1001-1500 chars", 1001, 1500),
        ("1501-2000 chars", 1501, 2000),
        ("2001-3000 chars", 2001, 3000),
        ("> 3000 chars", 3001, float("inf")),
    ]

    for name, lower, upper in buckets:
        count = sum(
            1 for length in total_lengths
            if lower <= length <= upper
        )

        print(f"{name}: {count}")

    # --------------------------------------------------
    # Programming relevance
    # --------------------------------------------------

    programming_keywords = [
        "python",
        "program",
        "code",
        "function",
        "class",
        "object",
        "variable",
        "list",
        "tuple",
        "dictionary",
        "dict",
        "set",
        "string",
        "array",
        "loop",
        "for loop",
        "while loop",
        "recursion",
        "algorithm",
        "sorting",
        "searching",
        "stack",
        "queue",
        "linked list",
        "tree",
        "graph",
        "exception",
        "debug",
        "error",
        "api",
        "json",
        "file",
        "database",
        "sql",
        "complexity",
        "programming",
        "javascript",
        "java",
        "c++",
        "c#",
        "html",
        "css",
        "git",
    ]

    def relevance_score(example):
        text = (
            example["instruction"]
            + " "
            + example["input"]
            + " "
            + example["output"]
        ).lower()

        score = 0

        for keyword in programming_keywords:
            if keyword in text:
                score += 1

        return score

    scores = [relevance_score(example) for example in data]

    print("\n========== PROGRAMMING RELEVANCE ==========")

    print("Examples with score 0:", sum(score == 0 for score in scores))
    print("Examples with score 1:", sum(score == 1 for score in scores))
    print("Examples with score 2:", sum(score == 2 for score in scores))
    print("Examples with score 3-4:", sum(3 <= score <= 4 for score in scores))
    print("Examples with score 5+:", sum(score >= 5 for score in scores))

    # --------------------------------------------------
    # Unique instructions
    # --------------------------------------------------

    instructions = [
        example["instruction"].strip().lower()
        for example in data
    ]

    unique_instructions = len(set(instructions))

    print("\n========== DUPLICATES ==========")
    print("Unique instructions:", unique_instructions)
    print("Duplicate instructions:", len(instructions) - unique_instructions)

    # --------------------------------------------------
    # Samples with low relevance
    # --------------------------------------------------

    print("\n========== LOW-RELEVANCE SAMPLES ==========")

    low_relevance = [
        (i, scores[i], data[i])
        for i in range(len(data))
        if scores[i] == 0
    ]

    for i, score, example in low_relevance[:10]:
        print(f"\nExample index: {i}")
        print("Instruction:", example["instruction"])
        print("Input:", example["input"])
        print("Output:", example["output"])

    # --------------------------------------------------
    # Long examples
    # --------------------------------------------------

    print("\n========== LONG EXAMPLES ==========")

    long_examples = [
        (i, total_lengths[i], data[i])
        for i in range(len(data))
        if total_lengths[i] > 2000
    ]

    print("Examples > 2000 chars:", len(long_examples))

    for i, length, example in long_examples[:5]:
        print(f"\nExample index: {i}")
        print("Total length:", length)
        print("Instruction:", example["instruction"])
        print("Output preview:", example["output"][:500])


if __name__ == "__main__":
    import sys
    if '--audit' in sys.argv:
        sys.argv.remove('--audit')
        from curation_report import main
        main()
    else:
        dataset = load_data()
        analyze_data(dataset)
