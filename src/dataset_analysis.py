from datasets import load_dataset


DATASET_NAME = "flwrlabs/code-alpaca-20k"


def load_data():
    print("Loading dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset:")
    print(dataset)

    return dataset


def analyze_data(dataset):
    data = dataset["train"]

    print("\n========== DATASET ANALYSIS ==========")

    # Basic information
    print("Total examples:", len(data))
    print("Columns:", data.column_names)

    # Empty inputs
    empty_inputs = sum(
        1 for example in data
        if not example["input"].strip()
    )

    print("\nExamples with empty input:", empty_inputs)
    print(
        "Examples with non-empty input:",
        len(data) - empty_inputs
    )

    # Instruction lengths
    instruction_lengths = [
        len(example["instruction"])
        for example in data
    ]

    # Output lengths
    output_lengths = [
        len(example["output"])
        for example in data
    ]

    print("\nInstruction length:")
    print("  Minimum:", min(instruction_lengths))
    print("  Maximum:", max(instruction_lengths))
    print(
        "  Average:",
        round(sum(instruction_lengths) / len(instruction_lengths), 2)
    )

    print("\nOutput length:")
    print("  Minimum:", min(output_lengths))
    print("  Maximum:", max(output_lengths))
    print(
        "  Average:",
        round(sum(output_lengths) / len(output_lengths), 2)
    )

    # Duplicate instructions
    instructions = [
        example["instruction"].strip().lower()
        for example in data
    ]

    unique_instructions = len(set(instructions))
    duplicate_count = len(instructions) - unique_instructions

    print("\nUnique instructions:", unique_instructions)
    print("Duplicate instructions:", duplicate_count)

    # Show examples
    print("\n========== SAMPLE EXAMPLES ==========")

    for i in range(5):
        print(f"\nExample {i + 1}:")
        print("Instruction:", data[i]["instruction"])
        print("Input:", data[i]["input"])
        print("Output:", data[i]["output"])


if __name__ == "__main__":
    dataset = load_data()
    analyze_data(dataset)