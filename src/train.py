import argparse
import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

MAX_SEQ_LENGTH = 512

TRAIN_FILE = Path("data/processed/verified_dataset.json")
OUTPUT_DIR = Path("models/adapter_candidate")


def load_data(train_file=TRAIN_FILE):
    """Load train and validation records from the configured dataset file."""

    from datasets import Dataset

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = Dataset.from_list(data["train"])
    validation_data = Dataset.from_list(data["validation"])

    print("Training examples:", len(train_data))
    print("Validation examples:", len(validation_data))

    return train_data, validation_data


def format_example(example):
    """Convert CodeAlpaca example into Qwen chat format."""

    user_message = example["instruction"]

    if example["input"].strip():
        user_message += "\n\nInput:\n" + example["input"]

    return {
        "prompt": [
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["output"],
            },
        ]
    }


def prepare_data(train_data, validation_data):

    train_data = train_data.map(
        format_example,
        remove_columns=train_data.column_names,
    )

    validation_data = validation_data.map(
        format_example,
        remove_columns=validation_data.column_names,
    )

    return train_data, validation_data


def load_model():

    from unsloth import FastLanguageModel

    print("\nLoading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    print("Model loaded!")

    return model, tokenizer


def add_lora(model):

    from unsloth import FastLanguageModel

    print("\nAdding LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,

        r=16,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        lora_alpha=16,
        lora_dropout=0,
        bias="none",

        use_gradient_checkpointing="unsloth",

        random_state=42,
    )

    return model


def build_training_config(output_dir):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=True,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        report_to="none",
        seed=42,
    )


def train(model, tokenizer, train_data, validation_data, output_dir=OUTPUT_DIR):

    from trl import SFTTrainer

    print("\nStarting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,

        train_dataset=train_data,
        eval_dataset=validation_data,

        args=build_training_config(output_dir),
    )

    trainer.train()

    return trainer


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("========== CodeMentor QLoRA Training ==========")

    train_data, validation_data = load_data(args.train_file)

    train_data, validation_data = prepare_data(
        train_data,
        validation_data,
    )

    print("\nExample after formatting:")
    print(train_data[0])

    model, tokenizer = load_model()

    model = add_lora(model)

    train(
        model,
        tokenizer,
        train_data,
        validation_data,
        args.output_dir,
    )

    print("\nSaving adapter...")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print("Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
