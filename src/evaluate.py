import json
import time
import gc
import torch
from unsloth import FastLanguageModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FINETUNED_MODEL = "models/adapter"

TEST_FILE = "data/test/test.json"

BASE_RESULTS = "results/baseline.json"
FINETUNED_RESULTS = "results/finetuned.json"
COMPARISON_RESULTS = "results/comparison.json"

MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4


def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["test"]


def create_prompt(example):
    prompt = example["instruction"]

    if example["input"].strip():
        prompt += "\n\nInput:\n" + example["input"]

    return prompt


def load_model(model_name, label):
    print(f"\nLoading {label} model...", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    print(f"{label} model loaded.", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    return model, tokenizer


def generate_batch(model, tokenizer, questions):
    messages = [
        [{"role": "user", "content": question}]
        for question in questions
    ]

    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            max_length=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []

    input_lengths = inputs["attention_mask"].sum(dim=1)

    for i in range(len(questions)):
        generated_tokens = outputs[i][input_lengths[i]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        responses.append(response)

    return responses


def evaluate_model(model, tokenizer, test_data, label):
    results = []

    total = len(test_data)

    for start in range(0, total, BATCH_SIZE):

        batch = test_data[start:start + BATCH_SIZE]

        questions = [
            create_prompt(example)
            for example in batch
        ]

        batch_number = start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"{label}: batch {batch_number}/{total_batches} "
            f"(examples {start + 1}-{start + len(batch)})",
            flush=True
        )

        start_time = time.time()

        responses = generate_batch(
            model,
            tokenizer,
            questions
        )

        elapsed = time.time() - start_time

        for example, response in zip(batch, responses):

            results.append({
                "instruction": example["instruction"],
                "input": example["input"],
                "reference": example["output"],
                "response": response,
                "generation_time_seconds": round(
                    elapsed / len(batch),
                    3
                )
            })

    return results


def save_results(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved: {filename}", flush=True)


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():

    print("Loading test dataset...", flush=True)

    test_data = load_test_data()

    print(
        f"Test examples: {len(test_data)}",
        flush=True
    )

    # =============================
    # BASE MODEL
    # =============================

    model, tokenizer = load_model(
        BASE_MODEL,
        "BASE"
    )

    baseline_results = evaluate_model(
        model,
        tokenizer,
        test_data,
        "BASE"
    )

    save_results(
        baseline_results,
        BASE_RESULTS
    )

    unload_model(model)

    # =============================
    # FINE-TUNED MODEL
    # =============================

    model, tokenizer = load_model(
        FINETUNED_MODEL,
        "FINE-TUNED"
    )

    finetuned_results = evaluate_model(
        model,
        tokenizer,
        test_data,
        "FINE-TUNED"
    )

    save_results(
        finetuned_results,
        FINETUNED_RESULTS
    )

    unload_model(model)

    # =============================
    # COMPARISON
    # =============================

    comparison = []

    for i in range(len(test_data)):

        comparison.append({
            "instruction": test_data[i]["instruction"],
            "input": test_data[i]["input"],
            "reference": test_data[i]["output"],
            "base_response": baseline_results[i]["response"],
            "finetuned_response": finetuned_results[i]["response"],
            "base_generation_time_seconds":
                baseline_results[i][
                    "generation_time_seconds"
                ],
            "finetuned_generation_time_seconds":
                finetuned_results[i][
                    "generation_time_seconds"
                ]
        })

    save_results(
        comparison,
        COMPARISON_RESULTS
    )

    print("\nEvaluation complete!", flush=True)
    print("Results saved in results/", flush=True)


if __name__ == "__main__":
    main()