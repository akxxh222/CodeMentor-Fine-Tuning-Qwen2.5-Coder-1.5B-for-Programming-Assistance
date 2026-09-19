import argparse
import torch
from unsloth import FastLanguageModel


BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
FINETUNED_MODEL = "models/adapter"

MAX_SEQ_LENGTH = 512


def load_model(model_type):
    if model_type == "base":
        model_name = BASE_MODEL
        print("Loading BASE model...")
    else:
        model_name = FINETUNED_MODEL
        print("Loading FINE-TUNED CodeMentor model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    print("Model loaded successfully!")
    print("GPU:", torch.cuda.get_device_name(0))

    return model, tokenizer


def generate_response(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            max_length=None,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    return response


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["base", "finetuned"],
        default="finetuned",
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    question = "Explain what a Python dictionary is to a beginner."

    response = generate_response(
        model,
        tokenizer,
        question,
    )

    print("\n--- RESPONSE ---")
    print(response)