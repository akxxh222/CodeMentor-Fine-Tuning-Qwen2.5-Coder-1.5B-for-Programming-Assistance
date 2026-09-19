"""Score structurally clean examples with the cached local Qwen model."""
import argparse
import json
from pathlib import Path

import torch

try:
    from .selection import build_judge_prompt, candidate_score
except ImportError:  # Direct script execution.
    from selection import build_judge_prompt, candidate_score


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_local_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("--batch-size and --max-length must be positive")
    if args.output.exists():
        parser.error("output already exists")
    records = [record for record in read_jsonl(args.input) if candidate_score(record) is not None]
    records.sort(key=lambda record: record["id"])

    model, tokenizer = load_local_model(args.model.resolve())
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pass_ids = tokenizer.encode(" PASS", add_special_tokens=False)
    fail_ids = tokenizer.encode(" FAIL", add_special_tokens=False)
    if len(pass_ids) != 1 or len(fail_ids) != 1:
        raise RuntimeError("PASS and FAIL must each be one tokenizer token")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": build_judge_prompt(record)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for record in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            if encoded["input_ids"].shape[1] > args.max_length:
                raise RuntimeError(f"judge prompt exceeds {args.max_length} tokens")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits[:, -1, [pass_ids[0], fail_ids[0]]].float()
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for record, (pass_probability, fail_probability) in zip(batch, probabilities):
                result = {
                    "id": record["id"],
                    "source_index": record["source_index"],
                    "verdict": "PASS" if pass_probability >= fail_probability else "FAIL",
                    "pass_probability": round(pass_probability, 8),
                    "fail_probability": round(fail_probability, 8),
                    "method": "qwen_next_token_pairwise",
                }
                handle.write(json.dumps(result) + "\n")
            handle.flush()
            if start == 0 or (start + len(batch)) % 500 == 0:
                print(f"Reviewed {start + len(batch)}/{len(records)}", flush=True)


if __name__ == "__main__":
    main()
