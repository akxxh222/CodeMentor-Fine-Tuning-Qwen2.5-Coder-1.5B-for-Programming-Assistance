---
base_model: unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit
library_name: peft
model_name: codementor-qwen2.5-coder-v2
tags:
- base_model:adapter:unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
pipeline_tag: text-generation
---

# CodeMentor Qwen2.5-Coder V2 Adapter

This is the active CodeMentor V2 LoRA adapter for [unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit](https://huggingface.co/unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit). It was trained locally with Unsloth and TRL on 1,000 syntax/compile-screened programming examples, using 100 validation examples and completion-only loss.

The training data is not semantically certified. See the repository-level `README.md`, `CURATION.md`, and `results/finetuned_vs_baseline.md` for evaluation scope and limitations.

## Quick start

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/adapter",
    max_seq_length=512,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

## Training procedure

 
This adapter was trained with QLoRA supervised fine-tuning for 2 epochs, a learning rate of `5e-5`, LoRA rank/alpha of 16/16, and assistant-completion-only loss. Its best recorded validation loss was `0.3921625912`.

### Framework versions

- PEFT 0.21.0
- TRL: 0.24.0
- Transformers: 5.5.0
- Pytorch: 2.11.0+cu128
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```
