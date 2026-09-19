from pathlib import Path

from src import train


def test_format_example_separates_prompt_from_completion():
    formatted = train.format_example(
        {
            "instruction": "Write a function.",
            "input": "n = 2",
            "output": "def answer(n): return n",
        }
    )

    assert formatted == {
        "prompt": [
            {
                "role": "user",
                "content": "Write a function.\n\nInput:\nn = 2",
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "def answer(n): return n",
            }
        ],
    }


def test_candidate_training_defaults_do_not_overwrite_current_adapter():
    assert train.TRAIN_FILE == Path("data/processed/verified_dataset.json")
    assert train.OUTPUT_DIR == Path("models/adapter_candidate")


def test_training_config_uses_completion_only_loss_and_lower_learning_rate():
    config = train.build_training_config(Path("models/test-candidate"))

    assert Path(config.output_dir) == Path("models/test-candidate")
    assert config.completion_only_loss is True
    assert config.learning_rate == 5e-5
    assert config.load_best_model_at_end is True
    assert config.metric_for_best_model == "eval_loss"
