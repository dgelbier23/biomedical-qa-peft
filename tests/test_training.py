"""CPU tests of training with the tiny Llama and tokenizer from conftest.py."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import HEADS, HIDDEN, KV_HEADS, LAYERS

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.evaluation.config import BASELINE_PROMPT
from biomedical_qa_peft.evaluation.inference import build_prompt
from biomedical_qa_peft.training.config import LoraSettings, PrefixConfig, TrainingConfig
from biomedical_qa_peft.training.lora import add_lora
from biomedical_qa_peft.training.prefix_tuning import add_prefix
from biomedical_qa_peft.training.trainer import (
    build_trainer,
    build_training_example,
    make_dataset,
    plot_training_curves,
    run_training,
    train,
)

FORMATTED = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: q?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\nSimplified Answer: no<|eot_id|><|pad|>"
)


def test_training_prompt_is_the_test_prompt_followed_by_the_answer() -> None:
    prompt, answer = build_training_example(FORMATTED, BASELINE_PROMPT)

    assert prompt == build_prompt(FORMATTED, BASELINE_PROMPT)[0]
    assert answer == "\nSimplified Answer: no<|eot_id|>"


def test_prefix_trains_the_projection_mlp_but_saves_only_its_output(tiny_llama) -> None:
    n = PrefixConfig().num_virtual_tokens
    kv_dim = HIDDEN // HEADS * KV_HEADS  # PEFT sizes the prefix to the key/value heads
    prefix_values = n * LAYERS * 2 * kv_dim

    projected = add_prefix(tiny_llama(), PrefixConfig())
    trainable = sum(p.numel() for p in projected.parameters() if p.requires_grad)
    # Embedding (n x kv_dim) -> Linear(kv_dim, hidden) -> tanh -> Linear(hidden, layers*2*kv_dim).
    # For Llama 3.2 3B this is 179,387,392 trainable values, not the 1,146,880 saved.
    assert trainable == n * kv_dim + (kv_dim * HIDDEN + HIDDEN) + (HIDDEN * prefix_values // n + prefix_values // n)
    assert projected.get_prompt_embedding_to_save("default").numel() == prefix_values
    assert projected.config.use_cache is False

    direct = add_prefix(tiny_llama(), PrefixConfig(prefix_projection=False))
    assert sum(p.numel() for p in direct.parameters() if p.requires_grad) == prefix_values


def test_lora_trains_query_and_value_adapters(tiny_llama) -> None:
    r, kv_dim = LoraSettings().r, HIDDEN // HEADS * KV_HEADS
    model = add_lora(tiny_llama(), LoraSettings())

    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    # A (r x in) + B (out x r) per adapted projection; q: hidden -> hidden, v: hidden -> kv_dim.
    # For Llama 3.2 3B this is 2,293,760.
    assert sum(trainable.values()) == LAYERS * (r * 2 * HIDDEN + r * (HIDDEN + kv_dim))
    assert all("lora_" in name and ("q_proj" in name or "v_proj" in name) for name in trainable)
    assert model.config.use_cache is False


def test_loss_covers_only_the_answer(tiny_llama, tokenizer, tmp_path: Path) -> None:
    tokenizer.padding_side = "right"
    dataset = make_dataset([("is the drug", "safe yes"), ("yes", "no")], tokenizer)
    trainer = build_trainer(tiny_llama(), tokenizer, dataset, dataset,
                            replace(TrainingConfig(), fp16=False), tmp_path)

    batch = trainer.data_collator([dataset[0], dataset[1]])

    assert batch["input_ids"].tolist() == [[3, 4, 5, 6, 7], [7, 8, 0, 0, 0]]
    # Prompt tokens and padding are -100; only answer tokens count towards the loss.
    assert batch["labels"].tolist() == [[-100, -100, -100, 6, 7], [-100, 8, -100, -100, -100]]


CONFIG = replace(
    TrainingConfig(), fp16=False, num_train_epochs=1, dataloader_num_workers=0,
    eval_steps=1, save_steps=1, logging_steps=1, warmup_steps=0,
)
EXAMPLES = [("is the drug safe", "yes"), ("it was tested in mice", "no"),
            ("is it safe", "yes"), ("the answer", "no")] * 2


def test_projected_prefix_cannot_reload_its_best_checkpoint_or_resume(
    tiny_llama, tokenizer, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset = make_dataset(EXAMPLES, tokenizer)
    tokenizer.padding_side = "left"
    trainer = build_trainer(add_prefix(tiny_llama(), PrefixConfig()), tokenizer,
                            dataset, dataset, CONFIG, tmp_path)

    state = train(trainer, tmp_path, resume=False)

    assert state.global_step == 2  # 8 rows / (batch 2 x accumulation 2)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-1", "checkpoint-2"]
    assert "Could not reload the best checkpoint" in capsys.readouterr().out
    assert tokenizer.padding_side == "left"  # restored after training
    with pytest.raises(FileExistsError):
        train(trainer, tmp_path, resume=False)

    train_fig, eval_fig = plot_training_curves(state.log_history)
    assert train_fig.axes[0].get_title() == "Training Loss by Steps"
    assert eval_fig.axes[0].get_title() == "Evaluation Loss by Steps"


def test_run_training_saves_the_best_checkpoint_curves_and_summary(
    tiny_llama, tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    # pyplot would pick a GUI backend, which crashes where Tk is not installed.
    monkeypatch.setattr(plt, "figure", lambda *a, **k: pytest.fail("pyplot used for training curves"))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = FORMATTED.replace("q?", "is the drug safe")
    for split in ("train", "validation"):
        pd.DataFrame({"formatted_text": [row] * 8}).to_json(
            data_dir / f"{split}.jsonl", orient="records", lines=True)
    run_dir = tmp_path / "run"

    # 4 steps keeping 2 checkpoints, so older ones are deleted along the way.
    summary = run_training(run_dir, add_prefix(tiny_llama(), PrefixConfig()), tokenizer,
                           data_dir, BASELINE_PROMPT, replace(CONFIG, num_train_epochs=2),
                           resume=False)

    best = run_dir / "checkpoints" / f"checkpoint-{summary['best_step']}"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert (run_dir / "adapter" / name).read_bytes() == (best / name).read_bytes()
    assert summary["final_step"] == 4 and summary["trainable_parameters"] > 0
    assert (run_dir / "training_loss.png").exists() and (run_dir / "eval_loss.png").exists()
    record = json.loads((run_dir / "training.json").read_text())
    assert record["training_config"]["learning_rate"] == 2e-5 and record["log_history"]


def test_lora_reloads_best_checkpoint_and_resumes(
    tiny_llama, tokenizer, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset = make_dataset(EXAMPLES, tokenizer)
    states = []
    for _ in range(2):
        model = add_lora(tiny_llama(), LoraSettings())
        states.append(train(build_trainer(model, tokenizer, dataset, dataset, CONFIG, tmp_path), tmp_path))

    out = capsys.readouterr().out
    assert "Could not reload" not in out and "Resuming from" in out
    # The second run resumed at the final checkpoint, so it trained no further steps.
    assert states[1].global_step == 2
    assert states[1].log_history[:-1] == states[0].log_history[:-1]  # per-step logs restored
    assert states[1].log_history[-1]["train_loss"] == 0.0  # summary of a run with no new steps
