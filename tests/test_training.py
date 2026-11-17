"""CPU tests with a tiny random Llama and an in-memory tokenizer: nothing is downloaded."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.evaluation.config import BASELINE_PROMPT
from biomedical_qa_peft.training.config import PrefixConfig, TrainingConfig
from biomedical_qa_peft.training.prefix_tuning import add_prefix
from biomedical_qa_peft.training.trainer import (
    build_trainer,
    build_training_text,
    make_dataset,
    plot_training_curves,
    run_training,
    train,
)

WORDS = "is the drug safe yes no answer it was tested in mice".split()
LAYERS, HIDDEN, HEADS, KV_HEADS = 2, 64, 4, 2


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>"
    )


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=len(WORDS) + 3, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        max_position_embeddings=128, pad_token_id=0,
    ))


def test_build_training_text_swaps_system_prompt_and_keeps_answer() -> None:
    formatted = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: q?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\nSimplified Answer: no<|eot_id|><|pad|>"
    )

    text = build_training_text(formatted, BASELINE_PROMPT)

    # <|begin_of_text|> stays (the tokenizer adds a second one), unlike test prompts.
    assert text == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|> "
        f"{BASELINE_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: q?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\nSimplified Answer: no<|eot_id|>"
    )


def test_prefix_trains_the_projection_mlp_but_saves_only_its_output() -> None:
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


def test_collator_masks_only_padding_in_labels(tokenizer, tmp_path: Path) -> None:
    tokenizer.padding_side = "right"
    dataset = make_dataset(["is the drug safe", "yes"], tokenizer)
    trainer = build_trainer(tiny_llama(), tokenizer, dataset, dataset,
                            replace(TrainingConfig(), fp16=False), tmp_path)

    batch = trainer.data_collator([dataset[0], dataset[1]])

    assert batch["input_ids"].tolist() == [[3, 4, 5, 6], [7, 0, 0, 0]]
    assert batch["labels"].tolist() == [[3, 4, 5, 6], [7, -100, -100, -100]]


CONFIG = replace(
    TrainingConfig(), fp16=False, num_train_epochs=1, dataloader_num_workers=0,
    eval_steps=1, save_steps=1, logging_steps=1, warmup_steps=0,
)
TEXTS = ["is the drug safe yes", "it was tested in mice no", "is it safe yes", "the answer no"] * 2


def test_projected_prefix_trains_keeps_final_weights_and_refuses_to_resume(
    tokenizer, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset = make_dataset(TEXTS, tokenizer)
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


def test_run_training_saves_adapter_curves_and_summary(
    tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    # pyplot would pick a GUI backend, which crashes where Tk is not installed.
    monkeypatch.setattr(plt, "figure", lambda *a, **k: pytest.fail("pyplot used for training curves"))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "validation"):
        pd.DataFrame({"formatted_text": TEXTS}).to_json(data_dir / f"{split}.jsonl", orient="records", lines=True)
    run_dir = tmp_path / "run"

    summary = run_training(run_dir, add_prefix(tiny_llama(), PrefixConfig()), tokenizer,
                           data_dir, BASELINE_PROMPT, CONFIG, resume=False)

    assert summary["final_step"] == 2 and summary["trainable_parameters"] > 0
    assert (run_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (run_dir / "training_loss.png").exists() and (run_dir / "eval_loss.png").exists()
    record = json.loads((run_dir / "training.json").read_text())
    assert record["training_config"]["learning_rate"] == 2e-5 and record["log_history"]


def test_unprojected_adapter_resumes_from_checkpoint(
    tokenizer, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dataset = make_dataset(TEXTS, tokenizer)
    states = []
    for _ in range(2):
        model = add_prefix(tiny_llama(), PrefixConfig(prefix_projection=False))
        states.append(train(build_trainer(model, tokenizer, dataset, dataset, CONFIG, tmp_path), tmp_path))

    assert "Resuming from" in capsys.readouterr().out
    # The second run resumed at the final checkpoint, so it trained no further steps.
    assert states[1].global_step == 2
    assert states[1].log_history[:-1] == states[0].log_history[:-1]  # per-step logs restored
    assert states[1].log_history[-1]["train_loss"] == 0.0  # summary of a run with no new steps
