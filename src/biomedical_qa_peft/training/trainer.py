"""Training text, datasets and the Hugging Face Trainer setup shared by prefix tuning and LoRA."""

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.runtime.env import describe_environment
from biomedical_qa_peft.training.config import TrainingConfig

_SYSTEM_TURN = re.compile(
    re.escape("<|start_header_id|>system<|end_header_id|>") + r"\s*" + re.escape(SYSTEM_PROMPT)
)


def build_training_text(formatted_text: str, system_prompt: str) -> str:
    """The full conversation (answer included) with the system prompt replaced.

    As in the original, <|begin_of_text|> is kept and the tokenizer adds a second one,
    so training text starts with two, while test prompts (evaluation.inference) start
    with one.
    """
    text = _SYSTEM_TURN.sub(f"<|start_header_id|>system<|end_header_id|> {system_prompt}", formatted_text)
    return text.replace("<|pad|>", "")


def make_dataset(texts: list[str], tokenizer: Any) -> Any:
    """Tokenised datasets.Dataset (input_ids, attention_mask); padding is left to the collator."""
    from datasets import Dataset

    return Dataset.from_dict({"text": texts}).map(
        lambda rows: tokenizer(
            rows["text"], add_special_tokens=True, truncation=False, padding=False
        ),
        batched=True,
        remove_columns=["text"],
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: TrainingConfig,
    output_dir: Path,
) -> Any:
    """Trainer with the original arguments, causal-LM collator and early stopping.

    The loss covers every token (system prompt, question and answer); only padding
    is masked out. Logging goes to the trainer's log history, not Weights & Biases.
    """
    from transformers import (
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="none",
        torch_empty_cache_steps=config.torch_empty_cache_steps,
        torch_compile=False,
        group_by_length=config.group_by_length,
        seed=config.seed,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )


def train(trainer: Any, output_dir: Path, resume: bool = True) -> Any:
    """Train and return the trainer state (steps, best checkpoint and its loss).

    With resume, continues from the newest checkpoint in output_dir. Pads on the right
    while training (the original's setting), then restores the tokenizer's padding side.

    Prefix tuning with projection cannot resume, and cannot reload its best checkpoint
    at the end, in transformers 4.47: checkpoints hold only the projected prefix, not
    the MLP that produces it. Train it with resume=False. When reloading the best
    checkpoint fails this way, the final weights are kept, as in the original run.
    """
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints and not resume:
        raise FileExistsError(f"{output_dir} already has checkpoints; delete them to train from scratch")
    if checkpoints:
        print(f"Resuming from {checkpoints[-1]}")
    tokenizer = trainer.processing_class
    padding_side, tokenizer.padding_side = tokenizer.padding_side, "right"
    try:
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    except RuntimeError as error:
        peft_config = getattr(trainer.model, "active_peft_config", None)
        if checkpoints or not getattr(peft_config, "is_prompt_learning", False) or "size mismatch" not in str(error):
            raise
        print(
            f"Could not reload the best checkpoint ({trainer.state.best_model_checkpoint}, "
            f"eval_loss {trainer.state.best_metric}); keeping the final weights from step "
            f"{trainer.state.global_step}, as the original run did."
        )
    finally:
        tokenizer.padding_side = padding_side
    return trainer.state


def run_training(
    run_dir: Path,
    peft_model: Any,
    tokenizer: Any,
    data_dir: Path,
    system_prompt: str,
    config: TrainingConfig,
    resume: bool = True,
) -> dict[str, Any]:
    """Train on data_dir's train/validation splits and save everything under run_dir.

    Saves checkpoints/, adapter/ (the final weights), the two loss-curve figures and
    training.json (summary, config, environment and full log history). Returns the summary.
    """
    datasets = {
        split: make_dataset(
            [
                build_training_text(text, system_prompt)
                for text in pd.read_json(data_dir / f"{split}.jsonl", lines=True)["formatted_text"]
            ],
            tokenizer,
        )
        for split in ("train", "validation")
    }
    checkpoint_dir = run_dir / "checkpoints"
    trainer = build_trainer(
        peft_model, tokenizer, datasets["train"], datasets["validation"], config, checkpoint_dir
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    start = time.time()
    state = train(trainer, checkpoint_dir, resume)
    minutes = (time.time() - start) / 60
    trainer.save_model(str(run_dir / "adapter"))

    for name, figure in zip(("training_loss.png", "eval_loss.png"), plot_training_curves(state.log_history)):
        figure.savefig(run_dir / name)
    eval_losses = [(log["step"], log["eval_loss"]) for log in state.log_history if "eval_loss" in log]
    best = state.best_model_checkpoint
    summary = {
        "best_step": int(best.rsplit("-", 1)[1]) if best else None,
        "best_eval_loss": state.best_metric,
        "final_step": state.global_step,
        "epoch": state.epoch,
        "final_eval_loss": eval_losses[-1][1] if eval_losses else None,
        "training_minutes": round(minutes, 2),
        "trainable_parameters": trainable,
    }
    record = {
        **summary,
        "system_prompt": system_prompt,
        "training_config": asdict(config),
        "environment": describe_environment(),
        "log_history": state.log_history,
    }
    (run_dir / "training.json").write_text(json.dumps(record, indent=2, default=str))
    return summary


def plot_training_curves(log_history: list[dict]) -> list[Any]:
    """The original's two figures: training loss and validation loss by step.

    Built without pyplot, so no GUI backend is needed (Tk can be missing on Windows).
    """
    from matplotlib.figure import Figure

    train = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
    evals = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

    train_fig = Figure(figsize=(10, 6))
    ax = train_fig.add_subplot()
    ax.plot(*zip(*train), "b-", label="Training Loss")
    ax.set_title("Training Loss by Steps")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    eval_fig = Figure(figsize=(12, 5))
    ax = eval_fig.add_subplot(1, 2, 1)
    ax.plot(*zip(*evals), "r-", label="Evaluation Loss")
    ax.set_title("Evaluation Loss by Steps")
    ax.set_xlabel("Evaluation Step")
    ax.set_ylabel("Loss")
    ax.grid(True)
    eval_fig.tight_layout()
    return [train_fig, eval_fig]
