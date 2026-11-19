"""Training text, datasets and the Hugging Face Trainer setup shared by prefix tuning and LoRA."""

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from biomedical_qa_peft.evaluation.inference import END_TAG, build_prompt
from biomedical_qa_peft.runtime.env import describe_environment
from biomedical_qa_peft.training.config import TrainingConfig

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def build_training_example(formatted_text: str, system_prompt: str) -> tuple[str, str]:
    """(prompt, answer): the prompt exactly as at test time (evaluation.inference.build_prompt)
    and the reference answer that follows it, ending with <|eot_id|>."""
    prompt, reference = build_prompt(formatted_text, system_prompt)
    return prompt, f"\n{reference}{END_TAG}"


def make_dataset(examples: list[tuple[str, str]], tokenizer: Any) -> Any:
    """Tokenised datasets.Dataset of (prompt, answer) pairs: input_ids, attention_mask and
    labels. Labels are -100 over the prompt, so the loss covers only the answer."""
    from datasets import Dataset

    def tokenise(rows: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        prompts = tokenizer(rows["prompt"], add_special_tokens=True)["input_ids"]
        answers = tokenizer(rows["answer"], add_special_tokens=False)["input_ids"]
        input_ids = [p + a for p, a in zip(prompts, answers)]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "labels": [[-100] * len(p) + a for p, a in zip(prompts, answers)],
        }

    prompts, answers = zip(*examples)
    return Dataset.from_dict({"prompt": list(prompts), "answer": list(answers)}).map(
        tokenise, batched=True, remove_columns=["prompt", "answer"]
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: TrainingConfig,
    output_dir: Path,
) -> Any:
    """Trainer with the original arguments and early stopping. The collator pads labels
    with -100, so padding adds no loss. Logging goes to the trainer's log history."""
    from transformers import (
        DataCollatorForSeq2Seq,
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
        activation_checkpointing=config.activation_checkpointing,
        train_sampling_strategy="group_by_length" if config.group_by_length else "random",
        seed=config.seed,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )


def train(trainer: Any, output_dir: Path, resume: bool = True) -> Any:
    """Train and return the trainer state (steps, best checkpoint and its loss).

    With resume, continues from the newest checkpoint in output_dir. Pads on the right
    while training (the original's setting), then restores the tokenizer's padding side.

    Prefix tuning with projection cannot resume, and cannot reload its best checkpoint
    at the end (transformers 4.47 and 5): checkpoints hold only the projected prefix, not
    the MLP that produces it. Train it with resume=False. When reloading fails this way,
    training still succeeds; run_training saves the best checkpoint's adapter files.
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
        # transformers 5 wraps the size-mismatch error in its own message.
        message = f"{error} {error.__cause__}"
        if checkpoints or not getattr(peft_config, "is_prompt_learning", False) or "size mismatch" not in message:
            raise
        print(
            f"Could not reload the best checkpoint ({trainer.state.best_model_checkpoint}, "
            f"eval_loss {trainer.state.best_metric}) into the training model; its saved "
            f"adapter files are still usable."
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

    Saves checkpoints/, adapter/ (copied from the checkpoint with the lowest validation
    loss), the two loss-curve figures and training.json (summary, config, environment
    and full log history). Returns the summary. After a resume, training_minutes covers
    only the resumed part.
    """
    datasets = {
        split: make_dataset(
            [
                build_training_example(text, system_prompt)
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
    best = state.best_model_checkpoint
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        shutil.copy2(Path(best) / name, run_dir / "adapter" / name)

    for name, figure in zip(("training_loss.png", "eval_loss.png"), plot_training_curves(state.log_history)):
        figure.savefig(run_dir / name)
    eval_losses = [(log["step"], log["eval_loss"]) for log in state.log_history if "eval_loss" in log]
    summary = {
        "best_step": int(best.rsplit("-", 1)[1]),
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
