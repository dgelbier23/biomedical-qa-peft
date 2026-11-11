"""PubMedQA loading, cleaning, balancing and splitting into train/validation/test."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from biomedical_qa_peft.data.config import DataConfig
from biomedical_qa_peft.data.prompts import count_tokens, format_example

VALID_DECISIONS = {"yes", "no", "maybe"}


def clean_context(contexts: list[str]) -> str:
    """Join a row's context passages into one string with single spaces."""
    return re.sub(r"\s+", " ", " ".join(contexts)).strip()


def load_pubmedqa(config: DataConfig) -> pd.DataFrame:
    """Download PubMedQA and return question, cleaned_context, long_answer, final_decision."""
    from datasets import load_dataset  # Imported here so tests do not need the Hub.

    ds = load_dataset(
        config.dataset_name, config.dataset_subset, split="train", revision=config.dataset_revision
    )
    return pd.DataFrame({
        "question": ds["question"],
        "cleaned_context": [clean_context(c["contexts"]) for c in ds["context"]],
        "long_answer": ds["long_answer"],
        "final_decision": ds["final_decision"],
    })


def _is_valid_text(text: str, config: DataConfig) -> bool:
    return len(text) >= config.min_chars and len(text.split()) >= config.min_words and "." in text


def filter_valid(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Keep rows whose context and long answer pass the length/full-stop rules,
    whose question contains "?", and whose decision is yes/no/maybe."""
    mask = (
        df["cleaned_context"].map(lambda t: _is_valid_text(t, config))
        & df["long_answer"].map(lambda t: _is_valid_text(t, config))
        & df["question"].str.contains("?", regex=False)
        & df["final_decision"].str.lower().str.strip().isin(VALID_DECISIONS)
    )
    return df[mask].reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row of each (question, context) pair.

    The original removed duplicates on four columns, then on these two; this one
    pass keeps exactly the same rows.
    """
    return df.drop_duplicates(subset=["question", "cleaned_context"], keep="first")


def balance_classes(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Sample the same number of rows from each label in config.labels, then shuffle."""
    groups = [df[df["final_decision"] == label] for label in config.labels]
    n = min(config.per_class, *(len(g) for g in groups))
    sampled = pd.concat([g.sample(n=n, random_state=config.seed) for g in groups])
    return sampled.sample(frac=1, random_state=config.seed).reset_index(drop=True)


def split(df: pd.DataFrame, config: DataConfig) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split by final_decision."""
    train, held_out = train_test_split(
        df, test_size=config.eval_fraction, random_state=config.seed,
        shuffle=True, stratify=df["final_decision"],
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, random_state=config.seed,
        shuffle=True, stratify=held_out["final_decision"],
    )
    return {"train": train, "validation": validation, "test": test}


def prepare_splits(
    df: pd.DataFrame, tokenizer: Any, config: DataConfig
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Run the full pipeline on loaded rows.

    Returns the splits, each with formatted_text and count_of_token columns, and
    the row count after each step.
    """
    counts = {"loaded": len(df)}
    df = filter_valid(df, config)
    counts["valid"] = len(df)
    df = drop_duplicates(df)
    counts["deduplicated"] = len(df)
    df = balance_classes(df, config)
    counts["balanced"] = len(df)

    df["formatted_text"] = df.apply(lambda row: format_example(row, tokenizer, config), axis=1)
    df["count_of_token"] = df["formatted_text"].map(lambda t: count_tokens(t, tokenizer))
    df = df[df["count_of_token"] <= config.max_tokens]
    counts["within_max_tokens"] = len(df)

    splits = split(df, config)
    counts.update({name: len(part) for name, part in splits.items()})
    return splits, counts


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path, config: DataConfig) -> None:
    """Write <split>.jsonl files and the data_config.json that produced them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True)
    (out_dir / "data_config.json").write_text(json.dumps(asdict(config), indent=2))
