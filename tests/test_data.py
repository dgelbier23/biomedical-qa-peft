import json
from pathlib import Path

import pandas as pd

from biomedical_qa_peft.data.config import DataConfig
from biomedical_qa_peft.data.prompts import format_example
from biomedical_qa_peft.data.pubmedqa import (
    balance_classes,
    clean_context,
    drop_duplicates,
    filter_valid,
    prepare_splits,
    save_splits,
    split,
)

CONFIG = DataConfig(system_prompt="Be helpful.")


class StubTokenizer:
    """Mimics the Llama 3.2 chat template (date lines, trimmed content) without the gated model.

    Counts one token per word, plus one when add_special_tokens is True.
    """

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool) -> str:
        assert tokenize is False
        text = "<|begin_of_text|>"
        for m in messages:
            text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            if m["role"] == "system":
                text += "Cutting Knowledge Date: December 2023\nToday Date: 11 Nov 2026\n\n"
            text += m["content"].strip() + "<|eot_id|>"
        return text

    def __call__(self, text: str, add_special_tokens: bool, return_attention_mask: bool) -> dict:
        return {"input_ids": [0] * (len(text.split()) + add_special_tokens)}


def make_rows(n: int, label: str, start: int = 0) -> list[dict[str, str]]:
    return [
        {
            "question": f"Does treatment {i} help?",
            "cleaned_context": f"Study {i} measured the outcome in many patients.",
            "long_answer": f"Treatment {i} improved the outcome in this study.",
            "final_decision": label,
        }
        for i in range(start, start + n)
    ]


def test_clean_context_joins_and_collapses_whitespace() -> None:
    assert clean_context(["  First  part.\n", "Second\tpart. "]) == "First part. Second part."


def test_filter_valid_drops_each_failing_rule() -> None:
    valid = make_rows(1, "yes")[0]
    df = pd.DataFrame([
        valid,
        {**valid, "cleaned_context": "Too short."},                       # < 30 chars
        {**valid, "cleaned_context": "Supercalifragilistic expialidocious."},  # < 5 words
        {**valid, "long_answer": "This long answer has no full stop at all"},
        {**valid, "question": "This question has no question mark."},
        {**valid, "final_decision": "unsure"},
        {**valid, "final_decision": " Maybe "},                           # valid after normalising
    ])

    kept = filter_valid(df, CONFIG)

    assert kept["final_decision"].tolist() == ["yes", " Maybe "]
    assert kept.index.tolist() == [0, 1]


def test_drop_duplicates_keeps_first_question_context_pair() -> None:
    row = make_rows(1, "yes")[0]
    df = pd.DataFrame([row, {**row, "final_decision": "no"}, make_rows(1, "no", start=1)[0]])

    kept = drop_duplicates(df)

    assert kept["final_decision"].tolist() == ["yes", "no"]
    assert kept.index.tolist() == [0, 2]


def test_balance_classes_is_equal_seeded_and_drops_other_labels() -> None:
    df = pd.DataFrame(make_rows(30, "yes") + make_rows(8, "no", 30) + make_rows(5, "maybe", 38))
    config = DataConfig(per_class=10)

    balanced = balance_classes(df, config)

    assert balanced["final_decision"].value_counts().to_dict() == {"yes": 8, "no": 8}
    assert balanced.equals(balance_classes(df, config))
    assert not balanced.equals(balance_classes(df, DataConfig(per_class=10, seed=0)))


def test_format_example_matches_original_layout() -> None:
    row = make_rows(1, "no")[0]

    text = format_example(row, StubTokenizer(), CONFIG)

    # Layout of the dissertation's saved formatted_text: one newline after each
    # header, no date lines, template whitespace removed.
    assert text == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Be helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "Question: Does treatment 0 help?\n"
        "Context: Study 0 measured the outcome in many patients.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "Simplified Answer: no\n"
        "Detailed Answer: Treatment 0 improved the outcome in this study.<|eot_id|>"
    )


def test_split_is_80_10_10_and_stratified() -> None:
    df = pd.DataFrame(make_rows(20, "yes") + make_rows(20, "no", 20))

    splits = split(df, CONFIG)

    for name, size in {"train": 32, "validation": 4, "test": 4}.items():
        assert splits[name]["final_decision"].value_counts().to_dict() == {
            "yes": size // 2, "no": size // 2,
        }
    assert pd.concat(splits.values()).index.is_unique  # no row in two splits


def test_prepare_splits_counts_each_step_and_saves(tmp_path: Path) -> None:
    rows = make_rows(25, "yes") + make_rows(25, "no", 25)
    rows[0] = {**rows[0], "long_answer": rows[0]["long_answer"] + " Extra words." * 20}  # too long
    rows.append({**rows[1], "question": "No question mark."})                             # invalid
    rows.append(rows[2])                                                                    # duplicate
    config = DataConfig(per_class=25, max_tokens=60)

    splits, counts = prepare_splits(pd.DataFrame(rows), StubTokenizer(), config)

    assert counts == {
        "loaded": 52, "valid": 51, "deduplicated": 50, "balanced": 50,
        "within_max_tokens": 49, "train": 39, "validation": 5, "test": 5,
    }
    assert list(splits["train"].columns) == [
        "question", "cleaned_context", "long_answer", "final_decision",
        "formatted_text", "count_of_token",
    ]

    save_splits(splits, tmp_path, config)

    assert len(pd.read_json(tmp_path / "test.jsonl", lines=True)) == counts["test"]
    assert json.loads((tmp_path / "data_config.json").read_text())["per_class"] == 25
