"""Llama 3 chat formatting and token counting for PubMedQA rows."""

import re
from collections.abc import Mapping
from textwrap import dedent
from typing import Any

from biomedical_qa_peft.data.config import DataConfig


def strip_chat_metadata(text: str) -> str:
    """Remove the date lines the Llama 3.2 template adds, and blank lines after headers.

    Without this, the formatted text would change with the day it was generated.
    """
    text = re.sub(r"Cutting Knowledge Date:.*\nToday Date:.*\n", "", text, flags=re.MULTILINE)
    return re.sub(r"(<\|end_header_id\|>)\s*\n+", r"\1\n", text)


def format_example(row: Mapping[str, Any], tokenizer: Any, config: DataConfig) -> str:
    """Full system/user/assistant conversation for one row, as training text."""
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": dedent(config.user_template.format(**row))},
        {"role": "assistant", "content": dedent(config.answer_template.format(**row))},
    ]
    return strip_chat_metadata(tokenizer.apply_chat_template(messages, tokenize=False))


def count_tokens(text: str, tokenizer: Any) -> int:
    """Token count as the dissertation computed it.

    Caveat: the text already starts with <|begin_of_text|> and add_special_tokens=True
    adds another, so this is one more than the true length. Kept because the
    max_tokens filter was applied to this count.
    """
    return len(tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"])
