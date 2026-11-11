"""Configuration for building the PubMedQA splits. Defaults reproduce the dissertation."""

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a medical AI assistant specialized in providing clear and concise answers "
    "to medical questions based on context."
)
# Exact strings from the original notebook, including trailing whitespace, which
# textwrap.dedent and the chat template then remove. Filled with str.format(**row).
USER_TEMPLATE = "Question: {question}\nContext: {cleaned_context}\n  "
ANSWER_TEMPLATE = "Simplified Answer: {final_decision}\nDetailed Answer: {long_answer}\n      "


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "qiaojin/PubMedQA"
    dataset_subset: str = "pqa_artificial"
    tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    # Context and long answer need at least this many characters and words, plus a ".".
    min_chars: int = 30
    min_words: int = 5
    # Classes kept, in the order they are sampled and concatenated (order affects the shuffle).
    labels: tuple[str, ...] = ("yes", "no")
    per_class: int = 5000
    # Rows above this token count are dropped (see prompts.count_tokens for a caveat).
    max_tokens: int = 1024
    # Fraction held out for validation + test, split equally between them.
    eval_fraction: float = 0.2
    seed: int = 42
    system_prompt: str = SYSTEM_PROMPT
    user_template: str = USER_TEMPLATE
    answer_template: str = ANSWER_TEMPLATE
