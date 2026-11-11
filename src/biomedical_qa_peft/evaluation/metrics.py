"""Evaluation metrics ported from the dissertation's evaluation_metrics.py.

Known flaws kept on purpose until the improvements stage: the NLI input joins
reference and prediction with "[SEP]" (not RoBERTa's separator), and the yes/no
judge is the model under test.
"""

import re
from collections.abc import Callable
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from biomedical_qa_peft.evaluation.config import JUDGE_PROMPT

LABELS = {"no": 0, "yes": 1, "unclear": 2}
NLI_LABELS = ("ENTAILMENT", "CONTRADICTION", "NEUTRAL")

_FORMAT = re.compile(
    r"^\s*Simplified Answer:\s*(yes|no)\s*\n+\s*Detailed Answer:\s*(.*?)\s*$", re.DOTALL
)
_SHORT_ANSWER = re.compile(
    r"(?:Simplified\s+)?Answer(?:\s*is|\s*:)?\s*(yes|no|unclear)\b", re.IGNORECASE
)


def has_valid_format(output: str) -> bool:
    """True for "Simplified Answer: yes|no" then "Detailed Answer: ..." on a new line."""
    return bool(_FORMAT.match(output.strip()))


def format_stats(outputs: list[str]) -> dict[str, float]:
    valid = sum(has_valid_format(o) for o in outputs)
    return {
        "total_outputs": len(outputs),
        "valid_format_count": valid,
        "invalid_format_count": len(outputs) - valid,
        "valid_format_percentage": 100 * valid / len(outputs) if outputs else 0.0,
    }


def extract_short_answer(text: str) -> str:
    """yes / no / unclear after "Answer", or "not_found"."""
    match = _SHORT_ANSWER.search(text)
    return match.group(1).lower() if match else "not_found"


def judge_short_answers(texts: list[str], model: Any, tokenizer: Any) -> list[str]:
    """Ask the LLM for yes / no / unclear for each text, greedily, in one batch.

    The chat template inserts today's date, so judge inputs differ slightly from the
    original run's.
    """
    messages = [
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": text}]
        for text in texts
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", truncation=True, padding=True, add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=4,
            attention_mask=(inputs != tokenizer.pad_token_id),
            do_sample=False,
        )
    answers = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
    return [
        m.group(0).lower() if (m := re.search(r"\b(yes|no|unclear)\b", a, re.IGNORECASE))
        else "unclear"
        for a in answers
    ]


def score_short_answers(
    predictions: list[str],
    references: list[str],
    judge: Callable[[list[str]], list[str]],
    batch_size: int,
) -> dict[str, Any]:
    """Accuracy and weighted precision/recall/F1 (0-1) over no/yes/unclear labels.

    The regex extracts each answer; texts it cannot read go to `judge` in batches.
    Returns the metrics and the integer labels (for a confusion matrix).
    """
    predicted = [LABELS.get(extract_short_answer(p), None) for p in predictions]
    expected = [LABELS.get(extract_short_answer(r), 2) for r in references]

    unread = [i for i, label in enumerate(predicted) if label is None]
    judged: list[str] = []
    for start in range(0, len(unread), batch_size):
        judged += judge([predictions[i] for i in unread[start:start + batch_size]])
    for i, answer in zip(unread, judged):
        predicted[i] = LABELS.get(answer.strip().lower(), 2)

    return {
        "accuracy": accuracy_score(expected, predicted),
        "precision": precision_score(expected, predicted, average="weighted", zero_division=0),
        "recall": recall_score(expected, predicted, average="weighted", zero_division=0),
        "f1": f1_score(expected, predicted, average="weighted", zero_division=0),
        "predictions": predicted,
        "references": expected,
    }


def nli_results(
    nli_pipeline: Callable, predictions: list[str], references: list[str], batch_size: int
) -> dict[str, dict[str, float]]:
    """Count and mean confidence of each NLI label for "reference [SEP] prediction"."""
    scores: dict[str, list[float]] = {label: [] for label in NLI_LABELS}
    for start in range(0, len(predictions), batch_size):
        inputs = [
            f"{ref} [SEP] {pred}"
            for ref, pred in zip(
                references[start:start + batch_size], predictions[start:start + batch_size]
            )
        ]
        for result in nli_pipeline(inputs):
            scores[result["label"].upper()].append(result["score"])
    return {
        "counts": {label: len(s) for label, s in scores.items()},
        "mean_confidence": {label: sum(s) / len(s) if s else 0.0 for label, s in scores.items()},
    }


def meteor(predictions: list[str], references: list[str]) -> float:
    """Mean METEOR over pairs; a pair that fails to score counts as 0, as in the original.

    Needs the NLTK punkt, punkt_tab, wordnet and omw-1.4 data.
    """
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for pred, ref in zip(predictions, references):
        try:
            scores.append(meteor_score([word_tokenize(ref)], word_tokenize(pred)))
        except Exception as e:  # noqa: BLE001 - original behaviour
            print(f"METEOR failed for a pair, scored 0: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def bertscore_f1(predictions: list[str], references: list[str], bertscore: Any, model: str) -> float:
    """Mean BERTScore F1."""
    results = bertscore.compute(predictions=predictions, references=references, model_type=model)
    return sum(results["f1"]) / len(results["f1"])
