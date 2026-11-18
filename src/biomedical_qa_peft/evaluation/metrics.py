"""Evaluation metrics: answer format, short (yes/no) answer and long answer."""

import re
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LABELS = {"no": 0, "yes": 1}
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


def score_short_answers(yes_probabilities: list[float], references: list[str]) -> dict[str, Any]:
    """Accuracy and weighted precision/recall/F1 (0-1) of yes/no answers.

    An answer is "yes" when P(yes) > 0.5. Returns the metrics and the integer labels.
    """
    predicted = [int(p > 0.5) for p in yes_probabilities]
    expected = [LABELS[extract_short_answer(r)] for r in references]
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
) -> dict[str, Any]:
    """NLI label of each pair (premise: reference, hypothesis: prediction), with counts and
    mean confidence per label."""
    labels: list[str] = []
    scores: dict[str, list[float]] = {label: [] for label in NLI_LABELS}
    for start in range(0, len(predictions), batch_size):
        pairs = [
            {"text": ref, "text_pair": pred}
            for ref, pred in zip(
                references[start:start + batch_size], predictions[start:start + batch_size]
            )
        ]
        for result in nli_pipeline(pairs):
            labels.append(result["label"].upper())
            scores[labels[-1]].append(result["score"])
    return {
        "labels": labels,
        "counts": {label: len(s) for label, s in scores.items()},
        "mean_confidence": {label: sum(s) / len(s) if s else 0.0 for label, s in scores.items()},
    }


def meteor_scores(predictions: list[str], references: list[str]) -> list[float]:
    """METEOR per pair; a pair that fails to score counts as 0, as in the original.

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
    return scores


def bertscore_scores(
    predictions: list[str], references: list[str], bertscore: Any, model: str
) -> list[float]:
    """BERTScore F1 per pair."""
    return bertscore.compute(predictions=predictions, references=references, model_type=model)["f1"]


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float], n_rows: int, samples: int, seed: int
) -> list[float]:
    """95% percentile-bootstrap interval of statistic(row indices) over resampled rows."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n_rows, n_rows)) for _ in range(samples)]
    return np.percentile(values, [2.5, 97.5]).tolist()
