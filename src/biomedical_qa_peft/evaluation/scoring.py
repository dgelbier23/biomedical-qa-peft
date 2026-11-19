"""Run every metric on a set of answers, summarise with confidence intervals, and save."""

import json
from dataclasses import asdict, dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from biomedical_qa_peft.evaluation.config import InferenceConfig
from biomedical_qa_peft.evaluation.inference import build_prompt, generate_responses, score_yes_no
from biomedical_qa_peft.evaluation.metrics import (
    NLI_LABELS,
    bertscore_scores,
    bootstrap_ci,
    detailed_answer,
    format_stats,
    meteor_scores,
    nli_results,
    score_short_answers,
)
from biomedical_qa_peft.runtime.env import describe_environment


@dataclass
class Scorers:
    nli_pipeline: Any
    bertscore: Any


def load_scorers(config: InferenceConfig) -> Scorers:
    """Download the NLTK data, NLI model and BERTScore. None of them is the model under test."""
    import evaluate
    import nltk
    from transformers import pipeline

    for package in ("omw-1.4", "punkt", "punkt_tab", "wordnet"):
        nltk.download(package, quiet=True)
    return Scorers(
        nli_pipeline=pipeline("text-classification", model=config.nli_model),
        bertscore=evaluate.load("bertscore"),
    )


def _long_answer(
    predictions: list[str], references: list[str], scorers: Scorers, config: InferenceConfig
) -> dict[str, Any]:
    """NLI, METEOR and BERTScore on the explanations only: the yes/no is scored separately,
    and left in, it dominates NLI (a wrong yes/no reads as a contradiction)."""
    predictions = [detailed_answer(p) for p in predictions]
    references = [detailed_answer(r) for r in references]
    return {
        "nli": nli_results(scorers.nli_pipeline, predictions, references, config.nli_batch_size),
        "meteor": meteor_scores(predictions, references),
        "bertscore": bertscore_scores(
            predictions, references, scorers.bertscore, config.bertscore_model
        ),
    }


def evaluate_responses(
    yes_probabilities: list[float],
    predictions: list[str],
    references: list[str],
    scorers: Scorers,
    config: InferenceConfig,
) -> dict[str, Any]:
    """Format, short-answer and long-answer evaluation, run concurrently."""
    if not len(yes_probabilities) == len(predictions) == len(references):
        raise ValueError("yes_probabilities, predictions and references differ in length")
    with ThreadPool(processes=2) as pool:
        long = pool.apply_async(_long_answer, (predictions, references, scorers, config))
        fmt = pool.apply_async(format_stats, (predictions,))
        return {
            "format": fmt.get(),
            "short_answer": score_short_answers(yes_probabilities, predictions, references),
            "long_answer": long.get(),
        }


def summarise(results: dict[str, Any]) -> dict[str, float]:
    """One table row: percentages (0-100), METEOR and BERTScore."""
    short = results["short_answer"]
    long = results["long_answer"]
    counts = long["nli"]["counts"]
    total = sum(counts.values())
    return {
        **{k: 100 * short[k]
           for k in ("accuracy", "precision", "recall", "f1", "written_accuracy")},
        **{label.lower(): 100 * counts[label] / total for label in NLI_LABELS},
        "meteor": float(np.mean(long["meteor"])),
        "bertscore": float(np.mean(long["bertscore"])),
        "format_adherence": results["format"]["valid_format_percentage"],
    }


def confidence_intervals(results: dict[str, Any], config: InferenceConfig) -> dict[str, list[float]]:
    """95% bootstrap intervals over test rows, in the same units as summarise."""
    predicted = np.array(results["short_answer"]["predictions"])
    expected = np.array(results["short_answer"]["references"])
    written = np.array(results["short_answer"]["written_correct"])
    entailed = np.array(results["long_answer"]["nli"]["labels"]) == "ENTAILMENT"
    meteor = np.array(results["long_answer"]["meteor"])
    bertscore = np.array(results["long_answer"]["bertscore"])
    statistics = {
        "accuracy": lambda i: 100 * np.mean(predicted[i] == expected[i]),
        "f1": lambda i: 100 * f1_score(expected[i], predicted[i], average="weighted", zero_division=0),
        "written_accuracy": lambda i: 100 * np.mean(written[i]),
        "entailment": lambda i: 100 * np.mean(entailed[i]),
        "meteor": lambda i: np.mean(meteor[i]),
        "bertscore": lambda i: np.mean(bertscore[i]),
    }
    return {
        name: bootstrap_ci(statistic, len(predicted), config.bootstrap_samples, config.seed)
        for name, statistic in statistics.items()
    }


def save_predictions(run_dir: Path, predictions: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_json(run_dir / "predictions.jsonl", orient="records", lines=True)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """The reference, prediction (and, from v2 on, yes_probability) of every test row."""
    return pd.read_json(run_dir / "predictions.jsonl", lines=True)


def save_results(run_dir: Path, **record: Any) -> None:
    """Write results.json: everything needed to interpret and repeat the run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(record, indent=2, default=str))


def yes_no_accuracy(
    model: Any,
    tokenizer: Any,
    formatted_texts: list[str],
    system_prompt: str | None,
    config: InferenceConfig,
) -> float:
    """Short-answer accuracy (0-100) of a system prompt, without generating long answers."""
    prompts, references = zip(*(build_prompt(t, system_prompt) for t in formatted_texts))
    probabilities = score_yes_no(model, tokenizer, list(prompts), config)
    return 100 * score_short_answers(probabilities, list(references), list(references))["accuracy"]


def run_method(
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    formatted_texts: list[str],
    system_prompt: str | None,
    scorers: Scorers,
    config: InferenceConfig,
    **metadata: Any,
) -> dict[str, float]:
    """Score yes/no answers, generate long answers, evaluate, and save under run_dir.

    If run_dir already holds predictions.jsonl (e.g. after a Colab disconnect during
    scoring), those predictions are reused, unless results.json shows they came from a
    different InferenceConfig. Delete the folder to start over. Rescoring keeps the
    generation time and environment of the run that made the predictions, and records its
    own as scoring_environment. Returns the summary row; results.json also holds the 95%
    confidence intervals.
    """
    prompts, references = zip(*(build_prompt(t, system_prompt) for t in formatted_texts))
    seconds, environment = None, describe_environment()
    if (run_dir / "predictions.jsonl").exists():
        results_file = run_dir / "results.json"
        if results_file.exists():
            previous = json.loads(results_file.read_text())
            if previous["inference_config"] != asdict(config):
                raise ValueError(f"{run_dir} holds predictions from another InferenceConfig; delete it")
            seconds, environment = previous["generation_seconds"], previous["environment"]
        print(f"Reusing saved predictions in {run_dir}")
        saved = load_predictions(run_dir)
        if saved["reference"].tolist() != list(references):
            raise ValueError(f"Saved predictions in {run_dir} are for different test rows")
    else:
        probabilities = score_yes_no(model, tokenizer, list(prompts), config)
        predictions, seconds = generate_responses(model, tokenizer, list(prompts), config)
        saved = pd.DataFrame({
            "reference": references, "prediction": predictions, "yes_probability": probabilities,
        })
        save_predictions(run_dir, saved)

    results = evaluate_responses(
        saved["yes_probability"].tolist(), saved["prediction"].tolist(), list(references),
        scorers, config,
    )
    summary = summarise(results)
    save_results(
        run_dir,
        summary=summary,
        confidence_intervals=confidence_intervals(results, config),
        system_prompt=system_prompt,
        inference_config=asdict(config),
        generation_seconds=seconds,
        environment=environment,
        scoring_environment=describe_environment(),
        rows=len(references),
        results=results,
        **metadata,
    )
    return summary
