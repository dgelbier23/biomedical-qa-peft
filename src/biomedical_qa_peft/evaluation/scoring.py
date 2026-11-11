"""Run every metric on a set of answers, summarise, compare and save (ports EvaluationManager)."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import pandas as pd

from biomedical_qa_peft.evaluation.config import DISSERTATION_RESULTS, TOLERANCES, InferenceConfig
from biomedical_qa_peft.evaluation.inference import build_prompt, generate_responses
from biomedical_qa_peft.evaluation.metrics import (
    NLI_LABELS,
    bertscore_f1,
    format_stats,
    judge_short_answers,
    meteor,
    nli_results,
    score_short_answers,
)
from biomedical_qa_peft.runtime.env import describe_environment


@dataclass
class Scorers:
    judge: Callable[[list[str]], list[str]]
    nli_pipeline: Callable
    bertscore: Any


def load_scorers(model: Any, tokenizer: Any, config: InferenceConfig) -> Scorers:
    """Download the NLTK data, NLI model and BERTScore; the judge reuses `model`."""
    import evaluate
    import nltk
    from transformers import pipeline

    for package in ("omw-1.4", "punkt", "punkt_tab", "wordnet"):
        nltk.download(package, quiet=True)
    return Scorers(
        judge=partial(judge_short_answers, model=model, tokenizer=tokenizer),
        nli_pipeline=pipeline("text-classification", model=config.nli_model),
        bertscore=evaluate.load("bertscore"),
    )


def _long_answer(
    predictions: list[str], references: list[str], scorers: Scorers, config: InferenceConfig
) -> dict[str, Any]:
    return {
        "nli": nli_results(scorers.nli_pipeline, predictions, references, config.nli_batch_size),
        "meteor": meteor(predictions, references),
        "bertscore": bertscore_f1(
            predictions, references, scorers.bertscore, config.bertscore_model
        ),
    }


def evaluate_responses(
    predictions: list[str], references: list[str], scorers: Scorers, config: InferenceConfig
) -> dict[str, Any]:
    """Format, short-answer and long-answer evaluation, run concurrently as in the original.

    Unlike the original, an error in any metric is raised instead of printed and skipped.
    """
    if len(predictions) != len(references):
        # The original's prompt search crashed BERTScore and silently truncated the
        # other metrics this way.
        raise ValueError(f"{len(predictions)} predictions but {len(references)} references")
    with ThreadPool(processes=3) as pool:
        fmt = pool.apply_async(format_stats, (predictions,))
        short = pool.apply_async(
            score_short_answers, (predictions, references, scorers.judge, config.judge_batch_size)
        )
        long = pool.apply_async(_long_answer, (predictions, references, scorers, config))
        return {"format": fmt.get(), "short_answer": short.get(), "long_answer": long.get()}


def summarise(results: dict[str, Any]) -> dict[str, float]:
    """The dissertation's table row: percentages (0-100), METEOR and BERTScore."""
    short = results["short_answer"]
    counts = results["long_answer"]["nli"]["counts"]
    total = sum(counts.values())
    return {
        **{k: 100 * short[k] for k in ("accuracy", "precision", "recall", "f1")},
        **{label.lower(): 100 * counts[label] / total for label in NLI_LABELS},
        "meteor": results["long_answer"]["meteor"],
        "bertscore": results["long_answer"]["bertscore"],
        "format_adherence": results["format"]["valid_format_percentage"],
    }


def compare_to_dissertation(summary: dict[str, float], method: str) -> dict[str, bool]:
    """Print each value beside the dissertation's; True where it is within tolerance."""
    within = {}
    for key, target in DISSERTATION_RESULTS[method].items():
        tolerance = TOLERANCES.get(key, TOLERANCES["percent"])
        within[key] = abs(summary[key] - target) <= tolerance
        print(
            f"{key:>17}: {summary[key]:9.4f}  (dissertation {target:9.4f} ± {tolerance})  "
            f"{'ok' if within[key] else 'DIFF'}"
        )
    return within


def save_predictions(run_dir: Path, predictions: list[str], references: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reference": references, "prediction": predictions}).to_json(
        run_dir / "predictions.jsonl", orient="records", lines=True
    )


def load_predictions(run_dir: Path) -> tuple[list[str], list[str]]:
    """(predictions, references) saved by save_predictions, e.g. to re-score a run."""
    df = pd.read_json(run_dir / "predictions.jsonl", lines=True)
    return df["prediction"].tolist(), df["reference"].tolist()


def save_results(run_dir: Path, **record: Any) -> None:
    """Write results.json: everything needed to interpret and repeat the run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(record, indent=2, default=str))


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
    """Generate answers for test rows, score them and save both under run_dir.

    If run_dir already holds predictions.jsonl (e.g. after a Colab disconnect during
    scoring), those predictions are reused instead of regenerated. Delete the folder
    to start over. Returns the summary row.
    """
    prompts, references = zip(*(build_prompt(t, system_prompt) for t in formatted_texts))
    seconds = None
    if (run_dir / "predictions.jsonl").exists():
        print(f"Reusing saved predictions in {run_dir}")
        predictions, saved_references = load_predictions(run_dir)
        if saved_references != list(references):
            raise ValueError(f"Saved predictions in {run_dir} are for different test rows")
    else:
        predictions, seconds = generate_responses(model, tokenizer, list(prompts), config)
        save_predictions(run_dir, predictions, list(references))

    results = evaluate_responses(predictions, list(references), scorers, config)
    summary = summarise(results)
    save_results(
        run_dir,
        summary=summary,
        system_prompt=system_prompt,
        inference_config=asdict(config),
        inference_seconds=seconds,
        environment=describe_environment(),
        rows=len(predictions),
        results=results,
        **metadata,
    )
    return summary
