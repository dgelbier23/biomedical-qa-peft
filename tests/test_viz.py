import json

import numpy as np
import pandas as pd
import pytest

from biomedical_qa_peft.evaluation.metrics import mcnemar_p
from biomedical_qa_peft.viz.explorer import SAMPLE_SIZES, explorer_data, outcome, render_explorer
from biomedical_qa_peft.viz.plotting import METHODS
from biomedical_qa_peft.viz.results import (
    plot_accuracy,
    plot_calibration,
    plot_confusion,
    plot_efficiency,
    plot_nli,
    plot_training_history,
)

REFERENCES = [1, 0, 1, 0, 1, 0, 1, 0]
# Per method: P(yes) per question; predictions are P(yes) > 0.5.
PROBABILITIES = {
    "baseline": [0.9, 0.2, 0.3, 0.6, 0.8, 0.1, 0.7, 0.4],
    "prompt_engineering": [0.6, 0.7, 0.4, 0.3, 0.9, 0.2, 0.2, 0.6],
    "prefix_tuning": [0.8, 0.1, 0.9, 0.2, 0.7, 0.3, 0.6, 0.1],
    "lora": [0.95, 0.05, 0.85, 0.15, 0.4, 0.2, 0.9, 0.1],
}


def fake_run(probabilities: list[float]) -> dict:
    predictions = [int(p > 0.5) for p in probabilities]
    accuracy = 100 * np.mean(np.equal(predictions, REFERENCES))
    return {
        "summary": {"accuracy": accuracy, "entailment": 40.0, "neutral": 50.0, "contradiction": 10.0},
        "confidence_intervals": {"accuracy": [accuracy - 5, accuracy + 5]},
        "results": {
            "short_answer": {"predictions": predictions, "references": REFERENCES},
            "long_answer": {"nli": {"labels": ["ENTAILMENT", "NEUTRAL"] * 4},
                            "meteor": [0.25] * 8, "bertscore": [0.85] * 8},
        },
    }


RUNS = {m: fake_run(p) for m, p in PROBABILITIES.items()}
TRAINING = {"best_step": 20, "best_eval_loss": 1.37, "log_history": [
    {"loss": 1.6, "step": 10}, {"eval_loss": 1.4, "step": 10},
    {"loss": 1.4, "step": 20}, {"eval_loss": 1.37, "step": 20}, {"train_loss": 1.5, "step": 20}]}


@pytest.mark.parametrize("figure", [
    lambda: plot_accuracy(RUNS),
    lambda: plot_nli(RUNS),
    lambda: plot_training_history({"prefix_tuning": TRAINING, "lora": TRAINING}),
    lambda: plot_calibration(RUNS, PROBABILITIES),
    lambda: plot_confusion(RUNS),
    lambda: plot_efficiency(RUNS, {"prefix_tuning": 179_387_392, "lora": 2_293_760}),
])
def test_every_figure_renders_to_json(figure) -> None:
    assert json.loads(figure().to_json())["data"]


def test_accuracy_figure_reports_mcnemar_against_the_baseline() -> None:
    correct = {m: np.equal(r["results"]["short_answer"]["predictions"], REFERENCES) for m, r in RUNS.items()}
    labels = [a.text for a in plot_accuracy(RUNS).layout.annotations]

    assert labels[0] == "reference"
    assert labels[3] == f"p = {mcnemar_p(correct['lora'], correct['baseline']):.2g}"


def test_confusion_matrix_counts_reference_rows_by_prediction() -> None:
    lora = plot_confusion(RUNS).data[3].z  # Predictions: 1 0 1 0 0 0 1 0.
    assert np.array(lora).tolist() == [[4, 0], [1, 3]]


def test_outcome_groups() -> None:
    right = dict.fromkeys(METHODS, True)
    assert outcome(right) == "all_right"
    assert outcome(dict.fromkeys(METHODS, False)) == "all_wrong"
    assert outcome(right | {"baseline": False}) == "adapters_fix"
    assert outcome(dict.fromkeys(METHODS, False) | {"baseline": True}) == "adapters_break"
    assert outcome(right | {"lora": False}) == "mixed"


def test_explorer_samples_each_group_and_embeds_safely() -> None:
    test = pd.DataFrame({
        "question": [f"Q{i}?" for i in range(8)],
        "cleaned_context": ["Context </script> here."] * 8,
        "final_decision": ["yes" if r else "no" for r in REFERENCES],
    })
    predictions = {m: pd.DataFrame({
        "reference": ["Simplified Answer: yes\nDetailed Answer: Because."] * 8,
        "prediction": ["Simplified Answer: no\nDetailed Answer: It did not."] * 8,
        "yes_probability": p,
    }) for m, p in PROBABILITIES.items()}

    data = explorer_data(test, RUNS, predictions)
    html = render_explorer(data)

    assert sum(data["population"].values()) == 8
    assert {r["cat"] for r in data["rows"]} <= set(SAMPLE_SIZES)
    assert len(data["rows"]) == sum(min(n, data["population"][g]) for g, n in SAMPLE_SIZES.items())
    assert data == explorer_data(test, RUNS, predictions)  # Same seed, same sample.
    assert data["rows"][0]["m"]["lora"]["x"] == "It did not."
    assert "/*__DATA__*/" not in html and "Context <\\/script> here." in html
