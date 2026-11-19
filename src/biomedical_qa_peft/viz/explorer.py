"""The answer explorer: one self-contained HTML page comparing the four methods' answers
on a sample of test questions (published from docs/ with GitHub Pages)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomedical_qa_peft.evaluation.metrics import detailed_answer, has_valid_format, stated_answer

TEMPLATE = Path(__file__).with_name("explorer.html")
# Questions per outcome. Disagreements are over-sampled; the page states the full-set counts.
SAMPLE_SIZES = {"adapters_fix": 25, "adapters_break": 15, "mixed": 15, "all_wrong": 15, "all_right": 30}


def outcome(correct: dict[str, bool]) -> str:
    """Which outcome group a question falls in, from each method's yes/no correctness."""
    if all(correct.values()):
        return "all_right"
    if not any(correct.values()):
        return "all_wrong"
    adapters = correct["prefix_tuning"], correct["lora"]
    if not correct["baseline"] and all(adapters):
        return "adapters_fix"
    if correct["baseline"] and not any(adapters):
        return "adapters_break"
    return "mixed"


def explorer_data(test: pd.DataFrame, runs: dict[str, dict], predictions: dict[str, pd.DataFrame],
                  seed: int = 42) -> dict[str, Any]:
    """A stratified sample of test questions (SAMPLE_SIZES, fewer if a group is smaller)
    with every method's answer and scores, and the full-set count of each group.

    ``test`` is the test split; ``runs`` and ``predictions`` map a method key to its
    ``results.json`` and ``predictions.jsonl``, in test order.
    """
    yes = (test["final_decision"] == "yes").to_numpy()
    groups = [outcome({m: (p["yes_probability"].iloc[i] > 0.5) == yes[i] for m, p in predictions.items()})
              for i in range(len(test))]
    rng = np.random.default_rng(seed)
    chosen = []
    for group, size in SAMPLE_SIZES.items():
        pool = [i for i, g in enumerate(groups) if g == group]
        chosen += sorted(rng.choice(pool, size=min(size, len(pool)), replace=False).tolist()) if pool else []

    rows = []
    for i in chosen:
        answers = {}
        for m, frame in predictions.items():
            raw = frame["prediction"].iloc[i]
            long_answer = runs[m]["results"]["long_answer"]
            answers[m] = {
                "p": round(float(frame["yes_probability"].iloc[i]), 4),
                "w": stated_answer(raw),
                "nli": long_answer["nli"]["labels"][i].lower(),
                "bs": round(long_answer["bertscore"][i], 3),
                "met": round(long_answer["meteor"][i], 3),
                "fmt": has_valid_format(raw),
                "x": detailed_answer(raw),
                "raw": raw,
            }
        reference = next(iter(predictions.values()))["reference"].iloc[i]
        rows.append({"i": i, "q": test["question"].iloc[i], "ctx": test["cleaned_context"].iloc[i],
                     "ref": test["final_decision"].iloc[i], "refx": detailed_answer(reference),
                     "cat": groups[i], "m": answers})
    return {"rows": rows, "population": {group: groups.count(group) for group in SAMPLE_SIZES}}


def render_explorer(data: dict[str, Any]) -> str:
    """The explorer page with ``data`` embedded (no external data files)."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # Cannot close the script tag.
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
