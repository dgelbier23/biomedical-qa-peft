from pathlib import Path

import pytest
import torch

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.evaluation import metrics, scoring
from biomedical_qa_peft.evaluation.config import (
    BASELINE_PROMPT,
    DISSERTATION_RESULTS,
    InferenceConfig,
)
from biomedical_qa_peft.evaluation.inference import (
    build_prompt,
    extract_response,
    generate_responses,
)
from biomedical_qa_peft.evaluation.metrics import (
    extract_short_answer,
    format_stats,
    has_valid_format,
    nli_results,
    score_short_answers,
)

REFERENCE = "Simplified Answer: no\nDetailed Answer: Copeptin did not differ between groups."
# Layout of a row's formatted_text in the saved splits.
FORMATTED = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Question: Is copeptin elevated?\nContext: Levels were measured.<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n{REFERENCE}<|eot_id|>"
)
USER_AND_ASSISTANT = (
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "Question: Is copeptin elevated?\nContext: Levels were measured.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)


def test_build_prompt_replaces_system_prompt_and_returns_reference() -> None:
    prompt, reference = build_prompt(FORMATTED, BASELINE_PROMPT)

    assert reference == REFERENCE
    assert prompt == (
        f"<|start_header_id|>system<|end_header_id|> {BASELINE_PROMPT}{USER_AND_ASSISTANT}"
    )


def test_build_prompt_without_system_prompt_keeps_dataset_prompt() -> None:
    prompt, _ = build_prompt(FORMATTED, None)

    assert prompt == FORMATTED.split(REFERENCE)[0].rstrip("\n")


def test_extract_response_takes_text_between_assistant_header_and_end() -> None:
    prompt, _ = build_prompt(FORMATTED, BASELINE_PROMPT)

    assert extract_response(f"<|pad|>{prompt}\n\n Answer: yes <|eot_id|><|pad|>") == "Answer: yes"
    assert extract_response(f"{prompt} cut off at max tokens") == "cut off at max tokens"
    with pytest.raises(ValueError):
        extract_response("no assistant header")


class CharTokenizer:
    """One token per character; id 0 is padding. Enough for the collator and decoding."""

    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

    def pad(self, features: list[dict], return_tensors: str, **kwargs) -> dict:
        width = max(len(f["input_ids"]) for f in features)
        pad = [width - len(f["input_ids"]) for f in features]  # left padding
        return {
            key: torch.tensor([[0] * n + f[key] for n, f in zip(pad, features)])
            for key in ("input_ids", "attention_mask")
        }

    def batch_decode(self, rows: torch.Tensor, skip_special_tokens: bool) -> list[str]:
        return ["".join(chr(i) for i in row.tolist() if i > 1) for row in rows]


class EchoModel(torch.nn.Module):
    """Answers every prompt with its own prompt length, so outputs reveal ordering."""

    device = torch.device("cpu")

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        answers = [f" n={int(m.sum())}<|eot_id|>" for m in attention_mask]
        width = max(len(a) for a in answers)
        new = torch.tensor([[0] * (width - len(a)) + [ord(c) for c in a] for a in answers])
        return torch.cat([input_ids, new], dim=1)


def test_generate_responses_keeps_prompt_order_across_batches() -> None:
    prompts = [f"q{'?' * i}<|start_header_id|>assistant<|end_header_id|>" for i in (3, 1, 2)]
    config = InferenceConfig(batch_size=2, num_workers=0)

    answers, seconds = generate_responses(EchoModel(), CharTokenizer(), prompts, config)

    assert answers == [f"n={len(p)}" for p in prompts]
    assert seconds >= 0


def test_format_check_accepts_reference_layout_only() -> None:
    assert has_valid_format(REFERENCE)
    # What no_repeat_ngram_size=2 forces the models to write instead:
    assert not has_valid_format("Simplified Answer : yes\nDetailedAnswer: Because.")
    assert format_stats([REFERENCE, "yes"]) == {
        "total_outputs": 2, "valid_format_count": 1,
        "invalid_format_count": 1, "valid_format_percentage": 50.0,
    }


@pytest.mark.parametrize(("text", "expected"), [
    ("Simplified Answer: Yes\nDetailed Answer: ...", "yes"),
    ("**Simplified Answer:** No", "not_found"),  # markdown bold breaks the regex
    ("The answer is unclear.", "unclear"),
    ("Probably yes.", "not_found"),
])
def test_extract_short_answer(text: str, expected: str) -> None:
    assert extract_short_answer(text) == expected


def test_score_short_answers_judges_only_unreadable_answers_in_batches() -> None:
    judged_batches: list[list[str]] = []

    def judge(texts: list[str]) -> list[str]:
        judged_batches.append(texts)
        return [" Yes " if "think so" in t else "maybe" for t in texts]

    predictions = ["Answer: no", "I think so", "No idea", "I think so too"]
    references = ["Simplified Answer: no", "Simplified Answer: yes",
                  "Simplified Answer: no", "Simplified Answer: yes"]

    result = score_short_answers(predictions, references, judge, batch_size=2)

    assert judged_batches == [["I think so", "No idea"], ["I think so too"]]
    assert result["predictions"] == [0, 1, 2, 1]  # judge's "maybe" counts as unclear
    assert result["references"] == [0, 1, 0, 1]
    assert result["accuracy"] == 0.75


def test_nli_results_pairs_reference_then_prediction_in_batches() -> None:
    seen: list[list[str]] = []

    def nli(inputs: list[str]) -> list[dict]:
        seen.append(inputs)
        return [{"label": "ENTAILMENT" if "same" in i else "neutral", "score": 0.5} for i in inputs]

    result = nli_results(nli, ["same", "b", "c"], ["r1", "r2", "r3"], batch_size=2)

    assert seen == [["r1 [SEP] same", "r2 [SEP] b"], ["r3 [SEP] c"]]
    assert result["counts"] == {"ENTAILMENT": 1, "CONTRADICTION": 0, "NEUTRAL": 2}


def test_evaluate_references_against_themselves_scores_perfectly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # METEOR needs NLTK data and BERTScore a download; both are stubbed here.
    monkeypatch.setattr(scoring, "meteor", lambda p, r: 1.0)

    class BertScore:
        def compute(self, predictions, references, model_type):
            return {"f1": [1.0] * len(predictions)}

    scorers = scoring.Scorers(
        judge=lambda texts: pytest.fail("judge should not be needed"),
        nli_pipeline=lambda inputs: [{"label": "ENTAILMENT", "score": 0.9} for _ in inputs],
        bertscore=BertScore(),
    )
    references = [REFERENCE, REFERENCE.replace("no", "yes", 1)]

    results = scoring.evaluate_responses(references, references, scorers, InferenceConfig())
    summary = scoring.summarise(results)

    assert summary == {
        "accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0,
        "entailment": 100.0, "contradiction": 0.0, "neutral": 0.0,
        "meteor": 1.0, "bertscore": 1.0, "format_adherence": 100.0,
    }
    with pytest.raises(ValueError):
        scoring.evaluate_responses(references, references[:1], scorers, InferenceConfig())

    scoring.save_predictions(tmp_path, references[::-1], references)
    assert scoring.load_predictions(tmp_path) == (references[::-1], references)


def test_run_method_saves_then_reuses_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scoring, "meteor", lambda p, r: 0.5)
    calls = []
    monkeypatch.setattr(
        scoring, "generate_responses",
        lambda model, tok, prompts, config: calls.append(prompts) or (["Answer: no"] * len(prompts), 1.0),
    )

    class BertScore:
        def compute(self, predictions, references, model_type):
            return {"f1": [0.5] * len(predictions)}

    scorers = scoring.Scorers(
        judge=lambda texts: ["unclear"] * len(texts),
        nli_pipeline=lambda inputs: [{"label": "NEUTRAL", "score": 0.9} for _ in inputs],
        bertscore=BertScore(),
    )
    args = (tmp_path, None, None, [FORMATTED, FORMATTED], BASELINE_PROMPT, scorers, InferenceConfig())

    first = scoring.run_method(*args, method="baseline")
    second = scoring.run_method(*args, method="baseline")

    assert len(calls) == 1  # second run reused predictions.jsonl
    assert first == second and first["accuracy"] == 100.0  # reference answer is "no"
    record = (tmp_path / "results.json").read_text()
    assert '"method": "baseline"' in record and '"seed": 42' in record
    assert '"gpu":' in record  # environment is recorded


def test_compare_to_dissertation_uses_tolerances(capsys: pytest.CaptureFixture) -> None:
    summary = dict(DISSERTATION_RESULTS["baseline"])
    summary["accuracy"] += 3.4   # within ±3.5 points
    summary["meteor"] += 0.02    # outside ±0.01

    within = scoring.compare_to_dissertation(summary, "baseline")

    assert within["accuracy"] and not within["meteor"]
    assert sum(within.values()) == len(within) - 1
    assert "DIFF" in capsys.readouterr().out
