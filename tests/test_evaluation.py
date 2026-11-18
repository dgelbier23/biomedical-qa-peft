from pathlib import Path

import pytest
import torch

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.evaluation import metrics, scoring
from biomedical_qa_peft.evaluation.config import BASELINE_PROMPT, InferenceConfig
from biomedical_qa_peft.evaluation.inference import (
    ANSWER_PREFIX,
    build_prompt,
    extract_response,
    generate_responses,
    score_yes_no,
)
from biomedical_qa_peft.evaluation.metrics import (
    bootstrap_ci,
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
    # What the dissertation's no_repeat_ngram_size=2 forced the models to write:
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


def test_score_yes_no_reads_next_token_probabilities_independent_of_batching(
    tiny_llama, tokenizer
) -> None:
    model = tiny_llama()
    prompts = ["is the drug safe", "it was tested in mice", "the answer"]
    with pytest.raises(ValueError):  # right padding would read a pad token's logits
        score_yes_no(model, tokenizer, prompts, InferenceConfig(num_workers=0))
    tokenizer.padding_side = "left"  # as load_base_model sets it
    one = score_yes_no(model, tokenizer, prompts, InferenceConfig(batch_size=1, num_workers=0))
    padded = score_yes_no(model, tokenizer, prompts, InferenceConfig(batch_size=3, num_workers=0))

    # By hand for the first prompt: next-token probabilities after "Simplified Answer:".
    ids = tokenizer(prompts[0] + ANSWER_PREFIX, return_tensors="pt")["input_ids"]
    p = model(input_ids=ids).logits[0, -1].softmax(-1)
    yes, no = (sum(p[tokenizer.convert_tokens_to_ids(w)] for w in pair)
               for pair in (("yes", "Yes"), ("no", "No")))

    assert one[0] == pytest.approx((yes / (yes + no)).item(), abs=1e-6)
    assert one == pytest.approx(padded, abs=1e-5)  # left padding does not change answers


def test_score_short_answers_thresholds_yes_probability() -> None:
    references = ["Simplified Answer: no", "Simplified Answer: yes", "Simplified Answer: yes"]

    result = score_short_answers([0.2, 0.9, 0.4], references)

    assert result["predictions"] == [0, 1, 0]
    assert result["references"] == [0, 1, 1]
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_nli_results_pairs_reference_premise_with_prediction_in_batches() -> None:
    seen: list[list[dict]] = []

    def nli(inputs: list[dict]) -> list[dict]:
        seen.append(inputs)
        return [{"label": "ENTAILMENT" if i["text_pair"] == "same" else "neutral", "score": 0.5}
                for i in inputs]

    result = nli_results(nli, ["same", "b", "c"], ["r1", "r2", "r3"], batch_size=2)

    assert seen == [[{"text": "r1", "text_pair": "same"}, {"text": "r2", "text_pair": "b"}],
                    [{"text": "r3", "text_pair": "c"}]]
    assert result["labels"] == ["ENTAILMENT", "NEUTRAL", "NEUTRAL"]
    assert result["counts"] == {"ENTAILMENT": 1, "CONTRADICTION": 0, "NEUTRAL": 2}


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable() -> None:
    values = __import__("numpy").array([0, 1] * 50)

    ci = bootstrap_ci(lambda i: values[i].mean(), len(values), samples=500, seed=0)

    assert ci[0] < 0.5 < ci[1] and ci[1] - ci[0] < 0.3
    assert ci == bootstrap_ci(lambda i: values[i].mean(), len(values), samples=500, seed=0)


class BertScore:
    def __init__(self, score: float) -> None:
        self.score = score

    def compute(self, predictions, references, model_type):
        return {"f1": [self.score] * len(predictions)}


def test_evaluate_references_against_themselves_scores_perfectly(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # METEOR needs NLTK data and BERTScore a download; both are stubbed here.
    monkeypatch.setattr(scoring, "meteor_scores", lambda p, r: [1.0] * len(p))
    scorers = scoring.Scorers(
        nli_pipeline=lambda inputs: [{"label": "ENTAILMENT", "score": 0.9} for _ in inputs],
        bertscore=BertScore(1.0),
    )
    references = [REFERENCE, REFERENCE.replace("no", "yes", 1)]
    config = InferenceConfig(bootstrap_samples=50)

    results = scoring.evaluate_responses([0.1, 0.9], references, references, scorers, config)

    assert scoring.summarise(results) == {
        "accuracy": 100.0, "precision": 100.0, "recall": 100.0, "f1": 100.0,
        "entailment": 100.0, "contradiction": 0.0, "neutral": 0.0,
        "meteor": 1.0, "bertscore": 1.0, "format_adherence": 100.0,
    }
    assert scoring.confidence_intervals(results, config)["accuracy"] == [100.0, 100.0]
    with pytest.raises(ValueError):
        scoring.evaluate_responses([0.1], references, references, scorers, config)


def test_run_method_saves_then_reuses_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scoring, "meteor_scores", lambda p, r: [0.5] * len(p))
    calls = []
    monkeypatch.setattr(
        scoring, "generate_responses",
        lambda model, tok, prompts, config: calls.append(prompts) or (["Answer: no"] * len(prompts), 1.0),
    )
    monkeypatch.setattr(scoring, "score_yes_no", lambda model, tok, prompts, config: [0.3] * len(prompts))
    scorers = scoring.Scorers(
        nli_pipeline=lambda inputs: [{"label": "NEUTRAL", "score": 0.9} for _ in inputs],
        bertscore=BertScore(0.5),
    )
    args = (tmp_path, None, None, [FORMATTED, FORMATTED], BASELINE_PROMPT, scorers,
            InferenceConfig(bootstrap_samples=50))

    first = scoring.run_method(*args, method="baseline")
    second = scoring.run_method(*args, method="baseline")

    assert len(calls) == 1  # second run reused predictions.jsonl
    assert first == second == scoring.load_summary(tmp_path)
    assert first["accuracy"] == 100.0  # P(yes) 0.3 -> "no", the reference answer
    assert scoring.load_predictions(tmp_path)["yes_probability"].tolist() == pytest.approx([0.3, 0.3])
    record = (tmp_path / "results.json").read_text()
    assert '"method": "baseline"' in record and '"confidence_intervals"' in record
    assert '"gpu":' in record  # environment is recorded
