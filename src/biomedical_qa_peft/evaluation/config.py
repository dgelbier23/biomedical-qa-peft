"""Inference and evaluation settings, prompts, and the dissertation's reported results.

Prompts are copied exactly from the original code, including leading and trailing
spaces. Candidates 1 and 2 end a line with a space, written as \x20 so editors
cannot strip it.
"""

from dataclasses import dataclass

from biomedical_qa_peft.data.config import DataConfig

# System prompt for the baseline, and for prefix tuning and LoRA at test time.
BASELINE_PROMPT = """ You are a medical AI assistant. Format your response like this:
Simplified Answer: {yes or no}
Detailed Answer: {answer explanation}
"""

# Prompt search: round 1 (candidates 1-3) and round 2 (candidates 4-5), each on the
# first 100 test rows.
CANDIDATE_PROMPTS = (
    """You are a biomedical AI assistant answering academic questions.
Your response format is:
***
Simplified Answer: {yes or no}\x20
Detailed Answer: {contextual explanation}
***""",
    """You are a biomedical AI assistant answering academic medical
questions.
Only use the information provided in the context.
Do not use outside knowledge

Please format your response in the following structure:
***
Simplified Answer: {yes or no}\x20
Detailed Answer: {a clear explaination grounded in the given
context, written in an academic tone}
***
Ensure your explaination is relevant to the question and based
only on the user-provided context. Be concise and avoid
unnecessary elaboration.""",
    """You are a specialized biomedical AI assistant.
Your role is to answer academic biomedical questions using only
the context provided by the user.
Your response must strictly follow this format:
***
Simplified Answer: {yes or no only}
Detailed Answer: {a clear, academic, context-grounded explanation}
***
Rules:
- Only use the context provided. Do not rely on external knowledge.
- Do not include disclaimers, citations, or speculative content.
- Do not change the output format or capitalization.
- Do not invent content beyond the provided context.
- If the context is insufficient, infer the most probable answer
  using only what is available.""",
    """You are a specialized biomedical AI assistant.
Your role is to answer academic biomedical questions using only
the context provided by the user.

Your response must strictly follow this format:
[ANSWER BEGIN]
Simplified Answer: {yes or no only}
Detailed Answer: {a clear, academic, context-grounded explanation}
[ANSWER END]

Rules:
- Only use the context provided. Do not rely on external knowledge.
- Do not include disclaimers, citations, or speculative content.
- Do not change the output format or capitalization.
- Do not invent content beyond the provided context.
- If the context is insufficient, infer the most probable answer
using only what is available.

Example:
Question: Does autophagy inhibition reduce cell injury during
caerulein-induced pancreatitis?
Context: Mice lacking Atg5 showed reduced pancreatic edema and
serum amylase levels after caerulein injection.
[ANSWER BEGIN]
Simplified Answer: yes
Detailed Answer: Inhibition of autophagy by Atg5 deletion reduces
pancreatic injury markers following caerulein stimulation,
suggesting a protective effect.
[ANSWER END]

Example:
Question: Does inhibition of CXCL2 reduce intracellular calcium oscillations in acinar cells?
Context: Treatment with a CXCL2 neutralizing antibody did not affect CCK-induced calcium oscillation amplitude or frequency in acinar cells.
[ANSWER BEGIN]
Simplified Answer: no
Detailed Answer: CXCL2 inhibition does not alter intracellular calcium oscillations during CCK stimulation, indicating no direct regulatory role.
[ANSWER END]
""",
    """You are a specialized biomedical AI assistant.
Your role is to answer academic medical questions using only the
context provided by the user.
You’ve answered many similar academic questions before and should
follow the same structure and reasoning style.
Your response must strictly follow this format:
[ANSWER BEGIN]
Simplified Answer: {yes or no only}
Detailed Answer: {a clear, academic, context-grounded explanation}
[ANSWER END]

Rules:
- Only use the context provided. Do not rely on external knowledge.
- Do not include disclaimers, citations, or speculative content.
- Do not change the output format or capitalization.
- Do not invent content beyond the provided context.
- If the context is insufficient, infer the most probable answer
using only what is available.
Be concise, accurate, and consistent in structure across all
responses.""",
)

# The prompt chosen for the prompt-engineering method: candidate 5 with a leading space.
OPTIMISED_PROMPT = " " + CANDIDATE_PROMPTS[4]

# System prompt for the yes/no judge used when the regex finds no short answer.
# The missing space before "Extract" is in the original.
JUDGE_PROMPT = (
    "Analyze the given text carefully. "
    "The text provided is a direct response to a question. Your task is to assess "
    "whether the overall response to the question was 'Yes' or 'No'."
    "Extract only an explicitly stated answer: Yes, No, or Unclear. "
    "Do NOT explain or provide reasoning. "
    "Do NOT interpret missing or ambiguous information: answer 'Unclear' if unsure. "
    "Respond with one single word and nothing else. "
    "Inaccurate or verbose outputs will be treated as incorrect."
)


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = DataConfig.tokenizer_name
    model_revision: str = DataConfig.tokenizer_revision
    pad_token: str = "<|pad|>"
    # Decoding settings from the original. Sampling was on through Llama's own
    # generation_config; here it is explicit. no_repeat_ngram_size=2 bans the second
    # "Answer:", so outputs can never match the required format (fixed in a later stage).
    max_new_tokens: int = 150
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 2
    # New: the original sampling was unseeded, so it cannot be repeated exactly.
    seed: int = 42
    # Throughput settings from the original.
    batch_size: int = 64
    num_workers: int = 4
    # Scorers.
    judge_batch_size: int = 16
    nli_model: str = "FacebookAI/roberta-large-mnli"
    nli_batch_size: int = 16
    bertscore_model: str = "distilbert-base-uncased"


# Read from the result figures in the original Main_Inference_comparison.ipynb
# (1,000 test rows, one unseeded run each). Percentages are 0-100.
DISSERTATION_RESULTS = {
    "baseline": {
        "accuracy": 82.60, "precision": 84.25, "recall": 82.60, "f1": 82.45,
        "entailment": 16.6, "contradiction": 9.5, "neutral": 73.9,
        "meteor": 0.23241, "bertscore": 0.78310, "format_adherence": 0.0,
    },
    "prompt_engineering": {
        "accuracy": 76.50, "precision": 77.08, "recall": 76.50, "f1": 76.65,
        "entailment": 19.6, "contradiction": 17.9, "neutral": 62.5,
        "meteor": 0.24394, "bertscore": 0.77747, "format_adherence": 0.0,
    },
    "prefix_tuning": {
        "accuracy": 69.90, "precision": 71.49, "recall": 69.90, "f1": 70.33,
        "entailment": 6.6, "contradiction": 26.9, "neutral": 66.5,
        "meteor": 0.23649, "bertscore": 0.78555, "format_adherence": 0.0,
    },
    "lora": {
        "accuracy": 65.50, "precision": 67.71, "recall": 65.50, "f1": 65.62,
        "entailment": 5.3, "contradiction": 24.2, "neutral": 70.5,
        "meteor": 0.23118, "bertscore": 0.78820, "format_adherence": 0.0,
    },
}

# Agreed tolerance for a sampled re-run to count as reproducing a result:
# about the 95% range between two runs of 1,000 rows.
TOLERANCES = {"percent": 3.5, "meteor": 0.01, "bertscore": 0.01}
