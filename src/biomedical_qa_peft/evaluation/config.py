"""Inference and evaluation settings, and the system prompts.

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

# Candidates for the prompt-engineering method, from the dissertation's prompt search.
# notebooks/03_prompt_search.ipynb picks one on the validation split.
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


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = DataConfig.tokenizer_name
    model_revision: str = DataConfig.tokenizer_revision
    pad_token: str = "<|pad|>"
    # Long answers: greedy decoding, so a run is repeatable. The dissertation sampled
    # (temperature 0.7) with no_repeat_ngram_size=2, which banned the second "Answer:"
    # and made the required format impossible. Its repetition_penalty=1.2 is also off:
    # it penalises " no", which the prompt's "{yes or no}" contains, but not " yes", so
    # models that write lowercase answers wrote "yes" far too often.
    max_new_tokens: int = 150
    repetition_penalty: float = 1.0
    # Seeds the added pad token's embedding (see inference.load_base_model).
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    nli_model: str = "FacebookAI/roberta-large-mnli"
    nli_batch_size: int = 16
    bertscore_model: str = "distilbert-base-uncased"
    # 95% confidence intervals: resamples of the test rows.
    bootstrap_samples: int = 1000
