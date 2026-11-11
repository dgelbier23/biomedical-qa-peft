"""Model loading, prompt building and batched generation, ported from the dissertation."""

import re
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from biomedical_qa_peft.data.config import SYSTEM_PROMPT
from biomedical_qa_peft.evaluation.config import InferenceConfig
from biomedical_qa_peft.runtime.device import set_seed

ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"
END_TAG = "<|eot_id|>"
_DATA_SYSTEM_HEADER = re.compile(
    re.escape("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
    + r"\s*"
    + re.escape(SYSTEM_PROMPT)
)


def load_base_model(config: InferenceConfig) -> tuple[Any, Any]:
    """Llama in 4-bit NF4 with an added <|pad|> token and left padding, as in the original.

    Needs a CUDA GPU (bitsandbytes). Adding the pad token resizes the embeddings;
    the new row is initialised randomly, so seed before calling this.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.model_revision, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": config.pad_token})
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"  # Required for batched generation.
    return model, tokenizer


def build_prompt(formatted_text: str, system_prompt: str | None) -> tuple[str, str]:
    """Split a training-format row into (generation prompt, reference answer).

    With system_prompt, the dataset's system turn is replaced as in the original:
    the replacement drops <|begin_of_text|> (the tokenizer adds it back) and puts a
    space, not a newline, after the header.
    """
    text = formatted_text.replace("<|pad|>", "")
    if system_prompt is not None:
        text = _DATA_SYSTEM_HEADER.sub(
            f"<|start_header_id|>system<|end_header_id|> {system_prompt}", text
        )
    before, after = text.split(ASSISTANT_TAG, 1)
    return before + ASSISTANT_TAG, after.replace(END_TAG, "").strip()


def extract_response(decoded: str) -> str:
    """The assistant's answer from a decoded generation (special tokens kept)."""
    start = decoded.find(ASSISTANT_TAG)
    if start == -1:
        # The original silently skipped these, which misaligns every later
        # prediction with its reference.
        raise ValueError(f"No assistant header in generated text: {decoded[:200]!r}")
    if not decoded.endswith(END_TAG):
        decoded += END_TAG
    start += len(ASSISTANT_TAG)
    return decoded[start:decoded.find(END_TAG, start)].strip()


def generate_responses(
    model: Any, tokenizer: Any, prompts: list[str], config: InferenceConfig
) -> tuple[list[str], float]:
    """Generate one answer per prompt, in order. Returns (answers, seconds taken).

    Seeds with config.seed first, so a run can be repeated.
    """
    from accelerate import Accelerator
    from transformers import DataCollatorWithPadding

    set_seed(config.seed)
    encodings = [
        tokenizer(p, add_special_tokens=True, truncation=False, padding=False) for p in prompts
    ]
    workers = config.num_workers
    loader = DataLoader(
        encodings,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    model.eval()
    model, loader = Accelerator().prepare(model, loader)

    responses: list[str] = []
    start = time.time()
    for batch in tqdm(loader, desc="Generating"):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses += [
            extract_response(text)
            for text in tokenizer.batch_decode(outputs, skip_special_tokens=False)
        ]
    return responses, time.time() - start
