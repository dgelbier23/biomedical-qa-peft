"""A tiny random Llama and an in-memory tokenizer: nothing is downloaded."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

# " Yes" and " No" exist so the yes/no scorer finds one token for each spelling.
WORDS = "is the drug safe yes no answer it was tested in mice Yes No".split()
LAYERS, HIDDEN, HEADS, KV_HEADS = 2, 64, 4, 2


@pytest.fixture
def tokenizer() -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>"
    )


@pytest.fixture
def tiny_llama() -> Callable[[], Any]:
    """Builds a fresh, identically initialised 2-layer Llama on each call."""
    from transformers import LlamaConfig, LlamaForCausalLM

    def build() -> Any:
        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(
            vocab_size=len(WORDS) + 3, hidden_size=HIDDEN, intermediate_size=128,
            num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
            max_position_embeddings=128, pad_token_id=0,
        ))

    return build
