"""Prefix tuning with PEFT, as set up in the dissertation."""

from typing import Any

from biomedical_qa_peft.training.config import PrefixConfig


def add_prefix(model: Any, config: PrefixConfig) -> Any:
    """Wrap model with trainable prefix vectors; the base model stays frozen.

    Turns off the KV cache (training only; turn it back on for generation) and lets
    gradients reach the inputs.
    """
    from peft import PrefixTuningConfig, TaskType, get_peft_model

    peft_model = get_peft_model(
        model,
        PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_hidden_size=model.config.hidden_size,
            prefix_projection=config.prefix_projection,
        ),
    )
    peft_model.config.use_cache = False
    peft_model.enable_input_require_grads()
    return peft_model
