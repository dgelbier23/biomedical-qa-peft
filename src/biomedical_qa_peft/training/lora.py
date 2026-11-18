"""LoRA with PEFT on the 4-bit base model, as set up in the dissertation."""

from typing import Any

from biomedical_qa_peft.training.config import LoraSettings


def add_lora(model: Any, settings: LoraSettings) -> Any:
    """Wrap model with trainable LoRA adapters; the base model stays frozen.

    prepare_model_for_kbit_training casts the non-quantised weights (norms, embeddings)
    to float32, as in the original. Turns off the KV cache (training only) and lets
    gradients reach the inputs, which gradient checkpointing needs.
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    peft_model = get_peft_model(
        prepare_model_for_kbit_training(model),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings.r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            bias="none",
            target_modules=list(settings.target_modules),
        ),
    )
    peft_model.config.use_cache = False
    peft_model.enable_input_require_grads()
    return peft_model
