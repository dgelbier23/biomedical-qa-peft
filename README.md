# biomedical-qa-peft

Comparing parameter-efficient fine-tuning methods for biomedical research question
answering: **Llama 3.2 3B Instruct** on **PubMedQA**, adapted with prompt engineering,
prefix tuning and LoRA, and scored with a custom evaluation framework.

A clean, reproducible rebuild of my BSc dissertation, *Comparing Parameter-Efficient
Fine-Tuning for Biomedical Research QA*.

> **Status: migration in progress.** The original dissertation code is being ported
> stage by stage. Behaviour is reproduced faithfully first, then improved. Results
> will be added here only once they have been reproduced in this repo.

## Pipeline

1. **Data** (`biomedical_qa_peft.data`, defaults in `DataConfig`):

   | Step | Rule | Rows after |
   |---|---|---|
   | Load | PubMedQA `pqa_artificial`; context passages joined | 211,269 |
   | Validity filter | Context and long answer: ≥30 characters, ≥5 words, contain a "."; question contains "?" | 210,798 |
   | Remove duplicates | Same question and context | 210,751 |
   | Balance | 5,000 "yes" + 5,000 "no", seed 42 | 10,000 |
   | Format | Llama 3 chat template (system / user / assistant) | – |
   | Length filter | ≤ 1,024 tokens | 9,998 |
   | Split | Stratified 80/10/10, seed 42 | 7,998 / 1,000 / 1,000 |

   Row counts are from the original dissertation run. **Verified:** splits regenerated
   with this code are byte-identical to the dissertation's (2026-11-11). The dataset
   and tokenizer revisions are pinned in `DataConfig` so they stay that way.
2. **Adaptation**: each method adapts the same base model.
   - Prompt engineering: search for the best prompt, with no weight updates.
   - Prefix tuning: train learned prefix vectors (PEFT).
   - LoRA: train low-rank adapters on a 4-bit quantised base model (PEFT + bitsandbytes).
3. **Evaluation**: run inference with every method on the same fixed test split and
   decoding settings, then score it with the evaluation framework (classification
   metrics, METEOR, BERTScore and NLI-based checks).

## Repository layout

```
src/biomedical_qa_peft/
  data/          PubMedQA loading, preprocessing, splitting, prompt templates
  training/      prefix tuning and LoRA
  evaluation/    inference and the evaluation metrics/manager
  runtime/       environment detection, devices, seeding
notebooks/       thin notebooks that call the package (run locally or in Colab)
tests/           CPU-only tests; never download the gated model
artifacts/       local run outputs (ignored by Git)
```

## Requirements

- **Hugging Face access to Llama 3.2**: it is a gated model. Request access on its
  Hugging Face model page.
- **A CUDA GPU for training and LLM inference**: use Google Colab. Data processing,
  metrics and tests run on CPU.
- **Hugging Face and Weights & Biases keys**: these are read from Colab secrets or
  environment variables, never written in notebook code.

## Setup

From the repo folder in PowerShell (no need to activate the environment):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q
```

In VS Code, select the `.venv` kernel for notebooks.

## Everyday workflow

1. Edit code in VS Code and run the tests.
2. Commit and push to GitHub.
3. In Colab, use Runtime > Restart session, then run all cells.
4. Results save to Google Drive. Download a run folder into `artifacts/runs/` to
   inspect it locally.

## Notebooks

Notebooks run locally (`.venv` kernel) or in Google Colab, unchanged.

- The first code cell of every notebook is the same bootstrap cell. In Colab it
  clones the latest `main` from GitHub using the `GITHUB_TOKEN` Colab secret
  (fine-grained, read-only). Locally it does nothing.
- In Colab, open notebooks via File > Open notebook > GitHub, pick a GPU runtime,
  and treat them as read-only: edit in VS Code, push, then re-run in Colab.
- Generated outputs go to `get_output_dir(kind)`, for example `"data"` or `"runs"`. That is
  `MyDrive/<repo name>/<kind>` in Colab and `artifacts/<kind>` locally (ignored by Git).

| Notebook | Runs on | Purpose |
|---|---|---|
| `01_environment_check` | local or Colab | versions, device, output folder |
| `02_prepare_data` | Colab (CPU is enough) | build the PubMedQA train/validation/test splits |

## Reproducibility

### Dependency versions

| Environment | File | torch | Hugging Face stack |
|---|---|---|---|
| Local (CPU) | `requirements.txt` | pinned to Colab's version | dissertation versions |
| Colab (GPU) | `requirements-colab.txt` | Colab's preinstalled build | dissertation versions |

The Hugging Face stack is pinned to the versions the dissertation used, because
newer releases can change results:

| Package | Version |
|---|---|
| transformers | 4.47.0 |
| peft | 0.15.2 |
| accelerate | 1.6.0 |
| bitsandbytes | 0.45.5 |
| datasets | 3.5.1 |
| evaluate | 0.4.3 |
| bert-score | 0.3.13 |

The dissertation itself ran on torch 2.7.0. This repo uses Colab's current torch
instead, because reinstalling torch in Colab would replace its CUDA build.

- Never `pip install torch` in Colab. Use the version Colab provides.
- When Colab's torch changes (shown by `notebooks/01_environment_check.ipynb`),
  update the local pin and reinstall:

```powershell
pip install "torch==<version>" --index-url https://download.pytorch.org/whl/cpu
```

### Experiments

Every result will be recorded with its seeds, prompt template, decoding settings and
adapter checkpoint. All methods are compared on the same fixed test split.

## Data, models and licences

- **No data, models or results are committed.** This covers the PubMedQA splits (`.jsonl`),
  base models, adapters, checkpoints, W&B runs and result pickles. The splits are
  regenerated from the PubMedQA source.
- **Llama 3.2 is released under the Llama licence**, so its weights and tokenizer files
  are never redistributed here. Trained adapters are not published.
- **Never commit secrets.** That includes API keys, tokens, `.env` and `.netrc` files.
