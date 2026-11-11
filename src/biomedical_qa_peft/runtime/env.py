"""Runtime environment: Colab detection, environment description, output folders."""

import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/biomedical_qa_peft/runtime/env.py -> repo root
DRIVE_ROOT = Path("/content/drive/MyDrive")


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def _git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip()


def describe_environment() -> dict[str, str | bool | None]:
    """Versions, hardware and code version, for printing and saving with each run.

    ``git_dirty`` is True when there are uncommitted changes, meaning
    ``git_commit`` alone does not fully describe the code that ran.
    """
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "in_colab": in_colab(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def get_output_dir(kind: str) -> Path:
    """Folder for one kind of generated output (e.g. "data", "runs"), created if missing.

    In Colab: mounts Google Drive and returns ``MyDrive/<repo name>/<kind>``, so outputs
    survive the runtime and stay separate per project. Locally:
    ``<repo>/artifacts/<kind>`` (ignored by Git).
    """
    if in_colab():
        from google.colab import drive

        drive.mount(str(DRIVE_ROOT.parent))
        out_dir = DRIVE_ROOT / REPO_ROOT.name / kind
    else:
        out_dir = REPO_ROOT / "artifacts" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
