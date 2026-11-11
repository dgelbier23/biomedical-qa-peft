import sys
import types
from pathlib import Path

import pytest

from biomedical_qa_peft.runtime import env


def test_describe_environment_locally() -> None:
    info = env.describe_environment()

    assert info["in_colab"] is False
    assert info["torch"]
    assert info["git_commit"]  # Tests run inside the repo, so a commit exists.
    assert isinstance(info["git_dirty"], bool)


def test_output_dir_locally_is_under_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(env, "REPO_ROOT", tmp_path)

    out_dir = env.get_output_dir("data")

    assert out_dir == tmp_path / "artifacts" / "data"
    assert out_dir.is_dir()


def test_output_dir_in_colab_mounts_drive_under_repo_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounted: list[str] = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mounted.append)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(env, "DRIVE_ROOT", tmp_path / "MyDrive")
    monkeypatch.setattr(env, "REPO_ROOT", tmp_path / "my-project")

    out_dir = env.get_output_dir("runs")

    assert mounted == [str(tmp_path)]
    assert out_dir == tmp_path / "MyDrive" / "my-project" / "runs"
    assert out_dir.is_dir()
