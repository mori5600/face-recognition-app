from pathlib import Path

import pytest

from app.domain.results import is_failure, unwrap_success
from app.infra.app_paths import AppPaths
from app.infra.download_models import ModelSource, download_models


def test_download_models_rejects_hash_mismatch_and_deletes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_paths(tmp_path)
    payload = b"model-bytes"

    monkeypatch.setattr(
        "app.infra.download_models.MODEL_SOURCES",
        {
            "fake-model.bin": ModelSource(
                url="https://example.invalid/fake-model.bin",
                expected_size=len(payload),
                sha256="0" * 64,
            )
        },
    )
    monkeypatch.setattr(
        "app.infra.download_models.urlopen",
        lambda url: _FakeResponse(payload),
    )

    result = download_models(paths)

    assert is_failure(result)
    assert "expected hash" in result.message
    assert not (paths.models_dir / "fake-model.bin").exists()


def test_download_models_reuses_cached_verified_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_paths(tmp_path)
    payload = b"verified-model"
    destination = paths.models_dir / "fake-model.bin"

    monkeypatch.setattr(
        "app.infra.download_models.MODEL_SOURCES",
        {
            "fake-model.bin": ModelSource(
                url="https://example.invalid/fake-model.bin",
                expected_size=len(payload),
                sha256="3326723577b96bc7d05ecdb8a32ac917ce7eb053152f145b609635ebdbaf8d46",
            )
        },
    )
    monkeypatch.setattr(
        "app.infra.download_models.urlopen",
        lambda url: _FakeResponse(payload),
    )

    first_result = download_models(paths)
    assert not is_failure(first_result)
    assert unwrap_success(first_result) == (destination,)

    def fail_if_called(url: str) -> _FakeResponse:
        raise AssertionError(f"urlopen should not be called for cached file: {url}")

    monkeypatch.setattr(
        "app.infra.download_models.urlopen",
        fail_if_called,
    )

    second_result = download_models(paths)
    assert not is_failure(second_result)
    assert unwrap_success(second_result) == (destination,)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        _ = exc_type
        _ = exc
        _ = traceback

    def read(self) -> bytes:
        return self._payload


def _build_paths(root_dir: Path) -> AppPaths:
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    default_paths = AppPaths.default()
    return AppPaths(
        root_dir=root_dir,
        models_dir=models_dir,
        data_dir=data_dir,
        database_path=data_dir / "people.db",
        yunet_model_path=models_dir / "face_detection_yunet_2023mar.onnx",
        sface_model_path=models_dir / "face_recognition_sface_2021dec.onnx",
        mediapipe_face_landmarker_path=models_dir / "face_landmarker.task",
        sqlite_schema_path=default_paths.sqlite_schema_path,
    )
