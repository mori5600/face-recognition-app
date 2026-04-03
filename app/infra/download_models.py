from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from app.domain.errors import InfraError
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.infra.app_paths import AppPaths


@dataclass(frozen=True)
class ModelSource:
    url: str
    expected_size: int


MODEL_SOURCES = {
    "face_detection_yunet_2023mar.onnx": ModelSource(
        url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true",
        expected_size=232589,
    ),
    "face_recognition_sface_2021dec.onnx": ModelSource(
        url="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
        expected_size=38696353,
    ),
}


def download_models(
    paths: AppPaths | None = None,
) -> Result[tuple[Path, ...], InfraError]:
    resolved_paths = paths or AppPaths.default()
    resolved_paths.models_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths.data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    for file_name, source in MODEL_SOURCES.items():
        destination = resolved_paths.models_dir / file_name
        if destination.exists() and destination.stat().st_size == source.expected_size:
            downloaded_files.append(destination)
            continue

        try:
            with urlopen(source.url) as response:
                destination.write_bytes(response.read())
        except OSError as exc:
            return Failure(InfraError(f"Failed to download {file_name}: {exc}"))

        if destination.stat().st_size != source.expected_size:
            return Failure(
                InfraError(f"Downloaded {file_name}, but the file size was unexpected.")
            )

        downloaded_files.append(destination)

    return Success(tuple(downloaded_files))


def main() -> int:
    result = download_models()
    if is_failure(result):
        print(f"[ERROR] {result.message}")
        return 1
    downloaded_files = unwrap_success(result)

    print("[OK] Model files are ready.")
    for file_path in downloaded_files:
        print(f" - {file_path}")
    return 0
