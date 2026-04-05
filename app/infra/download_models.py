import hashlib
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
    sha256: str


MODEL_SOURCES = {
    "face_detection_yunet_2023mar.onnx": ModelSource(
        url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true",
        expected_size=232589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": ModelSource(
        url="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
        expected_size=38696353,
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "face_landmarker.task": ModelSource(
        url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        expected_size=3758596,
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def model_file_is_ready(file_path: Path, source: ModelSource) -> bool:
    if not file_path.exists():
        return False
    if file_path.stat().st_size != source.expected_size:
        return False
    return _compute_sha256(file_path) == source.sha256


def download_models(
    paths: AppPaths | None = None,
) -> Result[tuple[Path, ...], InfraError]:
    resolved_paths = paths or AppPaths.default()
    resolved_paths.models_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths.data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    for file_name, source in MODEL_SOURCES.items():
        destination = resolved_paths.models_dir / file_name
        if model_file_is_ready(destination, source):
            downloaded_files.append(destination)
            continue

        try:
            with urlopen(source.url) as response:
                destination.write_bytes(response.read())
        except OSError as exc:
            return Failure(InfraError(f"Failed to download {file_name}: {exc}"))

        if not model_file_is_ready(destination, source):
            _delete_file_if_exists(destination)
            return Failure(
                InfraError(
                    f"Downloaded {file_name}, but the file contents did not match the expected hash."
                )
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


def _compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def _delete_file_if_exists(file_path: Path) -> None:
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        return
