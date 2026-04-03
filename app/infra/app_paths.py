from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    models_dir: Path
    data_dir: Path
    database_path: Path
    yunet_model_path: Path
    sface_model_path: Path
    sqlite_schema_path: Path

    @classmethod
    def default(cls) -> "AppPaths":
        root_dir = Path(__file__).resolve().parents[2]
        models_dir = root_dir / "models"
        data_dir = root_dir / "data"
        return cls(
            root_dir=root_dir,
            models_dir=models_dir,
            data_dir=data_dir,
            database_path=data_dir / "people.db",
            yunet_model_path=models_dir / "face_detection_yunet_2023mar.onnx",
            sface_model_path=models_dir / "face_recognition_sface_2021dec.onnx",
            sqlite_schema_path=root_dir / "app" / "infra" / "sqlite_schema.sql",
        )
