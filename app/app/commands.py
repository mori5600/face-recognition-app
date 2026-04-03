from dataclasses import dataclass


@dataclass(frozen=True)
class DoctorCommand:
    """Validate the local runtime and model files."""


@dataclass(frozen=True)
class UiCommand:
    """Launch the desktop application."""


@dataclass(frozen=True)
class DownloadModelsCommand:
    """Download the required ONNX models."""


@dataclass(frozen=True)
class CameraCheckCommand:
    """Capture one frame from a local camera."""

    camera_index: int = 0
