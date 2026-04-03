import os
from dataclasses import dataclass
from typing import cast

import cv2

from app.domain.errors import InfraError
from app.domain.raw_types import RawFrame
from app.domain.results import Failure, Result, Success


@dataclass
class CameraHandle:
    capture: cv2.VideoCapture


def open_camera(camera_index: int = 0) -> Result[CameraHandle, InfraError]:
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return Success(CameraHandle(capture=capture))
        capture.release()
    return Failure(InfraError(f"Failed to open camera {camera_index}."))


def read_frame(handle: CameraHandle) -> Result[RawFrame, InfraError]:
    ok, frame = handle.capture.read()
    if not ok or frame is None:
        return Failure(InfraError("Failed to read a frame from the camera."))
    typed_frame = cast(RawFrame, frame)
    return Success(typed_frame)


def close_camera(handle: CameraHandle) -> Result[None, InfraError]:
    try:
        handle.capture.release()
        return Success(None)
    except cv2.error as exc:
        return Failure(InfraError(f"Failed to close the camera: {exc}"))
