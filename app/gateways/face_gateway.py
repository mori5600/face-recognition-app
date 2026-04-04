from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.domain.entities import DetectedFace
from app.domain.errors import InfraError
from app.domain.ids import FaceId
from app.domain.raw_types import RawFrame
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.domain.value_objects import BoundingBox, Distance, FaceEncoding
from app.infra.app_paths import AppPaths
from app.infra.cv2_compat import (
    FR_NORM_L2,
    FaceDetectorProtocol,
    FaceRecognizerProtocol,
    create_face_detector,
    create_face_recognizer,
)


@dataclass(frozen=True)
class OpenCvFaceEngineConfig:
    yunet_model_path: Path
    sface_model_path: Path
    input_size: tuple[int, int] = (320, 320)
    score_threshold: float = 0.8
    nms_threshold: float = 0.3
    top_k: int = 5000

    @classmethod
    def from_app_paths(cls, paths: AppPaths) -> "OpenCvFaceEngineConfig":
        return cls(
            yunet_model_path=paths.yunet_model_path,
            sface_model_path=paths.sface_model_path,
        )


@dataclass
class OpenCvFaceEngine:
    detector: FaceDetectorProtocol
    recognizer: FaceRecognizerProtocol


def load_face_engine(
    config: OpenCvFaceEngineConfig,
) -> Result[OpenCvFaceEngine, InfraError]:
    if not config.yunet_model_path.exists():
        return Failure(
            InfraError(f"YuNet model was not found: {config.yunet_model_path}")
        )
    if not config.sface_model_path.exists():
        return Failure(
            InfraError(f"SFace model was not found: {config.sface_model_path}")
        )

    try:
        detector = create_face_detector(
            str(config.yunet_model_path),
            "",
            config.input_size,
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )
        recognizer = create_face_recognizer(str(config.sface_model_path), "")
        return Success(OpenCvFaceEngine(detector=detector, recognizer=recognizer))
    except Exception as exc:
        return Failure(InfraError(f"Failed to load OpenCV DNN models: {exc}"))


def detect_faces(
    engine: OpenCvFaceEngine,
    frame: RawFrame,
) -> Result[tuple[DetectedFace, ...], InfraError]:
    if frame.size == 0:
        return Failure(InfraError("The frame is empty."))

    frame_height, frame_width = frame.shape[:2]
    engine.detector.setInputSize((frame_width, frame_height))

    try:
        _, faces = engine.detector.detect(frame)
    except Exception as exc:
        return Failure(InfraError(f"Face detection failed: {exc}"))

    if faces is None:
        empty_faces: tuple[DetectedFace, ...] = ()
        return Success(empty_faces)

    faces_array = np.asarray(faces)
    detected_faces: list[DetectedFace] = []
    for face_row in faces_array:
        try:
            aligned_face = engine.recognizer.alignCrop(frame, face_row)
            feature = engine.recognizer.feature(aligned_face)
        except Exception as exc:
            return Failure(InfraError(f"Face feature extraction failed: {exc}"))

        left = int(round(face_row[0]))
        top = int(round(face_row[1]))
        width = int(round(face_row[2]))
        height = int(round(face_row[3]))
        right = min(frame_width, left + width)
        bottom = min(frame_height, top + height)
        left = max(0, left)
        top = max(0, top)

        bounding_box_result = BoundingBox.create((left, top, right, bottom))
        if is_failure(bounding_box_result):
            return Failure(InfraError(bounding_box_result.message))
        bounding_box = unwrap_success(bounding_box_result)

        encoding_result = FaceEncoding.create(np.asarray(feature, dtype=np.float32))
        if is_failure(encoding_result):
            return Failure(InfraError(encoding_result.message))
        encoding = unwrap_success(encoding_result)

        detected_faces.append(
            DetectedFace(
                face_id=FaceId.new(),
                bounding_box=bounding_box,
                encoding=encoding,
            ),
        )

    return Success(tuple(detected_faces))


def compare_distance(
    engine: OpenCvFaceEngine,
    a: FaceEncoding,
    b: FaceEncoding,
) -> Result[Distance, InfraError]:
    try:
        raw_distance = engine.recognizer.match(
            a.to_row_vector(),
            b.to_row_vector(),
            FR_NORM_L2,
        )
    except Exception as exc:
        return Failure(InfraError(f"Face distance comparison failed: {exc}"))

    distance_result = Distance.create(float(raw_distance))
    if is_failure(distance_result):
        return Failure(InfraError(distance_result.message))
    distance = unwrap_success(distance_result)
    return Success(distance)
