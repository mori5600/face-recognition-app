from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Protocol, cast

import mediapipe as mp

from app.domain.errors import InfraError
from app.domain.liveness import LivenessSignals
from app.domain.raw_types import RawFrame
from app.domain.results import Failure, Result, Success
from app.infra.cv2_compat import convert_bgr_to_rgb

LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263
NOSE_TIP_INDEX = 1


class LandmarkerCategoryProtocol(Protocol):
    category_name: str
    score: float


class LandmarkProtocol(Protocol):
    x: float
    y: float


class FaceLandmarkerResultProtocol(Protocol):
    face_landmarks: Sequence[Sequence[LandmarkProtocol]]
    face_blendshapes: Sequence[Sequence[LandmarkerCategoryProtocol]]


class FaceLandmarkerProtocol(Protocol):
    def detect_for_video(
        self,
        image: object,
        timestamp_ms: int,
    ) -> FaceLandmarkerResultProtocol: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class MediaPipeLivenessEngineConfig:
    model_path: Path


@dataclass(frozen=True)
class MediaPipeLivenessEngine:
    landmarker: FaceLandmarkerProtocol


def load_liveness_engine(
    config: MediaPipeLivenessEngineConfig,
) -> Result[MediaPipeLivenessEngine, InfraError]:
    if not config.model_path.exists():
        return Failure(
            InfraError(f"MediaPipe model file is missing: {config.model_path}")
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(config.model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    try:
        landmarker = cast(
            FaceLandmarkerProtocol,
            mp.tasks.vision.FaceLandmarker.create_from_options(options),
        )
    except Exception as exc:
        return Failure(InfraError(f"Failed to load MediaPipe Face Landmarker: {exc}"))

    return Success(MediaPipeLivenessEngine(landmarker=landmarker))


def close_liveness_engine(
    engine: MediaPipeLivenessEngine,
) -> Result[None, InfraError]:
    try:
        engine.landmarker.close()
    except Exception as exc:
        return Failure(InfraError(f"Failed to close MediaPipe Face Landmarker: {exc}"))
    return Success(None)


def detect_liveness_signals(
    engine: MediaPipeLivenessEngine,
    frame: RawFrame,
    timestamp_ms: int,
) -> Result[LivenessSignals | None, InfraError]:
    if frame.size == 0:
        return Failure(InfraError("The frame is empty."))

    rgb_frame = convert_bgr_to_rgb(frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        result = engine.landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as exc:
        return Failure(InfraError(f"MediaPipe liveness inference failed: {exc}"))

    if len(result.face_landmarks) == 0:
        return Success(None)

    blendshapes = result.face_blendshapes[0] if len(result.face_blendshapes) > 0 else ()
    categories = {
        category.category_name: float(category.score) for category in blendshapes
    }
    landmarks = result.face_landmarks[0]

    left_eye = landmarks[LEFT_EYE_OUTER_INDEX]
    right_eye = landmarks[RIGHT_EYE_OUTER_INDEX]
    nose_tip = landmarks[NOSE_TIP_INDEX]
    inter_eye_distance = max(abs(right_eye.x - left_eye.x), 1e-6)
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    turn_score = float((nose_tip.x - eye_mid_x) / inter_eye_distance)

    return Success(
        LivenessSignals(
            face_count=len(result.face_landmarks),
            blink_left=_get_shape_score(categories, "eyeBlinkLeft"),
            blink_right=_get_shape_score(categories, "eyeBlinkRight"),
            jaw_open=_get_shape_score(categories, "jawOpen"),
            turn_score=turn_score,
        )
    )


def _get_shape_score(categories: dict[str, float], name: str) -> float:
    return categories.get(name, 0.0)
