from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.app.runtime import FaceRecognitionRuntime
from app.domain.entities import (
    DetectedFace,
    MatchCandidate,
    MatchResult,
    RegisteredPerson,
)
from app.domain.ids import FaceId, PersonId
from app.domain.results import is_failure, unwrap_success
from app.domain.states import AppState, CameraState, MatchingState, PeopleState, UiState
from app.domain.statuses import (
    CameraStatus,
    DeferredActionKind,
    ExperimentStatus,
    LivenessStatus,
    MatchingStatus,
)
from app.domain.value_objects import (
    BoundingBox,
    DisplayName,
    Distance,
    FaceEncoding,
    Timestamp,
)
from app.gateways.face_gateway import OpenCvFaceEngine
from app.gateways.liveness_gateway import (
    FaceLandmarkerProtocol,
    FaceLandmarkerResultProtocol,
    LandmarkerCategoryProtocol,
    LandmarkProtocol,
    MediaPipeLivenessEngine,
)
from app.gateways.sqlite_gateway import initialize_database
from app.infra.app_paths import AppPaths
from app.infra.cv2_compat import FaceDetectorProtocol, FaceRecognizerProtocol
from app.ui.view_model import build_main_window_view_model


def test_experiment_first_match_shows_liveness_challenge(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)

    start_result = runtime.start_experiment()
    assert not is_failure(start_result)

    match_result = runtime.match_face()
    assert is_failure(match_result)
    assert runtime.state.experiment.status is ExperimentStatus.ACTIVE
    assert len(runtime.state.experiment.trials) == 0
    assert runtime.state.liveness.deferred_action is DeferredActionKind.MATCH

    view_model = build_main_window_view_model(runtime)

    assert view_model.phase_badge_text == "確認中"
    assert "生体確認" in view_model.phase_title


def test_liveness_challenge_takes_priority_over_previous_match_result(
    tmp_path: Path,
) -> None:
    runtime = _build_runtime(tmp_path)
    person = runtime.state.people.persons[0]

    runtime._state = replace(
        runtime.state,
        matching=MatchingState(
            status=MatchingStatus.SUCCESS,
            results=(
                MatchResult(
                    candidate=MatchCandidate(
                        person_id=person.person_id,
                        display_name=person.display_name,
                        distance=unwrap_success(Distance.create(0.591)),
                    ),
                    matched=True,
                ),
            ),
            last_error=None,
        ),
    )

    queued_result = runtime.match_face()
    assert is_failure(queued_result)

    view_model = build_main_window_view_model(runtime)

    assert view_model.phase_badge_text == "確認中"
    assert view_model.result_lines == ("まだ照合していません。",)


def test_experiment_verified_liveness_executes_match_automatically(
    tmp_path: Path,
) -> None:
    runtime = _build_runtime(tmp_path)

    start_result = runtime.start_experiment()
    assert not is_failure(start_result)

    queued_result = runtime.match_face()
    assert is_failure(queued_result)

    runtime._state = replace(
        runtime.state,
        liveness=replace(
            runtime.state.liveness,
            status=LivenessStatus.VERIFIED,
            verified_until_ms=9_999_999_999_999,
        ),
    )
    execute_result = runtime._execute_deferred_action_if_ready()
    assert not is_failure(execute_result)

    assert runtime.state.experiment.status is ExperimentStatus.ACTIVE
    assert len(runtime.state.experiment.trials) == 1
    assert runtime.state.experiment.trials[0].success is True
    assert runtime.state.experiment.trials[0].accepted_as_target is True
    assert runtime.state.liveness.status is LivenessStatus.IDLE

    summary_lines = runtime.experiment_summary_lines()
    assert "試行 1件" in summary_lines[1]


def _build_runtime(tmp_path: Path) -> FaceRecognitionRuntime:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    person = _registered_person("alice")
    face = _detected_face(person.encodings[0])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = AppState(
        camera=CameraState(
            status=CameraStatus.RUNNING,
            latest_frame=frame,
            detected_faces=(face,),
        ),
        people=PeopleState(persons=(person,)),
        ui=UiState(selected_person_id=person.person_id),
    )
    return FaceRecognitionRuntime(
        paths=paths,
        face_engine=OpenCvFaceEngine(
            detector=_StubDetector(),
            recognizer=_StubRecognizer(),
        ),
        liveness_engine=MediaPipeLivenessEngine(landmarker=_StubLandmarker()),
        initial_state=state,
    )


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


def _registered_person(name: str) -> RegisteredPerson:
    display_name = unwrap_success(DisplayName.create(name))
    encoding = unwrap_success(FaceEncoding.create(np.zeros(128, dtype=np.float32)))
    created_at = Timestamp.now()
    return RegisteredPerson(
        person_id=PersonId.new(),
        display_name=display_name,
        encodings=(encoding,),
        created_at=created_at,
        updated_at=created_at,
    )


def _detected_face(encoding: FaceEncoding) -> DetectedFace:
    bounding_box = unwrap_success(BoundingBox.create((100, 100, 240, 240)))
    return DetectedFace(
        face_id=FaceId.new(),
        bounding_box=bounding_box,
        encoding=encoding,
    )


class _StubDetector(FaceDetectorProtocol):
    def setInputSize(self, input_size: tuple[int, int]) -> None:
        _ = input_size

    def detect(self, image: np.ndarray) -> tuple[object, object | None]:
        _ = image
        return ((), None)


class _StubRecognizer(FaceRecognizerProtocol):
    def alignCrop(self, image: np.ndarray, face: object) -> np.ndarray:
        _ = face
        return image

    def feature(self, aligned_face: np.ndarray) -> np.ndarray:
        return aligned_face.reshape(1, -1)

    def match(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        distance_type: int,
    ) -> float:
        _ = distance_type
        return float(np.linalg.norm(feature1.reshape(-1) - feature2.reshape(-1)))


class _StubLandmarker(FaceLandmarkerProtocol):
    def detect_for_video(
        self,
        image: object,
        timestamp_ms: int,
    ) -> FaceLandmarkerResultProtocol:
        _ = image
        _ = timestamp_ms
        return _StubLandmarkerResult()

    def close(self) -> None:
        return None


class _StubLandmarkerResult(FaceLandmarkerResultProtocol):
    @property
    def face_landmarks(self) -> Sequence[Sequence[LandmarkProtocol]]:
        return ()

    @property
    def face_blendshapes(self) -> Sequence[Sequence[LandmarkerCategoryProtocol]]:
        return ()


class _StubLandmark(LandmarkProtocol):
    x = 0.0
    y = 0.0


class _StubCategory(LandmarkerCategoryProtocol):
    category_name = ""
    score = 0.0
