from dataclasses import replace

import cv2
import numpy as np

from app.domain.entities import DetectedFace, MatchResult, RegisteredPerson
from app.domain.errors import AppError, DomainError
from app.domain.ids import PersonId
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.domain.states import (
    AppState,
    CameraState,
    MatchingState,
    PeopleState,
    RegistrationState,
    UiState,
)
from app.domain.statuses import CameraStatus, MatchingStatus, RegistrationStatus
from app.domain.value_objects import DisplayName, Distance, FaceEncoding, Timestamp
from app.gateways.camera_gateway import (
    CameraHandle,
    close_camera,
    open_camera,
    read_frame,
)
from app.gateways.face_gateway import (
    OpenCvFaceEngine,
    OpenCvFaceEngineConfig,
    compare_distance,
    detect_faces,
    load_face_engine,
)
from app.gateways.sqlite_gateway import (
    delete_person,
    initialize_database,
    insert_encoding,
    insert_person,
    load_people,
    update_person_updated_at,
)
from app.infra.app_paths import AppPaths
from app.infra.download_models import MODEL_SOURCES, download_models
from app.strategy.face_selection import (
    CenterFaceSelector,
    LargestFaceSelector,
    SingleFaceOnlySelector,
)
from app.strategy.matching import (
    MatchingThreshold,
    NearestEncodingMatcher,
    NearestPersonMatcher,
)

FACE_SELECTOR_LABELS = {
    "single": "単顔のみ",
    "largest": "最大顔優先",
    "center": "中央顔優先",
}
FACE_SELECTOR_BY_LABEL = {label: key for key, label in FACE_SELECTOR_LABELS.items()}

MATCHING_MODE_LABELS = {
    "nearest_encoding": "全 encoding 最近傍",
    "nearest_person": "人物単位最近傍",
}
MATCHING_MODE_BY_LABEL = {label: key for key, label in MATCHING_MODE_LABELS.items()}

DEFAULT_MATCHING_THRESHOLD = 1.128


class FaceRecognitionRuntime:
    def __init__(
        self,
        paths: AppPaths,
        face_engine: OpenCvFaceEngine,
        initial_state: AppState,
    ) -> None:
        self._paths = paths
        self._face_engine = face_engine
        self._state = initial_state
        self._camera_handle: CameraHandle | None = None
        self._face_selector_key = "single"
        self._matching_mode_key = "nearest_encoding"
        self._matching_threshold = DEFAULT_MATCHING_THRESHOLD
        self._face_selectors = {
            "single": SingleFaceOnlySelector(),
            "largest": LargestFaceSelector(),
            "center": CenterFaceSelector(),
        }

    @property
    def state(self) -> AppState:
        return self._state

    @classmethod
    def bootstrap(
        cls, paths: AppPaths | None = None
    ) -> Result["FaceRecognitionRuntime", AppError]:
        resolved_paths = paths or AppPaths.default()
        init_result = initialize_database(resolved_paths)
        if is_failure(init_result):
            return Failure(AppError(init_result.message))

        if _models_need_download(resolved_paths):
            download_result = download_models(resolved_paths)
            if is_failure(download_result):
                return Failure(AppError(download_result.message))

        engine_result = load_face_engine(
            OpenCvFaceEngineConfig.from_app_paths(resolved_paths)
        )
        if is_failure(engine_result):
            return Failure(AppError(engine_result.message))

        people_result = load_people(resolved_paths)
        if is_failure(people_result):
            return Failure(AppError(people_result.message))
        people = unwrap_success(people_result)

        selected_person_id = (
            people.persons[0].person_id if len(people.persons) > 0 else None
        )
        initial_state = AppState(
            people=people,
            ui=UiState(
                message="準備完了です。カメラを開始してください。",
                selected_person_id=selected_person_id,
            ),
        )
        engine = unwrap_success(engine_result)
        runtime: FaceRecognitionRuntime = cls(
            paths=resolved_paths,
            face_engine=engine,
            initial_state=initial_state,
        )
        return Success(runtime)

    def shutdown(self) -> None:
        if self._camera_handle is None:
            return
        close_camera(self._camera_handle)
        self._camera_handle = None
        self._state = replace(
            self._state,
            camera=replace(
                self._state.camera,
                status=CameraStatus.STOPPED,
                latest_frame=None,
                detected_faces=(),
            ),
        )

    def start_camera(self, camera_index: int = 0) -> Result[AppState, AppError]:
        if self._camera_handle is not None:
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.RUNNING,
                    last_error=None,
                ),
                ui=replace(self._state.ui, message="カメラは既に起動しています。"),
            )
            return Success(self._state)

        open_result = open_camera(camera_index)
        if is_failure(open_result):
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.ERROR,
                    last_error=open_result.message,
                ),
                ui=replace(self._state.ui, message=open_result.message),
            )
            return Failure(AppError(open_result.message))
        camera_handle = unwrap_success(open_result)

        self._camera_handle = camera_handle
        self._state = replace(
            self._state,
            camera=replace(
                self._state.camera,
                status=CameraStatus.RUNNING,
                last_error=None,
            ),
            ui=replace(self._state.ui, message="カメラを開始しました。"),
        )
        return Success(self._state)

    def stop_camera(self) -> Result[AppState, AppError]:
        if self._camera_handle is None:
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.STOPPED,
                    latest_frame=None,
                    detected_faces=(),
                ),
                ui=replace(self._state.ui, message="カメラは停止中です。"),
            )
            return Success(self._state)

        close_result = close_camera(self._camera_handle)
        self._camera_handle = None
        if is_failure(close_result):
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.ERROR,
                    latest_frame=None,
                    detected_faces=(),
                ),
                ui=replace(self._state.ui, message=close_result.message),
            )
            return Failure(AppError(close_result.message))

        self._state = replace(
            self._state,
            camera=replace(
                self._state.camera,
                status=CameraStatus.STOPPED,
                latest_frame=None,
                detected_faces=(),
                last_error=None,
            ),
            ui=replace(self._state.ui, message="カメラを停止しました。"),
        )
        return Success(self._state)

    def update_frame(self) -> Result[AppState, AppError]:
        if self._camera_handle is None:
            return Failure(AppError("Camera is not started."))

        frame_result = read_frame(self._camera_handle)
        if is_failure(frame_result):
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.ERROR,
                    last_error=frame_result.message,
                ),
                ui=replace(self._state.ui, message=frame_result.message),
            )
            return Failure(AppError(frame_result.message))
        frame = unwrap_success(frame_result)

        face_result = detect_faces(self._face_engine, frame)
        if is_failure(face_result):
            self._state = replace(
                self._state,
                camera=replace(
                    self._state.camera,
                    status=CameraStatus.ERROR,
                    latest_frame=frame,
                    last_error=face_result.message,
                ),
                ui=replace(self._state.ui, message=face_result.message),
            )
            return Failure(AppError(face_result.message))
        detected_faces = unwrap_success(face_result)

        self._state = replace(
            self._state,
            camera=CameraState(
                status=CameraStatus.RUNNING,
                latest_frame=frame,
                detected_faces=detected_faces,
                last_error=None,
            ),
        )
        return Success(self._state)

    def register_face(self, draft_name: str) -> Result[AppState, AppError]:
        name_result = DisplayName.create(draft_name)
        if is_failure(name_result):
            return self._registration_failure(name_result.message)
        display_name = unwrap_success(name_result)

        face_result = self._select_face()
        if is_failure(face_result):
            return self._registration_failure(face_result.message)
        selected_face = unwrap_success(face_result)

        now = Timestamp.now()
        existing_person = self._find_person_by_name(display_name.value)
        if existing_person is None:
            person = RegisteredPerson(
                person_id=PersonId.new(),
                display_name=display_name,
                encodings=(selected_face.encoding,),
                created_at=now,
                updated_at=now,
            )
            insert_person_result = insert_person(self._paths, person)
            if is_failure(insert_person_result):
                return self._registration_failure(insert_person_result.message)

            insert_encoding_result = insert_encoding(
                self._paths, person.person_id, selected_face.encoding, now
            )
            if is_failure(insert_encoding_result):
                return self._registration_failure(insert_encoding_result.message)

            self._state = replace(
                self._state,
                people=PeopleState(persons=self._state.people.persons + (person,)),
                registration=RegistrationState(
                    draft_name=display_name.value,
                    status=RegistrationStatus.SUCCESS,
                    last_registered_person_id=person.person_id,
                    last_error=None,
                ),
                ui=replace(
                    self._state.ui,
                    message=f"{person.display_name.value} さんを新規登録しました。",
                    selected_person_id=person.person_id,
                ),
            )
            return Success(self._state)

        insert_encoding_result = insert_encoding(
            self._paths, existing_person.person_id, selected_face.encoding, now
        )
        if is_failure(insert_encoding_result):
            return self._registration_failure(insert_encoding_result.message)

        update_person_result = update_person_updated_at(
            self._paths, existing_person.person_id, now
        )
        if is_failure(update_person_result):
            return self._registration_failure(update_person_result.message)

        updated_person = RegisteredPerson(
            person_id=existing_person.person_id,
            display_name=existing_person.display_name,
            encodings=existing_person.encodings + (selected_face.encoding,),
            created_at=existing_person.created_at,
            updated_at=now,
        )
        updated_people = tuple(
            updated_person if person.person_id == updated_person.person_id else person
            for person in self._state.people.persons
        )
        self._state = replace(
            self._state,
            people=PeopleState(persons=updated_people),
            registration=RegistrationState(
                draft_name=display_name.value,
                status=RegistrationStatus.SUCCESS,
                last_registered_person_id=updated_person.person_id,
                last_error=None,
            ),
            ui=replace(
                self._state.ui,
                message=f"{updated_person.display_name.value} さんに特徴量を追加しました。",
                selected_person_id=updated_person.person_id,
            ),
        )
        return Success(self._state)

    def match_face(self) -> Result[AppState, AppError]:
        face_result = self._select_face()
        if is_failure(face_result):
            return self._matching_failure(face_result.message)
        selected_face = unwrap_success(face_result)

        match_result = self._current_matcher().match(
            selected_face,
            self._state.people,
            self._compare_distance,
        )
        if is_failure(match_result):
            return self._matching_failure(match_result.message)
        result = unwrap_success(match_result)

        selected_person_id = (
            result.candidate.person_id
            if result.candidate is not None
            else self._state.ui.selected_person_id
        )
        self._state = replace(
            self._state,
            matching=MatchingState(
                status=MatchingStatus.SUCCESS,
                results=(result,),
                last_error=None,
            ),
            ui=replace(
                self._state.ui,
                message=_build_match_message(result),
                selected_person_id=selected_person_id,
            ),
        )
        return Success(self._state)

    def delete_selected_person(self) -> Result[AppState, AppError]:
        selected_person = self.selected_person()
        if selected_person is None:
            return Failure(AppError("削除対象の人物が選択されていません。"))

        delete_result = delete_person(self._paths, selected_person.person_id)
        if is_failure(delete_result):
            return Failure(AppError(delete_result.message))

        updated_people = tuple(
            person
            for person in self._state.people.persons
            if person.person_id != selected_person.person_id
        )
        next_selected_person_id = (
            updated_people[0].person_id if len(updated_people) > 0 else None
        )
        self._state = replace(
            self._state,
            people=PeopleState(persons=updated_people),
            matching=MatchingState(
                status=MatchingStatus.IDLE,
                results=(),
                last_error=None,
            ),
            ui=replace(
                self._state.ui,
                message=f"{selected_person.display_name.value} さんを削除しました。",
                selected_person_id=next_selected_person_id,
            ),
        )
        return Success(self._state)

    def preview_frame(self) -> np.ndarray | None:
        frame = self._state.camera.latest_frame
        if frame is None:
            return None

        annotated = frame.copy()
        selected_face_id = self._selected_face_id()
        for detected_face in self._state.camera.detected_faces:
            color = (
                (60, 179, 113)
                if detected_face.face_id == selected_face_id
                else (247, 196, 31)
            )
            cv2.rectangle(
                annotated,
                (detected_face.bounding_box.left, detected_face.bounding_box.top),
                (detected_face.bounding_box.right, detected_face.bounding_box.bottom),
                color,
                2,
            )

        latest_match = (
            self._state.matching.results[0]
            if len(self._state.matching.results) > 0
            else None
        )
        if latest_match is not None and latest_match.candidate is not None:
            cv2.putText(
                annotated,
                _build_match_message(latest_match),
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def people_lines(self) -> tuple[str, ...]:
        if len(self._state.people.persons) == 0:
            return ("未登録です。",)

        selected_person_id = self._state.ui.selected_person_id
        lines: list[str] = []
        for person in self._state.people.persons:
            prefix = ">" if person.person_id == selected_person_id else " "
            lines.append(
                f"{prefix} {person.display_name.value} | encoding={len(person.encodings)} | updated={person.updated_at.value:%Y-%m-%d %H:%M:%S}"
            )
        return tuple(lines)

    def result_lines(self) -> tuple[str, ...]:
        if len(self._state.matching.results) == 0:
            return ("まだ照合していません。",)
        return tuple(
            _build_match_message(result) for result in self._state.matching.results
        )

    def status_lines(self) -> tuple[str, ...]:
        return (
            f"camera={self._state.camera.status}",
            f"faces={len(self._state.camera.detected_faces)}",
            f"people={len(self._state.people.persons)}",
            f"selector={FACE_SELECTOR_LABELS[self._face_selector_key]}",
            f"matcher={MATCHING_MODE_LABELS[self._matching_mode_key]}",
            f"threshold={self._matching_threshold:.3f}",
        )

    def can_target_face(self) -> bool:
        face_result = self._select_face()
        return not is_failure(face_result)

    def face_selector_labels(self) -> tuple[str, ...]:
        return tuple(FACE_SELECTOR_LABELS.values())

    def matching_mode_labels(self) -> tuple[str, ...]:
        return tuple(MATCHING_MODE_LABELS.values())

    def selected_face_selector_label(self) -> str:
        return FACE_SELECTOR_LABELS[self._face_selector_key]

    def selected_matching_mode_label(self) -> str:
        return MATCHING_MODE_LABELS[self._matching_mode_key]

    def matching_threshold_text(self) -> str:
        return f"{self._matching_threshold:.3f}"

    def set_face_selector_by_label(self, label: str) -> Result[AppState, AppError]:
        selector_key = FACE_SELECTOR_BY_LABEL.get(label)
        if selector_key is None:
            return Failure(AppError(f"Unknown face selector: {label}"))
        self._face_selector_key = selector_key
        self._state = replace(
            self._state,
            ui=replace(
                self._state.ui, message=f"顔選択方式を {label} に変更しました。"
            ),
        )
        return Success(self._state)

    def set_matching_mode_by_label(self, label: str) -> Result[AppState, AppError]:
        matching_key = MATCHING_MODE_BY_LABEL.get(label)
        if matching_key is None:
            return Failure(AppError(f"Unknown matching mode: {label}"))
        self._matching_mode_key = matching_key
        self._state = replace(
            self._state,
            ui=replace(self._state.ui, message=f"照合方式を {label} に変更しました。"),
        )
        return Success(self._state)

    def set_matching_threshold(self, raw_value: str) -> Result[AppState, AppError]:
        try:
            threshold = float(raw_value)
        except ValueError:
            return Failure(AppError("閾値は数値で入力してください。"))

        if threshold <= 0:
            return Failure(AppError("閾値は 0 より大きい値にしてください。"))

        self._matching_threshold = threshold
        self._state = replace(
            self._state,
            ui=replace(
                self._state.ui, message=f"照合閾値を {threshold:.3f} に変更しました。"
            ),
        )
        return Success(self._state)

    def person_choice_labels(self) -> tuple[str, ...]:
        if len(self._state.people.persons) == 0:
            return ("未登録",)
        return tuple(
            f"{person.display_name.value} ({len(person.encodings)})"
            for person in self._state.people.persons
        )

    def selected_person_label(self) -> str:
        selected_person = self.selected_person()
        if selected_person is None:
            return "未登録"
        return (
            f"{selected_person.display_name.value} ({len(selected_person.encodings)})"
        )

    def selected_person(self) -> RegisteredPerson | None:
        selected_person_id = self._state.ui.selected_person_id
        if selected_person_id is None:
            return None
        for person in self._state.people.persons:
            if person.person_id == selected_person_id:
                return person
        return None

    def set_selected_person_by_label(self, label: str) -> Result[AppState, AppError]:
        for person in self._state.people.persons:
            candidate_label = f"{person.display_name.value} ({len(person.encodings)})"
            if candidate_label == label:
                self._state = replace(
                    self._state,
                    ui=replace(
                        self._state.ui,
                        message=f"{person.display_name.value} さんを選択しました。",
                        selected_person_id=person.person_id,
                    ),
                )
                return Success(self._state)

        return Failure(AppError(f"Unknown person label: {label}"))

    def _registration_failure(self, message: str) -> Result[AppState, AppError]:
        self._state = replace(
            self._state,
            registration=RegistrationState(
                draft_name=self._state.registration.draft_name,
                status=RegistrationStatus.ERROR,
                last_registered_person_id=self._state.registration.last_registered_person_id,
                last_error=message,
            ),
            ui=replace(self._state.ui, message=message),
        )
        return Failure(AppError(message))

    def _matching_failure(self, message: str) -> Result[AppState, AppError]:
        self._state = replace(
            self._state,
            matching=MatchingState(
                status=MatchingStatus.ERROR,
                results=(),
                last_error=message,
            ),
            ui=replace(self._state.ui, message=message),
        )
        return Failure(AppError(message))

    def _find_person_by_name(self, display_name: str) -> RegisteredPerson | None:
        for person in self._state.people.persons:
            if person.display_name.value == display_name:
                return person
        return None

    def _frame_size(self) -> tuple[int, int] | None:
        frame = self._state.camera.latest_frame
        if frame is None:
            return None
        height, width = frame.shape[:2]
        return (width, height)

    def _select_face(self) -> Result[DetectedFace, DomainError]:
        selector = self._face_selectors[self._face_selector_key]
        return selector.select(
            self._state.camera.detected_faces,
            frame_size=self._frame_size(),
        )

    def _selected_face_id(self) -> object | None:
        selected_result = self._select_face()
        if is_failure(selected_result):
            return None
        selected_face = unwrap_success(selected_result)
        return selected_face.face_id

    def _current_matcher(self) -> NearestEncodingMatcher | NearestPersonMatcher:
        threshold = MatchingThreshold(self._matching_threshold)
        if self._matching_mode_key == "nearest_person":
            return NearestPersonMatcher(threshold)
        return NearestEncodingMatcher(threshold)

    def _compare_distance(
        self, a: FaceEncoding, b: FaceEncoding
    ) -> Result[Distance, DomainError]:
        distance_result = compare_distance(self._face_engine, a, b)
        if is_failure(distance_result):
            return Failure(DomainError(distance_result.message))
        distance = unwrap_success(distance_result)
        return Success(distance)


def _models_need_download(paths: AppPaths) -> bool:
    file_sizes = {
        paths.yunet_model_path.name: paths.yunet_model_path.stat().st_size
        if paths.yunet_model_path.exists()
        else -1,
        paths.sface_model_path.name: paths.sface_model_path.stat().st_size
        if paths.sface_model_path.exists()
        else -1,
    }
    for file_name, source in MODEL_SOURCES.items():
        if file_sizes.get(file_name) != source.expected_size:
            return True
    return False


def _build_match_message(result: MatchResult) -> str:
    if result.candidate is None:
        return "登録済みの人物がいません。"
    distance = result.candidate.distance.value
    if result.matched:
        return f"一致: {result.candidate.display_name.value} (distance={distance:.3f})"
    return f"不一致: 最も近い候補は {result.candidate.display_name.value} (distance={distance:.3f})"
