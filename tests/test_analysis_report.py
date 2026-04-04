from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from app.app.analysis_report import write_analysis_report
from app.domain.entities import RegisteredPerson
from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import ExperimentSessionId, ExperimentTrialId, LogId, PersonId
from app.domain.logs import AppLogEntry, AppLogEvent, AppLogLevel
from app.domain.results import is_failure, unwrap_success
from app.domain.statuses import ExperimentStatus
from app.domain.value_objects import DisplayName, Distance, FaceEncoding, Timestamp
from app.gateways.sqlite_gateway import (
    initialize_database,
    insert_encoding,
    insert_experiment_session,
    insert_experiment_trial,
    insert_log,
    insert_person,
)
from app.infra.app_paths import AppPaths


def test_write_analysis_report_generates_html_from_sqlite_data(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    started_at = _timestamp(datetime(2026, 4, 5, 12, 0, tzinfo=UTC))
    person = _registered_person("alice", started_at)

    insert_person_result = insert_person(paths, person)
    assert not is_failure(insert_person_result)
    insert_encoding_result = insert_encoding(
        paths,
        person.person_id,
        person.encodings[0],
        started_at,
    )
    assert not is_failure(insert_encoding_result)

    session = ExperimentSession(
        session_id=ExperimentSessionId.new(),
        started_at=started_at,
        completed_at=_timestamp(started_at.value + timedelta(minutes=1)),
        scenario=ExperimentScenario.GENUINE,
        target_person_id=person.person_id,
        target_person_name=person.display_name.value,
        face_selector_key="single",
        matching_mode_key="nearest_encoding",
        threshold=1.128,
    )
    session_result = insert_experiment_session(
        paths,
        session,
        ExperimentStatus.COMPLETED,
    )
    assert not is_failure(session_result)

    trial = ExperimentTrial(
        trial_id=ExperimentTrialId.new(),
        session_id=session.session_id,
        created_at=_timestamp(started_at.value + timedelta(seconds=30)),
        matched=True,
        accepted_as_target=True,
        success=True,
        candidate_person_id=person.person_id,
        candidate_person_name=person.display_name.value,
        distance=_distance(0.591),
    )
    trial_result = insert_experiment_trial(paths, trial)
    assert not is_failure(trial_result)

    log_entry = AppLogEntry(
        log_id=LogId.new(),
        created_at=_timestamp(started_at.value + timedelta(seconds=35)),
        level=AppLogLevel.INFO,
        event=AppLogEvent.MATCH_SUCCEEDED,
        message="alice と一致しました。",
        person_id=person.person_id,
        person_name=person.display_name.value,
        distance=_distance(0.591),
    )
    log_result = insert_log(paths, log_entry)
    assert not is_failure(log_result)

    report_result = write_analysis_report(paths)
    assert not is_failure(report_result)

    report_path = unwrap_success(report_result)
    assert report_path.exists()

    report_html = report_path.read_text(encoding="utf-8")
    assert "顔認証アプリの解析レポート" in report_html
    assert "本人受入試験" in report_html
    assert "alice" in report_html
    assert "0.591" in report_html


def test_write_analysis_report_handles_empty_database(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    report_result = write_analysis_report(paths)
    assert not is_failure(report_result)

    report_path = unwrap_success(report_result)
    report_html = report_path.read_text(encoding="utf-8")

    assert "評価実験の記録はまだありません。" in report_html
    assert "イベント履歴はまだありません。" in report_html
    assert "登録人物" in report_html


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


def _registered_person(name: str, created_at: Timestamp) -> RegisteredPerson:
    display_name = unwrap_success(DisplayName.create(name))
    encoding = unwrap_success(FaceEncoding.create(np.zeros(128, dtype=np.float32)))
    return RegisteredPerson(
        person_id=PersonId.new(),
        display_name=display_name,
        encodings=(encoding,),
        created_at=created_at,
        updated_at=created_at,
    )


def _timestamp(value: datetime) -> Timestamp:
    timestamp_result = Timestamp.create(value)
    if is_failure(timestamp_result):
        raise AssertionError(timestamp_result.message)
    return unwrap_success(timestamp_result)


def _distance(value: float) -> Distance:
    distance_result = Distance.create(value)
    if is_failure(distance_result):
        raise AssertionError(distance_result.message)
    return unwrap_success(distance_result)
