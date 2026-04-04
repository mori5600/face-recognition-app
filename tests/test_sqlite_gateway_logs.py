from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.domain.ids import LogId, PersonId
from app.domain.logs import AppLogEntry, AppLogEvent, AppLogLevel
from app.domain.results import is_failure, unwrap_success
from app.domain.value_objects import Distance, Timestamp
from app.gateways.sqlite_gateway import (
    initialize_database,
    insert_log,
    load_recent_logs,
)
from app.infra.app_paths import AppPaths


def test_insert_log_and_load_recent_logs_round_trip(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)

    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    base_time = datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)
    older_entry = AppLogEntry(
        log_id=LogId.new(),
        created_at=_timestamp(base_time),
        level=AppLogLevel.INFO,
        event=AppLogEvent.CAMERA_STARTED,
        message="カメラを開始しました。",
    )
    newer_entry = AppLogEntry(
        log_id=LogId.new(),
        created_at=_timestamp(base_time + timedelta(seconds=5)),
        level=AppLogLevel.INFO,
        event=AppLogEvent.MATCH_SUCCEEDED,
        message="alice と一致しました。",
        person_id=PersonId("person-alice"),
        person_name="alice",
        distance=_distance(0.591),
    )

    older_result = insert_log(paths, older_entry)
    newer_result = insert_log(paths, newer_entry)

    assert not is_failure(older_result)
    assert not is_failure(newer_result)

    load_result = load_recent_logs(paths, limit=10)
    assert not is_failure(load_result)

    log_state = unwrap_success(load_result)

    assert len(log_state.entries) == 2
    assert [entry.event for entry in log_state.entries] == [
        AppLogEvent.MATCH_SUCCEEDED,
        AppLogEvent.CAMERA_STARTED,
    ]
    assert log_state.entries[0].person_id == PersonId("person-alice")
    assert log_state.entries[0].person_name == "alice"
    assert log_state.entries[0].distance is not None
    assert log_state.entries[0].distance.value == pytest.approx(0.591)


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
