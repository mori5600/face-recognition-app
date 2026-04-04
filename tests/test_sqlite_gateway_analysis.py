from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

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
    load_analysis_snapshot,
)
from app.infra.app_paths import AppPaths


def test_load_analysis_snapshot_joins_sessions_trials_and_logs(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    started_at = _timestamp(datetime(2026, 4, 5, 12, 0, tzinfo=UTC))
    alice = _registered_person("alice", started_at)
    bob = _registered_person("bob", _timestamp(started_at.value + timedelta(minutes=3)))

    for person in (alice, bob):
        person_result = insert_person(paths, person)
        assert not is_failure(person_result)
        encoding_result = insert_encoding(
            paths,
            person.person_id,
            person.encodings[0],
            person.created_at,
        )
        assert not is_failure(encoding_result)

    session = ExperimentSession(
        session_id=ExperimentSessionId.new(),
        started_at=started_at,
        completed_at=_timestamp(started_at.value + timedelta(minutes=1)),
        scenario=ExperimentScenario.IMPOSTOR,
        target_person_id=alice.person_id,
        target_person_name=alice.display_name.value,
        face_selector_key="largest",
        matching_mode_key="nearest_person",
        threshold=0.912,
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
        created_at=_timestamp(started_at.value + timedelta(seconds=20)),
        matched=True,
        accepted_as_target=False,
        success=True,
        candidate_person_id=bob.person_id,
        candidate_person_name=bob.display_name.value,
        distance=_distance(0.734),
    )
    trial_result = insert_experiment_trial(paths, trial)
    assert not is_failure(trial_result)

    log_entry = AppLogEntry(
        log_id=LogId.new(),
        created_at=_timestamp(started_at.value + timedelta(seconds=25)),
        level=AppLogLevel.WARNING,
        event=AppLogEvent.MATCH_REJECTED,
        message="一致しませんでした。",
        person_id=bob.person_id,
        person_name=bob.display_name.value,
        distance=_distance(0.734),
    )
    log_result = insert_log(paths, log_entry)
    assert not is_failure(log_result)

    snapshot_result = load_analysis_snapshot(paths)
    assert not is_failure(snapshot_result)

    snapshot = unwrap_success(snapshot_result)
    assert len(snapshot.people) == 2
    assert len(snapshot.logs) == 1
    assert len(snapshot.sessions) == 1
    assert len(snapshot.trials) == 1

    analysis_session = snapshot.sessions[0]
    assert analysis_session.session.target_person_id == alice.person_id
    assert analysis_session.session.matching_mode_key == "nearest_person"
    assert analysis_session.status is ExperimentStatus.COMPLETED

    analysis_trial = snapshot.trials[0]
    assert analysis_trial.scenario is ExperimentScenario.IMPOSTOR
    assert analysis_trial.target_person_id == alice.person_id
    assert analysis_trial.target_person_name == "alice"
    assert analysis_trial.threshold == 0.912
    assert analysis_trial.trial.candidate_person_id == bob.person_id
    assert analysis_trial.trial.distance is not None
    assert analysis_trial.trial.distance.value == 0.734


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
