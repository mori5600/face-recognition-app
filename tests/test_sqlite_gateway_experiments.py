from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import ExperimentSessionId, ExperimentTrialId, PersonId
from app.domain.results import is_failure, unwrap_success
from app.domain.statuses import ExperimentStatus
from app.domain.value_objects import Distance, Timestamp
from app.gateways.sqlite_gateway import (
    initialize_database,
    insert_experiment_session,
    insert_experiment_trial,
    load_latest_experiment,
    update_experiment_session_status,
)
from app.infra.app_paths import AppPaths


def test_insert_and_load_latest_experiment_round_trip(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    started_at = _timestamp(datetime(2026, 4, 4, 12, 0, tzinfo=UTC))
    completed_at = _timestamp(datetime(2026, 4, 4, 12, 1, tzinfo=UTC))
    session = ExperimentSession(
        session_id=ExperimentSessionId.new(),
        started_at=started_at,
        completed_at=None,
        scenario=ExperimentScenario.GENUINE,
        target_person_id=PersonId("person-alice"),
        target_person_name="alice",
        face_selector_key="single",
        matching_mode_key="nearest_encoding",
        threshold=1.128,
    )

    insert_session_result = insert_experiment_session(
        paths,
        session,
        ExperimentStatus.ACTIVE,
    )
    assert not is_failure(insert_session_result)

    trial = ExperimentTrial(
        trial_id=ExperimentTrialId.new(),
        session_id=session.session_id,
        created_at=_timestamp(started_at.value + timedelta(seconds=5)),
        matched=True,
        accepted_as_target=True,
        success=True,
        candidate_person_id=PersonId("person-alice"),
        candidate_person_name="alice",
        distance=_distance(0.591),
    )
    insert_trial_result = insert_experiment_trial(paths, trial)
    assert not is_failure(insert_trial_result)

    update_status_result = update_experiment_session_status(
        paths,
        session.session_id,
        ExperimentStatus.COMPLETED,
        completed_at,
    )
    assert not is_failure(update_status_result)

    load_result = load_latest_experiment(paths)
    assert not is_failure(load_result)

    experiment_state = unwrap_success(load_result)
    assert experiment_state.status is ExperimentStatus.COMPLETED
    assert experiment_state.session is not None
    assert experiment_state.session.target_person_name == "alice"
    assert experiment_state.session.completed_at == completed_at
    assert len(experiment_state.trials) == 1
    assert experiment_state.trials[0].success is True
    assert experiment_state.trials[0].distance is not None
    assert experiment_state.trials[0].distance.value == pytest.approx(0.591)


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
    result = Timestamp.create(value)
    if is_failure(result):
        raise AssertionError(result.message)
    return unwrap_success(result)


def _distance(value: float) -> Distance:
    result = Distance.create(value)
    if is_failure(result):
        raise AssertionError(result.message)
    return unwrap_success(result)
