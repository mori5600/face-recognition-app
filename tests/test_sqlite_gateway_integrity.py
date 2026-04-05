import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.domain.entities import RegisteredPerson
from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import ExperimentSessionId, ExperimentTrialId, PersonId
from app.domain.results import is_failure, unwrap_success
from app.domain.statuses import ExperimentStatus
from app.domain.value_objects import DisplayName, FaceEncoding, Timestamp
from app.gateways.sqlite_gateway import (
    append_encoding_to_person,
    delete_person,
    initialize_database,
    insert_experiment_session,
    insert_experiment_trial,
    insert_person_with_encodings,
    load_people,
)
from app.infra.app_paths import AppPaths


def test_insert_person_with_encodings_rolls_back_on_encoding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    person = _registered_person("alice")

    def fail_insert_encoding(
        connection: sqlite3.Connection,
        person_id: PersonId,
        encoding: FaceEncoding,
        created_at: Timestamp,
    ) -> None:
        _ = connection
        _ = person_id
        _ = encoding
        _ = created_at
        raise sqlite3.IntegrityError("encoding insert failed")

    monkeypatch.setattr(
        "app.gateways.sqlite_gateway._insert_encoding_row",
        fail_insert_encoding,
    )

    insert_result = insert_person_with_encodings(paths, person)

    assert is_failure(insert_result)
    assert _count_rows(paths, "persons") == 0
    assert _count_rows(paths, "face_encodings") == 0


def test_append_encoding_to_person_rolls_back_on_timestamp_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    person = _registered_person("alice")
    initial_insert_result = insert_person_with_encodings(paths, person)
    assert not is_failure(initial_insert_result)

    next_encoding = unwrap_success(FaceEncoding.create(np.ones(128, dtype=np.float32)))

    def fail_update_timestamp(
        connection: sqlite3.Connection,
        person_id: PersonId,
        updated_at: Timestamp,
    ) -> None:
        _ = connection
        _ = person_id
        _ = updated_at
        raise sqlite3.IntegrityError("timestamp update failed")

    monkeypatch.setattr(
        "app.gateways.sqlite_gateway._update_person_timestamp_row",
        fail_update_timestamp,
    )

    append_result = append_encoding_to_person(
        paths,
        person.person_id,
        next_encoding,
        Timestamp.now(),
    )

    assert is_failure(append_result)
    assert _count_rows(paths, "face_encodings") == 1


def test_delete_person_cascades_to_face_encodings(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    person = _registered_person("alice")
    insert_result = insert_person_with_encodings(paths, person)
    assert not is_failure(insert_result)
    assert _count_rows(paths, "face_encodings") == 1

    delete_result = delete_person(paths, person.person_id)

    assert not is_failure(delete_result)
    assert _count_rows(paths, "persons") == 0
    assert _count_rows(paths, "face_encodings") == 0


def test_experiment_trial_cascades_when_session_is_deleted(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    started_at = _timestamp(datetime(2026, 4, 5, 10, 0, tzinfo=UTC))
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
    session_result = insert_experiment_session(
        paths,
        session,
        ExperimentStatus.ACTIVE,
    )
    assert not is_failure(session_result)

    trial = ExperimentTrial(
        trial_id=ExperimentTrialId.new(),
        session_id=session.session_id,
        created_at=_timestamp(datetime(2026, 4, 5, 10, 0, 30, tzinfo=UTC)),
        matched=True,
        accepted_as_target=True,
        success=True,
    )
    trial_result = insert_experiment_trial(paths, trial)
    assert not is_failure(trial_result)

    with _raw_connection(paths) as connection:
        connection.execute(
            "DELETE FROM experiment_sessions WHERE session_id = ?",
            (session.session_id.value,),
        )
        connection.commit()

    assert _count_rows(paths, "experiment_sessions") == 0
    assert _count_rows(paths, "experiment_trials") == 0


def test_initialize_database_repairs_legacy_orphans(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(paths.database_path) as connection:
        connection.executescript(_legacy_schema_sql())
        connection.execute(
            """
            INSERT INTO persons (person_id, display_name, created_at, updated_at)
            VALUES ('person-orphan', 'orphan', '2026-04-05T10:00:00+00:00', '2026-04-05T10:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO persons (person_id, display_name, created_at, updated_at)
            VALUES ('person-valid', 'valid', '2026-04-05T10:01:00+00:00', '2026-04-05T10:01:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO face_encodings (encoding_id, person_id, encoding_blob, created_at)
            VALUES ('encoding-valid', 'person-valid', ?, '2026-04-05T10:01:00+00:00')
            """,
            (sqlite3.Binary(np.zeros(128, dtype=np.float32).tobytes()),),
        )
        connection.execute(
            """
            INSERT INTO face_encodings (encoding_id, person_id, encoding_blob, created_at)
            VALUES ('encoding-orphan', 'missing-person', ?, '2026-04-05T10:02:00+00:00')
            """,
            (sqlite3.Binary(np.ones(128, dtype=np.float32).tobytes()),),
        )
        connection.commit()

    init_result = initialize_database(paths)

    assert not is_failure(init_result)
    assert _count_rows(paths, "persons") == 1
    assert _count_rows(paths, "face_encodings") == 1

    people_result = load_people(paths)
    assert not is_failure(people_result)
    people_state = unwrap_success(people_result)
    assert len(people_state.persons) == 1
    assert people_state.persons[0].display_name.value == "valid"


def test_load_people_detects_person_without_encoding(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    with _raw_connection(paths) as connection:
        connection.execute(
            """
            INSERT INTO persons (person_id, display_name, created_at, updated_at)
            VALUES ('person-broken', 'broken', '2026-04-05T10:00:00+00:00', '2026-04-05T10:00:00+00:00')
            """
        )
        connection.commit()

    people_result = load_people(paths)

    assert is_failure(people_result)
    assert "has no face encodings stored" in people_result.message


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
    timestamp = _timestamp(datetime(2026, 4, 5, 10, 0, tzinfo=UTC))
    display_name = unwrap_success(DisplayName.create(name))
    encoding = unwrap_success(FaceEncoding.create(np.zeros(128, dtype=np.float32)))
    return RegisteredPerson(
        person_id=PersonId.new(),
        display_name=display_name,
        encodings=(encoding,),
        created_at=timestamp,
        updated_at=timestamp,
    )


def _timestamp(value: datetime) -> Timestamp:
    timestamp_result = Timestamp.create(value)
    if is_failure(timestamp_result):
        raise AssertionError(timestamp_result.message)
    return unwrap_success(timestamp_result)


def _count_rows(paths: AppPaths, table_name: str) -> int:
    with _raw_connection(paths) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    if row is None:
        raise AssertionError(f"Failed to count rows in {table_name}")
    return int(row[0])


def _raw_connection(paths: AppPaths) -> sqlite3.Connection:
    connection = sqlite3.connect(paths.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _legacy_schema_sql() -> str:
    return """
    CREATE TABLE persons (
        person_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE face_encodings (
        encoding_id TEXT PRIMARY KEY,
        person_id TEXT NOT NULL,
        encoding_blob BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons (person_id)
    );

    CREATE TABLE experiment_sessions (
        session_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL,
        scenario TEXT NOT NULL,
        target_person_id TEXT NOT NULL,
        target_person_name TEXT NOT NULL,
        face_selector_key TEXT NOT NULL,
        matching_mode_key TEXT NOT NULL,
        threshold REAL NOT NULL,
        FOREIGN KEY (target_person_id) REFERENCES persons (person_id)
    );

    CREATE TABLE experiment_trials (
        trial_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        matched INTEGER NOT NULL,
        accepted_as_target INTEGER NOT NULL,
        success INTEGER NOT NULL,
        candidate_person_id TEXT,
        candidate_person_name TEXT,
        distance REAL,
        FOREIGN KEY (session_id) REFERENCES experiment_sessions (session_id)
    );
    """
