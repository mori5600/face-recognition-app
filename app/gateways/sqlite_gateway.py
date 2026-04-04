import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np

from app.domain.entities import RegisteredPerson
from app.domain.errors import InfraError
from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import (
    EncodingId,
    ExperimentSessionId,
    ExperimentTrialId,
    LogId,
    PersonId,
)
from app.domain.logs import AppLogEntry, AppLogEvent, AppLogLevel
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.domain.states import ExperimentState, LogState, PeopleState
from app.domain.statuses import ExperimentStatus
from app.domain.value_objects import DisplayName, Distance, FaceEncoding, Timestamp
from app.infra.app_paths import AppPaths

LOG_RETENTION_LIMIT = 5000


def initialize_database(paths: AppPaths) -> Result[None, InfraError]:
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        schema = paths.sqlite_schema_path.read_text(encoding="utf-8")
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.executescript(schema)
            connection.commit()
    except OSError as exc:
        return Failure(InfraError(f"Failed to read the SQLite schema: {exc}"))
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to initialize the database: {exc}"))

    return Success(None)


def load_people(paths: AppPaths) -> Result[PeopleState, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            person_rows = connection.execute(
                """
                SELECT person_id, display_name, created_at, updated_at
                FROM persons
                ORDER BY created_at ASC
                """,
            ).fetchall()
            encoding_rows = connection.execute(
                """
                SELECT person_id, encoding_blob
                FROM face_encodings
                ORDER BY created_at ASC
                """,
            ).fetchall()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to load people from SQLite: {exc}"))

    encodings_by_person: dict[str, list[FaceEncoding]] = defaultdict(list)
    for person_id, encoding_blob in encoding_rows:
        encoding_result = FaceEncoding.create(
            np.frombuffer(encoding_blob, dtype=np.float32).copy()
        )
        if is_failure(encoding_result):
            return Failure(
                InfraError(
                    f"Failed to decode an encoding for person {person_id}: {encoding_result.message}",
                ),
            )
        encoding = unwrap_success(encoding_result)
        encodings_by_person[person_id].append(encoding)

    persons: list[RegisteredPerson] = []
    for person_id, raw_display_name, created_at, updated_at in person_rows:
        name_result = DisplayName.create(raw_display_name)
        if is_failure(name_result):
            return Failure(
                InfraError(
                    f"Invalid display name stored for person {person_id}: {name_result.message}"
                ),
            )
        display_name_value = unwrap_success(name_result)

        created_at_result = Timestamp.create(datetime.fromisoformat(created_at))
        updated_at_result = Timestamp.create(datetime.fromisoformat(updated_at))
        if is_failure(created_at_result):
            return Failure(
                InfraError(
                    f"Invalid created_at stored for person {person_id}: {created_at_result.message}"
                ),
            )
        if is_failure(updated_at_result):
            return Failure(
                InfraError(
                    f"Invalid updated_at stored for person {person_id}: {updated_at_result.message}"
                ),
            )
        created_timestamp = unwrap_success(created_at_result)
        updated_timestamp = unwrap_success(updated_at_result)

        encodings = tuple(encodings_by_person.get(person_id, []))
        if len(encodings) == 0:
            continue

        persons.append(
            RegisteredPerson(
                person_id=PersonId(person_id),
                display_name=display_name_value,
                encodings=encodings,
                created_at=created_timestamp,
                updated_at=updated_timestamp,
            ),
        )

    return Success(PeopleState(persons=tuple(persons)))


def load_recent_logs(
    paths: AppPaths,
    limit: int = 50,
) -> Result[LogState, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            log_rows = connection.execute(
                """
                SELECT log_id, created_at, level, event_type, message, person_id, person_name, distance
                FROM event_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to load event logs from SQLite: {exc}"))

    entries: list[AppLogEntry] = []
    for (
        log_id,
        created_at,
        level,
        event_type,
        message,
        person_id,
        person_name,
        distance,
    ) in log_rows:
        created_at_result = Timestamp.create(datetime.fromisoformat(created_at))
        if is_failure(created_at_result):
            return Failure(
                InfraError(
                    f"Invalid created_at stored for log {log_id}: {created_at_result.message}"
                ),
            )
        distance_value: Distance | None = None
        if distance is not None:
            distance_result = Distance.create(float(distance))
            if is_failure(distance_result):
                return Failure(
                    InfraError(
                        f"Invalid distance stored for log {log_id}: {distance_result.message}"
                    ),
                )
            distance_value = unwrap_success(distance_result)
        try:
            log_level = AppLogLevel(level)
            log_event = AppLogEvent(event_type)
        except ValueError as exc:
            return Failure(
                InfraError(
                    f"Invalid event log enum value stored for log {log_id}: {exc}"
                )
            )

        entries.append(
            AppLogEntry(
                log_id=LogId(log_id),
                created_at=unwrap_success(created_at_result),
                level=log_level,
                event=log_event,
                message=message,
                person_id=PersonId(person_id) if person_id is not None else None,
                person_name=person_name,
                distance=distance_value,
            )
        )

    return Success(LogState(entries=tuple(entries)))


def load_latest_experiment(paths: AppPaths) -> Result[ExperimentState, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            session_row = connection.execute(
                """
                SELECT
                    session_id,
                    started_at,
                    completed_at,
                    status,
                    scenario,
                    target_person_id,
                    target_person_name,
                    face_selector_key,
                    matching_mode_key,
                    threshold
                FROM experiment_sessions
                ORDER BY started_at DESC
                LIMIT 1
                """,
            ).fetchone()

            if session_row is None:
                return Success(ExperimentState())

            trial_rows = connection.execute(
                """
                SELECT
                    trial_id,
                    created_at,
                    matched,
                    accepted_as_target,
                    success,
                    candidate_person_id,
                    candidate_person_name,
                    distance
                FROM experiment_trials
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_row[0],),
            ).fetchall()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to load experiment data from SQLite: {exc}"))

    session_result = _decode_experiment_session(session_row)
    if is_failure(session_result):
        return session_result
    session = unwrap_success(session_result)
    try:
        experiment_status = ExperimentStatus(str(session_row[3]))
    except ValueError as exc:
        return Failure(
            InfraError(
                f"Invalid experiment status stored for session {session_row[0]}: {exc}"
            )
        )

    trials: list[ExperimentTrial] = []
    for trial_row in trial_rows:
        trial_result = _decode_experiment_trial(session.session_id, trial_row)
        if is_failure(trial_result):
            return trial_result
        trials.append(unwrap_success(trial_result))

    latest_distance = trials[-1].distance if len(trials) > 0 else None
    last_success = trials[-1].success if len(trials) > 0 else None
    return Success(
        ExperimentState(
            status=experiment_status,
            session=session,
            trials=tuple(trials),
            latest_distance=latest_distance,
            last_success=last_success,
            scenario=session.scenario,
        )
    )


def insert_person(
    paths: AppPaths, person: RegisteredPerson
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO persons (person_id, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    person.person_id.value,
                    person.display_name.value,
                    person.created_at.value.isoformat(),
                    person.updated_at.value.isoformat(),
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to insert a person into SQLite: {exc}"))

    return Success(None)


def insert_log(paths: AppPaths, entry: AppLogEntry) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO event_logs (
                    log_id,
                    created_at,
                    level,
                    event_type,
                    message,
                    person_id,
                    person_name,
                    distance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.log_id.value,
                    entry.created_at.value.isoformat(),
                    entry.level.value,
                    entry.event.value,
                    entry.message,
                    entry.person_id.value if entry.person_id is not None else None,
                    entry.person_name,
                    entry.distance.value if entry.distance is not None else None,
                ),
            )
            connection.execute(
                """
                DELETE FROM event_logs
                WHERE log_id NOT IN (
                    SELECT log_id
                    FROM event_logs
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (LOG_RETENTION_LIMIT,),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to insert an event log into SQLite: {exc}"))

    return Success(None)


def insert_experiment_session(
    paths: AppPaths,
    session: ExperimentSession,
    status: ExperimentStatus,
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO experiment_sessions (
                    session_id,
                    started_at,
                    completed_at,
                    status,
                    scenario,
                    target_person_id,
                    target_person_name,
                    face_selector_key,
                    matching_mode_key,
                    threshold
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id.value,
                    session.started_at.value.isoformat(),
                    (
                        session.completed_at.value.isoformat()
                        if session.completed_at is not None
                        else None
                    ),
                    status.value,
                    session.scenario.value,
                    session.target_person_id.value,
                    session.target_person_name,
                    session.face_selector_key,
                    session.matching_mode_key,
                    session.threshold,
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to insert an experiment session into SQLite: {exc}")
        )

    return Success(None)


def update_experiment_session_status(
    paths: AppPaths,
    session_id: ExperimentSessionId,
    status: ExperimentStatus,
    completed_at: Timestamp | None,
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                UPDATE experiment_sessions
                SET status = ?, completed_at = ?
                WHERE session_id = ?
                """,
                (
                    status.value,
                    completed_at.value.isoformat()
                    if completed_at is not None
                    else None,
                    session_id.value,
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to update experiment session status in SQLite: {exc}")
        )

    return Success(None)


def insert_experiment_trial(
    paths: AppPaths,
    trial: ExperimentTrial,
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO experiment_trials (
                    trial_id,
                    session_id,
                    created_at,
                    matched,
                    accepted_as_target,
                    success,
                    candidate_person_id,
                    candidate_person_name,
                    distance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id.value,
                    trial.session_id.value,
                    trial.created_at.value.isoformat(),
                    1 if trial.matched else 0,
                    1 if trial.accepted_as_target else 0,
                    1 if trial.success else 0,
                    (
                        trial.candidate_person_id.value
                        if trial.candidate_person_id is not None
                        else None
                    ),
                    trial.candidate_person_name,
                    trial.distance.value if trial.distance is not None else None,
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to insert an experiment trial into SQLite: {exc}")
        )

    return Success(None)


def update_person_updated_at(
    paths: AppPaths, person_id: PersonId, updated_at: Timestamp
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                UPDATE persons
                SET updated_at = ?
                WHERE person_id = ?
                """,
                (updated_at.value.isoformat(), person_id.value),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to update person timestamp in SQLite: {exc}")
        )

    return Success(None)


def insert_encoding(
    paths: AppPaths,
    person_id: PersonId,
    encoding: FaceEncoding,
    created_at: Timestamp,
) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO face_encodings (encoding_id, person_id, encoding_blob, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    EncodingId.new().value,
                    person_id.value,
                    sqlite3.Binary(encoding.value.astype(np.float32).tobytes()),
                    created_at.value.isoformat(),
                ),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to insert a face encoding into SQLite: {exc}")
        )

    return Success(None)


def delete_person(paths: AppPaths, person_id: PersonId) -> Result[None, InfraError]:
    try:
        with closing(sqlite3.connect(paths.database_path)) as connection:
            connection.execute(
                """
                DELETE FROM face_encodings
                WHERE person_id = ?
                """,
                (person_id.value,),
            )
            connection.execute(
                """
                DELETE FROM persons
                WHERE person_id = ?
                """,
                (person_id.value,),
            )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to delete a person from SQLite: {exc}"))

    return Success(None)


def _decode_experiment_session(
    row: tuple[object, ...],
) -> Result[ExperimentSession, InfraError]:
    (
        session_id,
        started_at,
        completed_at,
        _status,
        scenario,
        target_person_id,
        target_person_name,
        face_selector_key,
        matching_mode_key,
        threshold,
    ) = row

    started_at_result = Timestamp.create(datetime.fromisoformat(str(started_at)))
    if is_failure(started_at_result):
        return Failure(
            InfraError(
                f"Invalid started_at stored for experiment session {session_id}: {started_at_result.message}"
            )
        )

    completed_timestamp: Timestamp | None = None
    if completed_at is not None:
        completed_at_result = Timestamp.create(
            datetime.fromisoformat(str(completed_at))
        )
        if is_failure(completed_at_result):
            return Failure(
                InfraError(
                    f"Invalid completed_at stored for experiment session {session_id}: {completed_at_result.message}"
                )
            )
        completed_timestamp = unwrap_success(completed_at_result)

    try:
        experiment_scenario = ExperimentScenario(str(scenario))
    except ValueError as exc:
        return Failure(
            InfraError(
                f"Invalid experiment scenario stored for session {session_id}: {exc}"
            )
        )
    if not isinstance(threshold, (float, int, str)):
        return Failure(
            InfraError(
                f"Invalid threshold type stored for experiment session {session_id}: {type(threshold).__name__}"
            )
        )

    return Success(
        ExperimentSession(
            session_id=ExperimentSessionId(str(session_id)),
            started_at=unwrap_success(started_at_result),
            completed_at=completed_timestamp,
            scenario=experiment_scenario,
            target_person_id=PersonId(str(target_person_id)),
            target_person_name=str(target_person_name),
            face_selector_key=str(face_selector_key),
            matching_mode_key=str(matching_mode_key),
            threshold=float(threshold),
        )
    )


def _decode_experiment_trial(
    session_id: ExperimentSessionId,
    row: tuple[object, ...],
) -> Result[ExperimentTrial, InfraError]:
    (
        trial_id,
        created_at,
        matched,
        accepted_as_target,
        success,
        candidate_person_id,
        candidate_person_name,
        distance,
    ) = row

    created_at_result = Timestamp.create(datetime.fromisoformat(str(created_at)))
    if is_failure(created_at_result):
        return Failure(
            InfraError(
                f"Invalid created_at stored for experiment trial {trial_id}: {created_at_result.message}"
            )
        )

    distance_value: Distance | None = None
    if distance is not None:
        if not isinstance(distance, (float, int, str)):
            return Failure(
                InfraError(
                    f"Invalid distance type stored for experiment trial {trial_id}: {type(distance).__name__}"
                )
            )
        distance_result = Distance.create(float(distance))
        if is_failure(distance_result):
            return Failure(
                InfraError(
                    f"Invalid distance stored for experiment trial {trial_id}: {distance_result.message}"
                )
            )
        distance_value = unwrap_success(distance_result)

    return Success(
        ExperimentTrial(
            trial_id=ExperimentTrialId(str(trial_id)),
            session_id=session_id,
            created_at=unwrap_success(created_at_result),
            matched=bool(matched),
            accepted_as_target=bool(accepted_as_target),
            success=bool(success),
            candidate_person_id=(
                PersonId(str(candidate_person_id))
                if candidate_person_id is not None
                else None
            ),
            candidate_person_name=(
                str(candidate_person_name)
                if candidate_person_name is not None
                else None
            ),
            distance=distance_value,
        )
    )
