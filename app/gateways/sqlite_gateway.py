import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np

from app.domain.analysis import AnalysisSession, AnalysisSnapshot, AnalysisTrial
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
type SqliteRow = tuple[object, ...]

FACE_ENCODINGS_TABLE_SQL = """
CREATE TABLE face_encodings (
    encoding_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    encoding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
)
"""

EXPERIMENT_SESSIONS_TABLE_SQL = """
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
    threshold REAL NOT NULL
)
"""

EXPERIMENT_TRIALS_TABLE_SQL = """
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
    FOREIGN KEY (session_id) REFERENCES experiment_sessions (session_id) ON DELETE CASCADE
)
"""

EXPERIMENT_SESSIONS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiment_sessions_started_at
ON experiment_sessions (started_at DESC)
"""

EXPERIMENT_TRIALS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiment_trials_session_created_at
ON experiment_trials (session_id, created_at ASC)
"""


def _open_connection(paths: AppPaths) -> sqlite3.Connection:
    connection = sqlite3.connect(paths.database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _foreign_keys(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple[str, str, str], ...]:
    rows = connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
    foreign_keys: list[tuple[str, str, str]] = []
    for row in rows:
        foreign_keys.append((str(row[2]), str(row[3]), str(row[6]).upper()))
    return tuple(foreign_keys)


def _requires_face_encodings_rebuild(connection: sqlite3.Connection) -> bool:
    return _foreign_keys(connection, "face_encodings") != (
        ("persons", "person_id", "CASCADE"),
    )


def _requires_experiment_sessions_rebuild(connection: sqlite3.Connection) -> bool:
    return len(_foreign_keys(connection, "experiment_sessions")) > 0


def _requires_experiment_trials_rebuild(connection: sqlite3.Connection) -> bool:
    return _foreign_keys(connection, "experiment_trials") != (
        ("experiment_sessions", "session_id", "CASCADE"),
    )


def _rebuild_face_encodings_table(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE face_encodings RENAME TO face_encodings_legacy")
    connection.execute(FACE_ENCODINGS_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO face_encodings (encoding_id, person_id, encoding_blob, created_at)
        SELECT
            legacy.encoding_id,
            legacy.person_id,
            legacy.encoding_blob,
            legacy.created_at
        FROM face_encodings_legacy AS legacy
        WHERE EXISTS (
            SELECT 1
            FROM persons
            WHERE persons.person_id = legacy.person_id
        )
        """
    )
    connection.execute("DROP TABLE face_encodings_legacy")


def _rebuild_experiment_sessions_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE experiment_sessions RENAME TO experiment_sessions_legacy"
    )
    connection.execute(EXPERIMENT_SESSIONS_TABLE_SQL)
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
        FROM experiment_sessions_legacy
        """
    )
    connection.execute("DROP TABLE experiment_sessions_legacy")
    connection.execute(EXPERIMENT_SESSIONS_INDEX_SQL)


def _rebuild_experiment_trials_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE experiment_trials RENAME TO experiment_trials_legacy"
    )
    connection.execute(EXPERIMENT_TRIALS_TABLE_SQL)
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
        SELECT
            legacy.trial_id,
            legacy.session_id,
            legacy.created_at,
            legacy.matched,
            legacy.accepted_as_target,
            legacy.success,
            legacy.candidate_person_id,
            legacy.candidate_person_name,
            legacy.distance
        FROM experiment_trials_legacy AS legacy
        WHERE EXISTS (
            SELECT 1
            FROM experiment_sessions
            WHERE experiment_sessions.session_id = legacy.session_id
        )
        """
    )
    connection.execute("DROP TABLE experiment_trials_legacy")
    connection.execute(EXPERIMENT_TRIALS_INDEX_SQL)


def _cleanup_orphan_rows(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        DELETE FROM face_encodings
        WHERE NOT EXISTS (
            SELECT 1
            FROM persons
            WHERE persons.person_id = face_encodings.person_id
        )
        """
    )
    connection.execute(
        """
        DELETE FROM experiment_trials
        WHERE NOT EXISTS (
            SELECT 1
            FROM experiment_sessions
            WHERE experiment_sessions.session_id = experiment_trials.session_id
        )
        """
    )
    connection.execute(
        """
        DELETE FROM persons
        WHERE NOT EXISTS (
            SELECT 1
            FROM face_encodings
            WHERE face_encodings.person_id = persons.person_id
        )
        """
    )


def _migrate_schema(connection: sqlite3.Connection) -> None:
    requires_rebuild = (
        _requires_face_encodings_rebuild(connection)
        or _requires_experiment_sessions_rebuild(connection)
        or _requires_experiment_trials_rebuild(connection)
    )

    if requires_rebuild:
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.execute("BEGIN")
            if _requires_face_encodings_rebuild(connection):
                _rebuild_face_encodings_table(connection)
            if _requires_experiment_sessions_rebuild(connection):
                _rebuild_experiment_sessions_table(connection)
            if _requires_experiment_trials_rebuild(connection):
                _rebuild_experiment_trials_table(connection)
            _cleanup_orphan_rows(connection)
            connection.execute("COMMIT")
        except sqlite3.Error:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.execute("PRAGMA foreign_keys = ON")
        return

    _cleanup_orphan_rows(connection)


def _insert_person_row(
    connection: sqlite3.Connection,
    person: RegisteredPerson,
) -> None:
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


def _insert_encoding_row(
    connection: sqlite3.Connection,
    person_id: PersonId,
    encoding: FaceEncoding,
    created_at: Timestamp,
) -> None:
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


def _update_person_timestamp_row(
    connection: sqlite3.Connection,
    person_id: PersonId,
    updated_at: Timestamp,
) -> None:
    cursor = connection.execute(
        """
        UPDATE persons
        SET updated_at = ?
        WHERE person_id = ?
        """,
        (updated_at.value.isoformat(), person_id.value),
    )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError(
            f"Unknown person_id referenced while updating timestamp: {person_id.value}"
        )


def initialize_database(paths: AppPaths) -> Result[None, InfraError]:
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        schema = paths.sqlite_schema_path.read_text(encoding="utf-8")
        with closing(_open_connection(paths)) as connection:
            connection.executescript(schema)
            _migrate_schema(connection)
            connection.commit()
    except OSError as exc:
        return Failure(InfraError(f"Failed to read the SQLite schema: {exc}"))
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to initialize the database: {exc}"))

    return Success(None)


def load_people(paths: AppPaths) -> Result[PeopleState, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
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
            return Failure(
                InfraError(
                    f"Person {person_id} has no face encodings stored. Run initialize_database() to repair the database."
                )
            )

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
        with closing(_open_connection(paths)) as connection:
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
    for log_row in log_rows:
        entry_result = _decode_log_row(log_row)
        if is_failure(entry_result):
            return Failure(entry_result.error)
        entries.append(unwrap_success(entry_result))

    return Success(LogState(entries=tuple(entries)))


def load_analysis_snapshot(paths: AppPaths) -> Result[AnalysisSnapshot, InfraError]:
    people_result = load_people(paths)
    if is_failure(people_result):
        return people_result
    people_state = unwrap_success(people_result)

    try:
        with closing(_open_connection(paths)) as connection:
            log_rows = connection.execute(
                """
                SELECT log_id, created_at, level, event_type, message, person_id, person_name, distance
                FROM event_logs
                ORDER BY created_at ASC
                """,
            ).fetchall()
            session_rows = connection.execute(
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
                ORDER BY started_at ASC
                """,
            ).fetchall()
            trial_rows = connection.execute(
                """
                SELECT
                    trial_id,
                    session_id,
                    created_at,
                    matched,
                    accepted_as_target,
                    success,
                    candidate_person_id,
                    candidate_person_name,
                    distance
                FROM experiment_trials
                ORDER BY created_at ASC
                """,
            ).fetchall()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to load analysis data from SQLite: {exc}"))

    logs: list[AppLogEntry] = []
    for log_row in log_rows:
        entry_result = _decode_log_row(log_row)
        if is_failure(entry_result):
            return Failure(entry_result.error)
        logs.append(unwrap_success(entry_result))

    sessions: list[AnalysisSession] = []
    sessions_by_id: dict[ExperimentSessionId, AnalysisSession] = {}
    for session_row in session_rows:
        session_result = _decode_analysis_session(session_row)
        if is_failure(session_result):
            return Failure(session_result.error)
        analysis_session = unwrap_success(session_result)
        sessions.append(analysis_session)
        sessions_by_id[analysis_session.session.session_id] = analysis_session

    trials: list[AnalysisTrial] = []
    for trial_row in trial_rows:
        trial_result = _decode_analysis_trial(trial_row, sessions_by_id)
        if is_failure(trial_result):
            return Failure(trial_result.error)
        trials.append(unwrap_success(trial_result))

    return Success(
        AnalysisSnapshot(
            people=people_state.persons,
            logs=tuple(logs),
            sessions=tuple(sessions),
            trials=tuple(trials),
        )
    )


def load_analysis_fingerprint(paths: AppPaths) -> Result[str, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM persons),
                    (SELECT COALESCE(MAX(updated_at), '') FROM persons),
                    (SELECT COUNT(*) FROM face_encodings),
                    (SELECT COALESCE(MAX(created_at), '') FROM face_encodings),
                    (SELECT COUNT(*) FROM event_logs),
                    (SELECT COALESCE(MAX(created_at), '') FROM event_logs),
                    (SELECT COUNT(*) FROM experiment_sessions),
                    (
                        SELECT COALESCE(MAX(COALESCE(completed_at, started_at)), '')
                        FROM experiment_sessions
                    ),
                    (SELECT COUNT(*) FROM experiment_trials),
                    (SELECT COALESCE(MAX(created_at), '') FROM experiment_trials)
                """,
            ).fetchone()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to load analysis fingerprint from SQLite: {exc}")
        )

    if row is None:
        return Success("empty")

    return Success("|".join(str(value) for value in row))


def load_latest_experiment(paths: AppPaths) -> Result[ExperimentState, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
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
        with closing(_open_connection(paths)) as connection:
            _insert_person_row(connection, person)
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(InfraError(f"Failed to insert a person into SQLite: {exc}"))

    return Success(None)


def insert_person_with_encodings(
    paths: AppPaths,
    person: RegisteredPerson,
) -> Result[None, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
            _insert_person_row(connection, person)
            for encoding in person.encodings:
                _insert_encoding_row(
                    connection,
                    person.person_id,
                    encoding,
                    person.created_at,
                )
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to insert a person with encodings into SQLite: {exc}")
        )

    return Success(None)


def insert_log(paths: AppPaths, entry: AppLogEntry) -> Result[None, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
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
        with closing(_open_connection(paths)) as connection:
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
        with closing(_open_connection(paths)) as connection:
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
        with closing(_open_connection(paths)) as connection:
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
        with closing(_open_connection(paths)) as connection:
            _update_person_timestamp_row(connection, person_id, updated_at)
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
        with closing(_open_connection(paths)) as connection:
            _insert_encoding_row(connection, person_id, encoding, created_at)
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(f"Failed to insert a face encoding into SQLite: {exc}")
        )

    return Success(None)


def append_encoding_to_person(
    paths: AppPaths,
    person_id: PersonId,
    encoding: FaceEncoding,
    updated_at: Timestamp,
) -> Result[None, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
            _insert_encoding_row(connection, person_id, encoding, updated_at)
            _update_person_timestamp_row(connection, person_id, updated_at)
            connection.commit()
    except sqlite3.Error as exc:
        return Failure(
            InfraError(
                f"Failed to append a face encoding to the person in SQLite: {exc}"
            )
        )

    return Success(None)


def delete_person(paths: AppPaths, person_id: PersonId) -> Result[None, InfraError]:
    try:
        with closing(_open_connection(paths)) as connection:
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


def _decode_log_row(
    row: SqliteRow,
) -> Result[AppLogEntry, InfraError]:
    (
        log_id,
        created_at,
        level,
        event_type,
        message,
        person_id,
        person_name,
        distance,
    ) = row

    created_at_result = Timestamp.create(datetime.fromisoformat(str(created_at)))
    if is_failure(created_at_result):
        return Failure(
            InfraError(
                f"Invalid created_at stored for log {log_id}: {created_at_result.message}"
            ),
        )
    distance_value: Distance | None = None
    if distance is not None:
        if not isinstance(distance, (float, int, str)):
            return Failure(
                InfraError(
                    f"Invalid distance type stored for log {log_id}: {type(distance).__name__}"
                )
            )
        distance_result = Distance.create(float(distance))
        if is_failure(distance_result):
            return Failure(
                InfraError(
                    f"Invalid distance stored for log {log_id}: {distance_result.message}"
                ),
            )
        distance_value = unwrap_success(distance_result)
    try:
        log_level = AppLogLevel(str(level))
        log_event = AppLogEvent(str(event_type))
    except ValueError as exc:
        return Failure(
            InfraError(f"Invalid event log enum value stored for log {log_id}: {exc}")
        )

    return Success(
        AppLogEntry(
            log_id=LogId(str(log_id)),
            created_at=unwrap_success(created_at_result),
            level=log_level,
            event=log_event,
            message=str(message),
            person_id=PersonId(str(person_id)) if person_id is not None else None,
            person_name=str(person_name) if person_name is not None else None,
            distance=distance_value,
        )
    )


def _decode_analysis_session(
    row: SqliteRow,
) -> Result[AnalysisSession, InfraError]:
    session_result = _decode_experiment_session(row)
    if is_failure(session_result):
        return session_result
    session = unwrap_success(session_result)
    try:
        status = ExperimentStatus(str(row[3]))
    except ValueError as exc:
        return Failure(
            InfraError(f"Invalid experiment status stored for session {row[0]}: {exc}")
        )
    return Success(AnalysisSession(session=session, status=status))


def _decode_analysis_trial(
    row: SqliteRow,
    sessions_by_id: dict[ExperimentSessionId, AnalysisSession],
) -> Result[AnalysisTrial, InfraError]:
    (
        _trial_id,
        session_id,
        created_at,
        matched,
        accepted_as_target,
        success,
        candidate_person_id,
        candidate_person_name,
        distance,
    ) = row
    session_key = ExperimentSessionId(str(session_id))
    analysis_session = sessions_by_id.get(session_key)
    if analysis_session is None:
        return Failure(
            InfraError(f"Unknown session_id stored for experiment trial: {session_id}")
        )
    trial_result = _decode_experiment_trial(
        analysis_session.session.session_id,
        (
            row[0],
            created_at,
            matched,
            accepted_as_target,
            success,
            candidate_person_id,
            candidate_person_name,
            distance,
        ),
    )
    if is_failure(trial_result):
        return trial_result
    trial = unwrap_success(trial_result)
    return Success(
        AnalysisTrial(
            trial=trial,
            scenario=analysis_session.session.scenario,
            target_person_id=analysis_session.session.target_person_id,
            target_person_name=analysis_session.session.target_person_name,
            threshold=analysis_session.session.threshold,
        )
    )


def _decode_experiment_session(
    row: SqliteRow,
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
    row: SqliteRow,
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
