CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_encodings (
    encoding_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    encoding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons (person_id)
);

CREATE TABLE IF NOT EXISTS event_logs (
    log_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    person_id TEXT,
    person_name TEXT,
    distance REAL
);

CREATE INDEX IF NOT EXISTS idx_event_logs_created_at
ON event_logs (created_at DESC);

CREATE TABLE IF NOT EXISTS experiment_sessions (
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

CREATE TABLE IF NOT EXISTS experiment_trials (
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

CREATE INDEX IF NOT EXISTS idx_experiment_sessions_started_at
ON experiment_sessions (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_experiment_trials_session_created_at
ON experiment_trials (session_id, created_at ASC);
