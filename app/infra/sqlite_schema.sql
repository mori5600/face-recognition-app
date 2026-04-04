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
