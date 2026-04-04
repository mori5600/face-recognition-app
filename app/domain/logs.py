from dataclasses import dataclass
from enum import StrEnum

from app.domain.ids import LogId, PersonId
from app.domain.value_objects import Distance, Timestamp


class AppLogLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AppLogEvent(StrEnum):
    CAMERA_STARTED = "camera_started"
    CAMERA_STOPPED = "camera_stopped"
    CAMERA_ERROR = "camera_error"
    LIVENESS_STARTED = "liveness_started"
    LIVENESS_VERIFIED = "liveness_verified"
    LIVENESS_FAILED = "liveness_failed"
    LIVENESS_EXPIRED = "liveness_expired"
    PERSON_REGISTERED = "person_registered"
    PERSON_UPDATED = "person_updated"
    PERSON_DELETED = "person_deleted"
    MATCH_SUCCEEDED = "match_succeeded"
    MATCH_REJECTED = "match_rejected"
    MATCH_FAILED = "match_failed"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_STOPPED = "experiment_stopped"


@dataclass(frozen=True)
class AppLogEntry:
    log_id: LogId
    created_at: Timestamp
    level: AppLogLevel
    event: AppLogEvent
    message: str
    person_id: PersonId | None = None
    person_name: str | None = None
    distance: Distance | None = None
