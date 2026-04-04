from enum import StrEnum


class CameraStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class RegistrationStatus(StrEnum):
    IDLE = "idle"
    SUCCESS = "success"
    ERROR = "error"


class MatchingStatus(StrEnum):
    IDLE = "idle"
    SUCCESS = "success"
    ERROR = "error"


class LivenessStatus(StrEnum):
    IDLE = "idle"
    CHALLENGE = "challenge"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class ExperimentStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class DeferredActionKind(StrEnum):
    MATCH = "match"
    REGISTER = "register"
