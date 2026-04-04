from dataclasses import dataclass, field

from app.domain.entities import DetectedFace, MatchResult, RegisteredPerson
from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import PersonId
from app.domain.liveness import LivenessChallengeStep
from app.domain.logs import AppLogEntry
from app.domain.raw_types import RawFrame
from app.domain.statuses import (
    CameraStatus,
    DeferredActionKind,
    ExperimentStatus,
    LivenessStatus,
    MatchingStatus,
    RegistrationStatus,
)
from app.domain.value_objects import Distance


@dataclass(frozen=True)
class CameraState:
    status: CameraStatus = CameraStatus.IDLE
    latest_frame: RawFrame | None = None
    detected_faces: tuple[DetectedFace, ...] = ()
    last_error: str | None = None


@dataclass(frozen=True)
class RegistrationState:
    draft_name: str = ""
    status: RegistrationStatus = RegistrationStatus.IDLE
    last_registered_person_id: PersonId | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class MatchingState:
    status: MatchingStatus = MatchingStatus.IDLE
    results: tuple[MatchResult, ...] = ()
    last_error: str | None = None


@dataclass(frozen=True)
class PeopleState:
    persons: tuple[RegisteredPerson, ...] = ()


@dataclass(frozen=True)
class LogState:
    entries: tuple[AppLogEntry, ...] = ()


@dataclass(frozen=True)
class ExperimentState:
    status: ExperimentStatus = ExperimentStatus.IDLE
    session: ExperimentSession | None = None
    trials: tuple[ExperimentTrial, ...] = ()
    latest_distance: Distance | None = None
    last_success: bool | None = None
    scenario: ExperimentScenario | None = None


@dataclass(frozen=True)
class LivenessState:
    status: LivenessStatus = LivenessStatus.IDLE
    requested_action: str | None = None
    deferred_action: DeferredActionKind | None = None
    deferred_name: str | None = None
    challenge_steps: tuple[LivenessChallengeStep, ...] = ()
    current_step_index: int = 0
    neutral_ready: bool = False
    started_at_ms: int | None = None
    verified_until_ms: int | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class UiState:
    message: str = ""
    selected_person_id: PersonId | None = None


@dataclass(frozen=True)
class AppState:
    camera: CameraState = field(default_factory=CameraState)
    registration: RegistrationState = field(default_factory=RegistrationState)
    matching: MatchingState = field(default_factory=MatchingState)
    people: PeopleState = field(default_factory=PeopleState)
    logs: LogState = field(default_factory=LogState)
    experiment: ExperimentState = field(default_factory=ExperimentState)
    liveness: LivenessState = field(default_factory=LivenessState)
    ui: UiState = field(default_factory=UiState)
