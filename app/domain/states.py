from dataclasses import dataclass, field

from app.domain.entities import DetectedFace, MatchResult, RegisteredPerson
from app.domain.ids import PersonId
from app.domain.raw_types import RawFrame


@dataclass(frozen=True)
class CameraState:
    status: str = "idle"
    latest_frame: RawFrame | None = None
    detected_faces: tuple[DetectedFace, ...] = ()
    last_error: str | None = None


@dataclass(frozen=True)
class RegistrationState:
    draft_name: str = ""
    status: str = "idle"
    last_registered_person_id: PersonId | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class MatchingState:
    status: str = "idle"
    results: tuple[MatchResult, ...] = ()
    last_error: str | None = None


@dataclass(frozen=True)
class PeopleState:
    persons: tuple[RegisteredPerson, ...] = ()


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
    ui: UiState = field(default_factory=UiState)
