from dataclasses import dataclass

from app.domain.ids import FaceId, PersonId
from app.domain.value_objects import (
    BoundingBox,
    DisplayName,
    Distance,
    FaceEncoding,
    Timestamp,
)


@dataclass(frozen=True)
class DetectedFace:
    face_id: FaceId
    bounding_box: BoundingBox
    encoding: FaceEncoding


@dataclass(frozen=True)
class RegisteredPerson:
    person_id: PersonId
    display_name: DisplayName
    encodings: tuple[FaceEncoding, ...]
    created_at: Timestamp
    updated_at: Timestamp


@dataclass(frozen=True)
class MatchCandidate:
    person_id: PersonId
    display_name: DisplayName
    distance: Distance


@dataclass(frozen=True)
class MatchResult:
    candidate: MatchCandidate | None
    matched: bool
