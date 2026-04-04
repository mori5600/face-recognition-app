from dataclasses import dataclass
from enum import StrEnum

from app.domain.ids import ExperimentSessionId, ExperimentTrialId, PersonId
from app.domain.value_objects import Distance, Timestamp


class ExperimentScenario(StrEnum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"


@dataclass(frozen=True)
class ExperimentTrial:
    trial_id: ExperimentTrialId
    session_id: ExperimentSessionId
    created_at: Timestamp
    matched: bool
    accepted_as_target: bool
    success: bool
    candidate_person_id: PersonId | None = None
    candidate_person_name: str | None = None
    distance: Distance | None = None


@dataclass(frozen=True)
class ExperimentSession:
    session_id: ExperimentSessionId
    started_at: Timestamp
    scenario: ExperimentScenario
    target_person_id: PersonId
    target_person_name: str
    face_selector_key: str
    matching_mode_key: str
    threshold: float
    completed_at: Timestamp | None = None
