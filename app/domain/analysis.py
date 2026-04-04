from dataclasses import dataclass

from app.domain.entities import RegisteredPerson
from app.domain.experiments import (
    ExperimentScenario,
    ExperimentSession,
    ExperimentTrial,
)
from app.domain.ids import PersonId
from app.domain.logs import AppLogEntry
from app.domain.statuses import ExperimentStatus


@dataclass(frozen=True)
class AnalysisSession:
    session: ExperimentSession
    status: ExperimentStatus


@dataclass(frozen=True)
class AnalysisTrial:
    trial: ExperimentTrial
    scenario: ExperimentScenario
    target_person_id: PersonId
    target_person_name: str
    threshold: float


@dataclass(frozen=True)
class AnalysisSnapshot:
    people: tuple[RegisteredPerson, ...] = ()
    logs: tuple[AppLogEntry, ...] = ()
    sessions: tuple[AnalysisSession, ...] = ()
    trials: tuple[AnalysisTrial, ...] = ()
