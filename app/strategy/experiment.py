from dataclasses import dataclass

from app.domain.entities import MatchResult
from app.domain.experiments import ExperimentScenario
from app.domain.ids import PersonId
from app.domain.value_objects import Distance


@dataclass(frozen=True)
class ExperimentTrialAssessment:
    matched: bool
    accepted_as_target: bool
    success: bool
    candidate_person_id: PersonId | None
    candidate_person_name: str | None
    distance: Distance | None


@dataclass(frozen=True)
class ExperimentMetrics:
    trial_count: int
    success_count: int
    target_accept_count: int
    success_rate: float | None
    target_accept_rate: float | None
    average_distance: Distance | None


def assess_experiment_trial(
    scenario: ExperimentScenario,
    target_person_id: PersonId,
    match_result: MatchResult,
) -> ExperimentTrialAssessment:
    candidate = match_result.candidate
    accepted_as_target = (
        match_result.matched
        and candidate is not None
        and candidate.person_id == target_person_id
    )
    success = (
        accepted_as_target
        if scenario is ExperimentScenario.GENUINE
        else not accepted_as_target
    )
    return ExperimentTrialAssessment(
        matched=match_result.matched,
        accepted_as_target=accepted_as_target,
        success=success,
        candidate_person_id=candidate.person_id if candidate is not None else None,
        candidate_person_name=(
            candidate.display_name.value if candidate is not None else None
        ),
        distance=candidate.distance if candidate is not None else None,
    )


def summarize_experiment_trials(
    assessments: tuple[ExperimentTrialAssessment, ...],
) -> ExperimentMetrics:
    trial_count = len(assessments)
    if trial_count == 0:
        return ExperimentMetrics(
            trial_count=0,
            success_count=0,
            target_accept_count=0,
            success_rate=None,
            target_accept_rate=None,
            average_distance=None,
        )

    success_count = sum(1 for assessment in assessments if assessment.success)
    target_accept_count = sum(
        1 for assessment in assessments if assessment.accepted_as_target
    )
    distances = tuple(
        assessment.distance.value
        for assessment in assessments
        if assessment.distance is not None
    )
    average_distance: Distance | None = None
    if len(distances) > 0:
        average_value = sum(distances) / len(distances)
        average_distance = Distance(average_value)

    return ExperimentMetrics(
        trial_count=trial_count,
        success_count=success_count,
        target_accept_count=target_accept_count,
        success_rate=success_count / trial_count,
        target_accept_rate=target_accept_count / trial_count,
        average_distance=average_distance,
    )
