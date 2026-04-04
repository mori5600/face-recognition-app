import pytest

from app.domain.entities import MatchCandidate, MatchResult
from app.domain.experiments import ExperimentScenario
from app.domain.ids import PersonId
from app.domain.results import is_failure, unwrap_success
from app.domain.value_objects import DisplayName, Distance
from app.strategy.experiment import assess_experiment_trial, summarize_experiment_trials

EXPECTED_TRIAL_COUNT = 2


def test_assess_experiment_trial_for_genuine_accepts_target_match() -> None:
    target_id = PersonId("person-a")
    match_result = MatchResult(
        candidate=MatchCandidate(
            person_id=target_id,
            display_name=_display_name("alice"),
            distance=_distance(0.591),
        ),
        matched=True,
    )

    assessment = assess_experiment_trial(
        ExperimentScenario.GENUINE,
        target_id,
        match_result,
    )

    assert assessment.accepted_as_target is True
    assert assessment.success is True
    assert assessment.candidate_person_name == "alice"


def test_assess_experiment_trial_for_impostor_rejects_target_acceptance() -> None:
    target_id = PersonId("person-a")
    match_result = MatchResult(
        candidate=MatchCandidate(
            person_id=target_id,
            display_name=_display_name("alice"),
            distance=_distance(0.744),
        ),
        matched=True,
    )

    assessment = assess_experiment_trial(
        ExperimentScenario.IMPOSTOR,
        target_id,
        match_result,
    )

    assert assessment.accepted_as_target is True
    assert assessment.success is False


def test_summarize_experiment_trials_returns_rates_and_average_distance() -> None:
    target_id = PersonId("person-a")
    genuine_assessment = assess_experiment_trial(
        ExperimentScenario.GENUINE,
        target_id,
        MatchResult(
            candidate=MatchCandidate(
                person_id=target_id,
                display_name=_display_name("alice"),
                distance=_distance(0.600),
            ),
            matched=True,
        ),
    )
    failed_assessment = assess_experiment_trial(
        ExperimentScenario.GENUINE,
        target_id,
        MatchResult(
            candidate=MatchCandidate(
                person_id=PersonId("person-b"),
                display_name=_display_name("bob"),
                distance=_distance(1.020),
            ),
            matched=False,
        ),
    )

    metrics = summarize_experiment_trials((genuine_assessment, failed_assessment))

    assert metrics.trial_count == EXPECTED_TRIAL_COUNT
    assert metrics.success_count == 1
    assert metrics.target_accept_count == 1
    assert metrics.success_rate == pytest.approx(0.5)
    assert metrics.target_accept_rate == pytest.approx(0.5)
    assert metrics.average_distance is not None
    assert metrics.average_distance.value == pytest.approx(0.81)


def _display_name(value: str) -> DisplayName:
    result = DisplayName.create(value)
    if is_failure(result):
        raise AssertionError(result.message)
    return unwrap_success(result)


def _distance(value: float) -> Distance:
    result = Distance.create(value)
    if is_failure(result):
        raise AssertionError(result.message)
    return unwrap_success(result)
