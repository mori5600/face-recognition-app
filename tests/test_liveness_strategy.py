from app.domain.liveness import (
    LivenessChallengeKind,
    LivenessChallengeStep,
    LivenessSignals,
)
from app.strategy.liveness import (
    create_liveness_challenge_steps,
    evaluate_liveness_step,
)

EXPECTED_STEP_COUNT = 2


def test_create_liveness_challenge_steps_returns_unique_steps() -> None:
    steps = create_liveness_challenge_steps(step_count=EXPECTED_STEP_COUNT)

    assert len(steps) == EXPECTED_STEP_COUNT
    assert len({step.kind for step in steps}) == EXPECTED_STEP_COUNT


def test_blink_step_requires_neutral_then_trigger() -> None:
    step = LivenessChallengeStep(
        kind=LivenessChallengeKind.BLINK,
        instruction="1回まばたきしてください。",
    )
    neutral_signals = LivenessSignals(
        face_count=1,
        blink_left=0.05,
        blink_right=0.07,
        jaw_open=0.0,
        turn_score=0.0,
    )
    blink_signals = LivenessSignals(
        face_count=1,
        blink_left=0.8,
        blink_right=0.85,
        jaw_open=0.0,
        turn_score=0.0,
    )

    progress = evaluate_liveness_step(step, neutral_signals, neutral_ready=False)
    assert progress.neutral_ready is True
    assert progress.completed is False

    progress = evaluate_liveness_step(step, blink_signals, neutral_ready=True)
    assert progress.neutral_ready is False
    assert progress.completed is True


def test_turn_step_ignores_missing_face() -> None:
    step = LivenessChallengeStep(
        kind=LivenessChallengeKind.TURN_SIDE,
        instruction="顔を左右どちらかへ向けて戻してください。",
    )
    no_face_signals = LivenessSignals(
        face_count=0,
        blink_left=0.0,
        blink_right=0.0,
        jaw_open=0.0,
        turn_score=0.4,
    )

    progress = evaluate_liveness_step(step, no_face_signals, neutral_ready=False)

    assert progress.neutral_ready is False
    assert progress.completed is False
