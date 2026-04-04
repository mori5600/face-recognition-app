from dataclasses import dataclass
from random import Random, SystemRandom

from app.domain.liveness import (
    LivenessChallengeKind,
    LivenessChallengeStep,
    LivenessSignals,
)

BLINK_NEUTRAL_THRESHOLD = 0.18
BLINK_TRIGGER_THRESHOLD = 0.55
MOUTH_NEUTRAL_THRESHOLD = 0.10
MOUTH_TRIGGER_THRESHOLD = 0.30
TURN_NEUTRAL_THRESHOLD = 0.08
TURN_TRIGGER_THRESHOLD = 0.18

DEFAULT_CHALLENGE_STEPS = (
    LivenessChallengeStep(
        kind=LivenessChallengeKind.BLINK,
        instruction="1回まばたきしてください。",
    ),
    LivenessChallengeStep(
        kind=LivenessChallengeKind.TURN_SIDE,
        instruction="顔を左右どちらかへ向けて戻してください。",
    ),
    LivenessChallengeStep(
        kind=LivenessChallengeKind.MOUTH_OPEN,
        instruction="口を開けて閉じてください。",
    ),
)


@dataclass(frozen=True)
class LivenessStepProgress:
    neutral_ready: bool
    completed: bool


def create_liveness_challenge_steps(
    rng: Random | None = None,
    step_count: int = 2,
) -> tuple[LivenessChallengeStep, ...]:
    generator = rng or SystemRandom()
    if step_count >= len(DEFAULT_CHALLENGE_STEPS):
        return tuple(
            generator.sample(DEFAULT_CHALLENGE_STEPS, len(DEFAULT_CHALLENGE_STEPS))
        )
    return tuple(generator.sample(DEFAULT_CHALLENGE_STEPS, step_count))


def evaluate_liveness_step(
    step: LivenessChallengeStep,
    signals: LivenessSignals | None,
    neutral_ready: bool,
) -> LivenessStepProgress:
    if signals is None or not signals.single_face_visible:
        return LivenessStepProgress(neutral_ready=neutral_ready, completed=False)

    is_neutral = _step_is_neutral(step.kind, signals)
    next_neutral_ready = neutral_ready or is_neutral
    is_triggered = _step_is_triggered(step.kind, signals)
    completed = next_neutral_ready and is_triggered
    return LivenessStepProgress(
        neutral_ready=False if completed else next_neutral_ready,
        completed=completed,
    )


def _step_is_neutral(
    kind: LivenessChallengeKind,
    signals: LivenessSignals,
) -> bool:
    if kind is LivenessChallengeKind.BLINK:
        return signals.blink_score < BLINK_NEUTRAL_THRESHOLD
    if kind is LivenessChallengeKind.MOUTH_OPEN:
        return signals.jaw_open < MOUTH_NEUTRAL_THRESHOLD
    return abs(signals.turn_score) < TURN_NEUTRAL_THRESHOLD


def _step_is_triggered(
    kind: LivenessChallengeKind,
    signals: LivenessSignals,
) -> bool:
    if kind is LivenessChallengeKind.BLINK:
        return signals.blink_score >= BLINK_TRIGGER_THRESHOLD
    if kind is LivenessChallengeKind.MOUTH_OPEN:
        return signals.jaw_open >= MOUTH_TRIGGER_THRESHOLD
    return abs(signals.turn_score) >= TURN_TRIGGER_THRESHOLD
