from dataclasses import dataclass
from enum import StrEnum


class LivenessChallengeKind(StrEnum):
    BLINK = "blink"
    TURN_SIDE = "turn_side"
    MOUTH_OPEN = "mouth_open"


@dataclass(frozen=True)
class LivenessChallengeStep:
    kind: LivenessChallengeKind
    instruction: str


@dataclass(frozen=True)
class LivenessSignals:
    face_count: int
    blink_left: float
    blink_right: float
    jaw_open: float
    turn_score: float

    @property
    def blink_score(self) -> float:
        return min(self.blink_left, self.blink_right)

    @property
    def single_face_visible(self) -> bool:
        return self.face_count == 1
