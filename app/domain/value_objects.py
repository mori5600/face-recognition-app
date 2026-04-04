from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from app.domain.errors import DomainError
from app.domain.raw_types import RawBoundingBox, RawEncoding, RawName
from app.domain.results import Failure, Result, Success

MAX_DISPLAY_NAME_LENGTH = 64
FACE_ENCODING_DIMENSION = 128


@dataclass(frozen=True)
class DisplayName:
    value: str

    @classmethod
    def create(cls, raw: RawName | str) -> Result["DisplayName", DomainError]:
        normalized = str(raw).strip()
        if normalized == "":
            return Failure(DomainError("DisplayName must not be empty."))
        if len(normalized) > MAX_DISPLAY_NAME_LENGTH:
            return Failure(
                DomainError(
                    f"DisplayName must be at most {MAX_DISPLAY_NAME_LENGTH} characters."
                ),
            )
        display_name = DisplayName(normalized)
        return Success(display_name)


@dataclass(frozen=True)
class FaceEncoding:
    value: np.ndarray

    @classmethod
    def create(
        cls,
        raw: RawEncoding | Iterable[float] | np.ndarray,
    ) -> Result["FaceEncoding", DomainError]:
        encoding = np.asarray(raw, dtype=np.float32).reshape(-1)
        if encoding.shape != (FACE_ENCODING_DIMENSION,):
            return Failure(
                DomainError(
                    f"FaceEncoding must have {FACE_ENCODING_DIMENSION} values, got {encoding.shape[0]}.",
                ),
            )
        if not np.isfinite(encoding).all():
            return Failure(
                DomainError("FaceEncoding must not contain NaN or infinity.")
            )
        face_encoding = FaceEncoding(encoding)
        return Success(face_encoding)

    def to_row_vector(self) -> np.ndarray:
        return self.value.reshape(1, -1)


@dataclass(frozen=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def create(cls, raw: RawBoundingBox) -> Result["BoundingBox", DomainError]:
        left, top, right, bottom = raw
        if min(left, top, right, bottom) < 0:
            return Failure(
                DomainError("BoundingBox must not contain negative coordinates.")
            )
        if left >= right:
            return Failure(DomainError("BoundingBox must satisfy left < right."))
        if top >= bottom:
            return Failure(DomainError("BoundingBox must satisfy top < bottom."))
        bounding_box = BoundingBox(left=left, top=top, right=right, bottom=bottom)
        return Success(bounding_box)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.left + (self.width / 2.0),
            self.top + (self.height / 2.0),
        )


@dataclass(frozen=True)
class Distance:
    value: float

    @classmethod
    def create(cls, raw: float) -> Result["Distance", DomainError]:
        if not np.isfinite(raw):
            return Failure(DomainError("Distance must be finite."))
        if raw < 0:
            return Failure(DomainError("Distance must be non-negative."))
        distance = Distance(float(raw))
        return Success(distance)


@dataclass(frozen=True)
class Timestamp:
    value: datetime

    @classmethod
    def now(cls) -> "Timestamp":
        return cls(datetime.now(UTC))

    @classmethod
    def create(cls, raw: datetime) -> Result["Timestamp", DomainError]:
        if not isinstance(raw, datetime):
            return Failure(DomainError("Timestamp requires a datetime value."))
        timestamp = Timestamp(raw)
        return Success(timestamp)
