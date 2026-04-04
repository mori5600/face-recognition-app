from dataclasses import dataclass
from typing import TypeGuard

from app.domain.errors import ErrorDetail


@dataclass(frozen=True)
class Success[T]:
    value: T


@dataclass(frozen=True)
class Failure[E: ErrorDetail]:
    error: E

    @property
    def message(self) -> str:
        return self.error.message


type Result[T, E: ErrorDetail] = Success[T] | Failure[E]


def is_failure[T, E: ErrorDetail](result: Result[T, E]) -> TypeGuard[Failure[E]]:
    return isinstance(result, Failure)


def is_success[T, E: ErrorDetail](result: Result[T, E]) -> TypeGuard[Success[T]]:
    return isinstance(result, Success)


def unwrap_success[T, E: ErrorDetail](result: Result[T, E]) -> T:
    if isinstance(result, Failure):
        raise RuntimeError(f"Tried to unwrap Failure: {result.message}")
    return result.value
