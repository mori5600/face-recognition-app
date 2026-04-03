from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeGuard, TypeVar

from app.domain.errors import ErrorDetail

T = TypeVar("T")
E = TypeVar("E", bound=ErrorDetail)


@dataclass(frozen=True)
class Success(Generic[T]):
    value: T


@dataclass(frozen=True)
class Failure(Generic[E]):
    error: E

    @property
    def message(self) -> str:
        return self.error.message


Result: TypeAlias = Success[T] | Failure[E]


def is_failure(result: Result[T, E]) -> TypeGuard[Failure[E]]:
    return isinstance(result, Failure)


def is_success(result: Result[T, E]) -> TypeGuard[Success[T]]:
    return isinstance(result, Success)


def unwrap_success(result: Result[T, E]) -> T:
    if isinstance(result, Failure):
        raise RuntimeError(f"Tried to unwrap Failure: {result.message}")
    return result.value
