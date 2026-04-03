from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDetail:
    message: str


@dataclass(frozen=True)
class DomainError(ErrorDetail):
    pass


@dataclass(frozen=True)
class InfraError(ErrorDetail):
    pass


@dataclass(frozen=True)
class AppError(ErrorDetail):
    pass
