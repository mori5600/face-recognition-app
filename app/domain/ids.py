from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class PersonId:
    value: str

    @classmethod
    def new(cls) -> "PersonId":
        return cls(str(uuid4()))


@dataclass(frozen=True)
class FaceId:
    value: str

    @classmethod
    def new(cls) -> "FaceId":
        return cls(str(uuid4()))


@dataclass(frozen=True)
class EncodingId:
    value: str

    @classmethod
    def new(cls) -> "EncodingId":
        return cls(str(uuid4()))


@dataclass(frozen=True)
class LogId:
    value: str

    @classmethod
    def new(cls) -> "LogId":
        return cls(str(uuid4()))


@dataclass(frozen=True)
class ExperimentSessionId:
    value: str

    @classmethod
    def new(cls) -> "ExperimentSessionId":
        return cls(str(uuid4()))


@dataclass(frozen=True)
class ExperimentTrialId:
    value: str

    @classmethod
    def new(cls) -> "ExperimentTrialId":
        return cls(str(uuid4()))
