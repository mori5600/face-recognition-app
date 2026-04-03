from dataclasses import dataclass
from typing import Callable

from app.domain.entities import DetectedFace, MatchCandidate, MatchResult
from app.domain.errors import DomainError
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.domain.states import PeopleState
from app.domain.value_objects import Distance, FaceEncoding


@dataclass(frozen=True)
class MatchingThreshold:
    value: float = 1.128


class NearestEncodingMatcher:
    def __init__(self, threshold: MatchingThreshold | None = None) -> None:
        self._threshold = threshold or MatchingThreshold()

    def match(
        self,
        face: DetectedFace,
        people: PeopleState,
        compare_distance: Callable[
            [FaceEncoding, FaceEncoding], Result[Distance, DomainError]
        ],
    ) -> Result[MatchResult, DomainError]:
        if len(people.persons) == 0:
            return Success(MatchResult(candidate=None, matched=False))

        best_candidate: MatchCandidate | None = None
        for person in people.persons:
            for encoding in person.encodings:
                distance_result = compare_distance(face.encoding, encoding)
                if is_failure(distance_result):
                    return Failure(distance_result.error)
                distance = unwrap_success(distance_result)

                candidate = MatchCandidate(
                    person_id=person.person_id,
                    display_name=person.display_name,
                    distance=distance,
                )
                if (
                    best_candidate is None
                    or candidate.distance.value < best_candidate.distance.value
                ):
                    best_candidate = candidate

        if best_candidate is None:
            return Success(MatchResult(candidate=None, matched=False))

        matched = best_candidate.distance.value <= self._threshold.value
        return Success(MatchResult(candidate=best_candidate, matched=matched))


class NearestPersonMatcher:
    def __init__(self, threshold: MatchingThreshold | None = None) -> None:
        self._threshold = threshold or MatchingThreshold()

    def match(
        self,
        face: DetectedFace,
        people: PeopleState,
        compare_distance: Callable[
            [FaceEncoding, FaceEncoding], Result[Distance, DomainError]
        ],
    ) -> Result[MatchResult, DomainError]:
        if len(people.persons) == 0:
            return Success(MatchResult(candidate=None, matched=False))

        best_candidate: MatchCandidate | None = None
        for person in people.persons:
            person_distances: list[float] = []
            for encoding in person.encodings:
                distance_result = compare_distance(face.encoding, encoding)
                if is_failure(distance_result):
                    return Failure(distance_result.error)
                distance = unwrap_success(distance_result)
                person_distances.append(distance.value)

            if len(person_distances) == 0:
                continue

            average_distance = sum(person_distances) / len(person_distances)
            distance_value = Distance.create(average_distance)
            if is_failure(distance_value):
                return Failure(distance_value.error)
            average_distance_value = unwrap_success(distance_value)

            candidate = MatchCandidate(
                person_id=person.person_id,
                display_name=person.display_name,
                distance=average_distance_value,
            )
            if (
                best_candidate is None
                or candidate.distance.value < best_candidate.distance.value
            ):
                best_candidate = candidate

        if best_candidate is None:
            return Success(MatchResult(candidate=None, matched=False))

        matched = best_candidate.distance.value <= self._threshold.value
        return Success(MatchResult(candidate=best_candidate, matched=matched))
