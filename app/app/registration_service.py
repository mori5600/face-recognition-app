from dataclasses import dataclass

from app.domain.entities import RegisteredPerson
from app.domain.errors import AppError
from app.domain.ids import PersonId
from app.domain.results import Failure, Result, Success, is_failure
from app.domain.value_objects import DisplayName, FaceEncoding, Timestamp
from app.gateways.sqlite_gateway import (
    append_encoding_to_person,
    insert_person_with_encodings,
)
from app.infra.app_paths import AppPaths


@dataclass(frozen=True)
class RegistrationPersistenceResult:
    updated_person: RegisteredPerson
    people: tuple[RegisteredPerson, ...]
    created: bool


def persist_registration(
    paths: AppPaths,
    people: tuple[RegisteredPerson, ...],
    display_name: DisplayName,
    encoding: FaceEncoding,
    now: Timestamp,
) -> Result[RegistrationPersistenceResult, AppError]:
    existing_person = _find_person_by_name(people, display_name.value)
    if existing_person is None:
        person = RegisteredPerson(
            person_id=PersonId.new(),
            display_name=display_name,
            encodings=(encoding,),
            created_at=now,
            updated_at=now,
        )
        insert_result = insert_person_with_encodings(paths, person)
        if is_failure(insert_result):
            return Failure(AppError(insert_result.message))
        return Success(
            RegistrationPersistenceResult(
                updated_person=person,
                people=(*people, person),
                created=True,
            )
        )

    append_result = append_encoding_to_person(
        paths,
        existing_person.person_id,
        encoding,
        now,
    )
    if is_failure(append_result):
        return Failure(AppError(append_result.message))

    updated_person = RegisteredPerson(
        person_id=existing_person.person_id,
        display_name=existing_person.display_name,
        encodings=(*existing_person.encodings, encoding),
        created_at=existing_person.created_at,
        updated_at=now,
    )
    updated_people = tuple(
        updated_person if person.person_id == updated_person.person_id else person
        for person in people
    )
    return Success(
        RegistrationPersistenceResult(
            updated_person=updated_person,
            people=updated_people,
            created=False,
        )
    )


def _find_person_by_name(
    people: tuple[RegisteredPerson, ...],
    display_name: str,
) -> RegisteredPerson | None:
    for person in people:
        if person.display_name.value == display_name:
            return person
    return None
