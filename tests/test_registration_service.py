from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.app.registration_service import persist_registration
from app.domain.results import is_failure, unwrap_success
from app.domain.value_objects import DisplayName, FaceEncoding, Timestamp
from app.gateways.sqlite_gateway import initialize_database
from app.infra.app_paths import AppPaths


def test_persist_registration_creates_new_person(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    now = _timestamp(datetime(2026, 4, 5, 12, 0, tzinfo=UTC))
    display_name = unwrap_success(DisplayName.create("alice"))
    encoding = unwrap_success(FaceEncoding.create(np.zeros(128, dtype=np.float32)))

    result = persist_registration(paths, (), display_name, encoding, now)

    assert not is_failure(result)
    persisted = unwrap_success(result)
    assert persisted.created is True
    assert len(persisted.people) == 1
    assert persisted.updated_person.display_name.value == "alice"


def test_persist_registration_appends_encoding_to_existing_person(
    tmp_path: Path,
) -> None:
    paths = _build_paths(tmp_path)
    init_result = initialize_database(paths)
    assert not is_failure(init_result)

    now = _timestamp(datetime(2026, 4, 5, 12, 0, tzinfo=UTC))
    display_name = unwrap_success(DisplayName.create("alice"))
    first_encoding = unwrap_success(
        FaceEncoding.create(np.zeros(128, dtype=np.float32))
    )
    second_encoding = unwrap_success(
        FaceEncoding.create(np.ones(128, dtype=np.float32))
    )
    seed_result = persist_registration(
        paths,
        (),
        display_name,
        first_encoding,
        now,
    )
    assert not is_failure(seed_result)
    existing_person = unwrap_success(seed_result).updated_person

    result = persist_registration(
        paths,
        (existing_person,),
        display_name,
        second_encoding,
        _timestamp(datetime(2026, 4, 5, 12, 1, tzinfo=UTC)),
    )

    assert not is_failure(result)
    persisted = unwrap_success(result)
    assert persisted.created is False
    assert len(persisted.people) == 1
    assert len(persisted.updated_person.encodings) == 2


def _build_paths(root_dir: Path) -> AppPaths:
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    default_paths = AppPaths.default()
    return AppPaths(
        root_dir=root_dir,
        models_dir=models_dir,
        data_dir=data_dir,
        database_path=data_dir / "people.db",
        yunet_model_path=models_dir / "face_detection_yunet_2023mar.onnx",
        sface_model_path=models_dir / "face_recognition_sface_2021dec.onnx",
        mediapipe_face_landmarker_path=models_dir / "face_landmarker.task",
        sqlite_schema_path=default_paths.sqlite_schema_path,
    )


def _timestamp(value: datetime) -> Timestamp:
    timestamp_result = Timestamp.create(value)
    if is_failure(timestamp_result):
        raise AssertionError(timestamp_result.message)
    return unwrap_success(timestamp_result)
