import os

import pytest

from app.domain.results import is_failure, unwrap_success
from app.infra.encoding_protection import (
    PROTECTED_ENCODING_BLOB_PREFIX,
    default_encoding_protector,
)


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is available only on Windows.")
def test_default_encoding_protector_round_trip() -> None:
    protector = default_encoding_protector()
    plaintext = b"face-encoding-bytes"

    protected_result = protector.protect(plaintext)

    assert not is_failure(protected_result)
    protected_payload = unwrap_success(protected_result)
    assert protected_payload.startswith(PROTECTED_ENCODING_BLOB_PREFIX)

    unprotected_result = protector.unprotect(protected_payload)

    assert not is_failure(unprotected_result)
    assert unwrap_success(unprotected_result) == plaintext
