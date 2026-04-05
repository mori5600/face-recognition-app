import ctypes
import os
from ctypes import POINTER, byref, c_ubyte, wintypes
from typing import Protocol

from app.domain.errors import InfraError
from app.domain.results import Failure, Result, Success

PROTECTED_ENCODING_BLOB_PREFIX = b"FRENC01"
_DPAPI_OPTIONAL_ENTROPY = b"face-recognition-app:face-encoding:v1"
_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class EncodingProtectorProtocol(Protocol):
    def is_protected(self, payload: bytes) -> bool: ...

    def protect(self, plaintext: bytes) -> Result[bytes, InfraError]: ...

    def unprotect(self, payload: bytes) -> Result[bytes, InfraError]: ...


class UnsupportedEncodingProtector:
    def is_protected(self, payload: bytes) -> bool:
        return payload.startswith(PROTECTED_ENCODING_BLOB_PREFIX)

    def protect(self, plaintext: bytes) -> Result[bytes, InfraError]:
        _ = plaintext
        return Failure(
            InfraError("Encoding protection is not supported on this platform.")
        )

    def unprotect(self, payload: bytes) -> Result[bytes, InfraError]:
        _ = payload
        return Failure(
            InfraError("Encoding protection is not supported on this platform.")
        )


if os.name == "nt":
    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", POINTER(c_ubyte)),
        ]

    _crypt_protect_data = _crypt32.CryptProtectData
    _crypt_protect_data.argtypes = [
        POINTER(_DataBlob),
        wintypes.LPCWSTR,
        POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        POINTER(_DataBlob),
    ]
    _crypt_protect_data.restype = wintypes.BOOL

    _crypt_unprotect_data = _crypt32.CryptUnprotectData
    _crypt_unprotect_data.argtypes = [
        POINTER(_DataBlob),
        POINTER(wintypes.LPWSTR),
        POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        POINTER(_DataBlob),
    ]
    _crypt_unprotect_data.restype = wintypes.BOOL

    _local_free = _kernel32.LocalFree
    _local_free.argtypes = [ctypes.c_void_p]
    _local_free.restype = ctypes.c_void_p


class WindowsDpapiEncodingProtector:
    def is_protected(self, payload: bytes) -> bool:
        return payload.startswith(PROTECTED_ENCODING_BLOB_PREFIX)

    def protect(self, plaintext: bytes) -> Result[bytes, InfraError]:
        if os.name != "nt":
            return Failure(
                InfraError("Windows DPAPI is not available on this platform.")
            )
        protected_result = _dpapi_protect(plaintext)
        if isinstance(protected_result, Failure):
            return protected_result
        return Success(PROTECTED_ENCODING_BLOB_PREFIX + protected_result.value)

    def unprotect(self, payload: bytes) -> Result[bytes, InfraError]:
        if os.name != "nt":
            return Failure(
                InfraError("Windows DPAPI is not available on this platform.")
            )
        if not self.is_protected(payload):
            return Failure(InfraError("The encoding payload is not protected."))
        return _dpapi_unprotect(payload[len(PROTECTED_ENCODING_BLOB_PREFIX) :])


def default_encoding_protector() -> EncodingProtectorProtocol:
    if os.name == "nt":
        return WindowsDpapiEncodingProtector()
    return UnsupportedEncodingProtector()


def _dpapi_protect(plaintext: bytes) -> Result[bytes, InfraError]:
    input_blob, _input_buffer = _make_blob(plaintext)
    entropy_blob, _entropy_buffer = _make_blob(_DPAPI_OPTIONAL_ENTROPY)
    output_blob = _DataBlob()
    if not _crypt_protect_data(
        byref(input_blob),
        "Face Recognition App Encoding",
        byref(entropy_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        byref(output_blob),
    ):
        return Failure(
            InfraError(
                f"Failed to protect the face encoding with DPAPI: {_windows_error_message()}"
            )
        )
    try:
        return Success(_blob_bytes(output_blob))
    finally:
        _free_blob(output_blob)


def _dpapi_unprotect(ciphertext: bytes) -> Result[bytes, InfraError]:
    input_blob, _input_buffer = _make_blob(ciphertext)
    entropy_blob, _entropy_buffer = _make_blob(_DPAPI_OPTIONAL_ENTROPY)
    output_blob = _DataBlob()
    description = wintypes.LPWSTR()
    if not _crypt_unprotect_data(
        byref(input_blob),
        byref(description),
        byref(entropy_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        byref(output_blob),
    ):
        return Failure(
            InfraError(
                f"Failed to unprotect the face encoding with DPAPI: {_windows_error_message()}"
            )
        )
    try:
        return Success(_blob_bytes(output_blob))
    finally:
        if bool(description):
            _local_free(ctypes.cast(description, ctypes.c_void_p))
        _free_blob(output_blob)


def _make_blob(data: bytes):
    if len(data) == 0:
        return (_DataBlob(0, None), None)
    buffer = (c_ubyte * len(data)).from_buffer_copy(data)
    return (_DataBlob(len(data), buffer), buffer)


def _blob_bytes(blob: "_DataBlob") -> bytes:
    if blob.cbData == 0 or not bool(blob.pbData):
        return b""
    return bytes(ctypes.string_at(blob.pbData, blob.cbData))


def _free_blob(blob: "_DataBlob") -> None:
    if bool(blob.pbData):
        _local_free(ctypes.cast(blob.pbData, ctypes.c_void_p))


def _windows_error_message() -> str:
    error_code = ctypes.get_last_error()
    if error_code == 0:
        return "unknown error"
    try:
        return f"{error_code}: {ctypes.FormatError(error_code).strip()}"
    except (ValueError, OSError):
        return str(error_code)
