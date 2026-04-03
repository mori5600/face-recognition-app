from dataclasses import dataclass

from app.domain.entities import DetectedFace
from app.domain.errors import DomainError
from app.domain.results import Failure, Result, Success


class SingleFaceOnlySelector:
    def select(
        self,
        faces: tuple[DetectedFace, ...],
        frame_size: tuple[int, int] | None = None,
    ) -> Result[DetectedFace, DomainError]:
        if len(faces) == 0:
            return Failure(DomainError("No face was detected."))
        if len(faces) > 1:
            return Failure(DomainError("Exactly one face is required."))
        return Success(faces[0])


class LargestFaceSelector:
    def select(
        self,
        faces: tuple[DetectedFace, ...],
        frame_size: tuple[int, int] | None = None,
    ) -> Result[DetectedFace, DomainError]:
        if len(faces) == 0:
            return Failure(DomainError("No face was detected."))
        selected_face = max(faces, key=lambda face: face.bounding_box.area)
        return Success(selected_face)


@dataclass(frozen=True)
class CenterFaceSelector:
    def select(
        self,
        faces: tuple[DetectedFace, ...],
        frame_size: tuple[int, int] | None = None,
    ) -> Result[DetectedFace, DomainError]:
        if len(faces) == 0:
            return Failure(DomainError("No face was detected."))
        if frame_size is None:
            return Failure(
                DomainError("Frame size is required for center-face selection.")
            )

        frame_width, frame_height = frame_size
        frame_center = (frame_width / 2.0, frame_height / 2.0)

        def distance_to_center(face: DetectedFace) -> float:
            face_center_x, face_center_y = face.bounding_box.center
            delta_x = face_center_x - frame_center[0]
            delta_y = face_center_y - frame_center[1]
            return (delta_x * delta_x) + (delta_y * delta_y)

        selected_face = min(faces, key=distance_to_center)
        return Success(selected_face)
