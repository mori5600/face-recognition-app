from collections.abc import Callable
from typing import Protocol, cast

import cv2
import numpy as np

from app.domain.raw_types import RawFrame

Point = tuple[int, int]
Color = tuple[int, int, int]


class VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, object | None]: ...

    def release(self) -> None: ...


class FaceDetectorProtocol(Protocol):
    def setInputSize(self, input_size: tuple[int, int]) -> None: ...

    def detect(self, image: RawFrame) -> tuple[object, object | None]: ...


class FaceRecognizerProtocol(Protocol):
    def alignCrop(self, image: RawFrame, face: object) -> RawFrame: ...

    def feature(self, aligned_face: RawFrame) -> np.ndarray: ...

    def match(
        self, feature1: np.ndarray, feature2: np.ndarray, distance_type: int
    ) -> float: ...


VideoCaptureFactory = Callable[[int, int], VideoCaptureProtocol]
FaceDetectorCreate = Callable[
    [str, str, tuple[int, int], float, float, int],
    FaceDetectorProtocol,
]
FaceRecognizerCreate = Callable[[str, str], FaceRecognizerProtocol]
RectangleFn = Callable[[RawFrame, Point, Point, Color, int], object]
PutTextFn = Callable[[RawFrame, str, Point, int, float, Color, int, int], object]
CvtColorFn = Callable[[RawFrame, int], RawFrame]

CAP_DSHOW = int(getattr(cv2, "CAP_DSHOW", 700))
CAP_ANY = int(getattr(cv2, "CAP_ANY", 0))
FONT_HERSHEY_SIMPLEX = int(getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0))
LINE_AA = int(getattr(cv2, "LINE_AA", 16))
COLOR_BGR2RGB = int(getattr(cv2, "COLOR_BGR2RGB", 4))
FR_NORM_L2 = int(getattr(cv2, "FaceRecognizerSF_FR_NORM_L2", 1))

_video_capture_factory = cast(VideoCaptureFactory, getattr(cv2, "VideoCapture"))
_face_detector_create = cast(
    FaceDetectorCreate,
    getattr(getattr(cv2, "FaceDetectorYN"), "create"),
)
_face_recognizer_create = cast(
    FaceRecognizerCreate,
    getattr(getattr(cv2, "FaceRecognizerSF"), "create"),
)
_rectangle = cast(RectangleFn, getattr(cv2, "rectangle"))
_put_text = cast(PutTextFn, getattr(cv2, "putText"))
_cvt_color = cast(CvtColorFn, getattr(cv2, "cvtColor"))


def create_video_capture(camera_index: int, backend: int) -> VideoCaptureProtocol:
    return _video_capture_factory(camera_index, backend)


def create_face_detector(
    model_path: str,
    config_path: str,
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> FaceDetectorProtocol:
    return _face_detector_create(
        model_path,
        config_path,
        input_size,
        score_threshold,
        nms_threshold,
        top_k,
    )


def create_face_recognizer(
    model_path: str,
    config_path: str,
) -> FaceRecognizerProtocol:
    return _face_recognizer_create(model_path, config_path)


def draw_rectangle(
    image: RawFrame,
    top_left: Point,
    bottom_right: Point,
    color: Color,
    thickness: int,
) -> None:
    _rectangle(image, top_left, bottom_right, color, thickness)


def put_text(
    image: RawFrame,
    text: str,
    origin: Point,
    font_face: int,
    font_scale: float,
    color: Color,
    thickness: int,
    line_type: int,
) -> None:
    _put_text(
        image,
        text,
        origin,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    )


def convert_bgr_to_rgb(image: RawFrame) -> RawFrame:
    return _cvt_color(image, COLOR_BGR2RGB)
