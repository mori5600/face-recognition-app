from app.app.commands import (
    CameraCheckCommand,
    DoctorCommand,
    DownloadModelsCommand,
    UiCommand,
)
from app.domain.results import is_failure, unwrap_success
from app.gateways.camera_gateway import close_camera, open_camera, read_frame
from app.gateways.face_gateway import OpenCvFaceEngineConfig, load_face_engine
from app.gateways.liveness_gateway import (
    MediaPipeLivenessEngineConfig,
    close_liveness_engine,
    load_liveness_engine,
)
from app.gateways.sqlite_gateway import initialize_database, load_people
from app.infra.app_paths import AppPaths
from app.infra.download_models import download_models
from app.ui.main_window import MainWindow


def handle_doctor(_: DoctorCommand) -> int:
    paths = AppPaths.default()
    print(f"Project root: {paths.root_dir}")
    print(f"Database:      {paths.database_path}")
    print(f"YuNet model:   {paths.yunet_model_path}")
    print(f"SFace model:   {paths.sface_model_path}")
    print(f"Liveness:      {paths.mediapipe_face_landmarker_path}")

    database_result = initialize_database(paths)
    if is_failure(database_result):
        print(f"[ERROR] {database_result.message}")
        return 1

    engine_result = load_face_engine(OpenCvFaceEngineConfig.from_app_paths(paths))
    if is_failure(engine_result):
        print(f"[ERROR] {engine_result.message}")
        print(
            "Run `uv run face-recognition-download-models` first if the model files are missing."
        )
        return 1

    liveness_result = load_liveness_engine(
        MediaPipeLivenessEngineConfig(paths.mediapipe_face_landmarker_path)
    )
    if is_failure(liveness_result):
        print(f"[ERROR] {liveness_result.message}")
        return 1
    liveness_engine = unwrap_success(liveness_result)

    people_result = load_people(paths)
    if is_failure(people_result):
        print(f"[ERROR] {people_result.message}")
        close_liveness_engine(liveness_engine)
        return 1
    people = unwrap_success(people_result)

    print("[OK] OpenCV FaceDetectorYN and FaceRecognizerSF loaded successfully.")
    print("[OK] MediaPipe Face Landmarker loaded successfully.")
    print(f"[OK] Loaded {len(people.persons)} registered people from SQLite.")
    close_liveness_engine(liveness_engine)
    return 0


def handle_download_models(_: DownloadModelsCommand) -> int:
    result = download_models()
    if is_failure(result):
        print(f"[ERROR] {result.message}")
        return 1
    downloaded_files = unwrap_success(result)

    print("[OK] Model files are ready.")
    for file_path in downloaded_files:
        print(f" - {file_path}")
    return 0


def handle_camera_check(command: CameraCheckCommand) -> int:
    open_result = open_camera(command.camera_index)
    if is_failure(open_result):
        print(f"[ERROR] {open_result.message}")
        return 1
    handle = unwrap_success(open_result)

    try:
        frame_result = read_frame(handle)
        if is_failure(frame_result):
            print(f"[ERROR] {frame_result.message}")
            return 1
        frame = unwrap_success(frame_result)

        height, width = frame.shape[:2]
        print(
            f"[OK] Captured frame from camera {command.camera_index}: {width}x{height}"
        )
        return 0
    finally:
        close_result = close_camera(handle)
        if is_failure(close_result):
            print(f"[WARN] {close_result.message}")


def handle_ui(_: UiCommand) -> int:
    window = MainWindow(auto_start_camera=True)
    window.mainloop()
    return 0
