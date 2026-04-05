import time
from pathlib import Path

from app.app.analysis_tasks import AnalysisReportCoordinator
from app.domain.errors import AppError
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.infra.app_paths import AppPaths


def test_analysis_report_coordinator_returns_completed_report_path(
    tmp_path: Path,
) -> None:
    paths = _build_paths(tmp_path)
    report_path = tmp_path / "data" / "reports" / "analysis-report.html"

    def fake_writer(
        paths_arg: AppPaths,
        open_in_browser: bool,
    ) -> Result[Path, AppError]:
        _ = paths_arg
        assert open_in_browser is True
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("ok", encoding="utf-8")
        return Success(report_path)

    coordinator = AnalysisReportCoordinator(paths, writer=fake_writer)

    start_result = coordinator.start()
    assert not is_failure(start_result)

    completed_result = _poll_until_done(coordinator)
    assert not is_failure(completed_result)
    completed_path = unwrap_success(completed_result)
    assert completed_path == report_path


def test_analysis_report_coordinator_returns_failure_from_worker(
    tmp_path: Path,
) -> None:
    paths = _build_paths(tmp_path)

    def fake_writer(
        paths_arg: AppPaths,
        open_in_browser: bool,
    ) -> Result[Path, AppError]:
        _ = paths_arg
        _ = open_in_browser
        return Failure(AppError("解析レポートの書き出しに失敗しました。"))

    coordinator = AnalysisReportCoordinator(paths, writer=fake_writer)

    start_result = coordinator.start()
    assert not is_failure(start_result)

    completed_result = _poll_until_done(coordinator)
    assert is_failure(completed_result)
    assert completed_result.message == "解析レポートの書き出しに失敗しました。"


def _poll_until_done(
    coordinator: AnalysisReportCoordinator,
) -> Result[Path | None, AppError]:
    for _ in range(50):
        result = coordinator.poll()
        if is_failure(result):
            return result
        completed_path = unwrap_success(result)
        if completed_path is not None:
            return Success(completed_path)
        time.sleep(0.01)
    raise AssertionError("Analysis report coordinator did not finish in time.")


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
