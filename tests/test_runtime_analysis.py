from pathlib import Path

from app.domain.errors import AppError
from app.domain.results import Failure, Result, Success, is_failure
from tests.test_runtime_experiment import _build_runtime


def test_open_analysis_report_starts_background_job_and_updates_ui_message(
    tmp_path: Path,
) -> None:
    runtime = _build_runtime(tmp_path)
    runtime._analysis_report_coordinator = _FakeAnalysisReportCoordinator()

    result = runtime.open_analysis_report()

    assert not is_failure(result)
    assert runtime.state.ui.message == "解析レポートを生成しています。"
    assert runtime.is_analysis_report_running() is True


def test_poll_background_tasks_updates_ui_message_on_success(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    report_path = tmp_path / "data" / "reports" / "analysis-report.html"
    runtime._analysis_report_coordinator = _FakeAnalysisReportCoordinator(
        poll_results=(Success(report_path),)
    )

    result = runtime.poll_background_tasks()

    assert not is_failure(result)
    assert runtime.state.ui.message == "解析レポートを開きました: analysis-report.html"


def test_open_analysis_report_propagates_start_failure_to_ui(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    runtime._analysis_report_coordinator = _FakeAnalysisReportCoordinator(
        start_result=Failure(AppError("解析レポートは既に生成中です。"))
    )

    result = runtime.open_analysis_report()

    assert is_failure(result)
    assert runtime.state.ui.message == "解析レポートは既に生成中です。"


def test_poll_background_tasks_propagates_failure_to_ui(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    runtime._analysis_report_coordinator = _FakeAnalysisReportCoordinator(
        poll_results=(Failure(AppError("解析レポートの書き出しに失敗しました。")),)
    )

    result = runtime.poll_background_tasks()

    assert is_failure(result)
    assert runtime.state.ui.message == "解析レポートの書き出しに失敗しました。"


class _FakeAnalysisReportCoordinator:
    def __init__(
        self,
        start_result: Result[None, AppError] | None = None,
        poll_results: tuple[Result[Path | None, AppError], ...] = (),
    ) -> None:
        self._start_result = Success(None) if start_result is None else start_result
        self._poll_results = list(poll_results)
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def start(self) -> Result[None, AppError]:
        if is_failure(self._start_result):
            return self._start_result
        self._running = True
        return self._start_result

    def poll(self) -> Result[Path | None, AppError]:
        if len(self._poll_results) == 0:
            return Success(None)

        self._running = False
        return self._poll_results.pop(0)
