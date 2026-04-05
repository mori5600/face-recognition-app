from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Protocol

from app.app.analysis_report import write_analysis_report
from app.domain.errors import AppError
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.infra.app_paths import AppPaths

type AnalysisReportWriteResult = Result[Path, AppError]
type AnalysisReportWriter = Callable[[AppPaths, bool], AnalysisReportWriteResult]


class AnalysisReportCoordinatorProtocol(Protocol):
    def is_running(self) -> bool: ...

    def start(self) -> Result[None, AppError]: ...

    def poll(self) -> Result[Path | None, AppError]: ...


class AnalysisReportCoordinator:
    def __init__(
        self,
        paths: AppPaths,
        writer: AnalysisReportWriter = write_analysis_report,
    ) -> None:
        self._paths = paths
        self._writer = writer
        self._result_queue: Queue[AnalysisReportWriteResult] = Queue(maxsize=1)
        self._thread: Thread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> Result[None, AppError]:
        if self.is_running():
            return Failure(AppError("解析レポートは既に生成中です。"))

        self._result_queue = Queue(maxsize=1)
        self._thread = Thread(target=self._run, name="analysis-report", daemon=True)
        self._thread.start()
        return Success(None)

    def poll(self) -> Result[Path | None, AppError]:
        try:
            result = self._result_queue.get_nowait()
        except Empty:
            return Success(None)

        self._thread = None
        if is_failure(result):
            return Failure(result.error)
        return Success(unwrap_success(result))

    def _run(self) -> None:
        try:
            result = self._writer(self._paths, True)
        except Exception as exc:
            result = Failure(AppError(f"解析レポートの書き出しに失敗しました: {exc}"))
        self._result_queue.put(result)
