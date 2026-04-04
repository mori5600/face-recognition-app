from pathlib import Path

import pytest

from app.domain.errors import AppError
from app.domain.results import Failure, Success, is_failure
from tests.test_runtime_experiment import _build_runtime


def test_open_analysis_report_updates_ui_message_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_runtime(tmp_path)
    report_path = tmp_path / "data" / "reports" / "analysis-report.html"

    def fake_write_analysis_report(
        paths: object,
        open_in_browser: bool = False,
    ) -> Success[Path]:
        _ = paths
        assert open_in_browser is True
        return Success(report_path)

    monkeypatch.setattr(
        "app.app.runtime.write_analysis_report",
        fake_write_analysis_report,
    )

    result = runtime.open_analysis_report()

    assert not is_failure(result)
    assert runtime.state.ui.message == "解析レポートを開きました: analysis-report.html"


def test_open_analysis_report_propagates_failure_to_ui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _build_runtime(tmp_path)

    def fake_write_analysis_report(
        paths: object,
        open_in_browser: bool = False,
    ) -> Failure[AppError]:
        _ = paths
        _ = open_in_browser
        return Failure(AppError("解析レポートの書き出しに失敗しました。"))

    monkeypatch.setattr(
        "app.app.runtime.write_analysis_report",
        fake_write_analysis_report,
    )

    result = runtime.open_analysis_report()

    assert is_failure(result)
    assert runtime.state.ui.message == "解析レポートの書き出しに失敗しました。"
