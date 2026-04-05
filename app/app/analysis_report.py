import json
import webbrowser
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from html import escape
from pathlib import Path
from statistics import fmean

import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

from app.domain.analysis import AnalysisSession, AnalysisSnapshot, AnalysisTrial
from app.domain.errors import AppError
from app.domain.experiments import ExperimentScenario
from app.domain.logs import AppLogEntry, AppLogLevel
from app.domain.results import Failure, Result, Success, is_failure, unwrap_success
from app.domain.statuses import ExperimentStatus
from app.gateways.sqlite_gateway import (
    load_analysis_fingerprint,
    load_analysis_snapshot,
)
from app.infra.app_paths import AppPaths

REPORT_DIRECTORY_NAME = "reports"
REPORT_FILE_NAME = "analysis-report.html"
REPORT_METADATA_FILE_NAME = "analysis-report.meta.json"
PLOTLY_BUNDLE_FILE_NAME = "plotly.min.js"
REPORT_VERSION = 3
RECENT_SESSION_LIMIT = 8
RECENT_LOG_LIMIT = 12
THRESHOLD_POINT_COUNT = 25
PRIMARY_COLOR = "#2F9E7A"
PRIMARY_SOFT = "#E8F6F0"
SUCCESS_COLOR = "#006d3c"
SUCCESS_SOFT = "#e8f5ee"
DANGER_COLOR = "#9f1d1d"
DANGER_SOFT = "#fef2f2"
NEUTRAL_COLOR = "#5f6874"
NEUTRAL_SOFT = "#f3f4f6"
TEXT_PRIMARY = "#333333"
TEXT_MUTED = "#5f6874"
BORDER_COLOR = "#d1d5db"
BACKGROUND = "#f3f4f6"


@dataclass(frozen=True)
class TrialMetrics:
    trial_count: int
    success_rate: float | None
    accepted_rate: float | None
    average_distance: float | None


@dataclass(frozen=True)
class ReportCacheMetadata:
    report_version: int
    fingerprint: str


def write_analysis_report(
    paths: AppPaths,
    open_in_browser: bool = False,
) -> Result[Path, AppError]:
    report_path = paths.data_dir / REPORT_DIRECTORY_NAME / REPORT_FILE_NAME
    report_directory = report_path.parent
    metadata_path = report_directory / REPORT_METADATA_FILE_NAME
    bundle_path = report_directory / PLOTLY_BUNDLE_FILE_NAME

    fingerprint_result = load_analysis_fingerprint(paths)
    if is_failure(fingerprint_result):
        return Failure(AppError(fingerprint_result.message))
    fingerprint = unwrap_success(fingerprint_result)

    report_directory.mkdir(parents=True, exist_ok=True)

    cached_metadata = _load_cache_metadata(metadata_path)
    if (
        cached_metadata is not None
        and cached_metadata.report_version == REPORT_VERSION
        and cached_metadata.fingerprint == fingerprint
        and report_path.exists()
        and bundle_path.exists()
    ):
        if open_in_browser:
            return _open_report(report_path)
        return Success(report_path)

    snapshot_result = load_analysis_snapshot(paths)
    if is_failure(snapshot_result):
        return Failure(AppError(snapshot_result.message))
    snapshot = unwrap_success(snapshot_result)

    try:
        _ensure_plotly_bundle(bundle_path)
        report_path.write_text(
            _build_analysis_html(snapshot, generated_at=datetime.now(UTC)),
            encoding="utf-8",
        )
        _write_cache_metadata(
            metadata_path,
            ReportCacheMetadata(
                report_version=REPORT_VERSION,
                fingerprint=fingerprint,
            ),
        )
    except OSError as exc:
        return Failure(AppError(f"解析レポートの書き出しに失敗しました: {exc}"))

    if open_in_browser:
        return _open_report(report_path)

    return Success(report_path)


def _build_analysis_html(snapshot: AnalysisSnapshot, generated_at: datetime) -> str:
    overview_cards = _build_overview_cards(snapshot)
    quality_figure = _build_quality_figure(snapshot)
    operations_figure = _build_operations_figure(
        snapshot,
        generated_at.astimezone().date(),
    )
    latest_sessions_table = _build_sessions_table(snapshot)
    recent_logs_table = _build_logs_table(snapshot.logs)

    quality_html = quality_figure.to_html(
        full_html=False,
        include_plotlyjs=PLOTLY_BUNDLE_FILE_NAME,
    )
    operations_html = operations_figure.to_html(full_html=False, include_plotlyjs=False)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>顔認証アプリ 解析レポート</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: {BACKGROUND};
      --surface: #ffffff;
      --surface-alt: #f8f9fb;
      --text: {TEXT_PRIMARY};
      --muted: {TEXT_MUTED};
      --border: {BORDER_COLOR};
      --accent: {PRIMARY_COLOR};
      --accent-soft: {PRIMARY_SOFT};
      --success: {SUCCESS_COLOR};
      --success-soft: {SUCCESS_SOFT};
      --danger: {DANGER_COLOR};
      --danger-soft: {DANGER_SOFT};
      --neutral-soft: {NEUTRAL_SOFT};
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Noto Sans JP", "BIZ UDPGothic", "Yu Gothic UI", "Meiryo", sans-serif;
      line-height: 1.6;
    }}

    .topbar {{
      height: 6px;
      background: var(--accent);
    }}

    .page {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 28px 40px;
    }}

    .header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
      margin-bottom: 24px;
    }}

    .eyebrow {{
      margin: 0 0 6px;
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2rem, 3vw, 2.9rem);
      line-height: 1.25;
    }}

    .header-meta {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .status-chip {{
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-left: 6px solid var(--accent);
      border-radius: 12px;
      background: var(--surface);
      color: var(--text);
      font-size: 0.95rem;
      white-space: nowrap;
    }}

    .section {{
      margin-top: 28px;
    }}

    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }}

    .section-title {{
      margin: 0;
      font-size: 1.2rem;
      font-weight: 700;
    }}

    .section-note {{
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 18px 16px;
    }}

    .card-label {{
      margin: 0;
      font-size: 0.86rem;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}

    .card-value {{
      margin: 10px 0 4px;
      font-size: 1.9rem;
      font-weight: 700;
      line-height: 1.1;
    }}

    .card-detail {{
      margin: 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}

    .plots {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}

    .plot-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 10px 2px;
    }}

    .grid-two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}

    thead th {{
      background: var(--surface-alt);
      color: var(--text);
      font-size: 0.9rem;
      font-weight: 700;
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }}

    tbody td {{
      padding: 12px 14px;
      border-top: 1px solid var(--border);
      font-size: 0.95rem;
      vertical-align: top;
    }}

    .pill {{
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
    }}

    .pill-info {{
      background: var(--accent-soft);
      color: var(--accent);
    }}

    .pill-success {{
      background: var(--success-soft);
      color: var(--success);
    }}

    .pill-danger {{
      background: var(--danger-soft);
      color: var(--danger);
    }}

    .pill-neutral {{
      background: var(--neutral-soft);
      color: var(--text);
    }}

    .empty {{
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
    }}

    @media (max-width: 980px) {{
      .header,
      .cards,
      .grid-two {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding-inline: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="topbar"></div>
  <main class="page">
    <header class="header">
      <div>
        <p class="eyebrow">Analysis Report</p>
        <h1>顔認証アプリの解析レポート</h1>
        <p class="header-meta">評価実験・照合距離・運用ログを同じ画面で確認できます。</p>
      </div>
      <div class="status-chip">生成日時 {escape(generated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))}</div>
    </header>

    <section class="section">
      <div class="section-header">
        <h2 class="section-title">概要</h2>
        <p class="section-note">直近の登録状況と評価データの総量をまとめています。</p>
      </div>
      <div class="cards">{overview_cards}</div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2 class="section-title">照合品質</h2>
        <p class="section-note">distance 分布と閾値スイープで、現在の閾値設定が妥当か確認します。</p>
      </div>
      <div class="plots">
        <div class="plot-card">{quality_html}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2 class="section-title">運用状況</h2>
        <p class="section-note">登録済み特徴量の量と、イベント発生傾向を確認します。</p>
      </div>
      <div class="plots">
        <div class="plot-card">{operations_html}</div>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2 class="section-title">直近の評価実験</h2>
        <p class="section-note">セッション単位で結果を比較します。</p>
      </div>
      {latest_sessions_table}
    </section>

    <section class="section">
      <div class="section-header">
        <h2 class="section-title">直近のイベント</h2>
        <p class="section-note">最新の操作履歴を一覧化しています。</p>
      </div>
      {recent_logs_table}
    </section>
  </main>
</body>
</html>
"""


def _build_overview_cards(snapshot: AnalysisSnapshot) -> str:
    encodings_count = sum(len(person.encodings) for person in snapshot.people)
    genuine_trials = tuple(
        analysis_trial
        for analysis_trial in snapshot.trials
        if analysis_trial.scenario is ExperimentScenario.GENUINE
    )
    impostor_trials = tuple(
        analysis_trial
        for analysis_trial in snapshot.trials
        if analysis_trial.scenario is ExperimentScenario.IMPOSTOR
    )
    genuine_metrics = _trial_metrics(genuine_trials)
    impostor_metrics = _trial_metrics(impostor_trials)
    average_distance = _average_distance(snapshot.trials)

    cards = (
        ("登録人物", str(len(snapshot.people)), f"特徴量 {encodings_count} 件"),
        (
            "評価セッション",
            str(len(snapshot.sessions)),
            f"試行 {len(snapshot.trials)} 件",
        ),
        (
            "本人受入率",
            _format_ratio(genuine_metrics.success_rate),
            f"本人試行 {genuine_metrics.trial_count} 件",
        ),
        (
            "他人誤受入率",
            _format_ratio(impostor_metrics.accepted_rate),
            f"他人試行 {impostor_metrics.trial_count} 件",
        ),
        (
            "平均 distance",
            _format_distance(average_distance),
            "候補距離が取れた試行だけで計算",
        ),
        ("イベント履歴", str(len(snapshot.logs)), "SQLite に保持している件数"),
    )

    fragments: list[str] = []
    for label, value, detail in cards:
        fragments.append(
            f"""
            <article class="card">
              <p class="card-label">{escape(label)}</p>
              <p class="card-value">{escape(value)}</p>
              <p class="card-detail">{escape(detail)}</p>
            </article>
            """
        )
    return "".join(fragments)


def _build_quality_figure(snapshot: AnalysisSnapshot) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("distance 分布", "閾値スイープ"),
        horizontal_spacing=0.12,
    )

    _add_distance_distribution_traces(figure, snapshot.trials)
    _add_threshold_sweep_traces(figure, snapshot.trials, snapshot.sessions)

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 40, "r": 20, "t": 56, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        font={"family": "Yu Gothic UI, Meiryo, sans-serif", "color": TEXT_PRIMARY},
    )
    figure.update_yaxes(title_text="distance", row=1, col=1)
    figure.update_yaxes(title_text="割合", tickformat=".0%", row=1, col=2)
    figure.update_xaxes(title_text="試験区分", row=1, col=1)
    figure.update_xaxes(title_text="閾値", row=1, col=2)
    return figure


def _build_operations_figure(
    snapshot: AnalysisSnapshot,
    today: date,
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("登録済み人物ごとの特徴量数", "日次イベント件数"),
        horizontal_spacing=0.12,
    )

    person_names = [person.display_name.value for person in snapshot.people]
    encoding_counts = [len(person.encodings) for person in snapshot.people]
    if len(person_names) == 0:
        person_names = ["未登録"]
        encoding_counts = [0]
    figure.add_trace(
        go.Bar(
            x=person_names,
            y=encoding_counts,
            marker_color=PRIMARY_COLOR,
            name="特徴量数",
        ),
        row=1,
        col=1,
    )

    _add_daily_log_traces(figure, snapshot.logs, today=today)

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 40, "r": 20, "t": 56, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        font={"family": "Yu Gothic UI, Meiryo, sans-serif", "color": TEXT_PRIMARY},
    )
    figure.update_yaxes(title_text="件数", row=1, col=1)
    figure.update_yaxes(title_text="件数", row=1, col=2)
    figure.update_xaxes(title_text="人物", row=1, col=1)
    figure.update_xaxes(title_text="日付", row=1, col=2)
    return figure


def _add_distance_distribution_traces(
    figure: go.Figure,
    trials: tuple[AnalysisTrial, ...],
) -> None:
    grouped_values = {
        ("本人受入試験", "正答"): [],
        ("本人受入試験", "誤判定"): [],
        ("他人拒否試験", "正答"): [],
        ("他人拒否試験", "誤判定"): [],
    }
    for analysis_trial in trials:
        distance = analysis_trial.trial.distance
        if distance is None:
            continue
        scenario_label = _scenario_label(analysis_trial.scenario)
        outcome_label = "正答" if analysis_trial.trial.success else "誤判定"
        grouped_values[(scenario_label, outcome_label)].append(distance.value)

    colors = {
        "正答": SUCCESS_COLOR,
        "誤判定": DANGER_COLOR,
    }
    for (scenario_label, outcome_label), values in grouped_values.items():
        if len(values) == 0:
            continue
        figure.add_trace(
            go.Box(
                x=[scenario_label] * len(values),
                y=values,
                name=f"{scenario_label} / {outcome_label}",
                marker_color=colors[outcome_label],
                boxmean=True,
                legendgroup=outcome_label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )


def _add_threshold_sweep_traces(
    figure: go.Figure,
    trials: tuple[AnalysisTrial, ...],
    sessions: tuple[AnalysisSession, ...],
) -> None:
    thresholds = _threshold_points(trials)
    genuine_trials = tuple(
        analysis_trial
        for analysis_trial in trials
        if analysis_trial.scenario is ExperimentScenario.GENUINE
    )
    impostor_trials = tuple(
        analysis_trial
        for analysis_trial in trials
        if analysis_trial.scenario is ExperimentScenario.IMPOSTOR
    )
    genuine_values = [
        _accepted_rate_at_threshold(genuine_trials, threshold)
        for threshold in thresholds
    ]
    impostor_values = [
        _accepted_rate_at_threshold(impostor_trials, threshold)
        for threshold in thresholds
    ]

    figure.add_trace(
        go.Scatter(
            x=list(thresholds),
            y=genuine_values,
            mode="lines",
            name="本人受入率",
            line={"color": SUCCESS_COLOR, "width": 3},
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=list(thresholds),
            y=impostor_values,
            mode="lines",
            name="他人誤受入率",
            line={"color": DANGER_COLOR, "width": 3},
        ),
        row=1,
        col=2,
    )

    latest_threshold = sessions[-1].session.threshold if len(sessions) > 0 else None
    if latest_threshold is not None:
        figure.add_trace(
            go.Scatter(
                x=[latest_threshold, latest_threshold],
                y=[0.0, 1.0],
                mode="lines",
                name=f"直近の閾値 {latest_threshold:.3f}",
                line={"color": PRIMARY_COLOR, "width": 2, "dash": "dash"},
                hovertemplate="閾値 %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )


def _add_daily_log_traces(
    figure: go.Figure,
    logs: tuple[AppLogEntry, ...],
    today: date,
) -> None:
    day_labels = [
        (today - timedelta(days=offset)).strftime("%m/%d")
        for offset in range(13, -1, -1)
    ]
    info_counts = [0] * len(day_labels)
    warning_counts = [0] * len(day_labels)
    error_counts = [0] * len(day_labels)
    index_by_label = {label: index for index, label in enumerate(day_labels)}

    for entry in logs:
        label = entry.created_at.value.astimezone().strftime("%m/%d")
        if label not in index_by_label:
            continue
        index = index_by_label[label]
        if entry.level is AppLogLevel.INFO:
            info_counts[index] += 1
        elif entry.level is AppLogLevel.WARNING:
            warning_counts[index] += 1
        else:
            error_counts[index] += 1

    figure.add_trace(
        go.Bar(
            x=day_labels,
            y=info_counts,
            name="Info",
            marker_color=PRIMARY_COLOR,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=day_labels,
            y=warning_counts,
            name="Warning",
            marker_color="#b7791f",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=day_labels,
            y=error_counts,
            name="Error",
            marker_color=DANGER_COLOR,
        ),
        row=1,
        col=2,
    )
    figure.update_layout(barmode="stack")


def _threshold_points(trials: tuple[AnalysisTrial, ...]) -> tuple[float, ...]:
    values = sorted(
        analysis_trial.trial.distance.value
        for analysis_trial in trials
        if analysis_trial.trial.distance is not None
    )
    if len(values) == 0:
        return tuple(
            round(0.4 + ((1.4 - 0.4) / (THRESHOLD_POINT_COUNT - 1)) * index, 3)
            for index in range(THRESHOLD_POINT_COUNT)
        )

    minimum = max(0.0, values[0] - 0.15)
    maximum = max(values[-1] + 0.15, 1.4)
    if minimum == maximum:
        return (round(minimum, 3),)
    step = (maximum - minimum) / (THRESHOLD_POINT_COUNT - 1)
    return tuple(
        round(minimum + step * index, 3) for index in range(THRESHOLD_POINT_COUNT)
    )


def _accepted_rate_at_threshold(
    trials: tuple[AnalysisTrial, ...],
    threshold: float,
) -> float | None:
    if len(trials) == 0:
        return None

    accepted_count = 0
    for analysis_trial in trials:
        distance = analysis_trial.trial.distance
        candidate_person_id = analysis_trial.trial.candidate_person_id
        if (
            distance is not None
            and candidate_person_id is not None
            and distance.value <= threshold
        ):
            accepted_as_target = candidate_person_id == analysis_trial.target_person_id
            if accepted_as_target:
                accepted_count += 1
    return accepted_count / len(trials)


def _build_sessions_table(snapshot: AnalysisSnapshot) -> str:
    if len(snapshot.sessions) == 0:
        return '<div class="empty">評価実験の記録はまだありません。</div>'

    rows: list[str] = []
    recent_sessions = snapshot.sessions[-RECENT_SESSION_LIMIT:]
    for analysis_session in reversed(recent_sessions):
        session = analysis_session.session
        badge_class = _experiment_status_badge_class(analysis_session.status)
        session_trials = tuple(
            analysis_trial
            for analysis_trial in snapshot.trials
            if analysis_trial.trial.session_id == session.session_id
        )
        metrics = _trial_metrics(session_trials)
        if session.scenario is ExperimentScenario.GENUINE:
            rate_text = f"本人受入率 {_format_ratio(metrics.success_rate)}"
        else:
            rate_text = f"他人誤受入率 {_format_ratio(metrics.accepted_rate)}"
        rows.append(
            f"""
            <tr>
              <td>{escape(session.started_at.value.astimezone().strftime("%Y-%m-%d %H:%M"))}</td>
              <td><span class="pill pill-info">{escape(_scenario_label(session.scenario))}</span></td>
              <td>{escape(session.target_person_name)}</td>
              <td><span class="pill {badge_class}">{escape(_experiment_status_text(analysis_session.status))}</span></td>
              <td>{escape(str(metrics.trial_count))}</td>
              <td>{escape(rate_text)}</td>
              <td>{escape(_format_distance(metrics.average_distance))}</td>
              <td>{escape(_format_threshold(session.threshold))}</td>
            </tr>
            """
        )

    return f"""
    <table>
      <thead>
        <tr>
          <th>開始日時</th>
          <th>試験区分</th>
          <th>対象人物</th>
          <th>状態</th>
          <th>試行数</th>
          <th>主要指標</th>
          <th>平均 distance</th>
          <th>閾値</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _build_logs_table(logs: tuple[AppLogEntry, ...]) -> str:
    if len(logs) == 0:
        return '<div class="empty">イベント履歴はまだありません。</div>'

    rows: list[str] = []
    recent_logs = logs[-RECENT_LOG_LIMIT:]
    for entry in reversed(recent_logs):
        rows.append(
            f"""
            <tr>
              <td>{escape(entry.created_at.value.astimezone().strftime("%Y-%m-%d %H:%M:%S"))}</td>
              <td><span class="pill {_log_level_badge_class(entry.level)}">{escape(entry.level.value)}</span></td>
              <td>{escape(entry.event.value)}</td>
              <td>{escape(entry.message)}</td>
            </tr>
            """
        )

    return f"""
    <table>
      <thead>
        <tr>
          <th>時刻</th>
          <th>レベル</th>
          <th>イベント</th>
          <th>内容</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _trial_metrics(trials: tuple[AnalysisTrial, ...]) -> TrialMetrics:
    if len(trials) == 0:
        return TrialMetrics(
            trial_count=0,
            success_rate=None,
            accepted_rate=None,
            average_distance=None,
        )
    success_count = sum(1 for analysis_trial in trials if analysis_trial.trial.success)
    accepted_count = sum(
        1 for analysis_trial in trials if analysis_trial.trial.accepted_as_target
    )
    distances = tuple(
        analysis_trial.trial.distance.value
        for analysis_trial in trials
        if analysis_trial.trial.distance is not None
    )
    average_distance = fmean(distances) if len(distances) > 0 else None
    return TrialMetrics(
        trial_count=len(trials),
        success_rate=success_count / len(trials),
        accepted_rate=accepted_count / len(trials),
        average_distance=average_distance,
    )


def _average_distance(trials: tuple[AnalysisTrial, ...]) -> float | None:
    distances = [
        analysis_trial.trial.distance.value
        for analysis_trial in trials
        if analysis_trial.trial.distance is not None
    ]
    if len(distances) == 0:
        return None
    return fmean(distances)


def _scenario_label(scenario: ExperimentScenario) -> str:
    if scenario is ExperimentScenario.GENUINE:
        return "本人受入試験"
    return "他人拒否試験"


def _experiment_status_text(status: ExperimentStatus) -> str:
    if status is ExperimentStatus.ACTIVE:
        return "計測中"
    if status is ExperimentStatus.COMPLETED:
        return "完了"
    if status is ExperimentStatus.ABORTED:
        return "中断"
    return "未開始"


def _experiment_status_badge_class(status: ExperimentStatus) -> str:
    if status is ExperimentStatus.ACTIVE:
        return "pill-info"
    if status is ExperimentStatus.COMPLETED:
        return "pill-success"
    if status is ExperimentStatus.ABORTED:
        return "pill-neutral"
    return "pill-neutral"


def _log_level_badge_class(level: AppLogLevel) -> str:
    if level is AppLogLevel.INFO:
        return "pill-info"
    if level is AppLogLevel.WARNING:
        return "pill-neutral"
    return "pill-danger"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_distance(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_threshold(value: float) -> str:
    return f"{value:.3f}"


def _ensure_plotly_bundle(bundle_path: Path) -> None:
    if bundle_path.exists():
        return
    bundle_path.write_text(get_plotlyjs(), encoding="utf-8")


def _load_cache_metadata(metadata_path: Path) -> ReportCacheMetadata | None:
    if not metadata_path.exists():
        return None
    try:
        raw_data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    report_version = raw_data.get("report_version")
    fingerprint = raw_data.get("fingerprint")
    if not isinstance(report_version, int) or not isinstance(fingerprint, str):
        return None
    return ReportCacheMetadata(
        report_version=report_version,
        fingerprint=fingerprint,
    )


def _write_cache_metadata(
    metadata_path: Path,
    metadata: ReportCacheMetadata,
) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "report_version": metadata.report_version,
                "fingerprint": metadata.fingerprint,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _open_report(report_path: Path) -> Result[Path, AppError]:
    try:
        opened = webbrowser.open(report_path.resolve().as_uri())
    except OSError as exc:
        return Failure(AppError(f"解析レポートを開けませんでした: {exc}"))
    if not opened:
        return Failure(AppError("解析レポートを開けませんでした。"))
    return Success(report_path)
