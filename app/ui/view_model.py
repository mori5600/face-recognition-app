from dataclasses import dataclass

from app.app.runtime import FaceRecognitionRuntime
from app.domain.statuses import (
    CameraStatus,
    LivenessStatus,
    MatchingStatus,
    RegistrationStatus,
)


@dataclass(frozen=True)
class MainWindowViewModel:
    phase_badge_text: str
    phase_tone: str
    phase_title: str
    phase_detail: str
    summary_text: str
    people_lines: tuple[str, ...]
    result_lines: tuple[str, ...]
    face_selector_labels: tuple[str, ...]
    selected_face_selector_label: str
    matching_mode_labels: tuple[str, ...]
    selected_matching_mode_label: str
    matching_threshold_text: str
    person_choice_labels: tuple[str, ...]
    selected_person_label: str
    can_register: bool
    can_match: bool
    can_delete_person: bool


def build_main_window_view_model(
    runtime: FaceRecognitionRuntime,
) -> MainWindowViewModel:
    people_count = len(runtime.state.people.persons)
    phase_badge_text, phase_tone, phase_title, phase_detail = _build_phase_fields(
        runtime
    )
    return MainWindowViewModel(
        phase_badge_text=phase_badge_text,
        phase_tone=phase_tone,
        phase_title=phase_title,
        phase_detail=phase_detail,
        summary_text=(
            f"カメラ {_camera_status_label(runtime.state.camera.status)} / "
            f"顔 {len(runtime.state.camera.detected_faces)}件 / "
            f"登録 {people_count}人"
        ),
        people_lines=runtime.people_lines(),
        result_lines=runtime.result_lines(),
        face_selector_labels=runtime.face_selector_labels(),
        selected_face_selector_label=runtime.selected_face_selector_label(),
        matching_mode_labels=runtime.matching_mode_labels(),
        selected_matching_mode_label=runtime.selected_matching_mode_label(),
        matching_threshold_text=runtime.matching_threshold_text(),
        person_choice_labels=runtime.person_choice_labels(),
        selected_person_label=runtime.selected_person_label(),
        can_register=runtime.can_target_face(),
        can_match=runtime.can_target_face() and people_count > 0,
        can_delete_person=people_count > 0,
    )


def _build_phase_fields(
    runtime: FaceRecognitionRuntime,
) -> tuple[str, str, str, str]:
    state = runtime.state

    if state.camera.status is not CameraStatus.RUNNING:
        return (
            "待機中",
            "neutral",
            "カメラを開始してください。",
            "開始後に顔を正面へ向けてください。",
        )

    if (
        state.matching.status is MatchingStatus.SUCCESS
        and len(state.matching.results) > 0
    ):
        result = state.matching.results[0]
        if result.candidate is None:
            return ("未登録", "neutral", "照合対象がありません。", "")
        if result.matched:
            return (
                "一致",
                "success",
                f"{result.candidate.display_name.value} と一致しました。",
                f"distance {result.candidate.distance.value:.3f}",
            )
        return (
            "不一致",
            "neutral",
            "一致しませんでした。",
            (
                f"最も近い候補 {result.candidate.display_name.value} / "
                f"distance {result.candidate.distance.value:.3f}"
            ),
        )

    if state.registration.status is RegistrationStatus.SUCCESS:
        return ("登録完了", "success", "顔を登録しました。", "")

    if state.liveness.status is LivenessStatus.CHALLENGE:
        step_count = len(state.liveness.challenge_steps)
        step_index = state.liveness.current_step_index
        instruction = (
            state.liveness.challenge_steps[step_index].instruction
            if step_index < step_count
            else "指示に従ってください。"
        )
        return (
            "確認中",
            "info",
            f"生体確認 {step_index + 1}/{step_count}",
            instruction,
        )

    if state.liveness.status is LivenessStatus.VERIFIED:
        return (
            "操作可能",
            "success",
            "生体確認が完了しました。",
            "登録または照合を実行してください。",
        )

    if state.liveness.status is LivenessStatus.FAILED:
        return (
            "再確認",
            "danger",
            "生体確認に失敗しました。",
            "もう一度登録または照合を押してください。",
        )

    if state.liveness.status is LivenessStatus.EXPIRED:
        return (
            "再確認",
            "neutral",
            "生体確認の有効時間が切れました。",
            "もう一度登録または照合を押してください。",
        )

    if state.matching.status is MatchingStatus.ERROR:
        return (
            "エラー",
            "danger",
            "照合に失敗しました。",
            state.matching.last_error or "",
        )

    if state.registration.status is RegistrationStatus.ERROR:
        return (
            "エラー",
            "danger",
            "登録に失敗しました。",
            state.registration.last_error or "",
        )

    return (
        "待機中",
        "neutral",
        "登録または照合を選択してください。",
        "写真対策のため、生体確認の後に実行します。",
    )


def _camera_status_label(status: CameraStatus) -> str:
    if status is CameraStatus.RUNNING:
        return "稼働中"
    if status is CameraStatus.STOPPED:
        return "停止中"
    if status is CameraStatus.ERROR:
        return "エラー"
    return "待機中"
