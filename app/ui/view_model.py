from dataclasses import dataclass

from app.app.runtime import FaceRecognitionRuntime


@dataclass(frozen=True)
class MainWindowViewModel:
    message: str
    status_lines: tuple[str, ...]
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
    return MainWindowViewModel(
        message=runtime.state.ui.message,
        status_lines=runtime.status_lines(),
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
