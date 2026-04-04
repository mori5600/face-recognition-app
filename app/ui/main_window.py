import tkinter.font as tkfont

import customtkinter as ctk
from PIL import Image

from app.app.runtime import FaceRecognitionRuntime
from app.domain.results import is_failure, unwrap_success
from app.domain.statuses import CameraStatus
from app.ui.view_model import MainWindowViewModel, build_main_window_view_model

PRIMARY_FONT_CANDIDATES = (
    "Noto Sans JP",
    "BIZ UDPGothic",
    "Yu Gothic UI",
    "Meiryo",
)
MONO_FONT_CANDIDATES = (
    "Noto Sans Mono",
    "Cascadia Mono",
    "Consolas",
)

APP_BG = "#f3f4f6"
CARD_BG = "#ffffff"
CARD_ALT_BG = "#f8f9fb"
TEXT_PRIMARY = "#333333"
TEXT_MUTED = "#5f6874"
ACCENT = "#0017c1"
ACCENT_HOVER = "#00118f"
ACCENT_ACTIVE = "#000060"
ACCENT_SOFT = "#eef3ff"
PREVIEW_BG = "#111827"
PREVIEW_TEXT = "#f8fafc"
BORDER = "#000000"
BORDER_SOFT = "#d1d5db"
DANGER = "#9f1d1d"
DANGER_HOVER = "#7f1d1d"
DANGER_SOFT = "#fef2f2"
SUCCESS = "#006D3C"
SUCCESS_SOFT = "#E8F5EE"
NEUTRAL_SOFT = "#F3F4F6"


class MainWindow(ctk.CTk):
    def __init__(self, auto_start_camera: bool = True) -> None:
        ctk.set_appearance_mode("light")
        super().__init__()
        self.title("Face Recognition App")
        self.geometry("1440x900")
        self.minsize(1220, 780)
        self.configure(fg_color=APP_BG)

        primary_font_family = self._pick_font_family(
            PRIMARY_FONT_CANDIDATES,
            fallback="Yu Gothic UI",
        )
        mono_font_family = self._pick_font_family(
            MONO_FONT_CANDIDATES,
            fallback="Consolas",
        )

        self._font_body = ctk.CTkFont(family=primary_font_family, size=14)
        self._font_small = ctk.CTkFont(family=primary_font_family, size=12)
        self._font_label = ctk.CTkFont(
            family=primary_font_family,
            size=12,
            weight="bold",
        )
        self._font_heading = ctk.CTkFont(
            family=primary_font_family,
            size=34,
            weight="bold",
        )
        self._font_section = ctk.CTkFont(
            family=primary_font_family,
            size=20,
            weight="bold",
        )
        self._font_result = ctk.CTkFont(
            family=primary_font_family,
            size=26,
            weight="bold",
        )
        self._font_stat = ctk.CTkFont(
            family=primary_font_family,
            size=18,
            weight="bold",
        )
        self._font_mono = ctk.CTkFont(family=mono_font_family, size=13)

        self._runtime: FaceRecognitionRuntime | None = None
        self._preview_image = None
        self._tick_after_id = None

        self._last_phase_signature = ""
        self._last_summary_text = ""
        self._last_people_lines = ()
        self._last_result_lines = ()
        self._last_face_selector_values = ()
        self._last_face_selector_label = ""
        self._last_matching_mode_values = ()
        self._last_matching_mode_label = ""
        self._last_person_choice_values = ()
        self._last_person_choice_label = ""
        self._last_register_enabled = None
        self._last_match_enabled = None
        self._last_delete_enabled = None

        runtime_result = FaceRecognitionRuntime.bootstrap()
        if is_failure(runtime_result):
            self._build_error_screen(runtime_result.message)
            return

        runtime = unwrap_success(runtime_result)
        self._runtime = runtime
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(0, self._maximize_window)

        if auto_start_camera:
            runtime.start_camera()

        self._refresh_view()
        self._schedule_tick()

    def _pick_font_family(
        self,
        candidates: tuple[str, ...],
        fallback: str,
    ) -> str:
        available = {family.casefold(): family for family in tkfont.families()}
        for candidate in candidates:
            if candidate.casefold() in available:
                return available[candidate.casefold()]
        return fallback

    def _maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except Exception:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            self.geometry(f"{screen_width}x{screen_height}+0+0")

    def _build_error_screen(self, message: str) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        frame = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER,
        )
        frame.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        accent_bar = ctk.CTkFrame(
            frame,
            height=6,
            corner_radius=0,
            fg_color=ACCENT,
        )
        accent_bar.grid(row=0, column=0, sticky="ew")

        title = ctk.CTkLabel(
            frame,
            text="初期化に失敗しました",
            font=self._font_heading,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.grid(row=1, column=0, sticky="w", padx=28, pady=(24, 14))

        body = ctk.CTkTextbox(
            frame,
            width=720,
            height=260,
            font=self._font_body,
            fg_color=CARD_ALT_BG,
            border_width=1,
            border_color=BORDER_SOFT,
            text_color=TEXT_PRIMARY,
        )
        body.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 28))
        body.insert("1.0", message)
        body.configure(state="disabled")

    def _build_layout(self) -> None:
        runtime = self._require_runtime()
        self.grid_columnconfigure(0, weight=15)
        self.grid_columnconfigure(1, weight=4, minsize=320)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self._header_frame = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=0,
        )
        self._header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._header_frame.grid_columnconfigure(0, weight=1)

        accent_bar = ctk.CTkFrame(
            self._header_frame,
            height=5,
            corner_radius=0,
            fg_color=ACCENT,
        )
        accent_bar.grid(row=0, column=0, sticky="ew")

        header_body = ctk.CTkFrame(self._header_frame, fg_color="transparent")
        header_body.grid(row=1, column=0, sticky="ew", padx=24, pady=12)

        title_block = ctk.CTkFrame(header_body, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="w")

        title = ctk.CTkLabel(
            title_block,
            text="顔認証アプリ",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.pack(anchor="w")

        self._preview_panel = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        self._preview_panel.grid(
            row=1, column=0, sticky="nsew", padx=(24, 8), pady=(16, 24)
        )
        self._preview_panel.grid_rowconfigure(2, weight=1)
        self._preview_panel.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkFrame(self._preview_panel, fg_color="transparent")
        preview_header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 12))
        preview_header.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(
            preview_header,
            text="プレビュー",
            font=self._font_label,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        preview_title.grid(row=0, column=0, sticky="w")

        self._phase_panel = ctk.CTkFrame(
            self._preview_panel,
            corner_radius=10,
            fg_color=CARD_ALT_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        self._phase_panel.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        self._phase_panel.grid_columnconfigure(0, weight=1)

        self._phase_badge = ctk.CTkLabel(
            self._phase_panel,
            text="待機中",
            font=self._font_small,
            text_color=ACCENT,
            fg_color=ACCENT_SOFT,
            corner_radius=8,
            padx=10,
            pady=4,
        )
        self._phase_badge.grid(row=0, column=0, sticky="w", padx=14, pady=(14, 6))

        self._phase_title_label = ctk.CTkLabel(
            self._phase_panel,
            text="カメラを開始してください。",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self._phase_title_label.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 4))

        self._phase_detail_label = ctk.CTkLabel(
            self._phase_panel,
            text="開始後に顔を正面へ向けてください。",
            font=self._font_body,
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self._phase_detail_label.grid(
            row=2,
            column=0,
            sticky="w",
            padx=14,
            pady=(0, 6),
        )

        self._phase_summary_label = ctk.CTkLabel(
            self._phase_panel,
            text="",
            font=self._font_small,
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self._phase_summary_label.grid(
            row=3,
            column=0,
            sticky="w",
            padx=14,
            pady=(0, 14),
        )

        self._preview_label = ctk.CTkLabel(
            self._preview_panel,
            text="カメラを開始するとプレビューが表示されます。",
            font=self._font_body,
            text_color=PREVIEW_TEXT,
            fg_color=PREVIEW_BG,
            corner_radius=8,
            justify="center",
        )
        self._preview_label.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 16))

        action_bar = ctk.CTkFrame(
            self._preview_panel,
            corner_radius=10,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        action_bar.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        action_bar.grid_columnconfigure(0, weight=1)
        action_bar.grid_columnconfigure(1, weight=1)
        action_bar.grid_columnconfigure(2, weight=1)
        action_bar.grid_columnconfigure(3, weight=1)

        self._name_entry = ctk.CTkEntry(
            action_bar,
            placeholder_text="登録したい名前を入力",
            font=self._font_body,
            fg_color="#ffffff",
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=46,
        )
        self._name_entry.grid(
            row=0, column=0, columnspan=3, sticky="ew", padx=(14, 8), pady=(14, 10)
        )

        self._register_button = ctk.CTkButton(
            action_bar,
            text="登録",
            command=self._handle_register,
            font=self._font_label,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=CARD_BG,
            corner_radius=8,
            height=46,
        )
        self._register_button.grid(
            row=0, column=3, sticky="ew", padx=(8, 14), pady=(14, 10)
        )

        self._start_button = ctk.CTkButton(
            action_bar,
            text="開始",
            command=self._handle_start_camera,
            font=self._font_label,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=CARD_BG,
            corner_radius=8,
            height=46,
        )
        self._start_button.grid(
            row=1, column=0, sticky="ew", padx=(14, 6), pady=(0, 14)
        )

        self._stop_button = ctk.CTkButton(
            action_bar,
            text="停止",
            command=self._handle_stop_camera,
            font=self._font_label,
            fg_color=CARD_BG,
            hover_color=CARD_ALT_BG,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            height=46,
        )
        self._stop_button.grid(row=1, column=1, sticky="ew", padx=6, pady=(0, 14))

        self._match_button = ctk.CTkButton(
            action_bar,
            text="照合",
            command=self._handle_match,
            font=self._font_label,
            fg_color=CARD_BG,
            hover_color=ACCENT_SOFT,
            text_color=ACCENT,
            border_width=1,
            border_color=ACCENT,
            corner_radius=8,
            height=46,
        )
        self._match_button.grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=(6, 14), pady=(0, 14)
        )

        self._dashboard_panel = ctk.CTkScrollableFrame(
            self,
            corner_radius=12,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
            scrollbar_button_color="#9ca3af",
            scrollbar_button_hover_color="#6b7280",
        )
        self._dashboard_panel.grid(
            row=1, column=1, sticky="nsew", padx=(8, 24), pady=(16, 24)
        )
        self._dashboard_panel.grid_columnconfigure(0, weight=1)

        settings_card = ctk.CTkFrame(
            self._dashboard_panel,
            corner_radius=10,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        settings_card.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        settings_card.grid_columnconfigure(0, weight=1)

        self._face_selector_menu = ctk.CTkOptionMenu(
            settings_card,
            values=list(runtime.face_selector_labels()),
            command=self._handle_face_selector_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=CARD_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=44,
        )
        self._face_selector_menu.grid(
            row=0, column=0, sticky="ew", padx=12, pady=(12, 8)
        )

        self._matching_mode_menu = ctk.CTkOptionMenu(
            settings_card,
            values=list(runtime.matching_mode_labels()),
            command=self._handle_matching_mode_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=CARD_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=44,
        )
        self._matching_mode_menu.grid(
            row=1, column=0, sticky="ew", padx=12, pady=(0, 8)
        )

        threshold_row = ctk.CTkFrame(settings_card, fg_color="transparent")
        threshold_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        threshold_row.grid_columnconfigure(0, weight=1)

        self._threshold_entry = ctk.CTkEntry(
            threshold_row,
            placeholder_text="閾値",
            font=self._font_body,
            fg_color="#ffffff",
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=44,
        )
        self._threshold_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._threshold_entry.insert(0, runtime.matching_threshold_text())

        self._threshold_apply_button = ctk.CTkButton(
            threshold_row,
            text="適用",
            width=90,
            command=self._handle_threshold_apply,
            font=self._font_label,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=CARD_BG,
            corner_radius=8,
            height=44,
        )
        self._threshold_apply_button.grid(row=0, column=1, sticky="ew")

        result_card = ctk.CTkFrame(
            self._dashboard_panel,
            corner_radius=10,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        result_card.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        result_card.grid_columnconfigure(0, weight=1)

        self._result_badge = ctk.CTkLabel(
            result_card,
            text="待機中",
            font=self._font_small,
            text_color=ACCENT,
            fg_color=ACCENT_SOFT,
            corner_radius=8,
            padx=10,
            pady=4,
        )
        self._result_badge.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self._result_primary_label = ctk.CTkLabel(
            result_card,
            text="まだ照合していません。",
            font=self._font_body,
            text_color=TEXT_PRIMARY,
            anchor="w",
            justify="left",
            wraplength=240,
        )
        self._result_primary_label.grid(
            row=1, column=0, sticky="nw", padx=12, pady=(0, 4)
        )

        self._result_detail_label = ctk.CTkLabel(
            result_card,
            text="",
            font=self._font_small,
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=240,
        )
        self._result_detail_label.grid(
            row=2, column=0, sticky="nw", padx=12, pady=(0, 12)
        )

        lower_grid = ctk.CTkFrame(self._dashboard_panel, fg_color="transparent")
        lower_grid.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 16))
        lower_grid.grid_columnconfigure(0, weight=1)

        people_card = ctk.CTkFrame(
            lower_grid,
            corner_radius=10,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
        )
        people_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        people_card.grid_columnconfigure(0, weight=1)

        self._person_menu = ctk.CTkOptionMenu(
            people_card,
            values=list(runtime.person_choice_labels()),
            command=self._handle_person_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=CARD_BG,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=44,
        )
        self._person_menu.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

        self._delete_person_button = ctk.CTkButton(
            people_card,
            text="削除",
            command=self._handle_delete_person,
            font=self._font_label,
            fg_color=CARD_BG,
            hover_color=DANGER_SOFT,
            text_color=DANGER,
            border_width=1,
            border_color=DANGER,
            corner_radius=8,
            height=44,
        )
        self._delete_person_button.grid(
            row=1, column=0, sticky="ew", padx=12, pady=(0, 8)
        )

        self._people_list = ctk.CTkScrollableFrame(
            people_card,
            corner_radius=10,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER_SOFT,
            height=180,
            scrollbar_button_color="#9ca3af",
            scrollbar_button_hover_color="#6b7280",
        )
        self._people_list.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        self._people_list.grid_columnconfigure(0, weight=1)

    def _require_runtime(self) -> FaceRecognitionRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("FaceRecognitionRuntime is not initialized.")
        return runtime

    def _schedule_tick(self) -> None:
        self._tick_after_id = self.after(100, self._tick)

    def _tick(self) -> None:
        if (
            self._runtime is not None
            and self._runtime.state.camera.status == CameraStatus.RUNNING
        ):
            self._runtime.update_frame()
        self._refresh_view()
        self._schedule_tick()

    def _refresh_view(self) -> None:
        if self._runtime is None:
            return

        view_model = build_main_window_view_model(self._runtime)
        self._refresh_metadata(view_model)

        preview_frame = self._runtime.preview_frame()
        if preview_frame is None:
            self._clear_preview_image()
            self._preview_label.configure(
                image=None, text="カメラを開始するとプレビューが表示されます。"
            )
            return

        image = Image.fromarray(preview_frame)
        image = self._fit_preview_image(image)
        self._preview_image = ctk.CTkImage(
            light_image=image, dark_image=image, size=image.size
        )
        self._preview_label.configure(image=self._preview_image, text="")

    def _clear_preview_image(self) -> None:
        self._preview_image = None

        # CustomTkinter does not clear the underlying tkinter.Label image when
        # image=None is passed, so clear it explicitly before updating text.
        self._preview_label._image = None
        self._preview_label._label.configure(image="")

    def _refresh_metadata(self, view_model: MainWindowViewModel) -> None:
        phase_signature = "|".join(
            (
                view_model.phase_badge_text,
                view_model.phase_tone,
                view_model.phase_title,
                view_model.phase_detail,
            )
        )
        if phase_signature != self._last_phase_signature:
            self._render_phase(view_model)
            self._last_phase_signature = phase_signature

        if view_model.summary_text != self._last_summary_text:
            self._phase_summary_label.configure(text=view_model.summary_text)
            self._last_summary_text = view_model.summary_text

        if view_model.people_lines != self._last_people_lines:
            self._render_people_list(view_model.people_lines)
            self._last_people_lines = view_model.people_lines

        if view_model.result_lines != self._last_result_lines:
            self._render_result(view_model.result_lines)
            self._last_result_lines = view_model.result_lines

        if view_model.face_selector_labels != self._last_face_selector_values:
            self._face_selector_menu.configure(
                values=list(view_model.face_selector_labels)
            )
            self._last_face_selector_values = view_model.face_selector_labels
        if view_model.selected_face_selector_label != self._last_face_selector_label:
            self._face_selector_menu.set(view_model.selected_face_selector_label)
            self._last_face_selector_label = view_model.selected_face_selector_label

        if view_model.matching_mode_labels != self._last_matching_mode_values:
            self._matching_mode_menu.configure(
                values=list(view_model.matching_mode_labels)
            )
            self._last_matching_mode_values = view_model.matching_mode_labels
        if view_model.selected_matching_mode_label != self._last_matching_mode_label:
            self._matching_mode_menu.set(view_model.selected_matching_mode_label)
            self._last_matching_mode_label = view_model.selected_matching_mode_label

        if view_model.person_choice_labels != self._last_person_choice_values:
            self._person_menu.configure(values=list(view_model.person_choice_labels))
            self._last_person_choice_values = view_model.person_choice_labels
        if view_model.selected_person_label != self._last_person_choice_label:
            self._person_menu.set(view_model.selected_person_label)
            self._last_person_choice_label = view_model.selected_person_label

        if view_model.can_register != self._last_register_enabled:
            self._register_button.configure(
                state="normal" if view_model.can_register else "disabled"
            )
            self._last_register_enabled = view_model.can_register
        if view_model.can_match != self._last_match_enabled:
            self._match_button.configure(
                state="normal" if view_model.can_match else "disabled"
            )
            self._last_match_enabled = view_model.can_match
        if view_model.can_delete_person != self._last_delete_enabled:
            self._delete_person_button.configure(
                state="normal" if view_model.can_delete_person else "disabled"
            )
            self._last_delete_enabled = view_model.can_delete_person

    def _render_phase(self, view_model: MainWindowViewModel) -> None:
        badge_fg, badge_text_color, panel_border = self._phase_colors(
            view_model.phase_tone
        )
        self._phase_panel.configure(border_color=panel_border)
        self._phase_badge.configure(
            text=view_model.phase_badge_text,
            fg_color=badge_fg,
            text_color=badge_text_color,
        )
        self._phase_title_label.configure(text=view_model.phase_title)
        self._phase_detail_label.configure(text=view_model.phase_detail)

    def _phase_colors(self, tone: str) -> tuple[str, str, str]:
        if tone == "success":
            return (SUCCESS_SOFT, SUCCESS, SUCCESS)
        if tone == "danger":
            return (DANGER_SOFT, DANGER, DANGER)
        if tone == "info":
            return (ACCENT_SOFT, ACCENT, ACCENT)
        return (NEUTRAL_SOFT, TEXT_PRIMARY, BORDER_SOFT)

    def _render_people_list(self, people_lines: tuple[str, ...]) -> None:
        for child in self._people_list.winfo_children():
            child.destroy()

        if len(people_lines) == 0:
            people_lines = ("未登録です。",)

        for row_index, line in enumerate(people_lines):
            is_selected = line.startswith(">")
            text = line[1:].strip() if is_selected else line.strip()
            title, detail = self._parse_people_line(text)

            item_frame = ctk.CTkFrame(
                self._people_list,
                fg_color=ACCENT_SOFT if is_selected else CARD_BG,
                border_width=1,
                border_color=ACCENT if is_selected else BORDER_SOFT,
                corner_radius=10,
            )
            item_frame.grid(row=row_index, column=0, sticky="ew", padx=8, pady=6)
            item_frame.grid_columnconfigure(0, weight=1)

            title_label = ctk.CTkLabel(
                item_frame,
                text=title,
                anchor="w",
                font=self._font_label,
                text_color=TEXT_PRIMARY,
            )
            title_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

            if is_selected:
                selected_badge = ctk.CTkLabel(
                    item_frame,
                    text="選択中",
                    font=self._font_small,
                    text_color=ACCENT,
                    fg_color=CARD_BG,
                    corner_radius=8,
                    padx=8,
                    pady=4,
                )
                selected_badge.grid(row=0, column=1, sticky="e", padx=12, pady=(10, 2))

            detail_label = ctk.CTkLabel(
                item_frame,
                text=detail,
                anchor="w",
                justify="left",
                wraplength=330,
                font=self._font_small,
                text_color=TEXT_MUTED,
            )
            detail_label.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                padx=12,
                pady=(0, 10),
            )

    def _parse_people_line(self, line: str) -> tuple[str, str]:
        if line == "未登録です。":
            return ("未登録です。", "顔を登録するとここに一覧表示されます。")

        parts = [part.strip() for part in line.split("|")]
        title = parts[0]
        detail_parts: list[str] = []
        for part in parts[1:]:
            if "=" not in part:
                detail_parts.append(part)
                continue

            key, value = part.split("=", 1)
            if key == "encoding":
                detail_parts.append(f"特徴量 {value} 件")
                continue
            if key == "updated":
                detail_parts.append(f"更新 {value}")
                continue
            detail_parts.append(f"{key} {value}")

        detail = " / ".join(detail_parts)
        if detail == "":
            detail = "登録済みの人物です。"
        return (title, detail)

    def _render_result(self, result_lines: tuple[str, ...]) -> None:
        if len(result_lines) == 0:
            self._result_badge.configure(
                text="待機中",
                fg_color=ACCENT_SOFT,
                text_color=ACCENT,
            )
            self._result_primary_label.configure(text="まだ照合していません。")
            self._result_detail_label.configure(text="")
            return

        primary_text = result_lines[0]
        detail_text = "\n".join(result_lines[1:])
        badge_text, badge_fg, badge_text_color = self._result_badge_style(primary_text)
        rendered_primary = self._result_primary_text(primary_text)
        rendered_detail = (
            detail_text if detail_text != "" else self._result_detail_text(primary_text)
        )

        self._result_badge.configure(
            text=badge_text,
            fg_color=badge_fg,
            text_color=badge_text_color,
        )
        self._result_primary_label.configure(text=rendered_primary)
        self._result_detail_label.configure(text=rendered_detail)

    def _result_badge_style(self, primary_text: str) -> tuple[str, str, str]:
        if primary_text.startswith("一致:"):
            return ("一致", SUCCESS, CARD_BG)
        if primary_text.startswith("不一致:"):
            return ("不一致", NEUTRAL_SOFT, TEXT_PRIMARY)
        if "登録済みの人物がいません" in primary_text:
            return ("未登録", NEUTRAL_SOFT, TEXT_PRIMARY)
        return ("情報", ACCENT_SOFT, ACCENT)

    def _result_primary_text(self, primary_text: str) -> str:
        for prefix in ("一致:", "不一致:"):
            if primary_text.startswith(prefix):
                return primary_text[len(prefix) :].strip()
        return primary_text

    def _result_detail_text(self, primary_text: str) -> str:
        if "distance=" in primary_text:
            distance_text = primary_text.split("distance=", 1)[1].rstrip(")")
            return f"distance {distance_text}"
        return ""

    def _fit_preview_image(self, image: Image.Image) -> Image.Image:
        max_width = max(480, self._preview_label.winfo_width() - 24)
        max_height = max(320, self._preview_label.winfo_height() - 24)

        source_width, source_height = image.size
        scale = min(max_width / source_width, max_height / source_height)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))

        if (target_width, target_height) == image.size:
            return image

        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def _handle_start_camera(self) -> None:
        if self._runtime is None:
            return
        self._runtime.start_camera()
        self._refresh_view()

    def _handle_stop_camera(self) -> None:
        if self._runtime is None:
            return
        self._runtime.stop_camera()
        self._refresh_view()

    def _handle_register(self) -> None:
        if self._runtime is None:
            return
        result = self._runtime.register_face(self._name_entry.get())
        if not is_failure(result):
            self._name_entry.delete(0, "end")
        self._refresh_view()

    def _handle_match(self) -> None:
        if self._runtime is None:
            return
        self._runtime.match_face()
        self._refresh_view()

    def _handle_face_selector_change(self, selected_label: str) -> None:
        if self._runtime is None:
            return
        self._runtime.set_face_selector_by_label(selected_label)
        self._refresh_view()

    def _handle_matching_mode_change(self, selected_label: str) -> None:
        if self._runtime is None:
            return
        self._runtime.set_matching_mode_by_label(selected_label)
        self._refresh_view()

    def _handle_threshold_apply(self) -> None:
        if self._runtime is None:
            return
        result = self._runtime.set_matching_threshold(self._threshold_entry.get())
        if not is_failure(result):
            self._threshold_entry.delete(0, "end")
            self._threshold_entry.insert(0, self._runtime.matching_threshold_text())
        self._refresh_view()

    def _handle_person_change(self, selected_label: str) -> None:
        if self._runtime is None or selected_label == "未登録":
            return
        self._runtime.set_selected_person_by_label(selected_label)
        self._refresh_view()

    def _handle_delete_person(self) -> None:
        if self._runtime is None:
            return
        self._runtime.delete_selected_person()
        self._refresh_view()

    def _on_close(self) -> None:
        if self._tick_after_id is not None:
            self.after_cancel(self._tick_after_id)
            self._tick_after_id = None
        if self._runtime is not None:
            self._runtime.shutdown()
        self.destroy()
