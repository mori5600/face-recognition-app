import customtkinter as ctk
from PIL import Image

from app.app.runtime import FaceRecognitionRuntime
from app.domain.results import is_failure, unwrap_success
from app.ui.view_model import build_main_window_view_model

UI_FONT_FAMILY = "Yu Gothic UI"
MONO_FONT_FAMILY = "Consolas"

APP_BG = "#ebe5dc"
CARD_BG = "#f8f4ee"
CARD_ALT_BG = "#f1ebe2"
TEXT_PRIMARY = "#1f2933"
TEXT_MUTED = "#5b6670"
ACCENT = "#1f5f8b"
ACCENT_SOFT = "#d7e7f2"
SUCCESS = "#2f7d5c"
WARNING = "#b7791f"
BORDER = "#d7d1c8"


class MainWindow(ctk.CTk):
    def __init__(self, auto_start_camera: bool = True) -> None:
        super().__init__()
        self.title("Face Recognition App")
        self.geometry("1440x900")
        self.minsize(1220, 780)
        self.configure(fg_color=APP_BG)

        self._font_body = ctk.CTkFont(family=UI_FONT_FAMILY, size=14)
        self._font_small = ctk.CTkFont(family=UI_FONT_FAMILY, size=12)
        self._font_heading = ctk.CTkFont(family=UI_FONT_FAMILY, size=30, weight="bold")
        self._font_section = ctk.CTkFont(family=UI_FONT_FAMILY, size=17, weight="bold")
        self._font_result = ctk.CTkFont(family=UI_FONT_FAMILY, size=22, weight="bold")
        self._font_mono = ctk.CTkFont(family=MONO_FONT_FAMILY, size=13)

        self._runtime: FaceRecognitionRuntime | None = None
        self._preview_image = None
        self._tick_after_id = None

        self._last_message_text = ""
        self._last_status_lines = ()
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
            corner_radius=24,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER,
        )
        frame.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")

        title = ctk.CTkLabel(
            frame,
            text="初期化に失敗しました",
            font=self._font_heading,
            text_color=TEXT_PRIMARY,
        )
        title.pack(anchor="w", padx=28, pady=(28, 14))

        body = ctk.CTkTextbox(
            frame,
            width=720,
            height=260,
            font=self._font_body,
            fg_color=CARD_ALT_BG,
            border_width=0,
            text_color=TEXT_PRIMARY,
        )
        body.pack(fill="both", expand=True, padx=28, pady=(0, 28))
        body.insert("1.0", message)
        body.configure(state="disabled")

    def _build_layout(self) -> None:
        runtime = self._require_runtime()
        self.grid_columnconfigure(0, weight=7)
        self.grid_columnconfigure(1, weight=4, minsize=430)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self._header_frame = ctk.CTkFrame(
            self,
            fg_color="transparent",
        )
        self._header_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(18, 12)
        )
        self._header_frame.grid_columnconfigure(0, weight=1)
        self._header_frame.grid_columnconfigure(1, weight=0)

        title_block = ctk.CTkFrame(self._header_frame, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="w")

        title = ctk.CTkLabel(
            title_block,
            text="顔認証実験アプリ",
            font=self._font_heading,
            text_color=TEXT_PRIMARY,
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            title_block,
            text="OpenCV DNN (YuNet + SFace) / SQLite / Windows local only",
            font=self._font_small,
            text_color=TEXT_MUTED,
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        self._message_banner = ctk.CTkLabel(
            self._header_frame,
            text="",
            font=self._font_body,
            text_color=TEXT_PRIMARY,
            fg_color=CARD_BG,
            corner_radius=16,
            padx=18,
            pady=10,
        )
        self._message_banner.grid(row=0, column=1, sticky="e")

        self._preview_panel = ctk.CTkFrame(
            self,
            corner_radius=28,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER,
        )
        self._preview_panel.grid(
            row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 24)
        )
        self._preview_panel.grid_rowconfigure(1, weight=1)
        self._preview_panel.grid_columnconfigure(0, weight=1)

        preview_header = ctk.CTkFrame(self._preview_panel, fg_color="transparent")
        preview_header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 14))
        preview_header.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(
            preview_header,
            text="カメラプレビュー",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
        )
        preview_title.grid(row=0, column=0, sticky="w")

        preview_hint = ctk.CTkLabel(
            preview_header,
            text="検出した顔は枠で表示されます",
            font=self._font_small,
            text_color=TEXT_MUTED,
        )
        preview_hint.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self._preview_label = ctk.CTkLabel(
            self._preview_panel,
            text="カメラを開始するとプレビューが表示されます。",
            font=self._font_body,
            text_color="#f6f3ee",
            fg_color="#243543",
            corner_radius=22,
        )
        self._preview_label.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))

        action_bar = ctk.CTkFrame(
            self._preview_panel,
            corner_radius=20,
            fg_color=CARD_ALT_BG,
        )
        action_bar.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 24))
        action_bar.grid_columnconfigure(0, weight=1)
        action_bar.grid_columnconfigure(1, weight=1)
        action_bar.grid_columnconfigure(2, weight=1)
        action_bar.grid_columnconfigure(3, weight=1)

        self._name_entry = ctk.CTkEntry(
            action_bar,
            placeholder_text="登録名を入力",
            font=self._font_body,
            fg_color="#ffffff",
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            height=42,
        )
        self._name_entry.grid(
            row=0, column=0, columnspan=4, sticky="ew", padx=16, pady=(16, 12)
        )

        self._start_button = ctk.CTkButton(
            action_bar,
            text="開始",
            command=self._handle_start_camera,
            font=self._font_body,
            fg_color=ACCENT,
            hover_color="#184d71",
            height=42,
        )
        self._start_button.grid(
            row=1, column=0, sticky="ew", padx=(16, 6), pady=(0, 16)
        )

        self._stop_button = ctk.CTkButton(
            action_bar,
            text="停止",
            command=self._handle_stop_camera,
            font=self._font_body,
            fg_color="#6a7a89",
            hover_color="#596775",
            height=42,
        )
        self._stop_button.grid(row=1, column=1, sticky="ew", padx=6, pady=(0, 16))

        self._register_button = ctk.CTkButton(
            action_bar,
            text="登録",
            command=self._handle_register,
            font=self._font_body,
            fg_color=SUCCESS,
            hover_color="#296a4e",
            height=42,
        )
        self._register_button.grid(row=1, column=2, sticky="ew", padx=6, pady=(0, 16))

        self._match_button = ctk.CTkButton(
            action_bar,
            text="照合",
            command=self._handle_match,
            font=self._font_body,
            fg_color=WARNING,
            hover_color="#99641a",
            height=42,
        )
        self._match_button.grid(
            row=1, column=3, sticky="ew", padx=(6, 16), pady=(0, 16)
        )

        self._dashboard_panel = ctk.CTkScrollableFrame(
            self,
            corner_radius=28,
            fg_color=CARD_BG,
            border_width=1,
            border_color=BORDER,
        )
        self._dashboard_panel.grid(
            row=1, column=1, sticky="nsew", padx=(12, 24), pady=(0, 24)
        )
        self._dashboard_panel.grid_columnconfigure(0, weight=1)

        dashboard_title = ctk.CTkLabel(
            self._dashboard_panel,
            text="ダッシュボード",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
        )
        dashboard_title.grid(row=0, column=0, sticky="w", padx=22, pady=(20, 12))

        self._status_card = ctk.CTkFrame(
            self._dashboard_panel,
            corner_radius=20,
            fg_color=CARD_ALT_BG,
        )
        self._status_card.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        self._status_card.grid_columnconfigure((0, 1), weight=1)
        self._status_value_labels = []
        for index in range(6):
            stat_frame = ctk.CTkFrame(
                self._status_card,
                corner_radius=16,
                fg_color="#ffffff",
                border_width=1,
                border_color=BORDER,
            )
            row = index // 2
            column = index % 2
            stat_frame.grid(row=row, column=column, sticky="ew", padx=10, pady=10)
            stat_frame.grid_columnconfigure(0, weight=1)

            key_label = ctk.CTkLabel(
                stat_frame,
                text="-",
                font=self._font_small,
                text_color=TEXT_MUTED,
            )
            key_label.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

            value_label = ctk.CTkLabel(
                stat_frame,
                text="-",
                font=self._font_body,
                text_color=TEXT_PRIMARY,
            )
            value_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
            self._status_value_labels.append((key_label, value_label))

        settings_card = ctk.CTkFrame(
            self._dashboard_panel,
            corner_radius=20,
            fg_color=CARD_ALT_BG,
        )
        settings_card.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        settings_card.grid_columnconfigure(0, weight=1)

        settings_title = ctk.CTkLabel(
            settings_card,
            text="照合設定",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
        )
        settings_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 10))

        self._face_selector_menu = ctk.CTkOptionMenu(
            settings_card,
            values=list(runtime.face_selector_labels()),
            command=self._handle_face_selector_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=ACCENT,
            button_color=ACCENT,
            button_hover_color="#184d71",
            height=40,
        )
        self._face_selector_menu.grid(
            row=1, column=0, sticky="ew", padx=16, pady=(0, 10)
        )

        self._matching_mode_menu = ctk.CTkOptionMenu(
            settings_card,
            values=list(runtime.matching_mode_labels()),
            command=self._handle_matching_mode_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=ACCENT,
            button_color=ACCENT,
            button_hover_color="#184d71",
            height=40,
        )
        self._matching_mode_menu.grid(
            row=2, column=0, sticky="ew", padx=16, pady=(0, 10)
        )

        threshold_row = ctk.CTkFrame(settings_card, fg_color="transparent")
        threshold_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        threshold_row.grid_columnconfigure(0, weight=1)

        self._threshold_entry = ctk.CTkEntry(
            threshold_row,
            placeholder_text="照合閾値",
            font=self._font_body,
            fg_color="#ffffff",
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            height=40,
        )
        self._threshold_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._threshold_entry.insert(0, runtime.matching_threshold_text())

        self._threshold_apply_button = ctk.CTkButton(
            threshold_row,
            text="適用",
            width=90,
            command=self._handle_threshold_apply,
            font=self._font_body,
            fg_color=ACCENT,
            hover_color="#184d71",
            height=40,
        )
        self._threshold_apply_button.grid(row=0, column=1, sticky="ew")

        lower_grid = ctk.CTkFrame(self._dashboard_panel, fg_color="transparent")
        lower_grid.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        lower_grid.grid_columnconfigure(0, weight=1)

        people_card = ctk.CTkFrame(
            lower_grid,
            corner_radius=20,
            fg_color=CARD_ALT_BG,
        )
        people_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        people_card.grid_columnconfigure(0, weight=1)

        people_title = ctk.CTkLabel(
            people_card,
            text="登録済み人物",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
        )
        people_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 10))

        self._person_menu = ctk.CTkOptionMenu(
            people_card,
            values=list(runtime.person_choice_labels()),
            command=self._handle_person_change,
            font=self._font_body,
            dropdown_font=self._font_body,
            fg_color=ACCENT,
            button_color=ACCENT,
            button_hover_color="#184d71",
            height=40,
        )
        self._person_menu.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        self._delete_person_button = ctk.CTkButton(
            people_card,
            text="選択人物を削除",
            command=self._handle_delete_person,
            font=self._font_body,
            fg_color="#8d4b4b",
            hover_color="#753d3d",
            height=40,
        )
        self._delete_person_button.grid(
            row=2, column=0, sticky="ew", padx=16, pady=(0, 10)
        )

        self._people_list = ctk.CTkScrollableFrame(
            people_card,
            corner_radius=16,
            fg_color="#ffffff",
            height=200,
        )
        self._people_list.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        self._people_list.grid_columnconfigure(0, weight=1)

        result_card = ctk.CTkFrame(
            lower_grid,
            corner_radius=20,
            fg_color=CARD_ALT_BG,
        )
        result_card.grid(row=1, column=0, sticky="ew")
        result_card.grid_columnconfigure(0, weight=1)

        result_title = ctk.CTkLabel(
            result_card,
            text="照合結果",
            font=self._font_section,
            text_color=TEXT_PRIMARY,
        )
        result_title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 10))

        self._result_primary_label = ctk.CTkLabel(
            result_card,
            text="まだ照合していません。",
            font=self._font_result,
            text_color=TEXT_PRIMARY,
            anchor="w",
            justify="left",
            wraplength=360,
        )
        self._result_primary_label.grid(
            row=1, column=0, sticky="nw", padx=16, pady=(0, 8)
        )

        self._result_detail_label = ctk.CTkLabel(
            result_card,
            text="",
            font=self._font_body,
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=360,
        )
        self._result_detail_label.grid(
            row=2, column=0, sticky="nw", padx=16, pady=(0, 16)
        )

    def _require_runtime(self) -> FaceRecognitionRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("FaceRecognitionRuntime is not initialized.")
        return runtime

    def _schedule_tick(self) -> None:
        self._tick_after_id = self.after(100, self._tick)

    def _tick(self) -> None:
        if self._runtime is not None and self._runtime.state.camera.status == "running":
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

    def _refresh_metadata(self, view_model) -> None:
        message_text = view_model.message
        if message_text != self._last_message_text:
            self._message_banner.configure(text=message_text)
            self._last_message_text = message_text

        if view_model.status_lines != self._last_status_lines:
            self._render_status_cards(view_model.status_lines)
            self._last_status_lines = view_model.status_lines

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

    def _render_status_cards(self, status_lines: tuple[str, ...]) -> None:
        normalized = list(status_lines)
        while len(normalized) < len(self._status_value_labels):
            normalized.append("-=-")

        for index, status_line in enumerate(
            normalized[: len(self._status_value_labels)]
        ):
            key_label, value_label = self._status_value_labels[index]
            if "=" in status_line:
                key, value = status_line.split("=", 1)
            else:
                key, value = "info", status_line
            key_label.configure(text=key.upper())
            value_label.configure(text=value)

    def _render_people_list(self, people_lines: tuple[str, ...]) -> None:
        for child in self._people_list.winfo_children():
            child.destroy()

        if len(people_lines) == 0:
            people_lines = ("未登録です。",)

        for row_index, line in enumerate(people_lines):
            is_selected = line.startswith(">")
            text = line[1:].strip() if is_selected else line.strip()
            label = ctk.CTkLabel(
                self._people_list,
                text=text,
                anchor="w",
                justify="left",
                wraplength=330,
                font=self._font_body,
                text_color=TEXT_PRIMARY if is_selected else TEXT_MUTED,
                fg_color=ACCENT_SOFT if is_selected else "transparent",
                corner_radius=12,
                padx=12,
                pady=10,
            )
            label.grid(row=row_index, column=0, sticky="ew", padx=8, pady=6)

    def _render_result(self, result_lines: tuple[str, ...]) -> None:
        if len(result_lines) == 0:
            self._result_primary_label.configure(text="まだ照合していません。")
            self._result_detail_label.configure(text="")
            return

        primary_text = result_lines[0]
        detail_text = "\n".join(result_lines[1:])
        self._result_primary_label.configure(text=primary_text)
        self._result_detail_label.configure(text=detail_text)

    def _fit_preview_image(self, image: Image.Image) -> Image.Image:
        max_width = max(480, self._preview_label.winfo_width() - 16)
        max_height = max(320, self._preview_label.winfo_height() - 16)

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
