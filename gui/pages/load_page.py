"""
Страница загрузки .eeg файлов и настройки параметров.
Компактный вид — всё укладывается без прокрутки.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QListWidget,
    QFrame, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QAbstractItemView, QRadioButton, QButtonGroup, QGroupBox,
    QMessageBox, QComboBox, QScrollArea, QSizePolicy,
)

from gui.styles import PRIMARY, TEXT_SECONDARY, ACCENT, BORDER


# ── File drop area ──────────────────────────────────────────────────────────

class _FileDropArea(QFrame):
    """Область drag-and-drop + кнопка для .eeg файлов."""

    files_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        lbl = QLabel("Перетащите .eeg / .dat файлы сюда или нажмите кнопку")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
        root.addWidget(lbl)

        self._list = QListWidget()
        self._list.setObjectName("fileDrop")
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.setMaximumHeight(100)
        root.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Выбрать файлы")
        btn_add.setObjectName("secondary")
        btn_add.clicked.connect(self._browse)
        btn_row.addWidget(btn_add)
        btn_rm = QPushButton("Удалить выбранные")
        btn_rm.setObjectName("link")
        btn_rm.clicked.connect(self._remove_selected)
        btn_row.addWidget(btn_rm)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── drag and drop ──
    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev: QDropEvent):
        if ev.mimeData().hasUrls():
            for url in ev.mimeData().urls():
                fp = url.toLocalFile()
                ext = Path(fp).suffix.lower()
                if ext in (".eeg", ".dat") and fp not in self._all():
                    self._list.addItem(fp)
            ev.acceptProposedAction()
            self.files_changed.emit()

    def _browse(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите EEG-файлы",
            str(Path.home()),
            "EEG files (*.eeg *.dat);;All Files (*)",
        )
        for f in files:
            if f not in self._all():
                self._list.addItem(f)
        if files:
            self.files_changed.emit()

    def _remove_selected(self):
        # Удаляем с конца, чтобы индексы не сдвигались
        rows = sorted(
            {self._list.row(item) for item in self._list.selectedItems()},
            reverse=True,
        )
        for row in rows:
            self._list.takeItem(row)
        self.files_changed.emit()

    def _all(self) -> set[str]:
        return {self._list.item(i).text() for i in range(self._list.count())}

    def get_files(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())]

    def clear(self):
        self._list.clear()
        self.files_changed.emit()


# ── Per-file emotion selector ───────────────────────────────────────────────

class _FileEmotionRow(QFrame):
    """Одна строка: имя .eeg файла + комбобокс эмоции."""

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.setStyleSheet(
            f"background: #fafbfc; border: 1px solid {BORDER}; "
            f"border-radius: 6px; padding: 3px 6px;"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)

        name_lbl = QLabel(Path(filepath).name)
        name_lbl.setStyleSheet(f"font-size: 11px; color: {PRIMARY}; font-weight: 600; border: none;")
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(name_lbl, stretch=1)

        self.combo = QComboBox()
        self.combo.addItem("Радостный / Возбуждённый (HVHA)", "HVHA")
        self.combo.addItem("Спокойный / Умиротворённый (HVLA)", "HVLA")
        self.combo.addItem("Злой / Напряжённый (LVHA)", "LVHA")
        self.combo.addItem("Грустный / Подавленный (LVLA)", "LVLA")
        self.combo.setCurrentIndex(1)  # HVLA по умолчанию
        self.combo.setMinimumWidth(240)
        self.combo.setStyleSheet("border: none;")
        lay.addWidget(self.combo)

    def get_emotion(self) -> str:
        """Возвращает выбранную эмоцию."""
        return self.combo.currentData()


# ── EEG file info card ──────────────────────────────────────────────────────

class _EEGInfoCard(QFrame):
    """Краткая информация о загруженном файле."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self._label = QLabel("Выберите файл для просмотра информации")
        self._label.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        self._label.setWordWrap(True)
        lay.addWidget(self._label)

    def show_info(self, summary: dict):
        lines = [
            f"<b>{summary.get('filename', '?')}</b>",
            f"Каналов: {summary.get('n_channels', '?')} "
            f"(ЭЭГ: {len(summary.get('eeg_channels', []))}), "
            f"Частота: {summary.get('srate', '?')} Гц, "
            f"Длит.: {summary.get('duration_sec', 0):.1f} с",
        ]
        eeg_ch = summary.get("eeg_channels", [])
        if eeg_ch:
            lines.append(
                f"Каналы: {', '.join(eeg_ch[:8])}{'...' if len(eeg_ch) > 8 else ''}"
            )
        self._label.setText("<br>".join(lines))

    def show_error(self, msg: str):
        self._label.setText(f"<span style='color:red;'>{msg}</span>")

    def clear_info(self):
        self._label.setText("Выберите файл для просмотра информации")


# ── Main Load Page ──────────────────────────────────────────────────────────

class LoadPage(QWidget):
    """
    Страница загрузки файлов. Компактная — без прокрутки.
    Signals:
        start_analysis(list[str], dict)
        go_back()
    """

    start_analysis = Signal(list, dict)
    go_back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 14, 32, 14)
        root.setSpacing(10)

        # ── Header ──
        hdr = QHBoxLayout()
        btn_back = QPushButton("Назад")
        btn_back.setObjectName("link")
        btn_back.clicked.connect(self.go_back.emit)
        hdr.addWidget(btn_back)
        hdr.addStretch()
        title = QLabel("Загрузка EEG-файлов")
        title.setStyleSheet(f"font-size:20px; font-weight:700; color:{PRIMARY};")
        hdr.addWidget(title)
        hdr.addStretch()
        spacer = QLabel()
        spacer.setFixedWidth(50)
        hdr.addWidget(spacer)
        root.addLayout(hdr)

        # ── Two-column: files | params ──
        columns = QHBoxLayout()
        columns.setSpacing(16)

        # LEFT — файлы
        left = QVBoxLayout()
        left.setSpacing(8)
        self._drop = _FileDropArea()
        self._drop.files_changed.connect(self._on_files_changed)
        left.addWidget(self._drop)
        self._eeg_info = _EEGInfoCard()
        left.addWidget(self._eeg_info)

        # .dat / DEAP — выбор количества триалов
        self._dat_trials_frame = QFrame()
        self._dat_trials_frame.setObjectName("card")
        self._dat_trials_frame.setVisible(False)
        dat_lay = QHBoxLayout(self._dat_trials_frame)
        dat_lay.setContentsMargins(10, 6, 10, 6)
        dat_lay.setSpacing(8)
        dat_info_lbl = QLabel("DEAP .dat — количество триалов для анализа:")
        dat_info_lbl.setStyleSheet(f"font-size:11px; color:{PRIMARY}; font-weight:600;")
        dat_lay.addWidget(dat_info_lbl)
        self._spin_trials = QSpinBox()
        self._spin_trials.setRange(1, 40)
        self._spin_trials.setValue(5)
        self._spin_trials.setToolTip("Каждый триал — 63-секундная запись ЭЭГ (всего 40 триалов в файле)")
        dat_lay.addWidget(self._spin_trials)
        dat_lay.addStretch()
        left.addWidget(self._dat_trials_frame)

        # Per-file emotion selectors (shown for .eeg files)
        self._emotion_label = QLabel("Эмоции для .eeg файлов:")
        self._emotion_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {PRIMARY}; margin-top: 4px;"
        )
        self._emotion_label.setVisible(False)
        left.addWidget(self._emotion_label)

        self._emotion_container = QVBoxLayout()
        self._emotion_container.setSpacing(4)
        self._emotion_container.setContentsMargins(0, 0, 0, 0)
        left.addLayout(self._emotion_container)
        self._emotion_rows: list[_FileEmotionRow] = []

        left.addStretch()
        columns.addLayout(left, stretch=1)

        # RIGHT — параметры
        right = QVBoxLayout()
        right.setSpacing(8)
        params_frame = QFrame()
        params_frame.setObjectName("card")
        form = QFormLayout(params_frame)
        form.setContentsMargins(14, 10, 14, 10)
        form.setSpacing(5)
        form.setLabelAlignment(Qt.AlignRight)

        # Счётчики треков по эмоциям (из датасетов, не меняются)
        self._TRACK_COUNTS = {
            "emopia": {"HVHA": 250, "HVLA": 265, "LVLA": 253, "LVHA": 310, "total": 1078},
            "maestro": {"HVHA": 888, "HVLA": 216, "LVLA": 119, "LVHA": 53, "total": 1276},
        }

        def _sec(text: str) -> QLabel:
            """Заголовок секции — единый стиль для всей формы."""
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"font-size:13px; font-weight:600; color:{PRIMARY}; "
                f"margin-top:8px; margin-bottom:2px;"
            )
            return lbl

        form.addRow(_sec("Параметры анализа"))

        self._spin_classical = QSpinBox()
        self._spin_classical.setRange(1, 500)
        self._spin_classical.setValue(10)
        self._spin_classical.setToolTip("Сколько произведений случайно отобрать для сравнения с ЭЭГ")
        form.addRow("Кол-во произведений:", self._spin_classical)

        self._spin_topk = QSpinBox()
        self._spin_topk.setRange(1, 100)
        self._spin_topk.setValue(10)
        self._spin_topk.setToolTip("Сколько наиболее похожих произведений показать в результате")
        form.addRow("Результатов (топ-N):", self._spin_topk)

        self._spin_jobs = QSpinBox()
        self._spin_jobs.setRange(0, 32)
        self._spin_jobs.setValue(0)
        self._spin_jobs.setSpecialValueText("Авто")
        self._spin_jobs.setToolTip("Число параллельных процессов (0 = определяется автоматически)")
        form.addRow("Параллельных процессов:", self._spin_jobs)

        self._spin_window = QDoubleSpinBox()
        self._spin_window.setRange(1.0, 60.0)
        self._spin_window.setValue(4.0)
        self._spin_window.setSuffix(" сек")
        self._spin_window.setToolTip("Длина скользящего окна для анализа ЭЭГ")
        form.addRow("Размер окна:", self._spin_window)

        self._spin_hop = QDoubleSpinBox()
        self._spin_hop.setRange(0.5, 30.0)
        self._spin_hop.setValue(2.0)
        self._spin_hop.setSuffix(" сек")
        self._spin_hop.setToolTip("Шаг сдвига скользящего окна")
        form.addRow("Шаг окна:", self._spin_hop)

        self._spin_max_seconds = QDoubleSpinBox()
        self._spin_max_seconds.setRange(0, 3600.0)
        self._spin_max_seconds.setValue(0)
        self._spin_max_seconds.setSuffix(" сек")
        self._spin_max_seconds.setSpecialValueText("Весь файл")
        self._spin_max_seconds.setToolTip("Ограничить длину анализируемого ЭЭГ (0 = весь файл)")
        form.addRow("Макс. длина ЭЭГ:", self._spin_max_seconds)

        # ── Набор произведений ──────────────────────────────────────────
        form.addRow(_sec("Набор произведений"))

        self._corpus_group = QButtonGroup(self)
        self._radio_both    = QRadioButton("MAESTRO + EMOPIA  (2354 треков)")
        self._radio_maestro = QRadioButton("Только MAESTRO  (1276 треков)")
        self._radio_emopia  = QRadioButton("Только EMOPIA  (1078 треков)")
        self._radio_both.setChecked(True)
        for idx, rb in enumerate((self._radio_both, self._radio_maestro, self._radio_emopia)):
            self._corpus_group.addButton(rb, idx)
            form.addRow(rb)

        # ── Режим сравнения ─────────────────────────────────────────────
        form.addRow(_sec("Режим сравнения"))

        self._search_group = QButtonGroup(self)
        self._radio_search_all  = QRadioButton("Среди всех выбранных произведений")
        self._radio_search_emo  = QRadioButton("Только среди произведений одной эмоции")
        self._radio_search_both = QRadioButton("Сравнить оба режима")
        self._radio_search_all.setChecked(True)
        for idx, rb in enumerate((self._radio_search_all, self._radio_search_emo, self._radio_search_both)):
            self._search_group.addButton(rb, idx)
            form.addRow(rb)

        # Выбор эмоции (появляется только в режимах 1 и 2)
        self._combo_search_emo = QComboBox()
        self._combo_search_emo.addItem("Радостный / Возбуждённый (HVHA)", "HVHA")
        self._combo_search_emo.addItem("Спокойный / Умиротворённый (HVLA)", "HVLA")
        self._combo_search_emo.addItem("Злой / Напряжённый (LVHA)", "LVHA")
        self._combo_search_emo.addItem("Грустный / Подавленный (LVLA)", "LVLA")
        self._combo_search_emo.setEnabled(False)
        form.addRow("Эмоция:", self._combo_search_emo)

        # Динамическая подпись с количеством доступных треков
        self._emo_avail_lbl = QLabel()
        self._emo_avail_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        self._emo_avail_lbl.setVisible(False)
        form.addRow(self._emo_avail_lbl)

        def _on_search_mode_changed():
            active = self._radio_search_emo.isChecked() or self._radio_search_both.isChecked()
            self._combo_search_emo.setEnabled(active)
            self._update_emo_count()

        self._radio_search_emo.toggled.connect(_on_search_mode_changed)
        self._radio_search_both.toggled.connect(_on_search_mode_changed)
        self._combo_search_emo.currentIndexChanged.connect(self._update_emo_count)
        self._corpus_group.buttonClicked.connect(self._update_emo_count)

        # ── Конкретный MIDI для сравнения ───────────────────────────────
        form.addRow(_sec("Конкретный MIDI для сравнения"))

        self._manual_midi_path: str | None = None
        self._manual_midi_lbl = QLabel("не выбран")
        self._manual_midi_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        form.addRow(self._manual_midi_lbl)

        midi_btn_row = QHBoxLayout()
        btn_pick_midi = QPushButton("Выбрать файл…")
        btn_pick_midi.setObjectName("secondary")
        btn_pick_midi.clicked.connect(self._pick_manual_midi)
        midi_btn_row.addWidget(btn_pick_midi)
        btn_clear_midi = QPushButton("Очистить")
        btn_clear_midi.setObjectName("link")
        btn_clear_midi.clicked.connect(self._clear_manual_midi)
        midi_btn_row.addWidget(btn_clear_midi)
        midi_btn_row.addStretch()
        form.addRow(midi_btn_row)

        right.addWidget(params_frame)
        right.addStretch()
        columns.addLayout(right, stretch=1)

        root.addLayout(columns, stretch=1)

        # ── Run button ──
        self._btn_run = QPushButton("Запустить анализ")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(44)
        self._btn_run.clicked.connect(self._on_start)
        root.addWidget(self._btn_run)

    # ------------------------------------------------------------------
    def _on_files_changed(self):
        files = self._drop.get_files()
        if not files:
            self._eeg_info.clear_info()
            self._rebuild_emotion_rows([])
            self._dat_trials_frame.setVisible(False)
            return

        # Показать блок триалов если есть .dat файлы
        dat_files = [f for f in files if f.lower().endswith(".dat")]
        self._dat_trials_frame.setVisible(bool(dat_files))

        # Rebuild per-file emotion combos for .eeg files
        eeg_files = [f for f in files if f.lower().endswith(".eeg")]
        self._rebuild_emotion_rows(eeg_files)
        last = files[-1]
        if last.lower().endswith(".eeg"):
            try:
                from src.neurosoft_loader import get_neurosoft_file_summary
                summary = get_neurosoft_file_summary(last)
                self._eeg_info.show_info(summary)
            except Exception as e:
                self._eeg_info.show_error(str(e))
        elif last.lower().endswith(".dat"):
            self._eeg_info.show_info({
                "filename": Path(last).name,
                "study_name": "DEAP Dataset",
                "srate": 128,
                "n_channels": 40,
                "eeg_channels": [f"ch{i}" for i in range(32)],
                "duration_sec": 63.0,
                "n_samples": 8064,
            })
        else:
            self._eeg_info.show_error("Неподдерживаемый формат файла")

    def _rebuild_emotion_rows(self, eeg_files: list[str]):
        """Пересоздаёт per-file emotion combos."""
        # Remove old rows
        for row in self._emotion_rows:
            self._emotion_container.removeWidget(row)
            row.deleteLater()
        self._emotion_rows.clear()

        if not eeg_files:
            self._emotion_label.setVisible(False)
            return

        self._emotion_label.setVisible(True)
        for fp in eeg_files:
            row = _FileEmotionRow(fp)
            self._emotion_container.addWidget(row)
            self._emotion_rows.append(row)

    def _get_file_emotions(self) -> dict[str, str | None]:
        """Возвращает {filepath: emotion} для каждого .eeg файла."""
        return {row.filepath: row.get_emotion() for row in self._emotion_rows}

    def _update_emo_count(self):
        """Обновляет подпись с количеством доступных треков при текущем выборе эмоции+набора."""
        active = self._radio_search_emo.isChecked() or self._radio_search_both.isChecked()
        if not active:
            self._emo_avail_lbl.setVisible(False)
            return

        emo = self._combo_search_emo.currentData()
        cid = self._corpus_group.checkedId()

        if cid == 0:
            n_e = self._TRACK_COUNTS["emopia"].get(emo, 0)
            n_m = self._TRACK_COUNTS["maestro"].get(emo, 0)
            total = n_e + n_m
            text = f"Доступно в группе: {total} (EMOPIA: {n_e}, MAESTRO: {n_m})"
        elif cid == 1:
            n_m = self._TRACK_COUNTS["maestro"].get(emo, 0)
            text = f"Доступно в группе: {n_m} треков MAESTRO"
        else:
            n_e = self._TRACK_COUNTS["emopia"].get(emo, 0)
            text = f"Доступно в группе: {n_e} треков EMOPIA"

        self._emo_avail_lbl.setText(text)
        self._emo_avail_lbl.setVisible(True)

    def _pick_manual_midi(self):
        """Выбор конкретного MIDI-файла для ручного сравнения."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите MIDI-файл для сравнения",
            str(Path.home()),
            "MIDI files (*.mid *.midi);;All Files (*)",
        )
        if path:
            self._manual_midi_path = path
            name = Path(path).name
            self._manual_midi_lbl.setText(name)
            self._manual_midi_lbl.setStyleSheet(f"font-size:11px; color:{ACCENT}; font-weight:600;")

    def _clear_manual_midi(self):
        self._manual_midi_path = None
        self._manual_midi_lbl.setText("не выбран")
        self._manual_midi_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")

    def _on_start(self):
        files = self._drop.get_files()
        if not files:
            QMessageBox.warning(self, "Нет файлов", "Загрузите хотя бы один файл.")
            return

        corpus_id = self._corpus_group.checkedId()
        only_emopia = (corpus_id == 2)
        only_maestro = (corpus_id == 1)

        search_id = self._search_group.checkedId()
        match_by_emotion = search_id in (1, 2)
        compare_modes = (search_id == 2)
        target_emotion = self._combo_search_emo.currentData() if match_by_emotion else None

        params = {
            "max_classical": self._spin_classical.value(),
            "top_k": self._spin_topk.value(),
            "n_jobs": self._spin_jobs.value() or None,
            "window_size": self._spin_window.value(),
            "hop_size": self._spin_hop.value(),
            "max_seconds": self._spin_max_seconds.value() or None,
            "max_trials": self._spin_trials.value(),
            "only_emopia": only_emopia,
            "only_maestro": only_maestro,
            "match_emotions": match_by_emotion,
            "compare_modes": compare_modes,
            "target_emotion": target_emotion,
            "manual_midi_path": self._manual_midi_path,
            "eeg_emotions": self._get_file_emotions(),
        }
        self.start_analysis.emit(files, params)

    def set_running(self, running: bool):
        """Блокирует/разблокирует кнопку запуска во время анализа."""
        if hasattr(self, '_btn_run'):
            self._btn_run.setEnabled(not running)
            self._btn_run.setText("Анализ идёт…" if running else "Запустить анализ")

    def apply_params(self, params: dict):
        """Восстанавливает параметры из предыдущего запуска."""
        if not params:
            return
        if "max_classical" in params and hasattr(self, '_spin_classical'):
            self._spin_classical.setValue(params["max_classical"])
        if "top_k" in params and hasattr(self, '_spin_topk'):
            self._spin_topk.setValue(params["top_k"])
        if "window_size" in params and hasattr(self, '_spin_window'):
            self._spin_window.setValue(params["window_size"])
        if "hop_size" in params and hasattr(self, '_spin_hop'):
            self._spin_hop.setValue(params["hop_size"])
        if "max_trials" in params and hasattr(self, '_spin_trials'):
            self._spin_trials.setValue(params["max_trials"])
        if params.get("only_emopia") and hasattr(self, '_radio_emopia'):
            self._radio_emopia.setChecked(True)
        elif params.get("only_maestro") and hasattr(self, '_radio_maestro'):
            self._radio_maestro.setChecked(True)
        elif hasattr(self, '_radio_both'):
            self._radio_both.setChecked(True)

    def reset(self):
        self._drop.clear()
        self._eeg_info.clear_info()
        self._dat_trials_frame.setVisible(False)
        self._rebuild_emotion_rows([])
        self._radio_both.setChecked(True)
        self._radio_search_all.setChecked(True)
        self._combo_search_emo.setEnabled(False)
        self._emo_avail_lbl.setVisible(False)
        self._clear_manual_midi()
