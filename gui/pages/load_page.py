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
    QCheckBox, QAbstractItemView,
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
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
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
        self.combo.addItem("Авто", "auto")
        self.combo.addItem("HVHA", "HVHA")
        self.combo.addItem("HVLA", "HVLA")
        self.combo.addItem("LVHA", "LVHA")
        self.combo.addItem("LVLA", "LVLA")
        self.combo.setFixedWidth(100)
        self.combo.setStyleSheet("border: none;")
        lay.addWidget(self.combo)

    def get_emotion(self) -> str | None:
        """Возвращает выбранную эмоцию или None для 'auto'."""
        val = self.combo.currentData()
        return val if val != "auto" else None


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

        section_lbl = QLabel("Параметры анализа")
        section_lbl.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{PRIMARY}; margin-bottom:2px;"
        )
        form.addRow(section_lbl)

        self._spin_classical = QSpinBox()
        self._spin_classical.setRange(1, 500)
        self._spin_classical.setValue(10)
        self._spin_classical.setToolTip("Произведений для сравнения")
        form.addRow("Произведений:", self._spin_classical)

        self._spin_topk = QSpinBox()
        self._spin_topk.setRange(1, 100)
        self._spin_topk.setValue(10)
        form.addRow("Топ-K:", self._spin_topk)

        self._spin_jobs = QSpinBox()
        self._spin_jobs.setRange(0, 32)
        self._spin_jobs.setValue(0)
        self._spin_jobs.setSpecialValueText("Авто")
        form.addRow("Процессы:", self._spin_jobs)

        self._spin_window = QDoubleSpinBox()
        self._spin_window.setRange(1.0, 60.0)
        self._spin_window.setValue(4.0)
        self._spin_window.setSuffix(" сек")
        form.addRow("Окно:", self._spin_window)

        self._spin_hop = QDoubleSpinBox()
        self._spin_hop.setRange(0.5, 30.0)
        self._spin_hop.setValue(2.0)
        self._spin_hop.setSuffix(" сек")
        form.addRow("Шаг:", self._spin_hop)

        self._spin_max_seconds = QDoubleSpinBox()
        self._spin_max_seconds.setRange(0, 3600.0)
        self._spin_max_seconds.setValue(0)
        self._spin_max_seconds.setSuffix(" сек")
        self._spin_max_seconds.setSpecialValueText("Весь файл")
        self._spin_max_seconds.setToolTip("Макс. длительность EEG (0 = весь файл)")
        form.addRow("Макс. длит.:", self._spin_max_seconds)

        self._chk_emopia = QCheckBox("Только EMOPIA")
        form.addRow(self._chk_emopia)

        self._chk_match = QCheckBox("Фильтр по эмоциям")
        form.addRow(self._chk_match)

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
            return
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

    def _on_start(self):
        files = self._drop.get_files()
        if not files:
            QMessageBox.warning(self, "Нет файлов", "Загрузите хотя бы один файл.")
            return
        params = {
            "max_classical": self._spin_classical.value(),
            "top_k": self._spin_topk.value(),
            "n_jobs": self._spin_jobs.value() or None,
            "window_size": self._spin_window.value(),
            "hop_size": self._spin_hop.value(),
            "max_seconds": self._spin_max_seconds.value() or None,
            "only_emopia": self._chk_emopia.isChecked(),
            "match_emotions": self._chk_match.isChecked(),
            "eeg_emotions": self._get_file_emotions(),  # per-file emotions
        }
        self.start_analysis.emit(files, params)

    def reset(self):
        self._drop.clear()
        self._eeg_info.clear_info()
        self._rebuild_emotion_rows([])
