"""
Страница анализа с прогрессом и логом.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QTextEdit, QFrame,
)

from gui.styles import PRIMARY, ACCENT, TEXT_SECONDARY, DANGER


class AnalysisPage(QWidget):
    """Страница с прогресс-баром и журналом выполнения.

    Signals:
        cancel_requested()
    """

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 20)
        root.setSpacing(16)

        # ── header ──
        title = QLabel("Анализ выполняется…")
        title.setStyleSheet(f"font-size:22px; font-weight:700; color:{PRIMARY};")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)
        self._title = title

        # ── stage label ──
        self._stage = QLabel("Инициализация…")
        self._stage.setAlignment(Qt.AlignCenter)
        self._stage.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:13px;")
        root.addWidget(self._stage)

        # ── progress bar ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setMinimumHeight(30)
        root.addWidget(self._progress)

        # ── log ──
        self._log = QTextEdit()
        self._log.setObjectName("console")
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Menlo", 11))
        root.addWidget(self._log, stretch=1)

        # ── cancel ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_cancel = QPushButton("Отменить")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.clicked.connect(self.cancel_requested.emit)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    def set_progress(self, pct: int, text: str):
        self._progress.setValue(pct)
        self._stage.setText(text)

    def append_log(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def reset(self):
        self._progress.setValue(0)
        self._stage.setText("Инициализация…")
        self._log.clear()
        self._title.setText("Анализ выполняется…")
        self._btn_cancel.setEnabled(True)

    def mark_done(self):
        self._title.setText("Анализ завершён")
        self._title.setStyleSheet(f"font-size:22px; font-weight:700; color:{ACCENT};")
        self._btn_cancel.setEnabled(False)

    def mark_error(self, msg: str):
        self._title.setText("Ошибка")
        self._title.setStyleSheet(f"font-size:22px; font-weight:700; color:{DANGER};")
        self._btn_cancel.setEnabled(False)
        self._log.append(f"<span style='color:{DANGER};'>ОШИБКА: {msg}</span>")
