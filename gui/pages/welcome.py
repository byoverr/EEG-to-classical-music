"""
Страница приветствия: история сравнений + кнопка «Новый анализ».
"""
from __future__ import annotations

import webbrowser
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QMessageBox,
)

from gui.history import load_history, delete_run
from gui.styles import PRIMARY, ACCENT, TEXT_SECONDARY, EMOTION_COLORS


class _HistoryCard(QFrame):
    """Одна карточка прошлого запуска."""

    open_requested = Signal(dict)   # entry
    delete_requested = Signal(int)  # id

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # --- header row ---
        top = QHBoxLayout()
        lbl_title = QLabel(entry.get("label", "Запуск"))
        lbl_title.setStyleSheet(f"font-size:14px; font-weight:600; color:{PRIMARY};")
        top.addWidget(lbl_title)
        top.addStretch()

        ts = entry.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_text = dt.strftime("%d.%m.%Y %H:%M")
        except Exception:
            ts_text = ts
        lbl_time = QLabel(ts_text)
        lbl_time.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        top.addWidget(lbl_time)
        root.addLayout(top)

        # --- meta ---
        n_results = entry.get("n_results", 0)
        best = entry.get("best_score", 0)
        files = entry.get("eeg_files", [])
        meta = f"Результатов: {n_results}  ·  Best: {best:.4f}  ·  Файлов: {len(files)}"
        lbl_meta = QLabel(meta)
        lbl_meta.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:12px;")
        root.addWidget(lbl_meta)

        # --- params ---
        params = entry.get("params", {})
        if params:
            parts = []
            if "max_classical" in params:
                parts.append(f"Произведений: {params['max_classical']}")
            if "window_size" in params:
                parts.append(f"Окно: {params['window_size']} сек")
            if "top_k" in params:
                parts.append(f"Топ-N: {params['top_k']}")
            if parts:
                lbl_p = QLabel("  ·  ".join(parts))
                lbl_p.setStyleSheet("color:#888; font-size:11px;")
                root.addWidget(lbl_p)

        # --- buttons ---
        btn_row = QHBoxLayout()
        btn_open = QPushButton("Открыть результаты")
        btn_open.setObjectName("secondary")
        btn_open.clicked.connect(lambda: self.open_requested.emit(self.entry))
        btn_row.addWidget(btn_open)

        report_dir = entry.get("report_dir", "")
        html = Path(report_dir) / "index.html" if report_dir else None
        if html and html.exists():
            btn_html = QPushButton("HTML")
            btn_html.setObjectName("link")
            btn_html.clicked.connect(lambda: webbrowser.open(f"file://{html}"))
            btn_row.addWidget(btn_html)

        btn_row.addStretch()
        btn_del = QPushButton("Удалить")
        btn_del.setObjectName("link")
        btn_del.setToolTip("Удалить из истории")
        btn_del.clicked.connect(lambda: self.delete_requested.emit(entry.get("id", 0)))
        btn_row.addWidget(btn_del)
        root.addLayout(btn_row)


class WelcomePage(QWidget):
    """Приветственная страница с историей."""

    new_analysis = Signal()           # кнопка «Новый анализ»
    open_history = Signal(dict)       # открыть прошлый запуск

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 20)
        root.setSpacing(0)

        # ── Title block ──
        title = QLabel("EEG → Classical Music")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size:32px; font-weight:700; color:{PRIMARY}; margin-bottom:2px;")
        root.addWidget(title)

        subtitle = QLabel("Анализ ЭЭГ-сигналов и сравнение с классическими произведениями")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:14px; margin-bottom:20px;")
        root.addWidget(subtitle)

        # ── New analysis button ──
        btn = QPushButton("Новый анализ")
        btn.setObjectName("primary")
        btn.setFixedHeight(50)
        btn.setMinimumWidth(280)
        btn.setMaximumWidth(400)
        btn.setStyleSheet(btn.styleSheet() + "font-size:16px;")
        btn.clicked.connect(self.new_analysis.emit)
        hlayout = QHBoxLayout()
        hlayout.addStretch()
        hlayout.addWidget(btn)
        hlayout.addStretch()
        root.addLayout(hlayout)

        root.addSpacing(24)

        # ── History section ──
        self._history_header = QLabel("Последние анализы")
        self._history_header.setStyleSheet("font-size:16px; font-weight:600; margin-bottom:8px;")
        root.addWidget(self._history_header)

        self._history_empty = QLabel("Нет сохранённых анализов. Нажмите «Новый анализ», чтобы начать.")
        self._history_empty.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:13px; padding:20px;")
        self._history_empty.setAlignment(Qt.AlignCenter)
        root.addWidget(self._history_empty)

        # scroll area for history cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setSpacing(10)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, stretch=1)

        self.refresh_history()

    # ------------------------------------------------------------------
    def refresh(self):
        """Alias для обратной совместимости с MainWindow."""
        self.refresh_history()

    def refresh_history(self):
        """Перезагрузить список из файла."""
        # Очистить
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        history = load_history()
        self._history_empty.setVisible(len(history) == 0)
        self._scroll.setVisible(len(history) > 0)

        for entry in history[:20]:
            card = _HistoryCard(entry)
            card.open_requested.connect(self._on_open)
            card.delete_requested.connect(self._on_delete)
            self._scroll_layout.addWidget(card)

        self._scroll_layout.addStretch()

    def _on_open(self, entry: dict):
        self.open_history.emit(entry)

    def _on_delete(self, run_id: int):
        reply = QMessageBox.question(
            self, "Удалить",
            "Удалить запись из истории?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            delete_run(run_id)
            self.refresh_history()
