#!/usr/bin/env python3
"""
Виджет отображения результатов: таблица + встроенные графики matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QScrollArea, QPushButton, QFileDialog,
    QSizePolicy, QGroupBox, QGridLayout,
)


class MetricCard(QGroupBox):
    """Карточка с одной метрикой (имя + значение)."""

    def __init__(self, title: str, value: str, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
            }
            QGroupBox::title { subcontrol-position: top center; }
        """)
        layout = QVBoxLayout(self)
        lbl = QLabel(value)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #1a73e8;")
        layout.addWidget(lbl)


class ResultsWidget(QWidget):
    """Виджет для отображения результатов пайплайна."""

    open_html_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results_df: pd.DataFrame | None = None
        self._report_dir: str = ""
        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        root = QVBoxLayout(self)

        # --- Summary cards ---
        self._cards_layout = QHBoxLayout()
        self._cards_layout.setSpacing(12)
        root.addLayout(self._cards_layout)

        # --- Tabs ---
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        # Tab 1: Table
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tabs.addTab(self._table, "Таблица результатов")

        # Tab 2: Similarity chart
        self._chart_similarity = QLabel()
        self._chart_similarity.setAlignment(Qt.AlignCenter)
        scroll1 = QScrollArea()
        scroll1.setWidget(self._chart_similarity)
        scroll1.setWidgetResizable(True)
        self._tabs.addTab(scroll1, "График сходства")

        # Tab 3: Emotion distribution
        self._chart_emotion = QLabel()
        self._chart_emotion.setAlignment(Qt.AlignCenter)
        scroll2 = QScrollArea()
        scroll2.setWidget(self._chart_emotion)
        scroll2.setWidgetResizable(True)
        self._tabs.addTab(scroll2, "Эмоции")

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self._btn_html = QPushButton("🌐 Открыть HTML-отчёт")
        self._btn_html.clicked.connect(self._open_html)
        btn_row.addWidget(self._btn_html)

        self._btn_csv = QPushButton("📄 Экспорт CSV")
        self._btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_csv)

        self._btn_folder = QPushButton("📁 Открыть папку отчёта")
        self._btn_folder.clicked.connect(self._open_folder)
        btn_row.addWidget(self._btn_folder)

        btn_row.addStretch()
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    def set_results(self, results_df: pd.DataFrame, report_dir: str):
        """Обновляет виджет с новыми результатами."""
        self._results_df = results_df
        self._report_dir = report_dir
        self._populate_cards(results_df)
        self._populate_table(results_df)
        self._load_charts(report_dir)

    # ------------------------------------------------------------------
    def _populate_cards(self, df: pd.DataFrame):
        # Clear existing cards
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if df.empty:
            return

        cards = [
            ("Всего результатов", str(len(df))),
            ("Сред. combined", f"{df['combined_similarity'].mean():.3f}" if 'combined_similarity' in df.columns else "—"),
            ("Макс. combined", f"{df['combined_similarity'].max():.3f}" if 'combined_similarity' in df.columns else "—"),
        ]
        if "eeg_emotion" in df.columns:
            cards.append(("Эмоции EEG", str(df["eeg_emotion"].nunique())))
        if "classical_dataset" in df.columns:
            cards.append(("Датасеты", ", ".join(df["classical_dataset"].unique())))

        for title, value in cards:
            self._cards_layout.addWidget(MetricCard(title, value))
        self._cards_layout.addStretch()

    # ------------------------------------------------------------------
    def _populate_table(self, df: pd.DataFrame):
        cols = [c for c in [
            "participant_id", "trial_idx", "variant", "eeg_emotion",
            "classical_piece", "classical_dataset", "classical_emotion",
            "combined_similarity", "contour_similarity",
            "harmony_similarity", "interval_similarity",
        ] if c in df.columns]

        display = df.head(50)[cols]
        self._table.setRowCount(len(display))
        self._table.setColumnCount(len(cols))

        # Friendly column names
        header_names = {
            "participant_id": "Участник",
            "trial_idx": "Триал",
            "variant": "Вариант",
            "eeg_emotion": "Эмоция EEG",
            "classical_piece": "Произведение",
            "classical_dataset": "Датасет",
            "classical_emotion": "Эмоция класс.",
            "combined_similarity": "Combined",
            "contour_similarity": "Contour",
            "harmony_similarity": "Harmony",
            "interval_similarity": "Interval",
        }
        self._table.setHorizontalHeaderLabels(
            [header_names.get(c, c) for c in cols]
        )

        for i, (_, row) in enumerate(display.iterrows()):
            for j, col in enumerate(cols):
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val) if val is not None else ""
                    # Truncate long piece names
                    if col == "classical_piece" and len(text) > 50:
                        text = text[:50] + "…"
                item = QTableWidgetItem(text)
                if col == "combined_similarity":
                    item.setData(Qt.UserRole, float(val) if val else 0.0)
                self._table.setItem(i, j, item)

        self._table.resizeColumnsToContents()
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)

    # ------------------------------------------------------------------
    def _load_charts(self, report_dir: str):
        d = Path(report_dir)
        sim_path = d / "similarity_chart.png"
        if sim_path.exists():
            px = QPixmap(str(sim_path))
            self._chart_similarity.setPixmap(
                px.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self._chart_similarity.setText("График не найден")

        emo_path = d / "emotion_distribution.png"
        if emo_path.exists():
            px = QPixmap(str(emo_path))
            self._chart_emotion.setPixmap(
                px.scaled(700, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self._chart_emotion.setText("График не найден")

    # ------------------------------------------------------------------
    def _open_html(self):
        html = Path(self._report_dir) / "index.html"
        if html.exists():
            self.open_html_requested.emit(str(html))
        else:
            # Try to open report_dir
            import webbrowser
            webbrowser.open(str(Path(self._report_dir)))

    def _export_csv(self):
        if self._results_df is None or self._results_df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "results.csv", "CSV (*.csv)"
        )
        if path:
            self._results_df.to_csv(path, index=False)

    def _open_folder(self):
        if self._report_dir:
            import subprocess, sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", self._report_dir])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", self._report_dir])
            else:
                subprocess.Popen(["xdg-open", self._report_dir])
