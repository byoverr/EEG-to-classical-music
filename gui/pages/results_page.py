"""
Страница с результатами сравнения — богатый card-based интерфейс,
аналогичный HTML-отчёту (и лучше).

Содержит:
- Summary cards (статистика, датасеты, эмоции)
- Фильтрация + поиск
- Вкладки сортировки (Combined, Contour, SFI, Harmony, EMOPIA, MAESTRO)
- Match cards: header + метрики + графики pitch + audio + metadata
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal, QSize, QProcess
from PySide6.QtGui import QFont, QPixmap, QColor, QPainter, QPen, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QTabWidget, QGridLayout, QLineEdit, QComboBox,
    QGroupBox, QFileDialog, QMessageBox, QSpacerItem,
)

from gui.styles import (
    PRIMARY, PRIMARY_DARK, ACCENT, DANGER, WARNING,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_PAGE, BG_CARD, BORDER,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _similarity_color(val: float) -> str:
    """Возвращает CSS-цвет для значения сходства 0…1."""
    if val >= 0.7:
        return "#1e8449"
    elif val >= 0.4:
        return "#b7950b"
    elif val >= 0.2:
        return "#ca6f1e"
    return "#922b21"


def _format_sim(val: float) -> str:
    if val is None or np.isnan(val):
        return "—"
    return f"{val:.3f}"


# ── Audio playback helper ──────────────────────────────────────────────────

# Track active audio processes so we can stop them
_active_audio_processes: list[subprocess.Popen] = []


def _stop_all_audio():
    """Stop any currently playing audio."""
    for proc in _active_audio_processes:
        try:
            proc.terminate()
        except Exception:
            pass
    _active_audio_processes.clear()


def _play_wav(wav_or_midi_path: str):
    """Play a WAV (or MIDI) file using the system player.

    Tries the .wav version first; falls back to original file.
    Uses FluidSynth for MIDI→WAV conversion if WAV is absent.
    """
    _stop_all_audio()
    path = Path(wav_or_midi_path)

    # prefer WAV
    wav_path = path.with_suffix(".wav")
    if not wav_path.exists() and path.suffix.lower() == ".mid":
        # Try converting on the fly
        try:
            from src.audio_converter import midi_to_wav, find_soundfont
            sf = find_soundfont()
            if sf:
                midi_to_wav(str(path), str(wav_path), sf)
        except Exception:
            pass

    play_file = str(wav_path) if wav_path.exists() else str(path)
    if not Path(play_file).exists():
        return

    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.Popen(["afplay", play_file])
        elif system == "Linux":
            proc = subprocess.Popen(["aplay", play_file])
        elif system == "Windows":
            proc = subprocess.Popen(
                ["cmd", "/c", "start", "", play_file], shell=False,
            )
        else:
            return
        _active_audio_processes.append(proc)
    except Exception:
        pass


def _parse_pitch_list(val) -> list:
    """Parse pitch data from a DataFrame cell (may be list, str, or ndarray)."""
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
    return []


# ── MetricTile ──────────────────────────────────────────────────────────────

class _MetricTile(QFrame):
    """Компактная плитка с одной числовой метрикой."""

    def __init__(self, label: str, value: float, accent_color: str = PRIMARY,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(44)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(1)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)

        color = _similarity_color(value) if value is not None else TEXT_SECONDARY
        val_text = _format_sim(value)
        v_lbl = QLabel(val_text)
        v_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{color};")
        v_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(v_lbl)

    def set_border_color(self, color: str):
        self.setStyleSheet(
            f"QFrame#metricCard {{"
            f"  background: {BG_CARD};"
            f"  border: 1px solid {BORDER};"
            f"  border-left: 3px solid {color};"
            f"  border-radius: 6px;"
            f"  padding: 4px 6px;"
            f"}}"
        )


# ── PitchChart (matplotlib-free, pure QPainter) ────────────────────────────

class _PitchChart(QWidget):
    """Мини-график сравнения последовательностей питчей (EEG vs Classical)."""

    def __init__(self, eeg_pitches: list, cla_pitches: list, parent=None):
        super().__init__(parent)
        self.eeg = eeg_pitches or []
        self.cla = cla_pitches or []
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin = 10

        # white background
        painter.fillRect(0, 0, w, h, QColor("#fafbfc"))
        # thin border
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)

        if not self.eeg and not self.cla:
            # Placeholder text
            painter.setPen(QColor(TEXT_SECONDARY))
            fnt = QFont()
            fnt.setPixelSize(11)
            painter.setFont(fnt)
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "Нет данных питчей")
            painter.end()
            return

        # data bounds
        all_p = self.eeg + self.cla
        lo, hi = min(all_p), max(all_p)
        if lo == hi:
            lo -= 6
            hi += 6
        rng = hi - lo

        def _draw_line(pitches: list, color: QColor, dash: bool = False):
            if len(pitches) < 2:
                return
            pen = QPen(color, 2)
            if dash:
                pen.setDashPattern([4, 4])
            painter.setPen(pen)
            n = len(pitches)
            for i in range(n - 1):
                x1 = margin + (w - 2 * margin) * i / (n - 1)
                y1 = h - margin - (h - 2 * margin) * (pitches[i] - lo) / rng
                x2 = margin + (w - 2 * margin) * (i + 1) / (n - 1)
                y2 = h - margin - (h - 2 * margin) * (pitches[i + 1] - lo) / rng
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        _draw_line(self.eeg, QColor(PRIMARY), dash=False)
        _draw_line(self.cla, QColor(DANGER), dash=True)

        # legend
        painter.setPen(QColor(TEXT_SECONDARY))
        fnt = QFont()
        fnt.setPixelSize(9)
        painter.setFont(fnt)
        painter.setPen(QColor(PRIMARY))
        painter.drawText(margin, 10, "EEG")
        painter.setPen(QColor(DANGER))
        painter.drawText(margin + 35, 10, "Classical")

        painter.end()


# ── MatchCard ───────────────────────────────────────────────────────────────

class _MatchCard(QFrame):
    """Одна карточка совпадения — composer, metrics, chart, badges."""

    def __init__(self, row: dict, rank: int, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.row = row

        root = QVBoxLayout(self)
        root.setSpacing(5)
        root.setContentsMargins(10, 8, 10, 8)

        # ── header ──
        hdr = QHBoxLayout()
        rank_lbl = QLabel(f"#{rank}")
        rank_lbl.setStyleSheet(
            f"font-size:13px; font-weight:700; color:white; "
            f"background:{PRIMARY}; border-radius:12px; "
            f"min-width:26px; min-height:26px; max-width:26px; max-height:26px;"
        )
        rank_lbl.setAlignment(Qt.AlignCenter)
        hdr.addWidget(rank_lbl)

        left_col = QVBoxLayout()
        left_col.setSpacing(0)
        composer = row.get("composer") or row.get("classical_composer", "—")
        title = row.get("title") or row.get("classical_title") or row.get("classical_piece", "—")
        if len(str(title)) > 55:
            title = str(title)[:52] + "…"
        comp_lbl = QLabel(str(composer))
        comp_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY};")
        left_col.addWidget(comp_lbl)
        title_lbl = QLabel(str(title))
        title_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        title_lbl.setWordWrap(True)
        left_col.addWidget(title_lbl)
        hdr.addLayout(left_col, stretch=1)

        # right meta
        right_col = QVBoxLayout()
        right_col.setSpacing(0)
        right_col.setAlignment(Qt.AlignRight | Qt.AlignTop)

        trial = row.get("trial", "")
        variant = row.get("variant") or row.get("processing", "")
        meta_str = f"{trial} · {variant}" if trial and variant else str(trial or variant or "")
        if meta_str.strip():
            meta_lbl = QLabel(meta_str)
            meta_lbl.setStyleSheet(f"font-size:10px; color:{TEXT_SECONDARY};")
            meta_lbl.setAlignment(Qt.AlignRight)
            right_col.addWidget(meta_lbl)

        # badges
        badges_row = QHBoxLayout()
        badges_row.setSpacing(4)
        badges_row.addStretch()

        dataset = str(row.get("classical_dataset", "")).upper()
        if dataset:
            b = self._badge(dataset, "#f0f0f0", TEXT_PRIMARY)
            badges_row.addWidget(b)

        eeg_emo = row.get("eeg_emotion", "")
        if eeg_emo and eeg_emo != "unknown":
            b = self._badge(f"EEG {eeg_emo}", "#f0f0f0", TEXT_PRIMARY)
            badges_row.addWidget(b)

        cla_emo = row.get("classical_emotion", "")
        if cla_emo and str(cla_emo) not in ("None", "nan", ""):
            b = self._badge(str(cla_emo), "#f0f0f0", TEXT_PRIMARY)
            badges_row.addWidget(b)

        emo_match = row.get("emotion_match")
        if emo_match is True:
            b = self._badge("Match", "#f0f0f0", TEXT_PRIMARY)
            badges_row.addWidget(b)
        elif emo_match is False:
            b = self._badge("Mismatch", "#f0f0f0", TEXT_PRIMARY)
            badges_row.addWidget(b)

        right_col.addLayout(badges_row)

        # VA badges
        eeg_v = row.get("eeg_valence") or row.get("valence")
        eeg_a = row.get("eeg_arousal") or row.get("arousal")
        if eeg_v is not None or eeg_a is not None:
            va_row = QHBoxLayout()
            va_row.addStretch()
            if eeg_v is not None:
                try:
                    va_row.addWidget(self._badge(f"V {float(eeg_v):.1f}", "#f0f0f0", TEXT_PRIMARY))
                except (ValueError, TypeError):
                    pass
            if eeg_a is not None:
                try:
                    va_row.addWidget(self._badge(f"A {float(eeg_a):.1f}", "#f0f0f0", TEXT_PRIMARY))
                except (ValueError, TypeError):
                    pass
            right_col.addLayout(va_row)

        hdr.addLayout(right_col)
        root.addLayout(hdr)

        # ── metrics grid (3 columns) ──
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(4)
        metrics_grid.setContentsMargins(0, 4, 0, 4)

        metric_defs = [
            ("Combined", "combined_similarity", ACCENT),
            ("Contour", "contour_similarity", PRIMARY),
            ("Interval", "interval_similarity", "#8e44ad"),
            ("Harmony", "harmony_similarity", "#2980b9"),
            ("SFI", "sfi_similarity", "#e67e22"),
            ("Dynamic", "stat_similarity", "#16a085"),
        ]
        tile_idx = 0
        for m_label, m_key, m_color in metric_defs:
            val = row.get(m_key)
            if val is not None:
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                if fv > 0:
                    tile = _MetricTile(m_label, fv)
                    tile.set_border_color(m_color)
                    metrics_grid.addWidget(tile, tile_idx // 3, tile_idx % 3)
                    tile_idx += 1
        root.addLayout(metrics_grid)

        # ── participant metadata (collapsible) ──
        pid = row.get("participant_id", "")
        stim_artist = row.get("stimulus_artist", "")
        stim_title = row.get("stimulus_title", "")
        if pid or stim_artist or stim_title:
            meta_frame = QFrame()
            meta_frame.setStyleSheet(
                f"background:#fafbfc; border:1px solid {BORDER}; border-radius:6px; padding:6px;"
            )
            ml = QGridLayout(meta_frame)
            ml.setSpacing(4)
            r = 0
            if pid:
                age = row.get("participant_age", "")
                gender = row.get("participant_gender", "")
                pid_text = str(pid).upper()
                if age:
                    pid_text += f", {age}y"
                if gender:
                    pid_text += f" ({gender})"
                ml.addWidget(QLabel("Участник:"), r, 0)
                ml.addWidget(QLabel(pid_text), r, 1)
                r += 1
            if stim_artist:
                ml.addWidget(QLabel("Стимул:"), r, 0)
                ml.addWidget(QLabel(str(stim_artist)), r, 1)
                r += 1
            if stim_title:
                ml.addWidget(QLabel("Трек:"), r, 0)
                ml.addWidget(QLabel(str(stim_title)), r, 1)
                r += 1
            root.addWidget(meta_frame)

    @staticmethod
    def _badge(text: str, bg: str, fg: str) -> QLabel:
        b = QLabel(text)
        b.setStyleSheet(
            f"background:{bg}; color:{fg}; font-size:9px; font-weight:700; "
            f"padding:2px 6px; border-radius:6px;"
        )
        b.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        return b


# ── SummaryCard ─────────────────────────────────────────────────────────────

class _SummaryCard(QFrame):
    """Карточка из summary-ряда (Total Matches, Emotions, Datasets…)."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("summaryCard")
        self.setMinimumWidth(160)
        self.setMaximumWidth(260)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 10, 12, 10)
        self._lay.setSpacing(4)

        hdr = QLabel(title.upper())
        hdr.setStyleSheet(
            f"font-size:10px; color:{TEXT_SECONDARY}; "
            f"text-transform:uppercase; letter-spacing:0.5px;"
        )
        self._lay.addWidget(hdr)

    def set_big_value(self, text: str):
        v = QLabel(str(text))
        v.setStyleSheet(f"font-size:22px; font-weight:700; color:{PRIMARY};")
        self._lay.addWidget(v)

    def add_line(self, text: str, color: str = TEXT_PRIMARY):
        l = QLabel(text)
        l.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY};")
        l.setWordWrap(True)
        self._lay.addWidget(l)


# ── карточки для скролл-секции ──────────────────────────────────────────────

def _build_cards_widget(rows: list[dict]) -> QWidget:
    """Строит виджет с карточками для одной вкладки (двухколоночная сетка)."""
    container = QWidget()
    container.setStyleSheet(f"background-color: {BG_CARD};")
    grid = QGridLayout(container)
    grid.setSpacing(14)
    grid.setContentsMargins(4, 4, 4, 4)

    for i, row_data in enumerate(rows):
        card = _MatchCard(row_data, rank=i + 1)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row_idx = i // 2
        col_idx = i % 2
        grid.addWidget(card, row_idx, col_idx)

    # Add stretch at the bottom
    grid.setRowStretch(len(rows) // 2 + 1, 1)
    return container


# ═══════════════════════════════════════════════════════════════════════════
# ResultsPage — основной виджет
# ═══════════════════════════════════════════════════════════════════════════

class ResultsPage(QWidget):
    """Страница результатов.

    Signals:
        go_home()       – вернуться на WelcomePage
        new_analysis()  – начать новый анализ
    """

    go_home = Signal()
    new_analysis = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._report_dir: Optional[str] = None
        self._build_skeleton()

    # ------------------------------------------------------------------
    def _build_skeleton(self):
        """Создаёт общий каркас (верхняя панель + контейнер контента)."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background-color: {BG_PAGE};")

        # ── top bar ──
        top_bar = QFrame()
        top_bar.setStyleSheet(
            f"QFrame {{ background-color: {BG_CARD}; border-bottom: 1px solid {BORDER}; }}"
        )
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(20, 10, 20, 10)
        top_lay.setSpacing(8)

        btn_home = QPushButton("Главная")
        btn_home.setObjectName("link")
        btn_home.clicked.connect(self.go_home.emit)
        top_lay.addWidget(btn_home)

        btn_new = QPushButton("Новый анализ")
        btn_new.setObjectName("secondary")
        btn_new.clicked.connect(self.new_analysis.emit)
        top_lay.addWidget(btn_new)

        top_lay.addStretch()

        # search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск по композитору / названию…")
        self._search.setMinimumWidth(220)
        self._search.textChanged.connect(self._apply_filter)
        top_lay.addWidget(self._search)

        # filters
        self._filter_emotion = QComboBox()
        self._filter_emotion.setObjectName("filterCombo")
        self._filter_emotion.addItem("Эмоция: Все", "all")
        for e in ("HVHA", "HVLA", "LVHA", "LVLA"):
            self._filter_emotion.addItem(f"EEG: {e}", e)
        self._filter_emotion.currentIndexChanged.connect(self._apply_filter)
        top_lay.addWidget(self._filter_emotion)

        self._filter_dataset = QComboBox()
        self._filter_dataset.setObjectName("filterCombo")
        self._filter_dataset.addItem("Датасет: Все", "all")
        self._filter_dataset.addItem("MAESTRO", "maestro")
        self._filter_dataset.addItem("EMOPIA", "emopia")
        self._filter_dataset.currentIndexChanged.connect(self._apply_filter)
        top_lay.addWidget(self._filter_dataset)

        top_lay.addSpacerItem(QSpacerItem(8, 0))

        btn_csv = QPushButton("CSV")
        btn_csv.setObjectName("secondary")
        btn_csv.setToolTip("Экспортировать результаты в CSV")
        btn_csv.clicked.connect(self._export_csv)
        top_lay.addWidget(btn_csv)

        btn_html = QPushButton("HTML")
        btn_html.setObjectName("secondary")
        btn_html.setToolTip("Открыть HTML-отчёт в браузере")
        btn_html.clicked.connect(self._open_html)
        top_lay.addWidget(btn_html)

        btn_folder = QPushButton("Папка")
        btn_folder.setObjectName("secondary")
        btn_folder.setToolTip("Открыть папку отчёта")
        btn_folder.clicked.connect(self._open_folder)
        top_lay.addWidget(btn_folder)

        root.addWidget(top_bar)

        # ── content area ──
        self._content_area = QVBoxLayout()
        self._content_area.setContentsMargins(20, 14, 20, 14)
        self._content_area.setSpacing(12)
        root.addLayout(self._content_area, stretch=1)

        # placeholder
        self._placeholder = QLabel("Загрузите или выберите результаты для просмотра.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:14px; padding:60px;")
        self._content_area.addWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_results(self, df: pd.DataFrame, report_dir: str):
        """Заполняет страницу данными из завершённого пайплайна."""
        self._df = df.copy()
        self._report_dir = report_dir
        self._populate()

    def load_from_history(self, entry: dict):
        """Загружает результаты из записи истории."""
        report_dir = entry.get("report_dir", "")
        # Предпочитаем display_results.json (содержит pitches и MIDI пути)
        json_path = Path(report_dir) / "display_results.json"
        csv_path = Path(report_dir) / "comparison_results.csv"
        if json_path.exists():
            try:
                df = pd.read_json(json_path)
                self.load_results(df, report_dir)
                return
            except Exception:
                pass
        if not csv_path.exists():
            QMessageBox.warning(self, "Ошибка", f"CSV не найден:\n{csv_path}")
            return
        df = pd.read_csv(csv_path)
        self.load_results(df, report_dir)

    # ------------------------------------------------------------------
    # Build the rich UI from dataframe
    # ------------------------------------------------------------------

    def _populate(self):
        """Строит UI из self._df."""
        # clear existing content
        self._clear_content()

        df = self._df
        if df is None or df.empty:
            self._content_area.addWidget(self._placeholder)
            return

        # ── Summary row ──
        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)

        # Total
        c_total = _SummaryCard("Всего совпадений")
        c_total.set_big_value(str(len(df)))
        summary_row.addWidget(c_total)

        # Mean similarity
        if "combined_similarity" in df.columns:
            avg = df["combined_similarity"].mean()
            mx = df["combined_similarity"].max()
            c_sim = _SummaryCard("Сходство")
            c_sim.set_big_value(f"{avg:.3f}")
            c_sim.add_line(f"Макс: {mx:.3f}", ACCENT)
            summary_row.addWidget(c_sim)

        # Datasets
        if "classical_dataset" in df.columns:
            ds_counts = df["classical_dataset"].fillna("unknown").value_counts()
            c_ds = _SummaryCard("Датасеты")
            for ds_name, cnt in ds_counts.items():
                c_ds.add_line(f"{str(ds_name).upper()}: {cnt}")
            summary_row.addWidget(c_ds)

        # EEG emotions
        if "eeg_emotion" in df.columns:
            emo_counts = df["eeg_emotion"].fillna("unknown").value_counts()
            c_emo = _SummaryCard("EEG эмоции")
            for emo, cnt in emo_counts.head(6).items():
                c_emo.add_line(f"{emo}: {cnt}")
            summary_row.addWidget(c_emo)

        # Top composers
        if "classical_composer" in df.columns or "composer" in df.columns:
            comp_col = "classical_composer" if "classical_composer" in df.columns else "composer"
            top3 = df[comp_col].fillna("Unknown").value_counts().head(3)
            c_comp = _SummaryCard("Топ-композиторы")
            for comp_name, cnt in top3.items():
                c_comp.add_line(f"{comp_name}: {cnt}")
            summary_row.addWidget(c_comp)

        summary_row.addStretch()
        self._content_area.addLayout(summary_row)

        # ── Tabs ──
        self._tabs = QTabWidget()
        self._tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self._tabs.setUsesScrollButtons(True)
        # НЕ используем setDocumentMode — он включает нативный рендер,
        # который на macOS dark mode даёт тёмный фон за вкладками.
        self._tabs.setStyleSheet(
            f"QTabWidget {{ background-color: {BG_CARD}; }}"
            f"QTabWidget::pane {{ background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}"
            f"QTabBar {{ background-color: {BG_CARD}; }}"
        )
        pal = self._tabs.palette()
        pal.setColor(pal.ColorRole.Window, QColor(BG_CARD))
        pal.setColor(pal.ColorRole.Base, QColor(BG_CARD))
        self._tabs.setPalette(pal)
        self._tabs.setAutoFillBackground(True)
        self._populate_tabs(df)
        self._content_area.addWidget(self._tabs, stretch=1)

    # ------------------------------------------------------------------

    def _populate_tabs(self, df: pd.DataFrame):
        """Создаёт вкладки с разной сортировкой."""
        sections: list[tuple[str, pd.DataFrame]] = []

        n = len(df)
        if "combined_similarity" in df.columns:
            sections.append(("Combined", df.nlargest(n, "combined_similarity")))
        if "contour_similarity" in df.columns:
            sections.append(("Contour", df.nlargest(n, "contour_similarity")))
        if "sfi_similarity" in df.columns and (df["sfi_similarity"] > 0).any():
            sections.append(("SFI", df.nlargest(n, "sfi_similarity")))
        if "harmony_similarity" in df.columns and (df["harmony_similarity"] > 0).any():
            sections.append(("Harmony", df.nlargest(n, "harmony_similarity")))
        if "interval_similarity" in df.columns and (df["interval_similarity"] > 0).any():
            sections.append(("Interval", df.nlargest(n, "interval_similarity")))

        # Dataset subsections
        if "classical_dataset" in df.columns:
            emopia = df[df["classical_dataset"] == "emopia"]
            maestro = df[df["classical_dataset"] == "maestro"]
            if len(emopia) > 0 and "combined_similarity" in df.columns:
                sections.append(("EMOPIA", emopia.nlargest(len(emopia), "combined_similarity")))
            if len(maestro) > 0 and "combined_similarity" in df.columns:
                sections.append(("MAESTRO", maestro.nlargest(len(maestro), "combined_similarity")))

        if not sections:
            # fallback — just show all rows
            sections.append(("All", df))

        for label, section_df in sections:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setStyleSheet(
                f"QScrollArea {{ background-color: {BG_CARD}; border: none; }}"
                f"QScrollArea > QWidget > QWidget {{ background-color: {BG_CARD}; }}"
            )
            scroll.setAutoFillBackground(True)
            pal_s = scroll.palette()
            pal_s.setColor(pal_s.ColorRole.Window, QColor(BG_CARD))
            scroll.setPalette(pal_s)
            rows = section_df.to_dict("records")
            widget = _build_cards_widget(rows)
            scroll.setWidget(widget)
            self._tabs.addTab(scroll, label)

        # store current tab data for filter
        self._tab_sections = sections

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filter(self):
        """Фильтрация по поиску + эмоции + датасету."""
        if self._df is None:
            return
        df = self._df.copy()

        # Search
        query = self._search.text().strip().lower()
        if query:
            def _matches(row):
                searchable = " ".join([
                    str(row.get("composer", "")),
                    str(row.get("classical_composer", "")),
                    str(row.get("title", "")),
                    str(row.get("classical_title", "")),
                    str(row.get("classical_piece", "")),
                ]).lower()
                return query in searchable
            mask = df.apply(_matches, axis=1)
            df = df[mask]

        # Emotion filter
        emo_val = self._filter_emotion.currentData()
        if emo_val and emo_val != "all" and "eeg_emotion" in df.columns:
            df = df[df["eeg_emotion"] == emo_val]

        # Dataset filter
        ds_val = self._filter_dataset.currentData()
        if ds_val and ds_val != "all" and "classical_dataset" in df.columns:
            df = df[df["classical_dataset"] == ds_val]

        # Rebuild tabs
        if hasattr(self, "_tabs"):
            # Save current tab index
            cur_idx = self._tabs.currentIndex()
            self._tabs.clear()
            self._populate_tabs(df)
            if cur_idx < self._tabs.count():
                self._tabs.setCurrentIndex(cur_idx)

    # ------------------------------------------------------------------
    # Export / Navigation
    # ------------------------------------------------------------------

    def _export_csv(self):
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV", "comparison_results.csv", "CSV (*.csv)"
        )
        if path:
            self._df.to_csv(path, index=False)
            QMessageBox.information(self, "Экспорт", f"Сохранено: {path}")

    def _open_html(self):
        if self._report_dir:
            html_path = Path(self._report_dir) / "index.html"
            if html_path.exists():
                webbrowser.open(html_path.as_uri())
            else:
                QMessageBox.warning(self, "HTML", "HTML-отчёт не найден.")

    def _open_folder(self):
        if self._report_dir and Path(self._report_dir).exists():
            if os.name == "nt":
                os.startfile(self._report_dir)
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", self._report_dir])
            else:
                subprocess.Popen(["xdg-open", self._report_dir])

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _clear_content(self):
        """Удаляет все виджеты из content_area."""
        while self._content_area.count():
            item = self._content_area.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                ResultsPage._clear_layout(item.layout())
