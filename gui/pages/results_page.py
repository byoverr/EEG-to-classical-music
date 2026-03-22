"""
Богатая страница результатов, сфокусированная на гипотезе об эмоциях,
но сохраняющая наглядные карточки, графики и аудио.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QSizePolicy, QTabWidget, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QGridLayout, QTextBrowser,
)

from gui.styles import PRIMARY, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, BG_PAGE, BG_CARD, BORDER, DANGER


_active_audio_processes: list[subprocess.Popen] = []
_registered_audio_buttons: list[tuple[QPushButton, str]] = []
_current_audio_token: str | None = None


def _stop_all_audio():
    for proc in _active_audio_processes:
        try:
            proc.terminate()
        except Exception:
            pass
    _active_audio_processes.clear()


def _reset_audio_buttons():
    alive = []
    for button, default_text in _registered_audio_buttons:
        try:
            button.setText(default_text)
            alive.append((button, default_text))
        except RuntimeError:
            continue
    _registered_audio_buttons[:] = alive


def _register_audio_button(button: QPushButton, default_text: str):
    _registered_audio_buttons.append((button, default_text))


def _play_audio(path_str: str):
    path = Path(path_str)
    if not path.exists():
        wav_path = path.with_suffix(".wav")
        path = wav_path if wav_path.exists() else path
    if path.suffix.lower() == ".mid" and not path.with_suffix(".wav").exists():
        try:
            from src.audio_converter import midi_to_wav, find_soundfont
            sf = find_soundfont()
            wav_path = path.with_suffix(".wav")
            if sf and midi_to_wav(str(path), str(wav_path), sf):
                path = wav_path
        except Exception:
            pass
    elif path.suffix.lower() == ".mid" and path.with_suffix(".wav").exists():
        path = path.with_suffix(".wav")
    if not path.exists():
        return None

    system = platform.system()
    try:
        if system == "Darwin":
            if path.suffix.lower() == ".wav":
                proc = subprocess.Popen(["afplay", str(path)])
            else:
                proc = subprocess.Popen(["open", str(path)])
        elif system == "Linux":
            if path.suffix.lower() == ".wav":
                proc = subprocess.Popen(["aplay", str(path)])
            else:
                proc = subprocess.Popen(["xdg-open", str(path)])
        elif system == "Windows":
            proc = subprocess.Popen(["cmd", "/c", "start", "", str(path)], shell=False)
        else:
            return None
        _active_audio_processes.append(proc)
        return str(path)
    except Exception:
        return None


def _toggle_audio(path_str: str, button: QPushButton, default_text: str):
    global _current_audio_token
    if _current_audio_token == path_str and _active_audio_processes:
        _stop_all_audio()
        _reset_audio_buttons()
        _current_audio_token = None
        return

    _stop_all_audio()
    _reset_audio_buttons()
    token = _play_audio(path_str)
    if token:
        button.setText("Остановить")
        _current_audio_token = path_str


def _fmt_score(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "—"


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def _clean_piece_label(value) -> str:
    text = str(value or "Unknown").strip()
    if "|" in text:
        text = text.split("|")[-1]
    return text


def _pretty_table_html(
    df: pd.DataFrame,
    *,
    rename_map: dict[str, str] | None = None,
    formatters: dict[str, callable] | None = None,
    max_rows: int | None = None,
    empty_text: str = "Нет данных.",
) -> str:
    if df is None or df.empty:
        return f"<p class='muted'>{escape(empty_text)}</p>"

    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    if rename_map:
        table_df = table_df.rename(columns=rename_map)
    if formatters:
        for col, formatter in formatters.items():
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(formatter)

    for col in table_df.columns:
        table_df[col] = table_df[col].apply(lambda x: escape(str(x)))

    return table_df.to_html(index=False, classes="pretty-table", border=0, escape=False)


def _browser_shell(body_html: str) -> str:
    return f"""
    <html>
      <head>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: {TEXT_PRIMARY};
            font-size: 12px;
            line-height: 1.45;
            margin: 0;
            padding: 0;
          }}
          h3 {{
            margin: 0 0 8px 0;
            font-size: 15px;
            color: {PRIMARY};
          }}
          h4 {{
            margin: 16px 0 8px 0;
            font-size: 13px;
            color: {TEXT_PRIMARY};
          }}
          p {{
            margin: 0 0 10px 0;
          }}
          ul {{
            margin: 8px 0 0 16px;
          }}
          li {{
            margin: 4px 0;
          }}
          .lead {{
            background: #f8fbff;
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 14px;
          }}
          .muted {{
            color: {TEXT_SECONDARY};
          }}
          .pretty-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
          }}
          .pretty-table th {{
            text-align: left;
            background: #f4f7fb;
            color: {TEXT_PRIMARY};
            font-weight: 600;
            padding: 8px 10px;
            border-bottom: 1px solid {BORDER};
          }}
          .pretty-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #eceff3;
            vertical-align: top;
          }}
          .pretty-table tr:nth-child(even) td {{
            background: #fbfcfd;
          }}
          .section {{
            margin-bottom: 18px;
          }}
        </style>
      </head>
      <body>{body_html}</body>
    </html>
    """


class _MetricTile(QFrame):
    def __init__(self, label: str, value, parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        lay.addWidget(lbl)

        val_lbl = QLabel(_fmt_score(value) if isinstance(value, (int, float)) else str(value))
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet(f"font-size:15px; font-weight:700; color:{TEXT_PRIMARY};")
        lay.addWidget(val_lbl)


class _SummaryCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("summaryCard")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        hdr = QLabel(title.upper())
        hdr.setStyleSheet(f"font-size:10px; color:{TEXT_SECONDARY};")
        lay.addWidget(hdr)
        self._lay = lay

    def set_big_value(self, text: str, color: str = PRIMARY):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"font-size:22px; font-weight:700; color:{color};")
        self._lay.addWidget(lbl)

    def add_line(self, text: str):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY};")
        self._lay.addWidget(lbl)


class _PitchChart(QWidget):
    def __init__(self, eeg_pitches: list[int], cla_pitches: list[int], parent=None):
        super().__init__(parent)
        self.eeg = eeg_pitches or []
        self.cla = cla_pitches or []
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin = 12
        painter.fillRect(0, 0, w, h, QColor("#fafbfc"))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 8, 8)

        all_p = self.eeg + self.cla
        if not all_p:
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "Нет pitch-данных для фрагмента")
            painter.end()
            return

        lo, hi = min(all_p), max(all_p)
        if lo == hi:
            lo -= 6
            hi += 6
        rng = hi - lo

        def _draw_line(pitches: list[int], color: str, dashed: bool = False):
            if len(pitches) < 2:
                return
            pen = QPen(QColor(color), 2)
            if dashed:
                pen.setDashPattern([4, 4])
            painter.setPen(pen)
            n = len(pitches)
            for i in range(n - 1):
                x1 = margin + (w - 2 * margin) * i / (n - 1)
                x2 = margin + (w - 2 * margin) * (i + 1) / (n - 1)
                y1 = h - margin - (h - 2 * margin) * (pitches[i] - lo) / rng
                y2 = h - margin - (h - 2 * margin) * (pitches[i + 1] - lo) / rng
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        _draw_line(self.eeg, PRIMARY, False)
        _draw_line(self.cla, DANGER, True)

        painter.setPen(QColor(PRIMARY))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(margin, 14, "EEG")
        painter.setPen(QColor(DANGER))
        painter.drawText(margin + 36, 14, "Music")
        painter.end()


class _MatchCard(QFrame):
    def __init__(self, row: dict, rank: int, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.setSpacing(2)
        rank_lbl = QLabel(f"#{rank}")
        rank_lbl.setStyleSheet(f"font-size:12px; font-weight:700; color:{TEXT_SECONDARY};")
        left.addWidget(rank_lbl)

        composer = str(row.get("composer") or row.get("classical_composer") or "Unknown")
        title = str(row.get("title") or row.get("classical_title") or row.get("classical_piece") or "Unknown")
        comp_lbl = QLabel(composer)
        comp_lbl.setStyleSheet(f"font-size:16px; font-weight:700; color:{PRIMARY};")
        left.addWidget(comp_lbl)
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
        left.addWidget(title_lbl)
        top.addLayout(left, stretch=1)

        score_box = QFrame()
        score_box.setStyleSheet(f"background:#f8fbff; border:1px solid {BORDER}; border-radius:10px;")
        s_lay = QVBoxLayout(score_box)
        s_lay.setContentsMargins(12, 8, 12, 8)
        s_lay.setSpacing(1)
        main_lbl = QLabel("Main score")
        main_lbl.setAlignment(Qt.AlignCenter)
        main_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        s_lay.addWidget(main_lbl)
        score_lbl = QLabel(_fmt_score(row.get("music_match_score", row.get("combined_similarity", 0.0))))
        score_lbl.setAlignment(Qt.AlignCenter)
        score_lbl.setStyleSheet(f"font-size:24px; font-weight:700; color:{ACCENT};")
        s_lay.addWidget(score_lbl)
        sub_lbl = QLabel("Music Match")
        sub_lbl.setAlignment(Qt.AlignCenter)
        sub_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        s_lay.addWidget(sub_lbl)
        top.addWidget(score_box)
        root.addLayout(top)

        badges = QLabel(
            f"EEG: <b>{row.get('eeg_emotion', '—')}</b> | "
            f"Music: <b>{row.get('classical_emotion', '—')}</b> | "
            f"Agreement: <b>{row.get('match_label', '—')}</b>"
        )
        badges.setTextFormat(Qt.RichText)
        badges.setWordWrap(True)
        badges.setStyleSheet(f"font-size:12px; color:{TEXT_PRIMARY};")
        root.addWidget(badges)

        grid = QGridLayout()
        grid.setSpacing(6)
        metric_items = [
            ("Music Match", row.get("music_match_score", row.get("combined_similarity", 0.0))),
            ("Feature Similarity", row.get("feature_similarity_score", 0.0)),
            ("Emotion", row.get("emotion_agreement_score", 0.0)),
        ]
        for idx, (label, value) in enumerate(metric_items):
            grid.addWidget(_MetricTile(label, value), idx // 3, idx % 3)
        root.addLayout(grid)

        chart = _PitchChart(
            row.get("eeg_pitches") or [],
            row.get("cla_pitches") or [],
        )
        root.addWidget(chart)

        frag_lbl = QLabel(
            f"EEG fragment: {float(row.get('eeg_fragment_start_sec', 0.0)):.1f}-"
            f"{float(row.get('eeg_fragment_end_sec', 0.0)):.1f} s | "
            f"Music fragment: {float(row.get('music_fragment_start_sec', 0.0)):.1f}-"
            f"{float(row.get('music_fragment_end_sec', 0.0)):.1f} s"
        )
        frag_lbl.setWordWrap(True)
        frag_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
        root.addWidget(frag_lbl)

        diag_lbl = QLabel(
            f"EEG melody: {int(row.get('eeg_note_count_total', 0))} notes | "
            f"recording {float(row.get('eeg_recording_duration_sec', 0.0)) / 60.0:.1f} min | "
            f"analysis {float(row.get('eeg_melody_span_sec', 0.0)):.1f} s | "
            f"silence {float(row.get('eeg_silence_ratio', 1.0)) * 100:.1f}%"
        )
        diag_lbl.setWordWrap(True)
        diag_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
        root.addWidget(diag_lbl)

        btn_row = QHBoxLayout()
        eeg_default = "Слушать EEG-мелодию"
        eeg_btn = QPushButton(eeg_default)
        eeg_btn.setObjectName("secondary")
        _register_audio_button(eeg_btn, eeg_default)
        eeg_btn.clicked.connect(
            lambda: _toggle_audio(
                str(row.get("eeg_wav_path") or row.get("eeg_midi_path") or ""),
                eeg_btn,
                eeg_default,
            )
        )
        btn_row.addWidget(eeg_btn)

        music_default = "Слушать найденный фрагмент"
        music_btn = QPushButton(music_default)
        music_btn.setObjectName("secondary")
        _register_audio_button(music_btn, music_default)
        music_btn.clicked.connect(
            lambda: _toggle_audio(
                str(row.get("classical_wav_path") or row.get("classical_midi_path") or ""),
                music_btn,
                music_default,
            )
        )
        btn_row.addWidget(music_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)


def _image_card(title: str, image_path: Path, fallback_text: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(12, 10, 12, 10)
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY};")
    lay.addWidget(title_lbl)
    img_lbl = QLabel()
    img_lbl.setAlignment(Qt.AlignCenter)
    if image_path.exists():
        px = QPixmap(str(image_path))
        img_lbl.setPixmap(px.scaled(880, 620, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    else:
        img_lbl.setText(fallback_text)
        img_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY}; padding:24px;")
    lay.addWidget(img_lbl)
    return frame


def _fit_text_browser(browser: QTextBrowser, min_height: int = 180):
    browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    browser.document().adjustSize()
    height = int(browser.document().size().height()) + 28
    browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    browser.setMinimumHeight(max(min_height, height))


def _wrap_scroll(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


class ResultsPage(QWidget):
    go_home = Signal()
    new_analysis = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None
        self._report_dir: Optional[str] = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{BG_PAGE};")

        top_bar = QFrame()
        top_bar.setStyleSheet(f"background:{BG_CARD}; border-bottom:1px solid {BORDER};")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(20, 10, 20, 10)

        btn_home = QPushButton("Главная")
        btn_home.setObjectName("link")
        btn_home.clicked.connect(self.go_home.emit)
        top_lay.addWidget(btn_home)

        btn_new = QPushButton("Новый анализ")
        btn_new.setObjectName("secondary")
        btn_new.clicked.connect(self.new_analysis.emit)
        top_lay.addWidget(btn_new)

        top_lay.addStretch()

        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск по композитору / названию / эмоции")
        self._search.textChanged.connect(self._apply_filter)
        top_lay.addWidget(self._search)

        self._filter_emotion = QComboBox()
        self._filter_emotion.addItem("Эмоция: Все", "all")
        for emo in ("HVHA", "HVLA", "LVHA", "LVLA"):
            self._filter_emotion.addItem(emo, emo)
        self._filter_emotion.currentIndexChanged.connect(self._apply_filter)
        top_lay.addWidget(self._filter_emotion)

        btn_csv = QPushButton("CSV")
        btn_csv.setObjectName("secondary")
        btn_csv.clicked.connect(self._export_csv)
        top_lay.addWidget(btn_csv)

        btn_folder = QPushButton("Папка")
        btn_folder.setObjectName("secondary")
        btn_folder.clicked.connect(self._open_folder)
        top_lay.addWidget(btn_folder)

        root.addWidget(top_bar)

        self._content = QVBoxLayout()
        self._content.setContentsMargins(20, 14, 20, 14)
        self._content.setSpacing(12)
        root.addLayout(self._content, stretch=1)

    def load_results(self, df: pd.DataFrame, report_dir: str):
        self._df = df.copy()
        self._report_dir = report_dir
        self._populate()

    def load_from_history(self, entry: dict):
        report_dir = entry.get("report_dir", "")
        json_path = Path(report_dir) / "display_results.json"
        csv_path = Path(report_dir) / "comparison_results.csv"
        if json_path.exists():
            try:
                self.load_results(pd.read_json(json_path), report_dir)
                return
            except Exception:
                pass
        if csv_path.exists():
            self.load_results(pd.read_csv(csv_path), report_dir)
            return
        QMessageBox.warning(self, "Ошибка", "Файл результатов не найден.")

    def _load_metrics(self) -> dict:
        if not self._report_dir:
            return {}
        path = Path(self._report_dir) / "hypothesis_metrics.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _table_html(self, csv_name: str, empty_text: str) -> str:
        if not self._report_dir:
            return f"<p>{empty_text}</p>"
        path = Path(self._report_dir) / csv_name
        if not path.exists():
            return f"<p>{empty_text}</p>"
        try:
            df = pd.read_csv(path)
            return df.to_html(index=False, border=0)
        except Exception:
            return f"<p>{empty_text}</p>"

    def _populate(self):
        self._clear_layout(self._content)
        if self._df is None or self._df.empty:
            lbl = QLabel("Результаты отсутствуют.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"font-size:14px; color:{TEXT_SECONDARY}; padding:48px;")
            self._content.addWidget(lbl)
            return

        df = self._df.copy()
        metrics = self._load_metrics()

        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)

        c_total = _SummaryCard("Всего результатов")
        c_total.set_big_value(str(len(df)))
        summary_row.addWidget(c_total)

        c_match = _SummaryCard("Emotion Match")
        c_match.set_big_value(_fmt_pct(metrics.get("emotion_match_rate", 0.0)), ACCENT)
        c_match.add_line(f"Macro-F1: {_fmt_score(metrics.get('macro_f1', 0.0))}")
        summary_row.addWidget(c_match)

        c_music = _SummaryCard("Music Match")
        c_music.set_big_value(_fmt_score(metrics.get("best_music_match_score", df.get("music_match_score", df.get("combined_similarity", pd.Series([0.0]))).max())), PRIMARY)
        c_music.add_line(f"Mean music match: {_fmt_score(metrics.get('mean_music_match_score', df.get('music_match_score', df.get('combined_similarity', pd.Series([0.0]))).mean()))}")
        summary_row.addWidget(c_music)

        composer_rows = []
        cohort_path = Path(self._report_dir or "") / "cohort_emotion_summary.csv"
        if cohort_path.exists():
            try:
                cohort_df = pd.read_csv(cohort_path)
                if {"eeg_emotion", "top_composer"}.issubset(cohort_df.columns):
                    composer_rows = cohort_df.to_dict("records")
            except Exception:
                composer_rows = []

        if not composer_rows and {"eeg_emotion", "composer"}.issubset(df.columns):
            for emotion, group in df.groupby("eeg_emotion"):
                counts = group["composer"].fillna("Unknown").value_counts()
                if counts.empty:
                    continue
                composer_rows.append({
                    "eeg_emotion": emotion,
                    "top_composer": counts.index[0],
                    "n_people": int(counts.sum()),
                    "consistency": float(counts.iloc[0] / max(counts.sum(), 1)),
                })

        if composer_rows:
            emotion_order = {"HVHA": 0, "HVLA": 1, "LVHA": 2, "LVLA": 3}
            composer_rows = sorted(
                composer_rows,
                key=lambda item: emotion_order.get(str(item.get("eeg_emotion", "")), 99),
            )
            composers_card = _SummaryCard("Top Composers")
            for item in composer_rows:
                emotion = str(item.get("eeg_emotion", "Unknown"))
                composer = str(item.get("top_composer", "Unknown"))
                composers_card.add_line(f"{emotion}: {composer}")
            summary_row.addWidget(composers_card)

        summary_row.addStretch()
        self._content.addLayout(summary_row)

        self._tabs = QTabWidget()
        self._content.addWidget(self._tabs, stretch=1)
        self._populate_tabs(df)

    def _populate_tabs(self, df: pd.DataFrame):
        self._tabs.clear()
        self._tabs.addTab(self._build_matches_tab(df), "Best Matches")
        self._tabs.addTab(_wrap_scroll(self._build_emotion_tab(df)), "Emotion Analysis")
        self._tabs.addTab(_wrap_scroll(self._build_summary_tab(df)), "Summary / Insights")

    def _build_matches_tab(self, df: pd.DataFrame) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(12)

        if "music_match_score" in df.columns:
            ranked = df.sort_values(["music_match_score", "cemms_score"], ascending=[False, False])
        elif "combined_similarity" in df.columns:
            ranked = df.sort_values(["combined_similarity", "cemms_score"], ascending=[False, False]) \
                if "cemms_score" in df.columns else df.sort_values("combined_similarity", ascending=False)
        else:
            ranked = df
        for rank, row in enumerate(ranked.head(5).to_dict("records"), start=1):
            lay.addWidget(_MatchCard(row, rank))
        lay.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_emotion_tab(self, df: pd.DataFrame) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        metrics = self._load_metrics()
        if "music_match_score" in df.columns:
            ranked = df.sort_values(["music_match_score", "cemms_score"], ascending=[False, False]).reset_index(drop=True)
        elif "combined_similarity" in df.columns:
            ranked = df.sort_values("combined_similarity", ascending=False).reset_index(drop=True)
        else:
            ranked = df.reset_index(drop=True)
        best_row = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)

        summary = QFrame()
        summary.setObjectName("card")
        s_lay = QHBoxLayout(summary)
        s_lay.setContentsMargins(12, 10, 12, 10)
        tiles = [
            ("Emotion Match Rate", _fmt_pct(metrics.get("emotion_match_rate", 0.0))),
            ("Macro-F1", _fmt_score(metrics.get("macro_f1", 0.0))),
            ("Top-K Accuracy", _fmt_pct(metrics.get("top_k_accuracy", 0.0))),
            ("Group Consistency", _fmt_pct(metrics.get("group_consistency_mean", 0.0))),
        ]
        for title, value in tiles:
            tile = _MetricTile(title, value)
            s_lay.addWidget(tile)
        root.addWidget(summary)

        verdict = QFrame()
        verdict.setObjectName("card")
        verdict_lay = QVBoxLayout(verdict)
        verdict_lay.setContentsMargins(14, 12, 14, 12)
        verdict_lay.setSpacing(6)
        eeg_emotion = str(best_row.get("eeg_emotion", "—"))
        music_emotion = str(best_row.get("classical_emotion", "—"))
        composer = str(best_row.get("composer", "Unknown"))
        title = str(best_row.get("title", "Unknown"))
        is_match = eeg_emotion == music_emotion and eeg_emotion not in {"", "—"}
        result_title = QLabel("Ключевой эмоциональный результат")
        result_title.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY};")
        verdict_lay.addWidget(result_title)
        verdict_text = QLabel(
            f"Для текущего EEG лучшим музыкальным соответствием стала композиция "
            f"<b>{escape(composer)} — {escape(title)}</b>. "
            f"EEG определен как <b>{escape(eeg_emotion)}</b>, композиция размечена как "
            f"<b>{escape(music_emotion)}</b>."
        )
        verdict_text.setTextFormat(Qt.RichText)
        verdict_text.setWordWrap(True)
        verdict_text.setStyleSheet(f"font-size:12px; color:{TEXT_PRIMARY};")
        verdict_lay.addWidget(verdict_text)
        verdict_state = QLabel("Эмоциональное совпадение подтверждено." if is_match else "Эмоциональное совпадение не подтверждено.")
        verdict_state.setStyleSheet(
            f"font-size:12px; font-weight:600; color:{ACCENT if is_match else DANGER};"
        )
        verdict_lay.addWidget(verdict_state)
        root.addWidget(verdict)

        report_dir = Path(self._report_dir or "")
        root.addWidget(_image_card(
            "Confusion Matrix",
            report_dir / "confusion_matrix.png",
            "Confusion matrix пока не построена.",
        ))
        root.addWidget(_image_card(
            "Emotion Distribution",
            report_dir / "emotion_distribution.png",
            "График распределения эмоций не найден.",
        ))

        cohort_df = pd.DataFrame()
        cohort_path = report_dir / "cohort_emotion_summary.csv"
        if cohort_path.exists():
            try:
                cohort_df = pd.read_csv(cohort_path)
            except Exception:
                cohort_df = pd.DataFrame()

        if not cohort_df.empty and "top_work" in cohort_df.columns:
            cohort_df["top_work"] = cohort_df["top_work"].apply(_clean_piece_label)

        top_matches_df = ranked.copy()
        if "classical_piece" in top_matches_df.columns:
            top_matches_df["classical_piece"] = top_matches_df["classical_piece"].apply(_clean_piece_label)
        top_matches_df = top_matches_df[[
            c for c in [
                "variant", "eeg_emotion", "composer", "title", "classical_emotion",
                "music_match_score", "feature_similarity_score", "emotion_agreement_score",
            ] if c in top_matches_df.columns
        ]]

        cohort = QTextBrowser()
        cohort.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;")
        cohort.setOpenExternalLinks(False)
        cohort.setHtml(_browser_shell(
            f"""
            <div class="section">
              <div class="lead">
                <p><b>Что показывает вкладка:</b> здесь собраны результаты эмоциональной проверки для лучших музыкальных совпадений, а не технические детали пайплайна.</p>
              </div>
            </div>
            <div class="section">
              <h3>Сводка по эмоциям</h3>
              <p class="muted">Таблица показывает, какое произведение и какой композитор чаще всего оказываются лучшим соответствием для каждой эмоции.</p>
              {_pretty_table_html(
                  cohort_df,
                  rename_map={
                      "eeg_emotion": "EEG Emotion",
                      "top_composer": "Top Composer",
                      "top_work": "Top Work",
                      "consistency": "Consistency",
                      "n_people": "Participants",
                      "mean_music_match_score": "Mean Music Match",
                  },
                  formatters={
                      "Consistency": lambda v: _fmt_pct(v),
                      "Mean Music Match": lambda v: _fmt_score(v),
                  },
                  empty_text="Пока нет групповой сводки по эмоциям.",
              )}
            </div>
            <div class="section">
              <h3>Top Musical Matches</h3>
              <p class="muted">Первые строки показывают лучшие произведения по музыкальному сходству и их эмоциональную интерпретацию.</p>
              {_pretty_table_html(
                  top_matches_df,
                  rename_map={
                      "variant": "Variant",
                      "eeg_emotion": "EEG Emotion",
                      "composer": "Composer",
                      "title": "Work",
                      "classical_emotion": "Music Emotion",
                      "music_match_score": "Music Match",
                      "feature_similarity_score": "Feature Similarity",
                      "emotion_agreement_score": "Emotion",
                  },
                  formatters={
                      "Music Match": lambda v: _fmt_score(v),
                      "Feature Similarity": lambda v: _fmt_score(v),
                      "Emotion": lambda v: _fmt_score(v),
                  },
                  max_rows=5,
                  empty_text="Таблица top matches недоступна.",
              )}
            </div>
            """
        ))
        _fit_text_browser(cohort, min_height=220)
        root.addWidget(cohort)
        root.addStretch()
        return widget

    def _build_summary_tab(self, df: pd.DataFrame) -> QWidget:
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        report_dir = Path(self._report_dir or "")
        root.addWidget(_image_card(
            "Music Ranking and Validation",
            report_dir / "music_match_chart.png",
            "График score не найден.",
        ))
        root.addWidget(_image_card(
            "Full EEG Melody Timeline",
            report_dir / "best_eeg_melody_timeline.png",
            "Диагностика полной EEG-мелодии пока не построена.",
        ))

        metrics = self._load_metrics()
        if "music_match_score" in df.columns:
            ranked = df.sort_values(["music_match_score", "cemms_score"], ascending=[False, False]).reset_index(drop=True)
        elif "combined_similarity" in df.columns:
            ranked = df.sort_values("combined_similarity", ascending=False).reset_index(drop=True)
        else:
            ranked = df.reset_index(drop=True)
        best_row = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)

        melody_diag_df = pd.DataFrame()
        melody_diag_path = report_dir / "melody_diagnostics.csv"
        if melody_diag_path.exists():
            try:
                melody_diag_df = pd.read_csv(melody_diag_path)
            except Exception:
                melody_diag_df = pd.DataFrame()

        if not melody_diag_df.empty:
            melody_diag_df["recording_duration_sec"] = melody_diag_df["recording_duration_sec"].apply(
                lambda v: f"{float(v) / 60.0:.1f} min"
            )
            melody_diag_df["span_sec"] = melody_diag_df["span_sec"].apply(lambda v: f"{float(v):.1f} s")
            melody_diag_df["silence_ratio"] = melody_diag_df["silence_ratio"].apply(_fmt_pct)
            melody_diag_df["timeline_compression"] = melody_diag_df["timeline_compression"].apply(lambda v: f"x{float(v):.1f}")

        top_works_df = pd.DataFrame(metrics.get("top_works", []))
        if not top_works_df.empty and "work" in top_works_df.columns:
            top_works_df["work"] = top_works_df["work"].apply(_clean_piece_label)

        composer_df = pd.DataFrame()
        cohort_path = report_dir / "cohort_emotion_summary.csv"
        if cohort_path.exists():
            try:
                composer_df = pd.read_csv(cohort_path)
            except Exception:
                composer_df = pd.DataFrame()
        if not composer_df.empty and "top_work" in composer_df.columns:
            composer_df["top_work"] = composer_df["top_work"].apply(_clean_piece_label)

        browser = QTextBrowser()
        browser.setStyleSheet(f"background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;")
        browser.setHtml(_browser_shell(
            f"""
            <div class="section">
              <div class="lead">
                <p><b>Главный вывод:</b> лучшим музыкальным соответствием для текущего EEG стало произведение
                <b>{escape(str(best_row.get('composer', 'Unknown')))} — {escape(str(best_row.get('title', 'Unknown')))}</b>
                с итоговым score <b>{_fmt_score(best_row.get('music_match_score', best_row.get('combined_similarity', 0.0)))}</b>.</p>
                <p>Эмоция EEG: <b>{escape(str(best_row.get('eeg_emotion', '—')))}</b>,
                эмоция найденной композиции: <b>{escape(str(best_row.get('classical_emotion', '—')))}</b>.</p>
              </div>
            </div>
            <div class="section">
              <h3>Ключевые наблюдения</h3>
              <ul>
                <li>Обработано результатов: <b>{int(metrics.get('n_results', len(df)))}</b>.</li>
                <li>Лучший musical match: <b>{_fmt_score(metrics.get('best_music_match_score', 0.0))}</b>.</li>
                <li>Emotion Match Rate: <b>{_fmt_pct(metrics.get('emotion_match_rate', 0.0))}</b>.</li>
              </ul>
            </div>
            <div class="section">
              <h3>Top Composers by Emotion</h3>
              {_pretty_table_html(
                  composer_df[[c for c in ["eeg_emotion", "top_composer", "top_work", "consistency"] if c in composer_df.columns]],
                  rename_map={
                      "eeg_emotion": "EEG Emotion",
                      "top_composer": "Top Composer",
                      "top_work": "Representative Work",
                      "consistency": "Consistency",
                  },
                  formatters={"Consistency": lambda v: _fmt_pct(v)},
                  empty_text="По эмоциям пока нет сводки по композиторам.",
              )}
            </div>
            <div class="section">
              <h3>Signal Coverage</h3>
              <p class="muted">Таблица показывает, как длинная EEG запись преобразована в анализируемую музыкальную линию.</p>
              {_pretty_table_html(
                  melody_diag_df[[c for c in ["variant", "note_count", "recording_duration_sec", "span_sec", "silence_ratio", "timeline_compression"] if c in melody_diag_df.columns]],
                  rename_map={
                      "variant": "Variant",
                      "note_count": "Notes",
                      "recording_duration_sec": "Recording Duration",
                      "span_sec": "Analysis Melody",
                      "silence_ratio": "Silence",
                      "timeline_compression": "Compression",
                  },
                  empty_text="Диагностика EEG-мелодии недоступна.",
              )}
            </div>
            <div class="section">
              <h3>Most Selected Works</h3>
              {_pretty_table_html(
                  top_works_df,
                  rename_map={"work": "Work", "count": "Count"},
                  empty_text="Топ произведений пока не сформирован.",
              )}
            </div>
            """
        ))
        _fit_text_browser(browser, min_height=320)
        root.addWidget(browser)
        root.addStretch()
        return widget

    def _apply_filter(self):
        if self._df is None:
            return
        df = self._df.copy()
        query = self._search.text().strip().lower()
        if query:
            def _matches(row):
                text = " ".join([
                    str(row.get("composer", "")),
                    str(row.get("title", "")),
                    str(row.get("classical_piece", "")),
                    str(row.get("eeg_emotion", "")),
                    str(row.get("classical_emotion", "")),
                ]).lower()
                return query in text
            df = df[df.apply(_matches, axis=1)]

        emotion = self._filter_emotion.currentData()
        if emotion and emotion != "all" and "eeg_emotion" in df.columns:
            df = df[df["eeg_emotion"] == emotion]
        self._populate_tabs(df)

    def _export_csv(self):
        if self._df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт CSV", "hypothesis_results.csv", "CSV (*.csv)")
        if path:
            self._df.to_csv(path, index=False)
            QMessageBox.information(self, "Экспорт", f"Сохранено: {path}")

    def _open_folder(self):
        if not self._report_dir:
            return
        if os.name == "nt":
            os.startfile(self._report_dir)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", self._report_dir])
        else:
            subprocess.Popen(["xdg-open", self._report_dir])

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                ResultsPage._clear_layout(child_layout)
