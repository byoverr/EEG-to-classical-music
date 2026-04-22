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

from gui.styles import (
    PRIMARY, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, BG_PAGE, BG_CARD,
    BORDER, BORDER_SOFT, DANGER, CARD_HIGHLIGHT_BG, TABLE_HEADER_BG,
    ROW_ALT_BG, MATCH_OK, MATCH_WARN, MATCH_BAD, MATCH_NONE, TEXT_MUTED,
)


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
            background: {CARD_HIGHLIGHT_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 14px;
          }}
          .muted {{
            color: {TEXT_MUTED};
          }}
          .pretty-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
          }}
          .pretty-table th {{
            text-align: left;
            background: {TABLE_HEADER_BG};
            color: {TEXT_PRIMARY};
            font-weight: 600;
            padding: 8px 10px;
            border-bottom: 1px solid {BORDER};
          }}
          .pretty-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid {BORDER_SOFT};
            vertical-align: top;
          }}
          .pretty-table tr:nth-child(even) td {{
            background: {ROW_ALT_BG};
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


# ──────────────────────────────────────────────────────────────────────────
# Виджеты для вкладки «Преобразование» (Signal → Music)
# ──────────────────────────────────────────────────────────────────────────

def _load_signal_snapshot(report_dir: Path) -> Optional[dict]:
    """Находит snapshot с реально детектированными мотивами.

    Среди всех снимков в report_dir/signal_snapshots/ предпочитаем тот,
    где мотивов максимум — чтобы вкладка «Преобразование» не показывала
    «пустую» волну без единой сработки порога. При равном числе мотивов
    предпочитаем pca → original → smoothed → прочие.
    """
    if not report_dir:
        return None
    snap_dir = Path(report_dir) / "signal_snapshots"
    if not snap_dir.exists():
        return None
    candidates = list(snap_dir.glob("*.json"))
    if not candidates:
        return None

    variant_priority = {"pca": 0, "original": 1, "smoothed": 2}
    snapshots: list[tuple[int, int, str, dict]] = []
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        motifs = data.get("motifs") or []
        # Вариант определяем по суффиксу имени файла
        stem = p.stem
        v_rank = 99
        for v, pr in variant_priority.items():
            if stem.endswith(f"_{v}"):
                v_rank = pr
                break
        # Больше мотивов — лучше; при равенстве — меньше v_rank
        snapshots.append((-len(motifs), v_rank, stem, data))

    if not snapshots:
        return None
    snapshots.sort(key=lambda t: (t[0], t[1], t[2]))
    # Если есть хоть один snapshot с >0 мотивов — гарантированно берём его,
    # иначе возвращаем лучший по порядку вариантов.
    return snapshots[0][3]


class _SignalChartBase(QWidget):
    """База для чартов EEG: общие вычисления осей, сетка, подписи."""

    # Обрезаем по EDGE_TRIM_SEC секунд с начала и конца окна — там всплывают
    # артефакты (переходные процессы фильтра, моргания).
    EDGE_TRIM_SEC = 2.0

    def __init__(self, snapshot: dict, show_motifs: bool = False,
                 show_thresholds: bool = False, max_seconds: float = 60.0, parent=None):
        super().__init__(parent)
        self.snap = snapshot or {}
        self.show_motifs = bool(show_motifs)
        self.show_thresholds = bool(show_thresholds)
        self.max_seconds = float(max_seconds)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _window(self):
        """Возвращает (times, values, t_min, t_max, v_min, v_max, motifs_in_window, std)."""
        times = self.snap.get("time") or []
        values = self.snap.get("signal") or []
        if not times or not values:
            return None
        t_max_total = float(times[-1])
        # Окно: пропускаем первые EDGE_TRIM_SEC секунд и обрезаем хвост на
        # EDGE_TRIM_SEC — именно там обычно скачки амплитуды.
        t_lo = min(self.EDGE_TRIM_SEC, max(0.0, t_max_total - 1.0))
        t_hi_target = min(self.max_seconds + t_lo, t_max_total)
        t_hi = max(t_lo + 1.0, t_hi_target - self.EDGE_TRIM_SEC)
        pts = [(t, v) for t, v in zip(times, values) if t_lo <= t <= t_hi]
        if not pts:
            return None
        ts = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        t_min = ts[0]
        t_max = ts[-1]
        v_min = min(vs)
        v_max = max(vs)
        if v_max - v_min < 1e-9:
            v_max = v_min + 1.0
        motifs = [m for m in (self.snap.get("motifs") or [])
                  if t_min <= float(m.get("onset_time", 0.0)) <= t_max]
        std = float(self.snap.get("signal_std", 0.0) or 0.0)
        return ts, vs, t_min, t_max, v_min, v_max, motifs, std

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#fafbfc"))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 8, 8)

        win = self._window()
        if not win:
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "Снимок EEG-сигнала не найден")
            painter.end()
            return

        ts, vs, t_min, t_max, v_min, v_max, motifs, std = win
        margin_l, margin_r, margin_t, margin_b = 44, 56, 16, 26
        plot_w = max(10, w - margin_l - margin_r)
        plot_h = max(10, h - margin_t - margin_b)

        # Робастная вертикальная шкала: mean ± K*std, чтобы пороги ложились
        # на ~1/3 высоты, а не прижимались к центру из-за редких выбросов.
        mean_v = float(self.snap.get("signal_mean", 0.0) or 0.0)
        th_low = float(self.snap.get("threshold_low_std", 0.0) or 0.0)
        th_high = float(self.snap.get("threshold_high_std", 0.0) or 0.0)
        if std > 0:
            k = max(3.0, th_high + 1.5, th_low + 2.0)
            y_min = mean_v - k * std
            y_max = mean_v + k * std
        else:
            y_min, y_max = v_min, v_max
        y_rng = max(y_max - y_min, 1e-9)

        def _x(t):
            return margin_l + plot_w * (t - t_min) / max(t_max - t_min, 1e-9)

        def _y(v):
            v = max(y_min, min(y_max, v))  # клип выбросов
            return margin_t + plot_h - plot_h * (v - y_min) / y_rng

        # Сетка и оси
        grid_pen = QPen(QColor("#eceff3"), 1)
        painter.setPen(grid_pen)
        for i in range(1, 5):
            y = margin_t + plot_h * i / 5
            painter.drawLine(margin_l, int(y), margin_l + plot_w, int(y))
        for i in range(1, 6):
            x = margin_l + plot_w * i / 6
            painter.drawLine(int(x), margin_t, int(x), margin_t + plot_h)

        # Сигнал
        painter.setPen(QPen(QColor(PRIMARY), 1.4))
        for i in range(len(ts) - 1):
            painter.drawLine(
                int(_x(ts[i])), int(_y(vs[i])),
                int(_x(ts[i + 1])), int(_y(vs[i + 1])),
            )

        # Пороги: рисуем ПОСЛЕ сигнала (поверх) с заметным стилем и подписями справа за областью графика.
        if self.show_thresholds and std > 0:
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            for th_std, col_hex, dash, width in [
                (th_low,  "#f9ab00", [8, 5], 2.0),   # нижний — оранжевый
                (th_high, "#d93025", [3, 4], 2.0),   # верхний — красный
            ]:
                if th_std <= 0:
                    continue
                for sign in (+1, -1):
                    v = mean_v + sign * th_std * std
                    if not (y_min <= v <= y_max):
                        continue
                    # Лёгкая белая подложка для контраста поверх сигнала
                    bg = QPen(QColor(255, 255, 255, 180), width + 1.2)
                    painter.setPen(bg)
                    y_line = int(_y(v))
                    painter.drawLine(margin_l, y_line, margin_l + plot_w, y_line)
                    # Сама пунктирная линия
                    pen = QPen(QColor(col_hex), width)
                    pen.setDashPattern(dash)
                    painter.setPen(pen)
                    painter.drawLine(margin_l, y_line, margin_l + plot_w, y_line)
                    # Подпись справа, за полем графика
                    label = f"{'+' if sign > 0 else '−'}{th_std:.2f}σ"
                    painter.setPen(QColor(col_hex))
                    painter.drawText(margin_l + plot_w + 4, y_line + 4, label)

        # Маркеры мотивов
        if self.show_motifs and motifs:
            amp_abs = [abs(float(m.get("peak_amplitude", 0.0))) for m in motifs]
            amp_max = max(amp_abs) if amp_abs else 1.0
            for m in motifs:
                onset = float(m.get("onset_time", 0.0))
                peak = float(m.get("peak_time", onset))
                duration = float(m.get("duration", 0.0))
                amp = float(m.get("peak_amplitude", 0.0))
                if onset > t_max:
                    continue
                end = min(onset + duration, t_max)
                x_on = _x(onset)
                x_end = _x(end)
                x_peak = _x(min(peak, t_max))
                norm = abs(amp) / max(amp_max, 1e-9)
                shade = QColor(ACCENT)
                shade.setAlphaF(0.10 + 0.20 * norm)
                painter.fillRect(
                    int(x_on), margin_t,
                    max(1, int(x_end - x_on)), plot_h,
                    shade,
                )
                # Метка пика
                painter.setPen(QPen(QColor(ACCENT), 2))
                painter.drawLine(int(x_peak), margin_t + 2,
                                 int(x_peak), margin_t + plot_h - 2)

        # Оси — подписи
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(margin_l, h - 6, f"{t_min:.1f} с")
        painter.drawText(margin_l + plot_w - 40, h - 6, f"{t_max:.1f} с")
        painter.drawText(4, margin_t + 10, f"{y_max:+.2f}")
        painter.drawText(4, margin_t + plot_h, f"{y_min:+.2f}")
        painter.drawText(4, margin_t + plot_h // 2, "мкВ")
        painter.end()


class _PianoRollChart(QWidget):
    """Piano roll: ноты на шкале time × pitch."""

    def __init__(self, midi_path: str, parent=None):
        super().__init__(parent)
        self.midi_path = midi_path or ""
        self.events = self._load_events()
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _load_events(self) -> list:
        if not self.midi_path or not Path(self.midi_path).exists():
            return []
        try:
            from src.midi_utils import extract_note_events
            return list(extract_note_events(self.midi_path) or [])
        except Exception:
            return []

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#fafbfc"))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 8, 8)

        if not self.events:
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "MIDI-мелодия недоступна")
            painter.end()
            return

        onsets = [float(e.get("onset", 0.0)) for e in self.events]
        ends = [o + float(e.get("duration", 0.0)) for o, e in zip(onsets, self.events)]
        pitches = [int(e.get("pitch", 60)) for e in self.events]
        vels = [int(e.get("velocity", 64)) for e in self.events]
        t_min, t_max = min(onsets), max(ends)
        p_min, p_max = min(pitches) - 1, max(pitches) + 1
        if t_max - t_min < 1e-9:
            t_max = t_min + 1.0
        if p_max - p_min < 1:
            p_max = p_min + 12

        margin_l, margin_r, margin_t, margin_b = 40, 12, 14, 22
        plot_w = max(10, w - margin_l - margin_r)
        plot_h = max(10, h - margin_t - margin_b)

        def _x(t):
            return margin_l + plot_w * (t - t_min) / (t_max - t_min)

        def _y(p):
            return margin_t + plot_h - plot_h * (p - p_min) / (p_max - p_min)

        # Сетка
        painter.setPen(QPen(QColor("#eceff3"), 1))
        for i in range(1, 5):
            y = margin_t + plot_h * i / 5
            painter.drawLine(margin_l, int(y), margin_l + plot_w, int(y))
        for i in range(1, 6):
            x = margin_l + plot_w * i / 6
            painter.drawLine(int(x), margin_t, int(x), margin_t + plot_h)

        # Ноты
        row_h = max(3, int(plot_h / max(p_max - p_min, 1)) - 1)
        for o, end, p, v in zip(onsets, ends, pitches, vels):
            x0 = _x(o)
            x1 = _x(end)
            y = _y(p) - row_h / 2
            alpha = int(120 + 135 * (max(20, min(127, v)) - 20) / 107)
            col = QColor(PRIMARY)
            col.setAlpha(max(120, min(255, alpha)))
            painter.setBrush(col)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(x0), int(y),
                                    max(2, int(x1 - x0)), max(3, row_h),
                                    2, 2)

        # Подписи
        painter.setPen(QColor(TEXT_SECONDARY))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(margin_l, h - 4, f"{t_min:.2f} с")
        painter.drawText(margin_l + plot_w - 40, h - 4, f"{t_max:.2f} с")
        painter.drawText(4, margin_t + 10, f"pitch {p_max}")
        painter.drawText(4, margin_t + plot_h, f"pitch {p_min}")
        painter.end()


class _MappingRulesCard(QFrame):
    """Карточка с правилами маппинга EEG → MIDI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)

        title = QLabel("Как параметры волны ЭЭГ становятся нотой")
        title.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY};")
        lay.addWidget(title)

        intro = QLabel(
            "В сигнале ЭЭГ двумя порогами выделяются устойчивые волны-мотивы "
            "(метод Destexhe & Foubert, 2022). Каждый мотив превращается в одну ноту MIDI. "
            "Ниже — что именно управляет каждой характеристикой ноты."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY}; margin-bottom:4px;")
        lay.addWidget(intro)

        rows = [
            ("Насколько высока волна\n(амплитуда, мкВ)",
             "Громкость ноты (velocity)",
             "Чем сильнее всплеск в ЭЭГ, тем громче нота. "
             "Диапазон нормируется по σ сигнала (0.5σ…2.5σ)."),
            ("Длина волны во времени\n(от начала до конца)",
             "Длительность ноты",
             "Длинная медленная волна даёт протяжную ноту; короткий острый всплеск — staccato."),
            ("Номер волны в последовательности\n(и её частотная полоса)",
             "Высота ноты (pitch)",
             "Порядок и частота мотива отображаются на ступени выбранной гаммы "
             "через сплайн-интерполяцию — подряд идущие мотивы образуют мелодическую линию."),
            ("Время нарастания волны (rise)",
             "Attack огибающей ADSR",
             "Медленно нарастающая волна → мягкая атака ноты; резкий фронт → чёткий удар."),
            ("Время спада волны (decay)",
             "Decay / Release ADSR",
             "Быстро затухающая волна даёт короткий хвост ноты; плавный спад — долгое угасание."),
        ]
        for left, right, hint in rows:
            row = QHBoxLayout()
            row.setSpacing(10)
            l = QLabel(left)
            l.setWordWrap(True)
            l.setStyleSheet(f"font-size:12px; color:{TEXT_PRIMARY};")
            l.setMinimumWidth(210)
            l.setMaximumWidth(230)
            a = QLabel("→")
            a.setStyleSheet(f"font-size:16px; font-weight:700; color:{ACCENT};")
            a.setAlignment(Qt.AlignCenter)
            rbox = QVBoxLayout()
            rbox.setSpacing(1)
            r = QLabel(right)
            r.setStyleSheet(f"font-size:12px; font-weight:600; color:{TEXT_PRIMARY};")
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
            rbox.addWidget(r)
            rbox.addWidget(h)
            rwrap = QWidget()
            rwrap.setLayout(rbox)
            row.addWidget(l)
            row.addWidget(a)
            row.addWidget(rwrap, stretch=1)
            wrap = QWidget()
            wrap.setLayout(row)
            lay.addWidget(wrap)

        note = QLabel(
            "Итого: громкость ноты берётся из силы всплеска, высота — из последовательности волн, "
            "длительность и форма звука (ADSR) — из того, как волна нарастает и затухает во времени."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY}; margin-top:6px;")
        lay.addWidget(note)


class _StageCard(QFrame):
    """Обёртка для этапа: заголовок, подзаголовок, чарт."""

    def __init__(self, title: str, subtitle: str, chart: QWidget,
                 extra_header: Optional[QWidget] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY};")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        if extra_header is not None:
            hdr.addWidget(extra_header)
        wrap = QWidget()
        wrap.setLayout(hdr)
        lay.addWidget(wrap)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
            sub.setWordWrap(True)
            lay.addWidget(sub)

        lay.addWidget(chart)


# ──────────────────────────────────────────────────────────────────────────
# Виджеты для вкладок «Эмоции» и «Отчёт» (Qt-нативные, без HTML-простынь)
# ──────────────────────────────────────────────────────────────────────────

# Русские подписи эмоций и эмодзи — используются по всей новой Эмоции-вкладке
_EMOTION_FULL = {
    "HVHA": ("Радость — высокая валентность / высокое возбуждение", ""),
    "HVLA": ("Спокойствие — высокая валентность / низкое возбуждение", ""),
    "LVLA": ("Грусть — низкая валентность / низкое возбуждение", ""),
    "LVHA": ("Напряжение — низкая валентность / высокое возбуждение", ""),
}


# Русские названия акустических признаков + подробные подсказки для тултипов
_FEATURE_RU = {
    "Tempo":       ("Темп",
                    "Темп — количество долей в минуту (BPM). Высокий темп "
                    "ассоциируется с возбуждением и энергичностью, низкий — "
                    "со спокойствием и грустью."),
    "Velocity":    ("Громкость (velocity)",
                    "Сила удара по ноте (MIDI velocity) — аналог громкости "
                    "исполнения. Громкая динамика связана с высоким "
                    "возбуждением, тихая — со спокойствием."),
    "Mode":        ("Лад",
                    "Мажор или минор. Мажорный лад обычно воспринимается как "
                    "«светлый», минорный — как «тёмный», печальный."),
    "Pitch Range": ("Диапазон высот",
                    "Разница между самой низкой и самой высокой нотой. "
                    "Широкий диапазон — экспрессивность и драматизм, "
                    "узкий — сдержанность, покой."),
    "Staccato":    ("Отрывистость (staccato)",
                    "Доля коротких отрывистых нот. Высокая — энергия, "
                    "острота; низкая — плавная игра legato."),
    "Rhythm Var.": ("Вариативность ритма",
                    "Насколько сильно меняются длительности нот. "
                    "Высокая — неустойчивый, «нервный» ритм; "
                    "низкая — ровное, предсказуемое движение."),
    "Consonance":  ("Консонантность",
                    "Гармоничность созвучий. Высокая — благозвучные "
                    "интервалы, ощущение покоя; низкая — диссонансы, "
                    "напряжение."),
}


def _info_label(text: str, tooltip: str) -> "QLabel":
    """Маленькая иконка-подсказка (?) рядом с заголовком."""
    lbl = QLabel(text + "  ⓘ")
    lbl.setToolTip(tooltip)
    lbl.setStyleSheet(
        f"font-size:14px; font-weight:700; color:{PRIMARY};"
    )
    return lbl


class _KPITile(QFrame):
    """Крупный KPI: большое число сверху, подпись снизу, опционально цвет."""

    def __init__(self, value: str, label: str, color: str = PRIMARY,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if tooltip:
            self.setToolTip(tooltip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        val = QLabel(value)
        val.setAlignment(Qt.AlignCenter)
        val.setStyleSheet(f"font-size:26px; font-weight:700; color:{color};")
        lay.addWidget(val)
        cap_text = label + ("  ⓘ" if tooltip else "")
        cap = QLabel(cap_text)
        cap.setAlignment(Qt.AlignCenter)
        cap.setWordWrap(True)
        cap.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        if tooltip:
            cap.setToolTip(tooltip)
        lay.addWidget(cap)


class _VerdictCard(QFrame):
    """Hero-карточка вердикта: цветной бейдж эмоции + top-K stats."""

    def __init__(
        self,
        eeg_emotion: str,
        top1_music_emotion: str,
        topk_match: int,
        topk_total: int,
        best_piece: str,
        best_score: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        from gui.styles import EMOTION_COLORS as EC
        emo_color = EC.get(eeg_emotion, PRIMARY)
        emo_label, emoji = _EMOTION_FULL.get(eeg_emotion, (eeg_emotion, ""))

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(16)

        # ── Левая колонка — бейдж эмоции ──
        badge = QFrame()
        badge.setStyleSheet(
            f"background:{emo_color}; border-radius:12px;"
        )
        badge.setMinimumWidth(180)
        badge.setMaximumWidth(220)
        b_lay = QVBoxLayout(badge)
        b_lay.setContentsMargins(14, 12, 14, 12)
        b_lay.setSpacing(2)
        # Эмодзи убраны по требованию UX — оставляем только код и русскую подпись.
        b_code = QLabel(eeg_emotion or "—")
        b_code.setAlignment(Qt.AlignCenter)
        b_code.setStyleSheet(
            "font-size:20px; font-weight:700; color:white; background:transparent;"
        )
        b_lay.addWidget(b_code)
        b_lbl = QLabel(emo_label.split(" — ")[0])
        b_lbl.setAlignment(Qt.AlignCenter)
        b_lbl.setStyleSheet(
            "font-size:11px; color:rgba(255,255,255,0.92); background:transparent;"
        )
        b_lay.addWidget(b_lbl)
        root.addWidget(badge)

        # ── Правая колонка — вердикт ──
        right = QVBoxLayout()
        right.setSpacing(6)

        title = QLabel("Ключевой эмоциональный результат")
        title.setStyleSheet(f"font-size:13px; font-weight:700; color:{PRIMARY};")
        right.addWidget(title)

        # Top-1 совпадение — цветной индикатор
        match_ok = eeg_emotion == top1_music_emotion and eeg_emotion not in ("", "—")
        ind_color = MATCH_OK if match_ok else MATCH_BAD
        ind_icon = "✓" if match_ok else "✗"
        top1_lbl = QLabel(
            f"<span style='color:{ind_color}; font-weight:700;'>{ind_icon}</span> "
            f"Top-1 музыка: <b>{escape(top1_music_emotion or '—')}</b>"
        )
        top1_lbl.setTextFormat(Qt.RichText)
        top1_lbl.setStyleSheet(f"font-size:13px; color:{TEXT_PRIMARY};")
        right.addWidget(top1_lbl)

        # Top-K coverage
        if topk_total > 0:
            topk_pct = int(round(100 * topk_match / topk_total))
            topk_lbl = QLabel(
                f"В top-{topk_total}: <b>{topk_match}/{topk_total}</b> "
                f"совпали по эмоции ({topk_pct}%)"
            )
            topk_lbl.setTextFormat(Qt.RichText)
            topk_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
            right.addWidget(topk_lbl)

        piece_lbl = QLabel(
            f"Лучшее соответствие: <b>{escape(best_piece)}</b> "
            f"<span style='color:{TEXT_SECONDARY};'>score {best_score:.3f}</span>"
        )
        piece_lbl.setTextFormat(Qt.RichText)
        piece_lbl.setWordWrap(True)
        piece_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_PRIMARY};")
        right.addWidget(piece_lbl)

        right.addStretch()
        root.addLayout(right, stretch=1)


class _ConfusionMatrixWidget(QFrame):
    """Qt-нативная confusion matrix: heatmap по строкам × столбцам, без PNG."""

    EMO_ORDER = ("HVHA", "HVLA", "LVLA", "LVHA")

    def __init__(self, cm: "pd.DataFrame", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        hdr_row = QHBoxLayout()
        hdr_row.addWidget(_info_label(
            "Матрица ошибок",
            "Строки — эмоция EEG, столбцы — эмоция найденной музыки.\n"
            "Ячейки на диагонали означают совпадение.\n"
            "Насыщенность цвета пропорциональна числу совпадений."
        ))
        hdr_row.addStretch()
        root.addLayout(hdr_row)

        if cm is None or cm.empty:
            empty = QLabel("Нет данных для confusion matrix.")
            empty.setStyleSheet(f"color:{TEXT_SECONDARY}; padding:16px;")
            empty.setAlignment(Qt.AlignCenter)
            root.addWidget(empty)
            return

        # Приводим к фиксированному порядку эмоций
        rows = [e for e in self.EMO_ORDER if e in cm.index]
        cols = [e for e in self.EMO_ORDER if e in cm.columns]
        if not rows or not cols:
            empty = QLabel("Нет данных для confusion matrix.")
            empty.setStyleSheet(f"color:{TEXT_SECONDARY}; padding:16px;")
            empty.setAlignment(Qt.AlignCenter)
            root.addWidget(empty)
            return

        max_val = float(cm.loc[rows, cols].values.max() or 1)

        grid = QGridLayout()
        grid.setSpacing(2)
        grid.setContentsMargins(0, 0, 0, 0)

        # Пустая ячейка, затем заголовки столбцов
        corner = QLabel("")
        corner.setFixedSize(60, 28)
        grid.addWidget(corner, 0, 0)
        for ci, col in enumerate(cols, start=1):
            h = QLabel(col)
            h.setAlignment(Qt.AlignCenter)
            h.setStyleSheet(
                f"font-size:11px; font-weight:700; color:{TEXT_PRIMARY}; "
                f"background:{TABLE_HEADER_BG}; padding:4px; border-radius:4px;"
            )
            grid.addWidget(h, 0, ci)

        # Строки
        for ri, row in enumerate(rows, start=1):
            # Заголовок строки
            rh = QLabel(row)
            rh.setAlignment(Qt.AlignCenter)
            rh.setStyleSheet(
                f"font-size:11px; font-weight:700; color:{TEXT_PRIMARY}; "
                f"background:{TABLE_HEADER_BG}; padding:4px; border-radius:4px;"
            )
            grid.addWidget(rh, ri, 0)
            # Значения
            for ci, col in enumerate(cols, start=1):
                val = int(cm.loc[row, col]) if col in cm.columns else 0
                intensity = val / max_val if max_val > 0 else 0.0
                is_diag = row == col
                base = PRIMARY if is_diag else DANGER
                # Приводим hex → rgba с переменной прозрачностью
                r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
                alpha = 0.12 + 0.88 * intensity
                bg = f"rgba({r},{g},{b},{alpha:.2f})"
                cell = QLabel(str(val))
                cell.setAlignment(Qt.AlignCenter)
                fg = "white" if intensity > 0.55 else TEXT_PRIMARY
                cell.setStyleSheet(
                    f"background:{bg}; color:{fg}; "
                    f"font-size:13px; font-weight:700; "
                    f"border-radius:4px; padding:10px;"
                )
                cell.setMinimumSize(60, 42)
                grid.addWidget(cell, ri, ci)

        grid_wrap = QHBoxLayout()
        grid_wrap.addStretch()
        inner = QWidget()
        inner.setLayout(grid)
        grid_wrap.addWidget(inner)
        grid_wrap.addStretch()
        root.addLayout(grid_wrap)

        # Подписи осей
        axis = QLabel(
            f"<span style='color:{TEXT_MUTED};'>Вертикаль — эмоция EEG,  горизонталь — эмоция музыки.</span>"
        )
        axis.setTextFormat(Qt.RichText)
        axis.setAlignment(Qt.AlignCenter)
        axis.setStyleSheet("font-size:11px;")
        root.addWidget(axis)


class _ExpectedActualRow(QFrame):
    """Одна строка в Портрете эмоции: Признак | Ожидаемо | Реально | Статус."""

    def __init__(
        self,
        feat: str,
        expected: str,
        actual_level: Optional[str],
        actual_raw: Optional[float],
        status: str,  # 'match' | 'close' | 'mismatch' | 'no_data'
        reason: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet(f"border-bottom:1px solid {BORDER_SOFT};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 6, 2, 6)
        lay.setSpacing(10)

        # Feature name — русская подпись + подробный тултип
        ru_name, feat_hint = _FEATURE_RU.get(feat, (feat, ""))
        tip_parts = [feat_hint, reason] if feat_hint else [reason]
        tip = "\n\n".join(p for p in tip_parts if p)
        name = QLabel(ru_name)
        name.setFixedWidth(140)
        name.setWordWrap(True)
        name.setStyleSheet(f"font-size:11px; font-weight:600; color:{TEXT_PRIMARY}; border:none;")
        name.setToolTip(tip)
        lay.addWidget(name)

        # Expected pill — русификация уровней
        _LEVEL_RU = {
            "high": "высокая", "low": "низкая", "moderate": "средняя",
            "major": "мажор", "minor": "минор",
            "wide": "широкий", "narrow": "узкий",
        }
        exp = QLabel(_LEVEL_RU.get(expected, expected))
        exp.setFixedWidth(88)
        exp.setAlignment(Qt.AlignCenter)
        exp.setStyleSheet(
            f"font-size:10px; font-weight:600; color:{PRIMARY}; "
            f"background:{CARD_HIGHLIGHT_BG}; "
            f"border:1px solid {BORDER_SOFT}; border-radius:8px; "
            f"padding:2px 6px;"
        )
        lay.addWidget(exp)

        # Actual cell
        status_colors = {
            "match": MATCH_OK, "close": MATCH_WARN,
            "mismatch": MATCH_BAD, "no_data": MATCH_NONE,
        }
        icon_map = {"match": "✓", "close": "~", "mismatch": "✗", "no_data": "—"}
        color = status_colors.get(status, MATCH_NONE)
        icon = icon_map.get(status, "—")
        if status == "no_data":
            actual_text = "нет данных"
        else:
            raw_disp = (
                "мажор" if feat == "Mode" and (actual_raw or 0) > 0.5 else
                "минор" if feat == "Mode" else
                f"{actual_raw:.2f}" if actual_raw is not None else ""
            )
            actual_text = _LEVEL_RU.get(actual_level or "", actual_level or "")
            if raw_disp and feat != "Mode":
                actual_text += f" ({raw_disp})"
        act = QLabel(f"<span style='color:{color}; font-weight:700;'>{icon}</span>  {escape(actual_text)}")
        act.setTextFormat(Qt.RichText)
        act.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY}; border:none;")
        lay.addWidget(act, stretch=1)


class _EmotionPortraitCard(QFrame):
    """Карточка-портрет эмоции: top work, top composer, expected vs actual."""

    def __init__(
        self,
        emo: str,
        n_participants: int,
        top_work: Optional[str],
        top_composer: Optional[str],
        expected_map: dict,            # {feat: (direction, reason)}
        actual_map: dict,              # {feat: (raw_value, level)}  — может быть пустым
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        from gui.styles import EMOTION_COLORS as EC
        color = EC.get(emo, PRIMARY)
        full_label, emoji = _EMOTION_FULL.get(emo, (emo, ""))

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        # Header: цветная полоса слева + код эмоции + n
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        accent = QFrame()
        accent.setFixedWidth(4)
        accent.setMinimumHeight(32)
        accent.setStyleSheet(f"background:{color}; border-radius:2px;")
        hdr.addWidget(accent)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        t1 = QLabel(emo)
        t1.setStyleSheet(f"font-size:15px; font-weight:700; color:{color};")
        title_col.addWidget(t1)
        t2 = QLabel(full_label.split(" — ")[0] if " — " in full_label else full_label)
        t2.setStyleSheet(f"font-size:10px; color:{TEXT_SECONDARY};")
        title_col.addWidget(t2)
        hdr.addLayout(title_col)
        hdr.addStretch()
        n_lbl = QLabel(f"n = {n_participants}")
        n_lbl.setStyleSheet(
            f"font-size:10px; color:{TEXT_SECONDARY}; "
            f"background:{TABLE_HEADER_BG}; padding:2px 8px; border-radius:8px;"
        )
        hdr.addWidget(n_lbl)
        root.addLayout(hdr)

        # Top work / composer
        if top_work:
            tw = QLabel(f"Произведение: <b>{escape(top_work)}</b>")
            tw.setTextFormat(Qt.RichText)
            tw.setWordWrap(True)
            tw.setStyleSheet(f"font-size:11px; color:{TEXT_PRIMARY};")
            root.addWidget(tw)
        if top_composer:
            tc = QLabel(f"Композитор: {escape(top_composer)}")
            tc.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
            root.addWidget(tc)

        # Expected vs actual — таблица через Qt-виджеты
        # Заголовок таблицы
        head = QHBoxLayout()
        head.setContentsMargins(2, 4, 2, 2)
        head.setSpacing(10)
        for txt, w in [("Признак", 140), ("Ожидаемо", 88), ("Реально", 0)]:
            h = QLabel(txt.upper())
            if w:
                h.setFixedWidth(w)
            h.setStyleSheet(f"font-size:9px; font-weight:700; color:{TEXT_MUTED};")
            head.addWidget(h, stretch=1 if w == 0 else 0)
        root.addLayout(head)

        # Direction compatibility map
        direction_compat = {
            "high": ["high", "moderate"],
            "low": ["low", "moderate"],
            "moderate": ["moderate", "low", "high"],
            "major": ["major"],
            "minor": ["minor"],
            "wide": ["high", "moderate"],
            "narrow": ["low", "moderate"],
        }

        for feat, (direction, reason) in expected_map.items():
            if feat in actual_map:
                raw_val, actual_level = actual_map[feat]
                compat = direction_compat.get(direction, [direction])
                if actual_level == direction:
                    status = "match"
                elif actual_level in compat:
                    status = "close"
                else:
                    status = "mismatch"
                row = _ExpectedActualRow(feat, direction, actual_level, raw_val, status, reason)
            else:
                row = _ExpectedActualRow(feat, direction, None, None, "no_data", reason)
            root.addWidget(row)


class _SignalCoverageBar(QFrame):
    """Горизонтальная визуализация: длина записи → длина проанализированной мелодии."""

    def __init__(self, variant: str, rec_min: float, melody_sec: float,
                 note_count: int, silence_ratio: float, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        # Header line
        compression = (rec_min * 60.0 / melody_sec) if melody_sec > 0 else 0.0
        hdr = QLabel(
            f"<b>{escape(variant)}</b>  "
            f"<span style='color:{TEXT_SECONDARY};'>"
            f"{rec_min:.1f} мин → ♪ {melody_sec:.1f} с  (×{compression:.0f})"
            f"</span>"
        )
        hdr.setTextFormat(Qt.RichText)
        hdr.setStyleSheet(f"font-size:12px; color:{TEXT_PRIMARY};")
        lay.addWidget(hdr)

        # Bar — ширина мелодии относительно записи (1.0 = вся запись)
        bar_row = QHBoxLayout()
        bar_row.setSpacing(6)
        bar_bg = QFrame()
        bar_bg.setFixedHeight(10)
        bar_bg.setStyleSheet(
            f"background:{ROW_ALT_BG}; border:1px solid {BORDER_SOFT}; border-radius:5px;"
        )
        bar_bg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bar_fill = QFrame(bar_bg)
        frac = min(1.0, max(0.0, (melody_sec / (rec_min * 60.0)) if rec_min > 0 else 0.0))
        # Отрисовка через setStyleSheet + resize в показовом событии — простая имитация
        bar_fill.setStyleSheet(
            f"background:{PRIMARY}; border-radius:5px;"
        )
        # Храним frac чтобы обновить в showEvent
        bar_bg._fill = bar_fill
        bar_bg._frac = frac
        bar_bg._last_w = 0

        def _resize_bar(ev=None, bg=bar_bg):
            w = bg.width() - 2
            h = bg.height() - 2
            bg._fill.setGeometry(1, 1, max(1, int(w * bg._frac)), max(1, h))
        bar_bg.resizeEvent = _resize_bar
        bar_row.addWidget(bar_bg, stretch=1)

        pct_lbl = QLabel(f"{int(frac*100)}%")
        pct_lbl.setFixedWidth(40)
        pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        bar_row.addWidget(pct_lbl)
        lay.addLayout(bar_row)

        # Stats line
        stats = QLabel(
            f"нот: <b>{note_count}</b>   "
            f"тишина: <b>{silence_ratio*100:.0f}%</b>"
        )
        stats.setTextFormat(Qt.RichText)
        stats.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY};")
        lay.addWidget(stats)


class _TopWorksBarChart(QWidget):
    """Qt-нативная горизонтальная столбчатая диаграмма top-N произведений."""

    def __init__(self, works: list[tuple[str, int]], parent=None):
        super().__init__(parent)
        self.works = [(str(w), int(c)) for w, c in works][:10]
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(max(60, 28 * max(1, len(self.works)) + 12))

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(BG_CARD))

        if not self.works:
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "Нет данных.")
            painter.end()
            return

        max_count = max(c for _, c in self.works) or 1
        row_h = 24
        gap = 4
        label_w = 260
        bar_x = label_w + 10
        bar_w_max = w - bar_x - 50

        painter.setFont(QFont("", 11))
        for i, (title, count) in enumerate(self.works):
            y = 6 + i * (row_h + gap)
            # Название (обрезка до label_w)
            painter.setPen(QColor(TEXT_PRIMARY))
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(title, Qt.ElideRight, label_w)
            painter.drawText(4, y, label_w, row_h, Qt.AlignLeft | Qt.AlignVCenter, elided)
            # Бар
            bw = max(2, int(bar_w_max * (count / max_count)))
            painter.setBrush(QColor(PRIMARY))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bar_x, y + 4, bw, row_h - 8, 3, 3)
            # Число справа
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(
                bar_x + bw + 6, y, 40, row_h,
                Qt.AlignLeft | Qt.AlignVCenter,
                str(count),
            )
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
        # Rank + signal variant badge on same row
        rank_row = QHBoxLayout()
        rank_row.setSpacing(6)
        rank_lbl = QLabel(f"#{rank}")
        rank_lbl.setStyleSheet(f"font-size:12px; font-weight:700; color:{TEXT_SECONDARY};")
        rank_row.addWidget(rank_lbl)

        variant = str(row.get("variant") or "")
        if variant:
            v_lbl = QLabel(variant)
            v_lbl.setStyleSheet(
                f"font-size:10px; color:{TEXT_SECONDARY}; padding:0 4px;"
            )
            rank_row.addWidget(v_lbl)

        # EEG source file (или participant/trial для DAT) — рядом с rank/variant
        _eeg_file_top = str(row.get("eeg_source_file") or "")
        _participant_top = str(row.get("participant_id") or "")
        _trial_top = row.get("trial_idx")
        if _eeg_file_top:
            _src_text = _eeg_file_top
        elif _participant_top:
            _src_text = _participant_top
            if _trial_top is not None and str(_trial_top) not in ("", "nan", "None"):
                _src_text += f" trial {int(float(_trial_top))}"
        else:
            _src_text = ""
        if _src_text:
            src_lbl = QLabel(_src_text)
            src_lbl.setStyleSheet(
                f"font-size:10px; color:{TEXT_SECONDARY}; padding:0 4px;"
            )
            rank_row.addWidget(src_lbl)

        rank_row.addStretch()
        left.addLayout(rank_row)

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
        score_box.setStyleSheet(
            f"background:{CARD_HIGHLIGHT_BG}; border:1px solid {BORDER_SOFT}; border-radius:10px;"
        )
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
        sub_lbl = QLabel("Совпадение музыки")
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
            ("Совпадение музыки", row.get("music_match_score", row.get("combined_similarity", 0.0))),
            ("Сходство признаков", row.get("feature_similarity_score", 0.0)),
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
        self._tabs: Optional[QTabWidget] = None
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

        c_match = _SummaryCard("Совпадение эмоций")
        c_match.set_big_value(_fmt_pct(metrics.get("emotion_match_rate", 0.0)), ACCENT)
        c_match.add_line(f"Macro-F1: {_fmt_score(metrics.get('macro_f1', 0.0))}")
        summary_row.addWidget(c_match)

        c_music = _SummaryCard("Совпадение музыки")
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
            composers_card = _SummaryCard("Топ-композиторы")
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
        self._tabs.addTab(self._build_matches_tab(df), "Лучшие совпадения")
        self._tabs.addTab(_wrap_scroll(self._build_transform_tab(df)), "Преобразование")
        self._tabs.addTab(_wrap_scroll(self._build_emotion_tab(df)), "Эмоции")
        self._tabs.addTab(_wrap_scroll(self._build_summary_tab(df)), "Отчёт")

    def _build_matches_tab(self, df: pd.DataFrame) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(12)

        if "music_match_score" in df.columns:
            ranked = df.sort_values("music_match_score", ascending=False)
        elif "combined_similarity" in df.columns:
            ranked = df.sort_values("combined_similarity", ascending=False)
        else:
            ranked = df
        for rank, row in enumerate(ranked.head(5).to_dict("records"), start=1):
            lay.addWidget(_MatchCard(row, rank))
        lay.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_transform_tab(self, df: pd.DataFrame) -> QWidget:
        """Вкладка «Преобразование»: сырой EEG → мотивы → MIDI-мелодия."""
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        report_dir = Path(self._report_dir or "")

        # Заголовок + лид
        header = QLabel("Signal → Music: как EEG становится мелодией")
        header.setStyleSheet(f"font-size:16px; font-weight:700; color:{PRIMARY};")
        root.addWidget(header)
        lead = QLabel(
            "Три этапа конвейера: исходный EEG-сигнал, детекция волновых мотивов "
            "двухпороговым методом и финальная MIDI-мелодия."
        )
        lead.setWordWrap(True)
        lead.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
        root.addWidget(lead)

        snap = _load_signal_snapshot(report_dir)

        # MIDI-путь для piano roll: либо из snapshot, либо из df
        midi_path = ""
        if snap:
            midi_path = str(snap.get("midi_path") or "")
        if not midi_path:
            # fallback — первый eeg_midi_path из df
            for _, row in df.iterrows():
                cand = str(row.get("eeg_midi_path") or row.get("eeg_midi") or "").strip()
                if cand and Path(cand).exists():
                    midi_path = cand
                    break

        if not snap:
            warn = QFrame()
            warn.setObjectName("card")
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(16, 14, 16, 14)
            msg = QLabel(
                "Снимок EEG-сигнала недоступен для этого запуска. "
                "Запустите новый анализ — первый и второй этапы заполнятся автоматически."
            )
            msg.setWordWrap(True)
            msg.setStyleSheet(f"font-size:12px; color:{TEXT_SECONDARY};")
            wl.addWidget(msg)
            root.addWidget(warn)

        # Этап 1 — Сырой EEG-сигнал
        if snap:
            fs = float(snap.get("fs", 0.0) or 0.0)
            duration = float(snap.get("duration_sec", 0.0) or 0.0)
            window = min(60.0, duration if duration > 0 else 60.0)
            subtitle1 = f"fs = {fs:.0f} Гц,  длительность записи = {duration:.1f} с   (показано первые {window:.1f} с)"
            chart1 = _SignalChartBase(snap, show_motifs=False, show_thresholds=False, max_seconds=window)
            root.addWidget(_StageCard(
                "Этап 1 — Сырой EEG-сигнал",
                subtitle1,
                chart1,
            ))

            # Этап 2 — Детекция wave-мотивов
            # Считаем мотивы, попадающие в показанное окно (после обрезки краёв).
            edge = _SignalChartBase.EDGE_TRIM_SEC
            t_lo = min(edge, max(0.0, duration - 1.0))
            t_hi = max(t_lo + 1.0, min(window, duration) - edge + t_lo)
            motifs_in_view = [
                m for m in (snap.get("motifs") or [])
                if t_lo <= float(m.get("onset_time", 0.0)) <= t_hi
            ]
            n_motifs = len(motifs_in_view)
            th_low = float(snap.get("threshold_low_std", 0.0) or 0.0)
            th_high = float(snap.get("threshold_high_std", 0.0) or 0.0)
            subtitle2 = (
                f"Нижний порог (upstroke) = ±{th_low:.2f}σ · верхний (peak) = ±{th_high:.2f}σ · "
                f"найдено мотивов в окне: {n_motifs}"
            )
            chart2 = _SignalChartBase(snap, show_motifs=True, show_thresholds=True, max_seconds=window)
            root.addWidget(_StageCard(
                "Этап 2 — Детекция wave-мотивов",
                subtitle2,
                chart2,
            ))

        # Этап 3 — MIDI-мелодия
        play_btn = None
        if midi_path and Path(midi_path).exists():
            play_btn = QPushButton("▶ Прослушать")
            play_btn.setObjectName("secondary")
            _register_audio_button(play_btn, "▶ Прослушать")
            play_btn.clicked.connect(lambda _=False, mp=midi_path, b=play_btn:
                                     _toggle_audio(mp, b, "▶ Прослушать"))
        chart3 = _PianoRollChart(midi_path)
        note_count = len(chart3.events)
        subtitle3 = f"Piano roll: {note_count} нот"
        root.addWidget(_StageCard(
            "Этап 3 — MIDI-мелодия",
            subtitle3,
            chart3,
            extra_header=play_btn,
        ))

        # Карточка правил маппинга
        root.addWidget(_MappingRulesCard())

        root.addStretch()
        return widget

    def _build_emotion_tab(self, df: pd.DataFrame) -> QWidget:
        """Hero + Confusion Matrix + 2×2 портреты + Radar/Bars."""
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        report_dir = Path(self._report_dir or "")
        metrics = self._load_metrics()

        ranking_col = "music_match_score" if "music_match_score" in df.columns else "combined_similarity"
        if ranking_col in df.columns:
            ranked = df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
        else:
            ranked = df.reset_index(drop=True)
        best_row = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)

        eeg_emotion = str(best_row.get("eeg_emotion", "—")) or "—"
        top1_music_emotion = str(best_row.get("classical_emotion", "—")) or "—"
        best_piece = f"{best_row.get('composer', 'Unknown')} — {best_row.get('title', 'Unknown')}"
        best_score = float(best_row.get("music_match_score", best_row.get("combined_similarity", 0.0)) or 0.0)

        topk_emotions = ranked.head(5)["classical_emotion"].dropna().astype(str) \
            if "classical_emotion" in ranked.columns else pd.Series(dtype=str)
        topk_match_count = int((topk_emotions == eeg_emotion).sum()) if eeg_emotion not in {"", "—"} else 0
        topk_total = len(topk_emotions)

        # ── 1. Hero: verdict + KPI tiles (горизонтально) ─────────────────
        hero = QHBoxLayout()
        hero.setSpacing(12)
        hero.addWidget(_VerdictCard(
            eeg_emotion=eeg_emotion,
            top1_music_emotion=top1_music_emotion,
            topk_match=topk_match_count,
            topk_total=topk_total,
            best_piece=best_piece,
            best_score=best_score,
        ), stretch=3)

        kpi_col = QVBoxLayout()
        kpi_col.setSpacing(8)
        kpi_row1 = QHBoxLayout(); kpi_row1.setSpacing(8)
        kpi_row2 = QHBoxLayout(); kpi_row2.setSpacing(8)
        kpi_row1.addWidget(_KPITile(
            _fmt_pct(metrics.get("emotion_match_rate", 0.0)),
            "Совпадение эмоций (Top-K)", ACCENT,
            tooltip=(
                "Доля испытаний, в которых эмоция ЭЭГ совпала хотя бы с одной из K "
                "ближайших по признакам классических пьес (по умолчанию K = 5). "
                "Смягчённая метрика: учитывается, что в топе всегда несколько "
                "подходящих кандидатов, а не только один."
            ),
        ))
        kpi_row1.addWidget(_KPITile(
            _fmt_pct(metrics.get("strict_top1_match_rate", 0.0)),
            "Точное совпадение (Top-1)", PRIMARY,
            tooltip=(
                "Доля испытаний, в которых эмоция ЭЭГ совпала с эмоцией самой "
                "первой (ближайшей) классической пьесы. Это строгий случай: "
                "метод ошибся, даже если правильная эмоция стояла в топе, но не "
                "на первом месте."
            ),
        ))
        kpi_row2.addWidget(_KPITile(
            _fmt_score(metrics.get("macro_f1", 0.0)),
            "Macro-F1", PRIMARY,
            tooltip=(
                "Средняя по всем четырём классам эмоций F1-мера (гармоническое "
                "среднее точности и полноты). \"Macro\" — значит каждая эмоция "
                "учитывается с равным весом, поэтому редкий класс не теряется "
                "за счёт частого. Диапазон: 0…1, чем выше — тем лучше."
            ),
        ))
        kpi_row2.addWidget(_KPITile(
            _fmt_pct(metrics.get("group_consistency_mean", 0.0)),
            "Согласованность испытуемых", PRIMARY,
            tooltip=(
                "Насколько похожи между собой результаты разных участников, "
                "которым показывали один и тот же стимул. Высокое значение — "
                "разные испытуемые при одинаковой эмоции получают похожий набор "
                "рекомендованных пьес; низкое — результат сильно зависит от "
                "индивидуальной реакции."
            ),
        ))
        kpi_col.addLayout(kpi_row1)
        kpi_col.addLayout(kpi_row2)
        kpi_wrap = QWidget()
        kpi_wrap.setLayout(kpi_col)
        hero.addWidget(kpi_wrap, stretch=2)
        root.addLayout(hero)

        # ── 2. Confusion matrix (Qt-native, без PNG) ─────────────────────
        cm_df = pd.DataFrame()
        if "eeg_emotion" in df.columns and "classical_emotion" in df.columns:
            try:
                cm_df = pd.crosstab(df["eeg_emotion"], df["classical_emotion"])
            except Exception:
                cm_df = pd.DataFrame()
        root.addWidget(_ConfusionMatrixWidget(cm_df))

        # ── 3. Подготовка EMOTION_RESEARCH и actual_vals ─────────────────
        # Research-backed expected feature directions per emotion
        EMOTION_RESEARCH = {
            "HVHA": {"expected": {
                "Tempo": ("high", "Быстрый темп → высокий arousal"),
                "Velocity": ("high", "Громкая динамика → excitement"),
                "Mode": ("major", "Мажор → позитивная валентность"),
                "Pitch Range": ("wide", "Широкий диапазон → экспрессивность"),
                "Staccato": ("high", "Короткие ноты → энергичность"),
                "Rhythm Var.": ("moderate", "Умеренная вариативность ритма"),
            }},
            "HVLA": {"expected": {
                "Tempo": ("low", "Медленный темп → низкий arousal"),
                "Velocity": ("moderate", "Умеренная громкость → спокойствие"),
                "Mode": ("major", "Мажор → позитивная валентность"),
                "Pitch Range": ("narrow", "Узкий диапазон → уравновешенность"),
                "Staccato": ("low", "Legato → плавность"),
                "Consonance": ("high", "Высокая консонантность → умиротворение"),
            }},
            "LVLA": {"expected": {
                "Tempo": ("low", "Медленный темп → апатия"),
                "Velocity": ("low", "Тихая динамика → грусть"),
                "Mode": ("minor", "Минор → негативная валентность"),
                "Pitch Range": ("narrow", "Небольшой диапазон, нисходящий контур"),
                "Staccato": ("low", "Legato → тягучесть"),
                "Consonance": ("moderate", "Умеренная консонантность"),
            }},
            "LVHA": {"expected": {
                "Tempo": ("high", "Быстрый темп → напряжение"),
                "Velocity": ("high", "Контрастная динамика → агрессия"),
                "Mode": ("minor", "Минор → негативная валентность"),
                "Pitch Range": ("wide", "Широкий диапазон, скачки"),
                "Rhythm Var.": ("high", "Нерегулярный ритм → нестабильность"),
                "Consonance": ("low", "Диссонансы → напряжение"),
            }},
        }

        # Загрузка фактических профилей
        profiles_df = pd.DataFrame()
        profiles_path = report_dir / "group_feature_profiles.csv"
        if profiles_path.exists():
            try:
                profiles_df = pd.read_csv(profiles_path)
            except Exception:
                pass
        if profiles_df.empty and not df.empty:
            try:
                from src.group_analysis import compute_group_profiles
                _gd = compute_group_profiles(df)
                if _gd:
                    profiles_df = _gd.get("feature_table", pd.DataFrame())
            except Exception:
                pass

        FEAT_COL_MAP = {
            "Tempo":       "Tempo",
            "Velocity":    "Velocity (Loudness)",
            "Mode":        "Mode",
            "Pitch Range": "Pitch Range",
            "Staccato":    "Staccato Ratio",
            "Rhythm Var.": "Rhythm Variability",
            "Consonance":  "Consonance",
        }

        actual_vals = {}
        if not profiles_df.empty and "Emotion" in profiles_df.columns:
            numeric_cols = [c for c in profiles_df.columns if c not in ("Emotion", "N")]
            col_mins, col_maxs = {}, {}
            for col in numeric_cols:
                vals = pd.to_numeric(profiles_df[col], errors="coerce").dropna()
                col_mins[col] = float(vals.min()) if len(vals) else 0.0
                col_maxs[col] = float(vals.max()) if len(vals) else 1.0
            for _, prof_row in profiles_df.iterrows():
                emo_key = str(prof_row["Emotion"])
                actual_vals[emo_key] = {}
                for feat_key, col_name in FEAT_COL_MAP.items():
                    if col_name not in prof_row.index:
                        continue
                    raw = pd.to_numeric(prof_row[col_name], errors="coerce")
                    if pd.isna(raw):
                        continue
                    raw = float(raw)
                    rng = col_maxs.get(col_name, 1.0) - col_mins.get(col_name, 0.0)
                    norm = (raw - col_mins.get(col_name, 0.0)) / rng if rng > 0 else 0.5
                    if col_name == "Mode":
                        level = "major" if raw > 0.5 else "minor"
                    elif norm >= 0.67:
                        level = "high"
                    elif norm >= 0.33:
                        level = "moderate"
                    else:
                        level = "low"
                    actual_vals[emo_key][feat_key] = (raw, level)

        # Top works / composers per emotion (для портретов)
        top_works_by_emo = {}
        top_composers_by_emo = {}
        try:
            from src.group_analysis import compute_group_profiles
            _gd = compute_group_profiles(df)
            if _gd:
                tw = _gd.get("top_works_by_emotion", {}) or {}
                tc = _gd.get("top_composers_by_emotion", {}) or {}
                for emo, lst in tw.items():
                    if lst:
                        top = lst[0]
                        top_works_by_emo[emo] = str(top.get("title", "") or "")
                for emo, lst in tc.items():
                    if lst:
                        top_composers_by_emo[emo] = str(lst[0].get("composer", "") or "")
        except Exception:
            pass

        # ── 4. 2×2 grid портретов ─────────────────────────────────────────
        portraits_title = QLabel("Портрет эмоции: ожидаемое vs реальное")
        portraits_title.setStyleSheet(f"font-size:14px; font-weight:700; color:{PRIMARY}; margin-top:4px;")
        portraits_title.setToolTip(
            "Ожидаемо — направление признака, предсказанное из исследований "
            "(Gabrielsson & Lindström 2001, Juslin & Sloboda 2010, Hunter 2010, Leman 2005). "
            "Реально — что получилось в найденных произведениях для этой EEG-эмоции."
        )
        root.addWidget(portraits_title)

        n_per_emotion = {}
        if "eeg_emotion" in df.columns:
            n_per_emotion = df.groupby("eeg_emotion").size().to_dict()

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(0, 0, 0, 0)
        positions = [("HVHA", 0, 0), ("HVLA", 0, 1), ("LVLA", 1, 0), ("LVHA", 1, 1)]
        for emo, row_, col_ in positions:
            card = _EmotionPortraitCard(
                emo=emo,
                n_participants=int(n_per_emotion.get(emo, 0)),
                top_work=top_works_by_emo.get(emo),
                top_composer=top_composers_by_emo.get(emo),
                expected_map=EMOTION_RESEARCH.get(emo, {}).get("expected", {}),
                actual_map=actual_vals.get(emo, {}),
            )
            grid.addWidget(card, row_, col_)
        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        root.addWidget(grid_wrap)

        # ── 5. Radar + Bars side-by-side ──────────────────────────────────
        radar_path = report_dir / "group_radar_chart.png"
        bars_path = report_dir / "group_feature_bars.png"
        if radar_path.exists() or bars_path.exists():
            charts_row = QHBoxLayout()
            charts_row.setSpacing(12)
            if radar_path.exists():
                charts_row.addWidget(_image_card(
                    "Радар признаков", radar_path, "Радар не найден"
                ), stretch=1)
            if bars_path.exists():
                charts_row.addWidget(_image_card(
                    "Наиболее различающиеся признаки", bars_path, "Bar chart не найден"
                ), stretch=1)
            root.addLayout(charts_row)

        root.addStretch()
        return widget

    def _build_summary_tab(self, df: pd.DataFrame) -> QWidget:
        """Executive summary + Signal coverage bars + Top works bar chart."""
        widget = QWidget()
        root = QVBoxLayout(widget)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        report_dir = Path(self._report_dir or "")
        metrics = self._load_metrics()

        if "music_match_score" in df.columns:
            ranked = df.sort_values("music_match_score", ascending=False).reset_index(drop=True)
        elif "combined_similarity" in df.columns:
            ranked = df.sort_values("combined_similarity", ascending=False).reset_index(drop=True)
        else:
            ranked = df.reset_index(drop=True)
        best_row = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)

        # ── 1. Executive summary — 3 крупные факта ────────────────────────
        exec_card = QFrame()
        exec_card.setObjectName("card")
        ex_lay = QVBoxLayout(exec_card)
        ex_lay.setContentsMargins(16, 14, 16, 14)
        ex_lay.setSpacing(10)

        exec_title = QLabel("Главное")
        exec_title.setStyleSheet(f"font-size:13px; font-weight:700; color:{PRIMARY};")
        ex_lay.addWidget(exec_title)

        # Fact 1: match rate
        match_rate = metrics.get("emotion_match_rate", 0.0)
        fact1 = QLabel(
            f"<span style='font-size:22px; font-weight:700; color:{ACCENT};'>{_fmt_pct(match_rate)}</span>"
            f"  произведений в top-K совпали по эмоции с EEG."
        )
        fact1.setTextFormat(Qt.RichText)
        fact1.setWordWrap(True)
        fact1.setStyleSheet(f"font-size:13px; color:{TEXT_PRIMARY};")
        ex_lay.addWidget(fact1)

        # Fact 2: best piece
        best_score = float(best_row.get("music_match_score", best_row.get("combined_similarity", 0.0)) or 0.0)
        best_piece = f"{best_row.get('composer', 'Unknown')} — {best_row.get('title', 'Unknown')}"
        fact2 = QLabel(
            f"Лучшее соответствие — <b>{escape(best_piece)}</b>  "
            f"<span style='font-size:16px; font-weight:700; color:{PRIMARY};'>({best_score:.3f})</span>"
        )
        fact2.setTextFormat(Qt.RichText)
        fact2.setWordWrap(True)
        fact2.setStyleSheet(f"font-size:13px; color:{TEXT_PRIMARY};")
        ex_lay.addWidget(fact2)

        # Fact 3: most stable emotion (highest consistency)
        stable_line = None
        cohort_path = report_dir / "cohort_emotion_summary.csv"
        if cohort_path.exists():
            try:
                cohort_df = pd.read_csv(cohort_path)
                if "consistency" in cohort_df.columns and not cohort_df.empty:
                    best = cohort_df.sort_values("consistency", ascending=False).iloc[0]
                    emo = str(best.get("eeg_emotion", "—"))
                    cons = float(best.get("consistency", 0.0) or 0.0)
                    if emo not in ("", "—") and cons > 0:
                        stable_line = QLabel(
                            f"Самая стабильная эмоция — <b>{escape(emo)}</b>  "
                            f"<span style='font-size:16px; font-weight:700; color:{PRIMARY};'>"
                            f"(consistency {cons*100:.0f}%)</span>"
                        )
            except Exception:
                stable_line = None
        if stable_line is not None:
            stable_line.setTextFormat(Qt.RichText)
            stable_line.setWordWrap(True)
            stable_line.setStyleSheet(f"font-size:13px; color:{TEXT_PRIMARY};")
            ex_lay.addWidget(stable_line)

        # Bottom stats line
        stats = QLabel(
            f"<span style='color:{TEXT_SECONDARY};'>"
            f"обработано: <b>{int(metrics.get('n_results', len(df)))}</b> результатов  ·  "
            f"Macro-F1: <b>{_fmt_score(metrics.get('macro_f1', 0.0))}</b>"
            f"</span>"
        )
        stats.setTextFormat(Qt.RichText)
        stats.setStyleSheet("font-size:11px;")
        ex_lay.addWidget(stats)

        root.addWidget(exec_card)

        # ── 2. Signal coverage — горизонтальные бары ─────────────────────
        coverage_title = QLabel("Как EEG-запись сжалась в мелодию")
        coverage_title.setStyleSheet(f"font-size:13px; font-weight:700; color:{PRIMARY};")
        coverage_title.setToolTip(
            "Длинная EEG-запись конвертируется в короткую музыкальную линию "
            "через извлечение мотивов; цифра ×N — коэффициент сжатия."
        )
        root.addWidget(coverage_title)

        melody_diag_df = pd.DataFrame()
        melody_diag_path = report_dir / "melody_diagnostics.csv"
        if melody_diag_path.exists():
            try:
                melody_diag_df = pd.read_csv(melody_diag_path)
            except Exception:
                melody_diag_df = pd.DataFrame()

        if not melody_diag_df.empty:
            for _, r in melody_diag_df.iterrows():
                root.addWidget(_SignalCoverageBar(
                    variant=str(r.get("variant", "")),
                    rec_min=float(r.get("recording_duration_sec", 0.0) or 0.0) / 60.0,
                    melody_sec=float(r.get("span_sec", 0.0) or 0.0),
                    note_count=int(r.get("note_count", 0) or 0),
                    silence_ratio=float(r.get("silence_ratio", 0.0) or 0.0),
                ))
        else:
            empty = QLabel("Диагностика EEG-мелодии недоступна.")
            empty.setStyleSheet(f"color:{TEXT_SECONDARY}; padding:10px;")
            root.addWidget(empty)

        # ── 3. Распределение эмоций в подобранной музыке ─────────────────
        emo_title = QLabel("Распределение эмоций в подобранной музыке")
        emo_title.setStyleSheet(f"font-size:13px; font-weight:700; color:{PRIMARY};")
        emo_title.setToolTip(
            "Для каждой из четырёх эмоций — сколько испытаний завершилось "
            "рекомендацией пьесы этого эмоционального класса, и насколько "
            "хорошо в среднем совпал подбор (music match score). "
            "Даёт быстрый ответ: \"какие эмоции метод вообще умеет находить "
            "в этом датасете и с каким качеством\"."
        )
        root.addWidget(emo_title)

        emo_card = QFrame()
        emo_card.setObjectName("card")
        e_lay = QVBoxLayout(emo_card)
        e_lay.setContentsMargins(14, 12, 14, 12)
        e_lay.setSpacing(8)

        from gui.styles import EMOTION_COLORS as _EC
        score_col = "music_match_score" if "music_match_score" in df.columns \
            else ("combined_similarity" if "combined_similarity" in df.columns else None)
        total = len(df)
        rows_data: list[tuple[str, int, float, float]] = []
        order = ["HVHA", "HVLA", "LVLA", "LVHA"]
        if total > 0 and "classical_emotion" in df.columns:
            for emo in order:
                sub = df[df["classical_emotion"] == emo]
                cnt = int(len(sub))
                share = cnt / total if total else 0.0
                avg = float(sub[score_col].mean()) if (score_col and cnt) else 0.0
                rows_data.append((emo, cnt, share, avg))
        max_share = max((r[2] for r in rows_data), default=1.0) or 1.0

        if not rows_data or total == 0:
            empty = QLabel("Нет данных о подобранных эмоциях.")
            empty.setStyleSheet(f"color:{TEXT_SECONDARY};")
            e_lay.addWidget(empty)
        else:
            for emo, cnt, share, avg in rows_data:
                full_lbl, _ = _EMOTION_FULL.get(emo, (emo, ""))
                ru_name = full_lbl.split(" — ")[0] if " — " in full_lbl else emo
                color = _EC.get(emo, PRIMARY)
                # Заголовок строки: код + русское имя
                head = QLabel(
                    f"<b style='color:{color};'>{emo}</b> "
                    f"<span style='color:{TEXT_SECONDARY};'>— {escape(ru_name)}</span>  "
                    f"<span style='color:{TEXT_PRIMARY};'>"
                    f"{cnt} ({share*100:.1f}%)</span>  "
                    f"<span style='color:{TEXT_MUTED};'>· ср. совпадение: "
                    f"{avg:.2f}</span>"
                )
                head.setTextFormat(Qt.RichText)
                head.setStyleSheet("font-size:11px;")
                e_lay.addWidget(head)

                # Бар — ширина пропорциональна доле эмоции
                bar_bg = QFrame()
                bar_bg.setFixedHeight(10)
                bar_bg.setStyleSheet(
                    f"background:{ROW_ALT_BG}; border:1px solid {BORDER_SOFT}; "
                    f"border-radius:5px;"
                )
                bar_row = QHBoxLayout()
                bar_row.setContentsMargins(0, 0, 0, 0)
                bar_row.setSpacing(0)
                bar_bg.setLayout(bar_row)
                if share > 0:
                    fg = QFrame()
                    # Доля эмоции в долях от максимума, stretch для остального
                    fg.setStyleSheet(
                        f"background:{color}; border-radius:5px;"
                    )
                    bar_row.addWidget(fg, stretch=max(1, int(round(share / max_share * 1000))))
                    bar_row.addStretch(max(1, int(round((1 - share / max_share) * 1000))))
                else:
                    bar_row.addStretch(1)
                e_lay.addWidget(bar_bg)

            note = QLabel(
                "Чем длиннее полоса, тем чаще метод относит подобранную музыку "
                "к этой эмоции. «Ср. совпадение» — средний music match score "
                "внутри класса: показывает, насколько уверенно признаки пьесы "
                "соответствуют целевой эмоции ЭЭГ."
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"font-size:11px; color:{TEXT_SECONDARY}; margin-top:4px;")
            e_lay.addWidget(note)

        root.addWidget(emo_card)

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

        # Сохраняем индекс активной вкладки и позицию прокрутки,
        # чтобы фильтр не сбрасывал пользователя в самый верх.
        prev_tab = self._tabs.currentIndex() if self._tabs is not None else 0
        prev_scroll = 0
        if self._tabs is not None and prev_tab >= 0:
            prev_widget = self._tabs.widget(prev_tab)
            if isinstance(prev_widget, QScrollArea):
                prev_scroll = prev_widget.verticalScrollBar().value()

        self._populate_tabs(df)

        # Восстанавливаем
        if 0 <= prev_tab < self._tabs.count():
            self._tabs.setCurrentIndex(prev_tab)
            cur_widget = self._tabs.widget(prev_tab)
            if isinstance(cur_widget, QScrollArea):
                # Отложенно, чтобы layout успел посчитать размеры
                from PySide6.QtCore import QTimer
                QTimer.singleShot(
                    0,
                    lambda w=cur_widget, v=prev_scroll: w.verticalScrollBar().setValue(v),
                )

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
