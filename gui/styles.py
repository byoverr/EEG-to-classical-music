"""
Единый файл стилей для GUI приложения EEG → Classical Music.
"""
from __future__ import annotations

# Брендовые цвета
PRIMARY = "#1a73e8"
PRIMARY_DARK = "#1558b0"
ACCENT = "#34a853"
DANGER = "#d93025"
WARNING = "#f9ab00"
TEXT_PRIMARY = "#202124"
TEXT_SECONDARY = "#3c4043"   # поднят контраст (было #5f6368 — на грани WCAG AA)
TEXT_MUTED = "#5f6368"        # для совсем «фоновых» подписей, где контраст не критичен
BG_PAGE = "#f8f9fa"
BG_CARD = "#ffffff"
BORDER = "#dadce0"
SHADOW = "rgba(0,0,0,0.08)"

# Design tokens — повторяющиеся акцентные подложки (избавляемся от хардкода в страницах)
CARD_HIGHLIGHT_BG = "#f8fbff"   # лёгкая голубая подложка: score-box, lead-callout
TABLE_HEADER_BG = "#f4f7fb"     # шапки таблиц
ROW_ALT_BG = "#fbfcfd"          # зебра-строки
BORDER_SOFT = "#eceff3"         # внутренние разделители
HOVER_BG = "#e8f0fe"            # ховер primary-элементов

# Индикаторы соответствия (ожидаемое vs реальное)
MATCH_OK = "#34a853"
MATCH_WARN = "#f9ab00"
MATCH_BAD = "#d93025"
MATCH_NONE = "#9aa0a6"

# Spacing scale (пиксели, одинаково для layout.setSpacing / setContentsMargins)
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 12
SPACING_LG = 16
SPACING_XL = 24

# Цвета эмоций
EMOTION_COLORS = {
    "HVHA": "#e8453c",   # красный — высокая энергия + позитив
    "HVLA": "#4285f4",   # синий — спокойствие + позитив
    "LVLA": "#9aa0a6",   # серый — грусть
    "LVHA": "#f9ab00",   # жёлтый — напряжение
}

GLOBAL_STYLESHEET = f"""
/* ── Общие ──────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {BG_PAGE};
    color: {TEXT_PRIMARY};
}}
QWidget#centralWidget {{
    background-color: {BG_PAGE};
}}
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel[class="subtitle"] {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

/* ── Кнопки ─────────────────────────────────── */
QPushButton#primary {{
    background-color: {PRIMARY};
    color: white;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    min-height: 38px;
}}
QPushButton#primary:hover {{ background-color: {PRIMARY_DARK}; }}
QPushButton#primary:disabled {{ background-color: #bbb; }}

QPushButton#danger {{
    background-color: {DANGER};
    color: white;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
}}
QPushButton#danger:hover {{ background-color: #a5281b; }}
QPushButton#danger:disabled {{ background-color: #bbb; }}

QPushButton#secondary {{
    background-color: transparent;
    color: {PRIMARY};
    font-size: 13px;
    font-weight: 600;
    border: 1px solid {PRIMARY};
    border-radius: 8px;
    padding: 8px 20px;
}}
QPushButton#secondary:hover {{ background-color: #e8f0fe; }}

QPushButton#link {{
    background: transparent;
    color: {PRIMARY};
    border: none;
    font-size: 13px;
    text-decoration: underline;
    padding: 4px;
}}
QPushButton#link:hover {{ color: {PRIMARY_DARK}; }}

/* ── Карточки ───────────────────────────────── */
QFrame#card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 10px;
}}
QFrame#card:hover {{
    border-color: {PRIMARY};
}}

QFrame#metricCard {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 2px 4px;
    min-width: 50px;
    min-height: 38px;
}}

QFrame#summaryCard {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 12px;
    min-width: 140px;
    max-width: 260px;
}}

/* ── Прогресс ───────────────────────────────── */
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    font-size: 12px;
    background: #f0f0f0;
    min-height: 26px;
}}
QProgressBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {PRIMARY}, stop:1 {ACCENT}
    );
    border-radius: 7px;
}}

/* ── Лог / консоль ──────────────────────────── */
QTextEdit#console {{
    background: #f5f6f8;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px;
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 11px;
}}

/* ── Таблица ────────────────────────────────── */
QTableWidget {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: #eee;
    font-size: 12px;
}}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{
    background: #f0f2f5;
    font-weight: 600;
    border: none;
    border-bottom: 2px solid {BORDER};
    padding: 6px 8px;
    font-size: 12px;
}}

/* ── Tabs ───────────────────────────────────── */
QTabWidget {{
    background-color: {BG_CARD};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_CARD};
}}
QTabBar {{
    background-color: {BG_CARD};
}}
QTabBar::tab {{
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid {BORDER};
    border-bottom: 2px solid transparent;
    color: {TEXT_PRIMARY};
    background-color: #f0f2f5;
    margin-right: 2px;
    border-radius: 6px 6px 0 0;
}}
QTabBar {{
    qproperty-expanding: 0;
}}
QTabBar::tab:selected {{
    color: {PRIMARY};
    border-color: {PRIMARY};
    border-bottom-color: {PRIMARY};
    font-weight: 600;
    background-color: {BG_CARD};
}}
QTabBar::tab:hover {{
    color: {PRIMARY};
    background-color: #e8f0fe;
    border-radius: 6px 6px 0 0;
}}

/* ── Scroll areas / GroupBoxes ──────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background-color: {BG_PAGE};
}}
QGroupBox {{
    font-weight: 600;
    font-size: 13px;
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 10px;
    padding: 14px;
    background: {BG_CARD};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: {TEXT_PRIMARY};
}}

/* ── Inputs ─────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    background-color: white;
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 6px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: white;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: #e8f0fe;
    selection-color: {TEXT_PRIMARY};
    padding: 4px;
    border-radius: 6px;
}}

/* ── Styled filter combo (top bar) ──────────── */
QComboBox#filterCombo {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    min-width: 130px;
}}
QComboBox#filterCombo:hover {{
    border-color: {PRIMARY};
}}
QComboBox#filterCombo:focus {{
    border-color: {PRIMARY};
    background-color: #e8f0fe;
}}
QComboBox#filterCombo::drop-down {{
    border: none;
    padding-right: 8px;
}}
QComboBox#filterCombo::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {PRIMARY};
    margin-right: 8px;
}}

/* ── Drop ares ──────────────────────────────── */
QListWidget#fileDrop {{
    border: 2px dashed {BORDER};
    border-radius: 10px;
    padding: 10px;
    background: #fafbfc;
    font-size: 12px;
}}
QListWidget#fileDrop:hover {{ border-color: {PRIMARY}; }}

/* ── Tooltip ────────────────────────────────── */
QToolTip {{
    background: #333;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
}}
"""
