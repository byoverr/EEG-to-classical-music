#!/usr/bin/env python3
"""
Точка входа для десктопного приложения EEG → Classical Music.

Запуск:
    python -m gui.app
    или
    python gui/app.py
"""
import sys
import os
from pathlib import Path

# Гарантируем, что корень проекта в sys.path
_app_dir = Path(__file__).resolve().parent
_project_root = _app_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Для PyInstaller: если запущены из bundle, исправляем пути
if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).parent)


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from PySide6.QtCore import Qt

    # High-DPI support (для macOS Retina и Windows HiDPI)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("EEG Emotion Validation")
    app.setOrganizationName("SPbGETU LETI")
    app.setApplicationVersion("2.0.0")

    # Базовый шрифт
    font = QFont()
    font.setPointSize(13)
    app.setFont(font)

    # Единый stylesheet из gui/styles.py
    from gui.styles import GLOBAL_STYLESHEET
    app.setStyleSheet(GLOBAL_STYLESHEET)

    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
