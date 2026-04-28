#!/usr/bin/env python3
"""
Главное окно приложения — мульти-страничная навигация через QStackedWidget.

Страницы:
  0 — WelcomePage  (история + кнопка «Новый анализ»)
  1 — LoadPage     (загрузка .eeg/.dat + параметры)
  2 — AnalysisPage (прогресс-бар + лог)
  3 — ResultsPage  (богатый card-based вид результатов)
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QMessageBox,
)

from gui.pages.welcome import WelcomePage
from gui.pages.load_page import LoadPage
from gui.pages.analysis_page import AnalysisPage
from gui.pages.results_page import ResultsPage
from gui.history import save_run


# Индексы страниц
PAGE_WELCOME = 0
PAGE_LOAD = 1
PAGE_ANALYSIS = 2
PAGE_RESULTS = 3


class MainWindow(QMainWindow):
    """Главное окно с навигацией между страницами."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Преобразование сигналов ЭЭГ в музыкальные структуры")
        self.resize(1200, 820)
        self.setMinimumSize(900, 600)

        # ── Центральный виджет ──
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        lay.addWidget(self._stack)

        # ── Страницы ──
        self._welcome = WelcomePage()
        self._load = LoadPage()
        self._analysis = AnalysisPage()
        self._results = ResultsPage()

        self._stack.addWidget(self._welcome)   # 0
        self._stack.addWidget(self._load)       # 1
        self._stack.addWidget(self._analysis)   # 2
        self._stack.addWidget(self._results)    # 3

        # ── Сигналы навигации ──
        self._welcome.new_analysis.connect(self._go_load)
        self._welcome.open_history.connect(self._open_history_entry)

        self._load.go_back.connect(self._go_welcome)
        self._load.start_analysis.connect(self._start_pipeline)

        self._analysis.cancel_requested.connect(self._cancel_pipeline)

        self._results.go_home.connect(self._go_welcome)
        self._results.new_analysis.connect(self._go_load)

        # Worker
        self._worker = None
        self._current_files: list[str] = []
        self._current_params: dict = {}
        self._cancelled = False  # флаг, чтобы finished_ok после cancel не перетирал UI

        self._go_welcome()

    # ------------------------------------------------------------------
    # Навигация
    # ------------------------------------------------------------------

    def _go_welcome(self):
        self._welcome.refresh()
        self._stack.setCurrentIndex(PAGE_WELCOME)

    def _go_load(self):
        # Восстанавливаем ранее введённые параметры, чтобы форма не сбрасывалась
        if self._current_params:
            self._load.apply_params(self._current_params)
        self._stack.setCurrentIndex(PAGE_LOAD)

    def _go_analysis(self):
        self._analysis.reset()
        self._stack.setCurrentIndex(PAGE_ANALYSIS)

    def _go_results(self):
        self._stack.setCurrentIndex(PAGE_RESULTS)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _start_pipeline(self, files: list, params: dict):
        """Запуск пайплайна с параметрами от LoadPage."""
        # Не допускаем повторный запуск, пока предыдущий worker жив
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "Анализ уже идёт",
                "Предыдущий анализ ещё не завершён. Дождитесь окончания или нажмите «Отмена».",
            )
            return

        self._current_files = files
        self._current_params = params
        self._cancelled = False
        self._load.set_running(True)
        self._go_analysis()

        from gui.worker import PipelineWorker

        # Определяем dataset_source из radio-кнопок LoadPage
        if params.get("only_emopia"):
            _ds = "emopia"
        elif params.get("only_maestro"):
            _ds = "maestro"
        else:
            _ds = params.get("dataset_source", "both")

        self._worker = PipelineWorker(
            eeg_files=files,
            max_classical=params.get("max_classical", 10),
            max_trials=params.get("max_trials", 5),
            top_k=params.get("top_k", 10),
            n_jobs=params.get("n_jobs"),
            dataset_source=_ds,
            only_emopia=params.get("only_emopia", False),
            match_emotions=params.get("match_emotions", False),
            analysis_mode=params.get("analysis_mode", "single"),
            window_size=params.get("window_size", 4.0),
            hop_size=params.get("hop_size", 2.0),
            max_seconds=params.get("max_seconds"),
            eeg_emotions=params.get("eeg_emotions"),
            seed=params.get("seed"),
            compare_modes=params.get("compare_modes", False),
            target_emotion=params.get("target_emotion"),
            manual_midi_path=params.get("manual_midi_path"),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_error.connect(self._on_error)
        self._worker.start()

    def _disconnect_worker(self):
        """Безопасно отвязывает сигналы worker'а перед уничтожением."""
        if self._worker is None:
            return
        for sig, slot in (
            (self._worker.progress, self._on_progress),
            (self._worker.log_message, self._on_log),
            (self._worker.finished_ok, self._on_finished),
            (self._worker.finished_error, self._on_error),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                # Сигнал уже отключён либо C++-объект уничтожен — это нормально
                pass

    def _cancel_pipeline(self):
        self._cancelled = True
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._analysis.mark_error("Отменено пользователем.")
            self._disconnect_worker()
            self._worker.quit()
            self._worker.wait(3000)
            self._worker.deleteLater()
            self._worker = None
        self._load.set_running(False)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, pct: int, text: str):
        self._analysis.set_progress(pct, text)

    def _on_log(self, msg: str):
        self._analysis.append_log(msg)

    def _on_finished(self, results_df, report_dir: str):
        # Если пользователь отменил прогон — поздний finished_ok игнорируем
        if self._cancelled:
            return
        self._analysis.mark_done()
        self._load.set_running(False)

        # Сохраняем в историю
        best_score = 0.0
        n_results = 0
        try:
            n_results = len(results_df)
            if "music_match_score" in results_df.columns:
                best_score = float(results_df["music_match_score"].max())
            elif "combined_similarity" in results_df.columns:
                best_score = float(results_df["combined_similarity"].max())
        except Exception:
            pass

        save_run(
            eeg_files=self._current_files,
            params=self._current_params,
            report_dir=report_dir,
            n_results=n_results,
            best_score=best_score,
        )

        # Показываем результаты
        self._results.load_results(results_df, report_dir)
        self._go_results()

    def _on_error(self, msg: str):
        if self._cancelled:
            return
        self._analysis.mark_error(msg)
        self._load.set_running(False)
        QMessageBox.critical(self, "Ошибка пайплайна", msg[:800])

    # ------------------------------------------------------------------
    # Open from history
    # ------------------------------------------------------------------

    def _open_history_entry(self, entry: dict):
        """Открывает результаты из записи истории."""
        report_dir = entry.get("report_dir", "")
        csv_path = Path(report_dir) / "comparison_results.csv"
        if not csv_path.exists():
            QMessageBox.warning(
                self, "Ошибка",
                f"Файл результатов не найден:\n{csv_path}\n\n"
                "Возможно, папка отчёта была удалена.",
            )
            return
        self._results.load_from_history(entry)
        self._go_results()

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._disconnect_worker()
            self._worker.quit()
            self._worker.wait(2000)
            self._worker.deleteLater()
            self._worker = None
        super().closeEvent(ev)
