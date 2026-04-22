#!/usr/bin/env python3
"""
Фоновый поток (QThread) для выполнения пайплайна EEG → MIDI → Сравнение.

Адаптирует логику scripts/run_comparison.py с прогресс-сигналами для GUI.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

# ---------------------------------------------------------------------------
# Значения этапов (для прогресс-бара)
# ---------------------------------------------------------------------------
STAGE_LOAD_DATASETS = (0, "Загрузка датасетов…")
STAGE_PSEUDO_LABEL = (10, "Подготовка эмоциональных меток…")
STAGE_CLASSICAL_WINDOWS = (20, "Расчёт окон классики…")
STAGE_EEG_PROCESSING = (40, "Обработка EEG и генерация мелодий…")
STAGE_COMPILE_RESULTS = (75, "Проверка эмоционального соответствия…")
STAGE_PREPARE_RESULTS = (85, "Подготовка результатов…")
STAGE_GENERATE_EXPORTS = (92, "Экспорт метрик и графиков…")
STAGE_DONE = (100, "Готово!")


class PipelineWorker(QThread):
    """Выполняет пайплайн в фоновом потоке.

    Signals
    -------
    progress(int, str)
        (process_percent, stage_description)
    log_message(str)
        Текстовое сообщение для лога в GUI.
    finished_ok(pd.DataFrame, str)
        (results_df, report_dir)  — успешное завершение.
    finished_error(str)
        Сообщение об ошибке.
    """

    progress = Signal(int, str)
    log_message = Signal(str)
    finished_ok = Signal(object, str)      # DataFrame, report_dir
    finished_error = Signal(str)

    def __init__(
        self,
        eeg_files: list[str],
        max_participants: int = 3,
        max_trials: int = 5,
        max_classical: int = 10,
        top_k: int = 10,
        n_jobs: Optional[int] = None,
        only_emopia: bool = False,
        dataset_source: Optional[str] = None,
        match_emotions: bool = False,
        analysis_mode: str = "single",
        window_size: float = 4.0,
        hop_size: float = 2.0,
        max_seconds: Optional[float] = None,
        eeg_emotions: Optional[dict[str, str | None]] = None,
        seed: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.eeg_files = eeg_files
        self.max_participants = max_participants
        self.max_trials = max_trials
        self.max_classical = max_classical
        self.top_k = top_k
        self.n_jobs = n_jobs
        # dataset_source приоритетнее, но если не задан — читаем из only_emopia для обратной совместимости.
        # Значения: "both" | "emopia" | "maestro"
        if dataset_source is None:
            dataset_source = "emopia" if only_emopia else "both"
        if dataset_source not in ("both", "emopia", "maestro"):
            dataset_source = "both"
        self.dataset_source = dataset_source
        self.only_emopia = (dataset_source == "emopia")
        self.match_emotions = match_emotions
        self.analysis_mode = analysis_mode
        self.window_size = window_size
        self.hop_size = hop_size
        self._max_seconds = max_seconds
        self._eeg_emotions = eeg_emotions or {}  # {filepath: emotion_str | None}
        self.seed = seed
        self._cancelled = False

    # ------------------------------------------------------------------
    def cancel(self):
        self._cancelled = True

    # ------------------------------------------------------------------
    @staticmethod
    def _create_unique_run_dir(runs_dir: Path) -> Path:
        """Создаёт уникальную папку run_NNN в runs_dir."""
        runs_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(runs_dir.glob("run_*"))
        max_num = 0
        for d in existing:
            try:
                num = int(d.name.split("_", 1)[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
        new_id = f"run_{max_num + 1:03d}"
        new_dir = runs_dir / new_id
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _emit(self, pct: int, text: str):
        self.progress.emit(pct, text)

    # ------------------------------------------------------------------
    def run(self):  # noqa: C901 — сознательно длинный метод
        """Основная точка входа потока."""
        try:
            self._run_pipeline()
        except Exception as exc:
            tb = traceback.format_exc()
            self.finished_error.emit(f"{exc}\n\n{tb}")

    # ------------------------------------------------------------------
    def _run_pipeline(self):  # noqa: C901
        import warnings
        warnings.filterwarnings("ignore")

        import random
        import shutil
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # --- Импорты проекта ---
        from src.config import (
            DEAP_DIR, MAESTRO_DIR, EMOPIA_DIR,
            EEG_THRESHOLD_LOW_STD, EEG_THRESHOLD_HIGH_STD,
            EEG_MIN_WAVE_DURATION, EEG_MAX_WAVE_DURATION, EEG_SCALE_KEY,
            EMOPIA_MAX_TRACKS, RUNS_DIR, USE_PSEUDO_LABELING,
            MATCH_FRAGMENT_DURATION, MAESTRO_PSEUDO_LABELS_PATH,
            RANDOM_SEED,
        )
        from src.deap_loader import load_deap_participant_data, get_emotion_labels
        from src.maestro_loader import get_maestro_midi_files, get_maestro_metadata
        from src.emopia_loader import (
            get_emopia_midi_files, get_emopia_metadata, deap_to_emotion_quadrant,
            get_all_emotions, get_emopia_track_info,
        )
        from src.track_features import (
            load_feature_cache, save_feature_cache,
            get_or_compute_features, features_to_vector,
        )
        from src.eeg_preprocessing import prepare_signal_data
        from src.eeg_processing import (
            detect_wave_motifs,
            detect_wave_motifs_segmented,
            map_motifs_to_adsr_sounds,
            compress_timed_events,
        )
        from src.midi_utils import (
            create_midi_with_precise_timing, extract_melody_with_time,
            create_midi_from_notes, create_comparison_midi,
        )
        from src.evaluation import (
            add_hypothesis_scores,
            compute_hypothesis_metrics,
            save_hypothesis_artifacts,
        )

        # Параметры окна/хопа передаём через локальные переменные ниже.
        # НЕ пишем в src.config — глобальная мутация ломает параллельные прогоны.

        # --- Фиксация seed для воспроизводимости (shuffle и np.random) ---
        effective_seed = int(self.seed) if self.seed is not None else int(RANDOM_SEED)
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        self._log(f"Seed: {effective_seed}")
        self._log(f"Источник классики: {self.dataset_source}")

        # Вспомогательные, скопированы из run_comparison.py -----------------
        from scripts.run_comparison import (
            create_eeg_variants, _process_classical_file,
            _process_trial, _init_worker,
        )

        if self._cancelled:
            return

        # ===== [1/5] Загрузка датасетов ====================================
        self._emit(*STAGE_LOAD_DATASETS)
        self._log("═" * 50)
        self._log("EEG → Emotion Validation Pipeline")
        self._log(f"Режим анализа: {self.analysis_mode}")
        self._log("═" * 50)

        run_dir = self._create_unique_run_dir(RUNS_DIR)
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Делим лимит между датасетами, если берём оба источника
        use_maestro = self.dataset_source in ("both", "maestro")
        use_emopia = self.dataset_source in ("both", "emopia")
        if self.dataset_source == "both":
            # поровну, остаток — в maestro
            emopia_limit = self.max_classical // 2
            maestro_limit = self.max_classical - emopia_limit
        elif self.dataset_source == "emopia":
            emopia_limit, maestro_limit = self.max_classical, 0
        else:  # "maestro"
            emopia_limit, maestro_limit = 0, self.max_classical

        maestro_files: list[str] = []
        if use_maestro and maestro_limit > 0:
            self._log("[MAESTRO] Загрузка…")
            all_maestro = get_maestro_midi_files(str(MAESTRO_DIR), max_files=None)
            random.shuffle(all_maestro)
            maestro_files = all_maestro[: maestro_limit]
            self._log(f"[MAESTRO] {len(maestro_files)} произведений")

        emopia_files: list[str] = []
        if use_emopia and emopia_limit > 0:
            try:
                self._log("[EMOPIA] Загрузка…")
                all_emopia = get_emopia_midi_files(str(EMOPIA_DIR), max_files=EMOPIA_MAX_TRACKS)
                random.shuffle(all_emopia)
                emopia_files = all_emopia[: emopia_limit]
                self._log(f"[EMOPIA] {len(emopia_files)} произведений")
            except Exception as e:
                self._log(f"[EMOPIA] Ошибка: {e}")

        all_classical = maestro_files + emopia_files
        self._log(f"Всего произведений: {len(all_classical)}")
        if not all_classical:
            self.finished_error.emit("Не найдены MIDI файлы для сравнения.")
            return

        # Создаём словари meta --
        classical_dict = {}
        classical_meta_map = {}
        for path in maestro_files:
            meta = get_maestro_metadata(Path(path).name)
            composer = meta.get("composer", "Unknown")
            title = meta.get("title", Path(path).stem)[:30]
            track_id = meta.get("track_id", Path(path).stem)
            name = f"maestro|{track_id}|{composer} - {title}"
            classical_dict[name] = path
            classical_meta_map[name] = {
                "dataset": "maestro", "track_id": track_id,
                "composer": composer, "title": title,
                "emotion": meta.get("emotion"),
                "emotion_source": meta.get("emotion_source"),
            }

        for path in emopia_files:
            track_id = Path(path).stem
            meta = get_emopia_metadata(track_id)
            emotion = meta.get("emotion", "Unknown")
            info = get_emopia_track_info(str(EMOPIA_DIR), track_id)
            title_e = info.get("title") or track_id
            uploader = info.get("uploader") or "YouTube"
            clip_idx = info.get("clip_idx")
            clip_suffix = f" (clip {clip_idx})" if clip_idx is not None else ""
            name = f"emopia|{track_id}|{uploader} - {title_e}{clip_suffix} [{emotion}]"
            classical_dict[name] = path
            classical_meta_map[name] = {
                "dataset": "emopia", "track_id": track_id,
                "composer": uploader, "title": f"{title_e}{clip_suffix}",
                "emotion": emotion, "emotion_source": "ground_truth",
            }

        if self._cancelled:
            return

        # ===== [1b] Pseudo-labeling ========================================
        self._emit(*STAGE_PSEUDO_LABEL)
        maestro_pseudo_emotions = {}
        if USE_PSEUDO_LABELING and emopia_files:
            self._log("Pseudo-labeling MAESTRO…")
            if MAESTRO_PSEUDO_LABELS_PATH.exists():
                try:
                    pseudo_df = pd.read_csv(MAESTRO_PSEUDO_LABELS_PATH)
                    for _, row in pseudo_df.iterrows():
                        maestro_pseudo_emotions[str(row["track_id"])] = {
                            "emotion": row.get("emotion"),
                            "confidence": float(row.get("confidence", 0)),
                        }
                    self._log(f"Загружены псевдо-метки: {len(maestro_pseudo_emotions)}")
                except Exception as e:
                    self._log(f"Не удалось загрузить псевдо-метки: {e}")

            feature_cache_path = report_dir / "feature_cache.json"
            feature_cache = load_feature_cache(feature_cache_path)
            emopia_vectors, emopia_emotions_list, emopia_paths_list = [], [], []

            if not maestro_pseudo_emotions:
                for path in emopia_files:
                    feats = get_or_compute_features(path, feature_cache)
                    if feats is None:
                        continue
                    emopia_vectors.append(features_to_vector(feats))
                    emopia_paths_list.append(path)
                    emopia_emotions_list.append(
                        get_emopia_metadata(Path(path).stem).get("emotion", None)
                    )

            if emopia_vectors:
                emopia_matrix = np.vstack(emopia_vectors)
                mean = emopia_matrix.mean(axis=0)
                std = emopia_matrix.std(axis=0) + 1e-8
                emopia_matrix = (emopia_matrix - mean) / std

                for path in maestro_files:
                    feats = get_or_compute_features(path, feature_cache)
                    if feats is None:
                        continue
                    vec = (features_to_vector(feats) - mean) / std
                    dists = np.linalg.norm(emopia_matrix - vec, axis=1)
                    best_idx = int(np.argmin(dists))
                    best_emotion = emopia_emotions_list[best_idx]
                    best_dist = float(dists[best_idx])
                    confidence = 1.0 / (1.0 + best_dist)
                    maestro_pseudo_emotions[Path(path).stem] = {
                        "emotion": best_emotion, "confidence": confidence,
                    }

            save_feature_cache(feature_cache_path, feature_cache)

        if self._cancelled:
            return

        # ===== [2/5] Окна классических произведений ========================
        self._emit(*STAGE_CLASSICAL_WINDOWS)
        self._log("Расчёт окон классических произведений…")
        import music21 as m21
        from src.MIDIComparator import extract_window_features

        COMPARISON_WINDOW_SIZE = self.window_size
        COMPARISON_HOP_SIZE = self.hop_size

        classical_windows_cache = {}
        n_jobs = self.n_jobs or max(1, multiprocessing.cpu_count() - 1)

        total_classical = len(classical_dict)
        completed = 0
        if n_jobs > 1 and total_classical > 3:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(_process_classical_file, (name, path, COMPARISON_WINDOW_SIZE, COMPARISON_HOP_SIZE)): name
                    for name, path in classical_dict.items()
                }
                for future in as_completed(futures):
                    name, windows, error = future.result()
                    completed += 1
                    if not error:
                        classical_windows_cache[name] = windows
                    else:
                        self._log(f"  Ошибка {name[:40]}: {error}")
                    pct = int(20 + 20 * completed / max(total_classical, 1))
                    self._emit(pct, f"Классика: {completed}/{total_classical}")
                    if self._cancelled:
                        return
        else:
            for name, path in classical_dict.items():
                try:
                    midi = m21.converter.parse(path)
                    total_dur = midi.duration.quarterLength
                    windows = []
                    st = 0.0
                    wid = 0
                    while st + COMPARISON_WINDOW_SIZE <= total_dur:
                        et = st + COMPARISON_WINDOW_SIZE
                        ws = midi.flatten().getElementsByOffset(
                            st, et, includeEndBoundary=False,
                            mustFinishInSpan=False, mustBeginInSpan=True,
                        )
                        feats = extract_window_features(ws, st, et)
                        if feats and feats.get("note_count", 0) >= 8:
                            feats["window_id"] = wid
                            feats["start_time"] = st
                            feats["end_time"] = et
                            feats["source"] = name
                            windows.append(feats)
                            wid += 1
                        st += COMPARISON_HOP_SIZE
                    classical_windows_cache[name] = windows
                except Exception as e:
                    self._log(f"  Ошибка {name[:40]}: {e}")
                completed += 1
                pct = int(20 + 20 * completed / max(total_classical, 1))
                self._emit(pct, f"Классика: {completed}/{total_classical}")
                if self._cancelled:
                    return

        total_windows = sum(len(w) for w in classical_windows_cache.values())
        self._log(f"Кэшировано: {total_windows} окон")

        if self._cancelled:
            return

        # ===== [3/5] Обработка EEG триалов =================================
        self._emit(*STAGE_EEG_PROCESSING)
        self._log("Обработка EEG триалов…")

        eeg_midi_dir = run_dir / "eeg_midi"
        eeg_midi_dir.mkdir(parents=True, exist_ok=True)

        # Определяем файлы участников
        if self.eeg_files:
            participant_files = [Path(f) for f in self.eeg_files]
        else:
            participant_files = sorted(DEAP_DIR.glob("s*.dat"))[: self.max_participants]

        # ── Разделяем .eeg и .dat файлы ──
        eeg_neurosoft = [p for p in participant_files if p.suffix.lower() == ".eeg"]
        dat_deap = [p for p in participant_files if p.suffix.lower() != ".eeg"]

        trial_tasks = []
        all_results = []

        # ── [3a] Обработка .eeg файлов через neurosoft_loader ──
        if eeg_neurosoft:
            self._log(f"Neurosoft .eeg файлов: {len(eeg_neurosoft)}")
            neuro_results = self._process_neurosoft_files(
                eeg_neurosoft, eeg_midi_dir, classical_windows_cache,
                classical_dict, classical_meta_map,
            )
            all_results.extend(neuro_results)
            self._log(f"Результатов от .eeg: {len(neuro_results)}")

        if self._cancelled:
            return

        # ── [3b] Обработка .dat файлов через DEAP (как раньше) ──
        if dat_deap:
            for pf in dat_deap[: self.max_participants]:
                for trial_idx in range(min(self.max_trials, 40)):
                    trial_tasks.append(
                        (str(pf), trial_idx, eeg_midi_dir, self.match_emotions,
                         COMPARISON_WINDOW_SIZE, COMPARISON_HOP_SIZE)
                    )

        self._log(f"DEAP триалов: {len(trial_tasks)}")
        total_tasks = len(trial_tasks)

        # Устанавливаем глобальный кэш для worker'ов
        import scripts.run_comparison as _rc
        _rc._classical_cache = classical_windows_cache
        _rc._classical_path_map = classical_dict
        _rc._classical_meta_map = classical_meta_map

        if n_jobs > 1 and total_tasks > 1:
            # ThreadPoolExecutor: threads share _classical_cache in memory — no spawn/pickling issues
            from concurrent.futures import ThreadPoolExecutor
            completed_t = 0
            with ThreadPoolExecutor(max_workers=min(n_jobs, total_tasks)) as executor:
                futures = {executor.submit(_process_trial, task): task for task in trial_tasks}
                for future in as_completed(futures):
                    pid, tidx, results, error = future.result()
                    completed_t += 1
                    if error:
                        self._log(f"  {pid}/trial{tidx}: ошибка — {error}")
                    else:
                        all_results.extend(results)
                        if results:
                            best = max(results, key=lambda x: x["combined_similarity"])
                            self._log(
                                f"  [{completed_t}/{total_tasks}] {pid}/trial{tidx}: "
                                f"{best['classical_piece'][:30]} ({best['combined_similarity']:.3f})"
                            )
                    pct = int(40 + 35 * completed_t / max(total_tasks, 1))
                    self._emit(pct, f"EEG: {completed_t}/{total_tasks}")
                    if self._cancelled:
                        return
        else:
            for i, task in enumerate(trial_tasks):
                pid, tidx, results, error = _process_trial(task)
                if not error:
                    all_results.extend(results)
                    if results:
                        best = max(results, key=lambda x: x["combined_similarity"])
                        self._log(
                            f"  [{i+1}/{total_tasks}] {pid}/trial{tidx}: "
                            f"{best['classical_piece'][:30]} ({best['combined_similarity']:.3f})"
                        )
                else:
                    self._log(f"  {pid}/trial{tidx}: ошибка — {error}")
                pct = int(40 + 35 * (i + 1) / max(total_tasks, 1))
                self._emit(pct, f"EEG: {i+1}/{total_tasks}")
                if self._cancelled:
                    return

        if not all_results:
            self.finished_error.emit("Нет результатов.")
            return

        # ===== [4/5] Компиляция результатов ================================
        self._emit(*STAGE_COMPILE_RESULTS)
        self._log("Формирование результатов…")

        results_df = pd.DataFrame(all_results)

        # Resolve emotions
        if "classical_dataset" in results_df.columns:
            def _resolve_emotion(row):
                ds = row.get("classical_dataset", "maestro")
                tid = row.get("classical_track_id", "")
                if ds == "emopia" and tid:
                    m = get_emopia_metadata(tid)
                    return m.get("emotion"), "ground_truth"
                if ds == "maestro" and tid:
                    m = get_maestro_metadata(row.get("classical_midi_path", tid))
                    if m.get("emotion"):
                        return m.get("emotion"), m.get("emotion_source") or "predicted"
                    if USE_PSEUDO_LABELING and tid:
                        pseudo = maestro_pseudo_emotions.get(tid)
                        if pseudo:
                            return pseudo.get("emotion"), "predicted"
                return None, None

            emo = results_df.apply(lambda r: _resolve_emotion(r), axis=1)
            results_df["classical_emotion"] = emo.apply(lambda x: x[0])
            results_df["classical_emotion_source"] = emo.apply(lambda x: x[1])

        results_df["emotion_match"] = None
        if {"eeg_emotion", "classical_emotion"}.issubset(results_df.columns):
            mask = (
                results_df["eeg_emotion"].notna()
                & results_df["classical_emotion"].notna()
            )
            results_df.loc[mask, "emotion_match"] = (
                results_df.loc[mask, "eeg_emotion"].astype(str)
                == results_df.loc[mask, "classical_emotion"].astype(str)
            )

        results_df = add_hypothesis_scores(results_df)
        ranking_col = "music_match_score" if "music_match_score" in results_df.columns else "combined_similarity"
        if ranking_col in results_df.columns:
            results_df = results_df.sort_values(ranking_col, ascending=False).reset_index(drop=True)

        csv_path = report_dir / "comparison_results.csv"
        results_df.to_csv(csv_path, index=False)
        self._log(f"CSV: {csv_path}")

        metrics, confusion, cohort_df, feature_summary = compute_hypothesis_metrics(
            results_df,
            top_k=self.top_k,
            analysis_mode=self.analysis_mode,
        )
        save_hypothesis_artifacts(report_dir, metrics, confusion, cohort_df, feature_summary)

        # ── Group Analysis (этапы 3-4: группировка по эмоциям) ──
        try:
            from src.group_analysis import compute_group_profiles, save_group_analysis_artifacts
            self._log("Групповой анализ по эмоциям…")
            group_data = compute_group_profiles(results_df, top_k=self.top_k)
            if group_data:
                ga_files = save_group_analysis_artifacts(report_dir, group_data)
                self._log(f"Group analysis: {len(ga_files)} artifacts saved")
                for emo in group_data.get("emotions", []):
                    n = group_data["n_per_emotion"].get(emo, 0)
                    works = group_data.get("top_works_by_emotion", {}).get(emo, [])
                    top_work = works[0]["title"] if works else "—"
                    self._log(f"  {emo} (n={n}): top work = {top_work}")
        except Exception as e:
            self._log(f"Group analysis warning: {e}")

        if metrics:
            self._log(
                "Emotion metrics: "
                f"music={metrics.get('best_music_match_score', 0.0):.3f}, "
                f"match={metrics.get('emotion_match_rate', 0.0):.3f}, "
                f"top_k={metrics.get('top_k_accuracy', 0.0):.3f}, "
                f"macro_f1={metrics.get('macro_f1', 0.0):.3f}"
            )

        top_results = results_df.head(self.top_k)

        if self._cancelled:
            return

        # ===== [5/5] Подготовка результатов + MIDI фрагменты ===============
        self._emit(*STAGE_PREPARE_RESULTS)
        self._log("Подготовка материалов результатов…")

        matches_out = report_dir / "matches"
        matches_out.mkdir(parents=True, exist_ok=True)

        # Use the actual comparison window size for display fragments
        window_size = max(COMPARISON_WINDOW_SIZE, MATCH_FRAGMENT_DURATION)
        comp_rows = self._build_comp_rows(
            top_results, classical_dict, classical_meta_map,
            maestro_pseudo_emotions, matches_out, window_size,
        )
        comp_df = pd.DataFrame(comp_rows)

        # Дедупликация
        dedup_cols = ["composer", "title", "eeg_emotion", "trial", "variant"]
        if all(c in comp_df.columns for c in dedup_cols):
            sort_col = "music_match_score" if "music_match_score" in comp_df.columns else "combined_similarity"
            if sort_col in comp_df.columns:
                comp_df = comp_df.sort_values(sort_col, ascending=False)
            comp_df = comp_df.drop_duplicates(subset=dedup_cols, keep="first")

        # Сохраняем display_results.json (включая pitches и MIDI пути)
        try:
            comp_df.to_json(
                report_dir / "display_results.json", orient="records",
                force_ascii=False, indent=2,
            )
        except Exception as e:
            self._log(f"Ошибка сохранения display JSON: {e}")

        # ===== Экспорт графиков и CSV ======================================
        self._emit(*STAGE_GENERATE_EXPORTS)
        self._export_charts(results_df, report_dir)

        self._emit(*STAGE_DONE)
        self._log("═" * 50)
        self._log("Пайплайн завершён!")
        # Отправляем comp_df (с pitches и MIDI путями) для GUI
        display_df = comp_df if not comp_df.empty else results_df
        self.finished_ok.emit(display_df, str(report_dir))

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _process_neurosoft_files(
        self,
        eeg_files: list,
        eeg_midi_dir: Path,
        classical_windows_cache: dict,
        classical_dict: dict,
        classical_meta_map: dict,
    ) -> list[dict]:
        """Обрабатывает .eeg (Neurosoft) файлы: загрузка → signal_data → MIDI → сравнение."""
        import music21 as m21
        from src.neurosoft_loader import load_neurosoft_eeg, prepare_neurosoft_signal_data
        from src.eeg_processing import (
            detect_wave_motifs,
            detect_wave_motifs_segmented,
            map_motifs_to_adsr_sounds,
            compress_timed_events,
        )
        from src.midi_utils import create_midi_with_precise_timing
        from src.config import (
            EEG_THRESHOLD_LOW_STD, EEG_THRESHOLD_HIGH_STD,
            EEG_MIN_WAVE_DURATION, EEG_MAX_WAVE_DURATION, EEG_SCALE_KEY,
        )
        from src.MIDIComparator import (
            extract_window_features, interval_similarity,
            pitch_class_similarity, contour_similarity_strict,
            dtw_distance, dynamic_range_similarity,
        )
        from src.emopia_loader import deap_to_emotion_quadrant
        from src.config import EMOTION_THRESHOLD
        import pandas as pd

        max_sec = getattr(self, '_max_seconds', None)
        # Если пользователь не указал ограничение — берём всю запись
        if max_sec is not None:
            self._log(f"Ограничение EEG: {max_sec:.0f} сек")
        else:
            self._log("EEG: используется вся запись (без ограничения)")

        all_results = []
        total_files = len(eeg_files)

        for fi, eeg_path in enumerate(eeg_files):
            if self._cancelled:
                return all_results

            participant_id = eeg_path.stem
            self._log(f"[EEG] {participant_id} ({fi+1}/{total_files})")

            try:
                info, signals = load_neurosoft_eeg(
                    str(eeg_path), eeg_only=True, max_seconds=max_sec,
                )
                srate = info.get("srate", 250)
                self._log(f"  Каналов: {signals.shape[0]}, сэмплов: {signals.shape[1]}, частота: {srate} Гц")
            except Exception as e:
                self._log(f"  Ошибка чтения {eeg_path.name}: {e}")
                import traceback
                self._log(traceback.format_exc())
                continue

            try:
                signal_data = prepare_neurosoft_signal_data(signals, srate)
            except Exception as e:
                self._log(f"  Ошибка предобработки: {e}")
                import traceback
                self._log(traceback.format_exc())
                continue

            def _prepare_analysis_signal(raw_signal: np.ndarray) -> np.ndarray:
                raw = np.asarray(raw_signal)
                bad_mask = ~np.isfinite(raw)
                bad_count = int(bad_mask.sum())
                if bad_count > 0:
                    # Сигналим, что в сыром сигнале были NaN/Inf — это маскировка проблем препроцессинга.
                    self._log(
                        f"  [warn] заменено {bad_count} NaN/Inf из {raw.size} "
                        f"отсчётов ({bad_count / max(raw.size, 1):.1%})"
                    )
                sig = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
                sig = sig - np.median(sig)
                baseline_window = max(5, min(int(srate * 0.5), max(5, len(sig) // 20)))
                kernel = np.ones(baseline_window, dtype=float) / float(baseline_window)
                baseline = np.convolve(sig, kernel, mode="same")
                sig = sig - baseline
                scale = np.std(sig) + 1e-8
                sig = sig / scale
                sig = np.clip(sig, -5.0, 5.0)
                return sig

            def _select_best_channel(signal_matrix: np.ndarray) -> int:
                if signal_matrix.ndim == 1 or signal_matrix.shape[0] == 1:
                    return 0
                probe_len = min(signal_matrix.shape[1], int(srate * 90))
                best_idx = 0
                best_score = -1.0
                for ch_idx in range(signal_matrix.shape[0]):
                    probe = _prepare_analysis_signal(signal_matrix[ch_idx, :probe_len])
                    if np.std(probe) < 1e-4:
                        continue
                    probe_motifs = detect_wave_motifs(
                        probe,
                        fs=srate,
                        threshold1_std=max(0.2, EEG_THRESHOLD_LOW_STD * 0.8),
                        threshold2_std=max(0.5, EEG_THRESHOLD_HIGH_STD * 0.8),
                        min_duration=EEG_MIN_WAVE_DURATION,
                        max_duration=EEG_MAX_WAVE_DURATION,
                    )
                    edge_len = min(len(probe) // 8, int(srate * 3))
                    if edge_len > 0 and len(probe) > edge_len * 2:
                        edge_energy = float(
                            np.mean(np.abs(np.concatenate([probe[:edge_len], probe[-edge_len:]])))
                        )
                        center_energy = float(np.mean(np.abs(probe[edge_len:-edge_len])))
                    else:
                        edge_energy = float(np.mean(np.abs(probe)))
                        center_energy = edge_energy
                    score = len(probe_motifs) + 0.2 * float(np.std(probe)) - 0.15 * max(edge_energy - center_energy, 0.0)
                    if score > best_score:
                        best_score = score
                        best_idx = ch_idx
                return best_idx

            def _build_multichannel_composite(signal_matrix: np.ndarray, fallback_idx: int) -> np.ndarray:
                if signal_matrix.ndim == 1 or signal_matrix.shape[0] == 1:
                    return _prepare_analysis_signal(signal_matrix[0, :] if signal_matrix.ndim > 1 else signal_matrix)
                prepared_channels = np.vstack([
                    _prepare_analysis_signal(signal_matrix[ch_idx, :])
                    for ch_idx in range(signal_matrix.shape[0])
                ])
                composite = np.sqrt(np.mean(prepared_channels ** 2, axis=0))
                composite = composite - np.median(composite)
                scale = np.std(composite) + 1e-8
                composite = np.clip(composite / scale, -5.0, 5.0)
                if np.std(composite) < 1e-3:
                    return _prepare_analysis_signal(signal_matrix[fallback_idx, :])
                return composite

            def _motif_distribution(motifs: list[dict], total_sec: float) -> tuple[list[int], float]:
                if not motifs or total_sec <= 0:
                    return [0, 0, 0, 0, 0], 0.0
                onsets = np.array([float(m.get("onset_time", 0.0)) for m in motifs], dtype=float)
                bins = np.linspace(0.0, total_sec, 6)
                hist, _ = np.histogram(onsets, bins=bins)
                edge_share = float((hist[0] + hist[-1]) / max(hist.sum(), 1))
                return hist.astype(int).tolist(), edge_share

            def _score_motif_candidate(motifs: list[dict], total_sec: float) -> tuple[float, list[int], float]:
                hist, edge_share = _motif_distribution(motifs, total_sec)
                coverage = sum(1 for value in hist if value > 0)
                score = float(len(motifs)) + 0.8 * float(coverage) - 2.2 * float(edge_share)
                return score, hist, edge_share

            def _filter_edge_motifs(motifs: list[dict], total_sec: float) -> list[dict]:
                edge_guard = max(0.0, min(2.0, total_sec * 0.03))
                if edge_guard <= 0:
                    return motifs
                return [
                    motif for motif in motifs
                    if edge_guard <= float(motif.get("onset_time", 0.0)) <= max(edge_guard, total_sec - edge_guard)
                ]

            def _select_motifs(analysis_signal: np.ndarray, variant_name: str) -> tuple[list[dict], str, list[int], float]:
                total_sec = len(analysis_signal) / max(srate, 1)
                if total_sec <= 0:
                    return [], "none", [0, 0, 0, 0, 0], 0.0

                detect_args = {
                    "fs": srate,
                    "threshold1_std": max(0.2, EEG_THRESHOLD_LOW_STD * 0.85),
                    "threshold2_std": max(0.5, EEG_THRESHOLD_HIGH_STD * 0.85),
                    "min_duration": EEG_MIN_WAVE_DURATION,
                    "max_duration": EEG_MAX_WAVE_DURATION,
                }

                candidates: list[tuple[str, list[dict]]] = []
                raw_global = _filter_edge_motifs(detect_wave_motifs(analysis_signal, **detect_args), total_sec)
                if raw_global:
                    candidates.append(("global", raw_global))

                raw_segmented = _filter_edge_motifs(
                    detect_wave_motifs_segmented(
                        analysis_signal,
                        segment_duration=max(10.0, self.window_size * 2.5),
                        hop_duration=max(5.0, self.window_size * 1.25),
                        **detect_args,
                    ),
                    total_sec,
                )
                if raw_segmented:
                    candidates.append(("segmented", raw_segmented))

                envelope_signal = _prepare_analysis_signal(np.abs(analysis_signal))
                env_segmented = _filter_edge_motifs(
                    detect_wave_motifs_segmented(
                        envelope_signal,
                        fs=srate,
                        threshold1_std=0.15,
                        threshold2_std=0.40,
                        min_duration=EEG_MIN_WAVE_DURATION,
                        max_duration=EEG_MAX_WAVE_DURATION,
                        segment_duration=max(10.0, self.window_size * 2.5),
                        hop_duration=max(5.0, self.window_size * 1.25),
                    ),
                    total_sec,
                )
                if env_segmented:
                    candidates.append(("envelope_segmented", env_segmented))

                env_global = _filter_edge_motifs(
                    detect_wave_motifs(
                        envelope_signal,
                        fs=srate,
                        threshold1_std=0.15,
                        threshold2_std=0.40,
                        min_duration=EEG_MIN_WAVE_DURATION,
                        max_duration=EEG_MAX_WAVE_DURATION,
                    ),
                    total_sec,
                )
                if env_global:
                    candidates.append(("envelope_global", env_global))

                if not candidates:
                    return [], "none", [0, 0, 0, 0, 0], 0.0

                scored = [
                    (name, motifs, *_score_motif_candidate(motifs, total_sec))
                    for name, motifs in candidates
                ]
                scored.sort(key=lambda item: (item[2], len(item[1])), reverse=True)
                chosen_name, chosen_motifs, _, hist, edge_share = scored[0]
                self._log(
                    f"  {variant_name}: motif mode={chosen_name}, count={len(chosen_motifs)}, bins={hist}"
                )
                return chosen_motifs, chosen_name, hist, edge_share

            best_channel_idx = _select_best_channel(signal_data["original"])
            self._log(f"  Канал fallback для sonification: {best_channel_idx + 1}")

            composite_original = _build_multichannel_composite(signal_data["original"], best_channel_idx)
            composite_smoothed = _build_multichannel_composite(signal_data["smoothed"], best_channel_idx)
            self._log("  Sonification: multichannel composite signal")

            # Создаём варианты MIDI (original, smoothed, pca)
            variants = {
                "original": composite_original,
                "smoothed": composite_smoothed,
                "pca": _build_multichannel_composite(signal_data["pca"], 0),
            }

            COMPARISON_WINDOW_SIZE = self.window_size
            COMPARISON_HOP_SIZE = self.hop_size
            # EEG MIDI очень разреженный (десятки нот на минуты),
            # поэтому порог нот для окна должен быть низким
            MIN_NOTES_EEG = 3
            MIN_NOTES_CLASSICAL = 8
            # Per-file emotion: lookup by full path or stem
            _user_eeg_emotion = (
                self._eeg_emotions.get(str(eeg_path))
                or self._eeg_emotions.get(eeg_path.name)
                or "EEG"
            )
            self._log(f"  Эмоция: {_user_eeg_emotion}")

            for variant_name, sig_arr in variants.items():
                if self._cancelled:
                    return all_results

                # Извлекаем 1D сигнал
                if sig_arr.ndim > 1:
                    analysis_signal = sig_arr[0, :]
                else:
                    analysis_signal = sig_arr
                analysis_signal = _prepare_analysis_signal(analysis_signal)

                motifs, motif_mode, hist, edge_share = _select_motifs(analysis_signal, variant_name)
                if not motifs:
                    self._log(f"  {variant_name}: нет мотивов")
                    continue

                raw_music_events = map_motifs_to_adsr_sounds(motifs, scale_key=EEG_SCALE_KEY)
                if not raw_music_events:
                    self._log(f"  {variant_name}: нет событий")
                    continue

                music_events = compress_timed_events(
                    raw_music_events,
                    min_gap_sec=0.08,
                    max_gap_sec=max(0.55, min(0.90, self.window_size / 4.0)),
                    min_duration_sec=0.10,
                    max_duration_sec=0.85,
                )
                raw_span = max(
                    (
                        float(ev.get("onset", 0.0)) + float(ev.get("duration", 0.0))
                        for ev in raw_music_events
                    ),
                    default=0.0,
                )
                analysis_span = max(
                    (
                        float(ev.get("onset", 0.0)) + float(ev.get("duration", 0.0))
                        for ev in music_events
                    ),
                    default=0.0,
                )
                compression = (raw_span / analysis_span) if analysis_span > 1e-8 else 0.0
                recording_duration_sec = len(analysis_signal) / max(float(srate), 1.0)
                self._log(
                    f"  {variant_name}: {len(music_events)} событий, motif_mode={motif_mode}, "
                    f"bins={hist}, edge={edge_share:.2f}, timeline x{compression:.1f}"
                )

                midi_filename = f"{participant_id}_eeg_{variant_name}.mid"
                midi_path = eeg_midi_dir / midi_filename
                try:
                    create_midi_with_precise_timing(music_events, str(midi_path))
                except Exception as e:
                    self._log(f"  {variant_name}: ошибка создания MIDI: {e}")
                    continue

                # --- Снимок сигнала и мотивов для вкладки «Преобразование» ---
                try:
                    self._save_signal_snapshot(
                        report_dir=eeg_midi_dir.parent / "report",
                        participant_id=participant_id,
                        variant_name=variant_name,
                        signal=analysis_signal,
                        fs=float(srate),
                        motifs=motifs,
                        threshold_low_std=float(EEG_THRESHOLD_LOW_STD),
                        threshold_high_std=float(EEG_THRESHOLD_HIGH_STD),
                        midi_path=str(midi_path),
                    )
                except Exception as e:
                    self._log(f"  {variant_name}: не удалось сохранить snapshot: {e}")

                # Извлекаем окна из EEG MIDI
                try:
                    eeg_midi = m21.converter.parse(str(midi_path))
                    total_dur = eeg_midi.duration.quarterLength
                    if total_dur <= 0:
                        self._log(f"  {variant_name}: MIDI пуст (dur=0)")
                        continue
                except Exception as e:
                    self._log(f"  {variant_name}: ошибка парсинга MIDI: {e}")
                    continue

                self._log(f"  {variant_name}: MIDI длительность={total_dur:.1f} qL")

                # --- Стратегия: адаптивное окно для EEG ---
                # EEG MIDI разреженный, используем более широкие окна
                # или, если нот мало, берём весь MIDI целиком как одно "окно"
                eeg_windows = []
                eeg_flat = eeg_midi.flatten()

                total_notes_in_midi = len([n for n in eeg_flat.notes if hasattr(n, 'pitch')])
                self._log(f"  {variant_name}: нот в MIDI: {total_notes_in_midi}, длит.: {total_dur:.1f}")

                if total_notes_in_midi < MIN_NOTES_EEG:
                    self._log(f"  {variant_name}: слишком мало нот ({total_notes_in_midi}), пропуск")
                    continue

                # Если нот мало (< 20) или MIDI короткий — используем как одно окно
                if total_notes_in_midi < 20 or total_dur < COMPARISON_WINDOW_SIZE * 2:
                    feats = extract_window_features(eeg_flat, 0.0, total_dur)
                    if feats and feats.get("note_count", 0) >= MIN_NOTES_EEG:
                        feats["window_id"] = 0
                        feats["start_time"] = 0.0
                        feats["end_time"] = total_dur
                        feats["source"] = "EEG"
                        eeg_windows.append(feats)
                    self._log(f"  {variant_name}: цельное окно, нот: {feats.get('note_count', 0) if feats else 0}")
                else:
                    # Для более длинных MIDI используем скользящее окно
                    # с адаптивным размером: минимум COMPARISON_WINDOW_SIZE,
                    # но увеличиваем если нот слишком мало
                    adaptive_window = max(COMPARISON_WINDOW_SIZE, total_dur / 5)
                    adaptive_hop = adaptive_window / 2
                    wid = 0
                    start = 0.0
                    while start + adaptive_window <= total_dur:
                        end = start + adaptive_window
                        ws = eeg_flat.getElementsByOffset(
                            start, end,
                            includeEndBoundary=False,
                            mustFinishInSpan=False,
                            mustBeginInSpan=True,
                        )
                        feats = extract_window_features(ws, start, end)
                        if feats and feats.get("note_count", 0) >= MIN_NOTES_EEG:
                            feats["window_id"] = wid
                            feats["start_time"] = start
                            feats["end_time"] = end
                            feats["source"] = "EEG"
                            eeg_windows.append(feats)
                            wid += 1
                        start += adaptive_hop
                    self._log(f"  {variant_name}: скользящее окно ({adaptive_window:.1f}с), окон: {wid}")

                if not eeg_windows:
                    self._log(f"  {variant_name}: нет валидных окон, пропуск")
                    continue

                self._log(f"  {variant_name}: сравнение {len(eeg_windows)} EEG окон x {len(classical_windows_cache)} произведений")

                # Сравнение с классикой
                variant_results = []
                for classical_name, classical_wins in classical_windows_cache.items():
                    if not classical_wins:
                        continue
                    dataset_info = classical_meta_map.get(classical_name, {})
                    valid_cw = [w for w in classical_wins if w.get("note_count", 0) >= MIN_NOTES_CLASSICAL]
                    if not valid_cw:
                        continue

                    for eeg_win in eeg_windows:
                        best_score = -1.0
                        best_idx = 0
                        best_metrics = {}
                        eeg_nc = eeg_win.get("note_count", 0)

                        for j, cw in enumerate(valid_cw):
                            cnc = cw.get("note_count", 0)
                            # Для EEG-сравнения не штрафуем жёстко за разницу в количестве нот,
                            # потому что EEG MIDI принципиально разреженнее
                            nr = min(eeg_nc, cnc) / max(eeg_nc, cnc)
                            if nr < 0.01:
                                continue
                            # Мягкий штраф: sqrt от ratio вместо линейного
                            nr_penalty = max(nr ** 0.3, 0.3)

                            eeg_ivl = eeg_win.get("intervals_raw", np.array([0]))
                            cla_ivl = cw.get("intervals_raw", np.array([0]))
                            ivl_sim = interval_similarity(eeg_ivl, cla_ivl)

                            eeg_pc = eeg_win.get("pitch_class_hist", np.zeros(12))
                            cla_pc = cw.get("pitch_class_hist", np.zeros(12))
                            harm_sim = pitch_class_similarity(eeg_pc, cla_pc)

                            eeg_p = eeg_win.get("pitches_raw", np.array([60]))
                            cla_p = cw.get("pitches_raw", np.array([60]))
                            cont_sim = contour_similarity_strict(eeg_p, cla_p)
                            dyn_sim = dynamic_range_similarity(eeg_p, cla_p)

                            eeg_ioi = eeg_win.get("ioi_raw", np.array([0.0]))
                            cla_ioi = cw.get("ioi_raw", np.array([0.0]))
                            # simple rhythm
                            def _r(a, b):
                                if not np.isfinite(a) or not np.isfinite(b):
                                    return 0.0
                                return float(min(a, b) / (max(a, b) + 1e-8))
                            rhythm_sim = _r(
                                float(np.mean(eeg_ioi)) if len(eeg_ioi) > 0 else 0.0,
                                float(np.mean(cla_ioi)) if len(cla_ioi) > 0 else 0.0,
                            )

                            sfi_diff = abs(
                                eeg_win.get("sfi_pitch", 0) - cw.get("sfi_pitch", 0)
                            )
                            sfi_sim = float(np.exp(-sfi_diff))

                            combined = (
                                0.30 * cont_sim
                                + 0.20 * ivl_sim
                                + 0.20 * rhythm_sim
                                + 0.20 * harm_sim
                                + 0.10 * dyn_sim
                            ) * nr_penalty
                            if not np.isfinite(combined):
                                continue

                            cla_emotion = dataset_info.get("emotion")
                            emotion_match = None

                            if combined > best_score:
                                best_score = combined
                                best_idx = j
                                best_metrics = {
                                    "contour_similarity": cont_sim,
                                    "interval_similarity": ivl_sim,
                                    "rhythm_similarity": rhythm_sim,
                                    "harmony_similarity": harm_sim,
                                    "sfi_similarity": sfi_sim,
                                    "dynamic_similarity": dyn_sim,
                                    "eeg_note_count": eeg_nc,
                                    "cla_note_count": cnc,
                                    "emotion_match": emotion_match,
                                    "eeg_window_pitches": eeg_win.get("pitches_raw", np.array([])).tolist(),
                                    "eeg_window_durations": eeg_win.get("durations_raw", np.array([])).tolist(),
                                    "eeg_window_ioi": eeg_win.get("ioi_raw", np.array([])).tolist(),
                                    "eeg_window_velocities": eeg_win.get("velocities_raw", np.array([])).tolist(),
                                    "classical_window_pitches": cw.get("pitches_raw", np.array([])).tolist(),
                                    "classical_window_durations": cw.get("durations_raw", np.array([])).tolist(),
                                    "classical_window_ioi": cw.get("ioi_raw", np.array([])).tolist(),
                                    "classical_window_velocities": cw.get("velocities_raw", np.array([])).tolist(),
                                }

                        if best_score < 0:
                            continue
                        bm = valid_cw[best_idx]
                        meta = classical_meta_map.get(classical_name, {})
                        variant_results.append({
                            "eeg_window_id": eeg_win["window_id"],
                            "eeg_start_time": eeg_win["start_time"],
                            "eeg_emotion": _user_eeg_emotion,
                            "classical_piece": classical_name,
                            "classical_dataset": dataset_info.get("dataset", "maestro"),
                            "classical_track_id": dataset_info.get("track_id", ""),
                            "classical_midi_path": classical_dict.get(classical_name, ""),
                            "classical_title": meta.get("title"),
                            "classical_composer": meta.get("composer"),
                            "classical_emotion": meta.get("emotion"),
                            "classical_emotion_source": meta.get("emotion_source"),
                            "classical_window_id": bm["window_id"],
                            "classical_start_time": bm["start_time"],
                            "combined_similarity": best_score,
                            **best_metrics,
                        })

                if variant_results:
                    vdf = pd.DataFrame(variant_results)
                    # Отфильтровать окна с очень малым числом нот
                    if "eeg_note_count" in vdf.columns:
                        vdf = vdf[vdf["eeg_note_count"] >= 5]
                    if vdf.empty:
                        continue
                    best = vdf.nlargest(self.top_k, "combined_similarity")
                    for _, row in best.iterrows():
                        all_results.append({
                            "participant_id": participant_id,
                            "trial_idx": 0,
                            "variant": variant_name,
                            "valence": 5.0,
                            "arousal": 5.0,
                            "eeg_emotion": _user_eeg_emotion,
                            "eeg_source_file": eeg_path.name,
                            "eeg_midi": str(midi_path),
                            "eeg_recording_duration_sec": float(recording_duration_sec),
                            "eeg_raw_event_span_sec": float(raw_span),
                            "eeg_analysis_span_sec": float(analysis_span),
                            "eeg_timeline_compression": float(compression),
                            **{k: row.get(k) for k in row.index},
                        })

            pct = int(40 + 35 * (fi + 1) / max(total_files, 1))
            self._emit(pct, f"EEG: {fi+1}/{total_files}")

        return all_results

    def _build_comp_rows(
        self, top_results, classical_dict, classical_meta_map,
        maestro_pseudo_emotions, matches_out, window_size,
    ):
        """Строит строки display-results для GUI и экспортируемых артефактов."""
        import shutil
        from pathlib import Path
        from src.midi_utils import (
            extract_melody_with_time, extract_note_events,
            create_midi_from_notes, create_midi_from_note_events, create_comparison_midi,
        )
        from src.emopia_loader import get_emopia_metadata
        from src.config import PLAYBACK_TEMPO_MULTIPLIER, USE_PSEUDO_LABELING

        comp_rows = []

        def _window_arrays_to_events(pitches, durations, ioi, velocities):
            pitches = [int(p) for p in (pitches or [])]
            durations = [float(d) for d in (durations or [])]
            ioi = [float(x) for x in (ioi or [])]
            velocities = [int(v) for v in (velocities or [])]
            if not pitches:
                return []
            if not durations:
                durations = [0.5] * len(pitches)
            if len(durations) < len(pitches):
                durations = durations + [durations[-1] if durations else 0.5] * (len(pitches) - len(durations))
            if len(velocities) < len(pitches):
                velocities = velocities + [80] * (len(pitches) - len(velocities))

            events = []
            onset = 0.0
            for idx, pitch in enumerate(pitches):
                if idx > 0:
                    step = ioi[idx - 1] if idx - 1 < len(ioi) else durations[idx - 1]
                    onset += max(0.05, float(step))
                events.append({
                    "pitch": pitch,
                    "onset": onset,
                    "duration": max(0.08, float(durations[idx])),
                    "velocity": int(velocities[idx]),
                })
            return events

        def _event_stats(events):
            if not events:
                return {
                    "note_count": 0,
                    "span_sec": 0.0,
                    "active_ratio": 0.0,
                    "silence_ratio": 1.0,
                }
            ordered = sorted(events, key=lambda ev: float(ev.get("onset", 0.0)))
            span_start = float(ordered[0].get("onset", 0.0))
            span_end = max(
                float(ev.get("onset", 0.0)) + float(ev.get("duration", 0.0))
                for ev in ordered
            )
            span_sec = max(span_end - span_start, 1e-8)
            active_sec = sum(max(0.0, float(ev.get("duration", 0.0))) for ev in ordered)
            silence_sec = 0.0
            for prev, curr in zip(ordered, ordered[1:]):
                prev_end = float(prev.get("onset", 0.0)) + float(prev.get("duration", 0.0))
                curr_onset = float(curr.get("onset", 0.0))
                silence_sec += max(0.0, curr_onset - prev_end)
            active_ratio = float(min(active_sec / span_sec, 1.0))
            silence_ratio = float(min(max(silence_sec / span_sec, 0.0), 1.0))
            return {
                "note_count": int(len(ordered)),
                "span_sec": float(span_sec),
                "active_ratio": active_ratio,
                "silence_ratio": silence_ratio,
            }

        for rank, (_, row) in enumerate(
            top_results.reset_index(drop=True).iterrows(), start=1,
        ):
            participant = row.get("participant_id", "unknown")
            trial = int(row.get("trial_idx", 0))
            variant = row.get("variant", "")
            classical_name = row.get("classical_piece")
            classical_dataset = row.get("classical_dataset", "maestro")
            classical_track_id = row.get("classical_track_id", "")
            eeg_emotion = row.get("eeg_emotion", "Unknown")
            eeg_midi_path = Path(str(row.get("eeg_midi")))
            eeg_start = float(row.get("eeg_start_time", 0.0))
            classical_start = float(row.get("classical_start_time", 0.0))

            if "|" in str(classical_name):
                parts = str(classical_name).split("|", 2)
                classical_display = parts[2] if len(parts) > 2 else classical_name
            else:
                classical_display = classical_name

            composer = row.get("classical_composer")
            title = row.get("classical_title")
            if not composer or not title:
                try:
                    if " - " in str(classical_display):
                        composer, title = str(classical_display).split(" - ", 1)
                    else:
                        composer = classical_dataset.upper()
                        title = str(classical_display)
                except Exception:
                    composer = "Unknown"
                    title = str(classical_display)

            classical_emotion = row.get("classical_emotion")
            classical_emotion_source = row.get("classical_emotion_source")
            if not classical_emotion:
                if classical_dataset == "emopia" and classical_track_id:
                    meta = get_emopia_metadata(classical_track_id)
                    classical_emotion = meta.get("emotion")
                    classical_emotion_source = "ground_truth"
                elif (
                    classical_dataset == "maestro"
                    and classical_track_id
                    and USE_PSEUDO_LABELING
                ):
                    pseudo = maestro_pseudo_emotions.get(classical_track_id)
                    if pseudo:
                        classical_emotion = pseudo.get("emotion")
                        classical_emotion_source = "predicted"

            def _sanitize(s, max_len=40):
                for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
                    s = s.replace(ch, "_")
                while "__" in s:
                    s = s.replace("__", "_")
                return s.strip("_")[:max_len]

            safe_title = _sanitize(str(title), 40)
            safe_comp = _sanitize(str(composer), 30)
            prefix = f"{rank:02d}_"
            variant_prefix = f"{variant}_" if variant else ""

            # EEG fragment
            eeg_window_events = _window_arrays_to_events(
                row.get("eeg_window_pitches"),
                row.get("eeg_window_durations"),
                row.get("eeg_window_ioi"),
                row.get("eeg_window_velocities"),
            )
            eeg_note_events = extract_note_events(str(eeg_midi_path)) if eeg_midi_path.exists() else []
            eeg_notes = (
                extract_melody_with_time(str(eeg_midi_path))
                if eeg_midi_path.exists()
                else []
            )
            eeg_diag = _event_stats(eeg_note_events)
            eeg_events_frag = list(eeg_window_events)
            if not eeg_events_frag:
                eeg_events_frag = [
                    ev for ev in eeg_note_events
                    if ev["onset"] < eeg_start + window_size and (ev["onset"] + ev["duration"]) >= eeg_start
                ]
            if not eeg_events_frag and eeg_note_events:
                center = eeg_start + window_size / 2.0
                eeg_events_frag = sorted(
                    eeg_note_events,
                    key=lambda ev: abs((ev["onset"] + ev["duration"] / 2.0) - center),
                )[:80]
                eeg_events_frag = sorted(eeg_events_frag, key=lambda ev: ev["onset"])
            if not eeg_events_frag and eeg_note_events:
                eeg_events_frag = eeg_note_events[:80]
            eeg_frag = [int(ev["pitch"]) for ev in eeg_events_frag] if eeg_events_frag else [
                int(p) for p, _ in eeg_notes[:50]
            ]

            # Classical fragment
            classical_path = classical_dict.get(classical_name)
            cla_frag = []
            cla_events_frag = []
            if classical_path:
                cla_window_events = _window_arrays_to_events(
                    row.get("classical_window_pitches"),
                    row.get("classical_window_durations"),
                    row.get("classical_window_ioi"),
                    row.get("classical_window_velocities"),
                )
                cla_note_events = extract_note_events(str(classical_path))
                cla_notes = extract_melody_with_time(str(classical_path))
                cla_events_frag = list(cla_window_events)
                if not cla_events_frag:
                    cla_events_frag = [
                        ev for ev in cla_note_events
                        if ev["onset"] < classical_start + window_size and (ev["onset"] + ev["duration"]) >= classical_start
                    ]
                if not cla_events_frag and cla_note_events:
                    center = classical_start + window_size / 2.0
                    cla_events_frag = sorted(
                        cla_note_events,
                        key=lambda ev: abs((ev["onset"] + ev["duration"] / 2.0) - center),
                    )[:120]
                    cla_events_frag = sorted(cla_events_frag, key=lambda ev: ev["onset"])
                if not cla_events_frag and cla_note_events:
                    cla_events_frag = cla_note_events[:120]
                cla_frag = [int(ev["pitch"]) for ev in cla_events_frag] if cla_events_frag else [
                    int(p) for p, _ in cla_notes[:50]
                ]

            # Sync lengths
            if eeg_frag and cla_frag:
                mn = max(min(len(eeg_frag), len(cla_frag)), 10)
                eeg_frag = eeg_frag[:mn]
                cla_frag = cla_frag[:mn]
                if eeg_events_frag:
                    eeg_events_frag = eeg_events_frag[:mn]
                if cla_events_frag:
                    cla_events_frag = cla_events_frag[:mn]

            tempo = int(120 * PLAYBACK_TEMPO_MULTIPLIER)

            eeg_mid_name = f"{prefix}{variant_prefix}EEG_{safe_comp}_{safe_title}.mid"
            eeg_mid_out = matches_out / eeg_mid_name
            if eeg_events_frag:
                create_midi_from_note_events(eeg_events_frag, str(eeg_mid_out), tempo_bpm=tempo)
            elif eeg_frag:
                create_midi_from_notes(eeg_frag, str(eeg_mid_out), tempo_bpm=tempo)
            elif eeg_midi_path.exists():
                try:
                    shutil.copy(str(eeg_midi_path), str(eeg_mid_out))
                except Exception:
                    pass

            cla_mid_name = f"{prefix}{variant_prefix}Classical_{safe_comp}_{safe_title}.mid"
            cla_mid_out = matches_out / cla_mid_name
            if cla_events_frag:
                create_midi_from_note_events(cla_events_frag, str(cla_mid_out), tempo_bpm=tempo)
            elif cla_frag:
                create_midi_from_notes(cla_frag, str(cla_mid_out), tempo_bpm=tempo)
            elif classical_path:
                try:
                    shutil.copy(str(classical_path), str(cla_mid_out))
                except Exception:
                    pass

            cmp_mid_name = f"{prefix}{variant_prefix}Comparison_{safe_comp}_{safe_title}.mid"
            cmp_mid_out = matches_out / cmp_mid_name
            if eeg_frag or cla_frag:
                try:
                    create_comparison_midi(
                        eeg_frag or [], cla_frag or [],
                        str(cmp_mid_out), tempo_bpm=tempo,
                    )
                except Exception:
                    pass

            # Convert MIDI to WAV for GUI playback
            try:
                from src.audio_converter import midi_to_wav, find_soundfont
                sf = find_soundfont()
                if sf:
                    for mid_file in [eeg_mid_out, cla_mid_out, cmp_mid_out]:
                        wav_file = mid_file.with_suffix(".wav")
                        if mid_file.exists() and not wav_file.exists():
                            try:
                                midi_to_wav(str(mid_file), str(wav_file), sf)
                            except Exception:
                                pass
            except Exception:
                pass

            comp_rows.append({
                "file": cla_mid_out.name,
                "eeg_source_file": str(row.get("eeg_source_file") or ""),
                "eeg_midi_path": str(eeg_mid_out),
                "classical_midi_path": str(cla_mid_out),
                "comparison_midi_path": str(cmp_mid_out),
                "composer": composer, "title": title,
                "variant": variant, "trial": f"Trial {trial}",
                "processing": variant,
                "eeg_valence": row.get("valence"),
                "eeg_arousal": row.get("arousal"),
                "eeg_emotion": eeg_emotion,
                "participant_id": participant,
                "classical_dataset": classical_dataset,
                "classical_emotion": classical_emotion,
                "classical_emotion_source": classical_emotion_source,
                "emotion_match": row.get("emotion_match", None),
                "combined_similarity": float(row.get("combined_similarity", 0.0)),
                "music_match_score": float(row.get("music_match_score", row.get("combined_similarity", 0.0))),
                "emotion_agreement_score": float(row.get("emotion_agreement_score", 0.0)),
                "feature_similarity_score": float(row.get("feature_similarity_score", 0.0)),
                "fragment_alignment_score": float(row.get("fragment_alignment_score", row.get("combined_similarity", 0.0))),
                "match_label": row.get("match_label", "Неопределено"),
                "contour_similarity": float(row.get("contour_similarity", 0.0)),
                "interval_similarity": float(row.get("interval_similarity", 0.0)),
                "correlation_similarity": float(row.get("correlation_similarity", 0.0)),
                "harmony_similarity": float(row.get("harmony_similarity", 0.0)),
                "sfi_similarity": float(row.get("sfi_similarity", 0.0)),
                "stat_similarity": float(row.get("stat_similarity", row.get("dynamic_similarity", 0.0))),
                "melodic_similarity": float(row.get("combined_similarity", 0.0)),
                "eeg_pitches": eeg_frag,
                "cla_pitches": cla_frag,
                "eeg_note_count_total": eeg_diag["note_count"],
                "eeg_melody_span_sec": eeg_diag["span_sec"],
                "eeg_active_ratio": eeg_diag["active_ratio"],
                "eeg_silence_ratio": eeg_diag["silence_ratio"],
                "eeg_recording_duration_sec": float(row.get("eeg_recording_duration_sec", 0.0)),
                "eeg_raw_event_span_sec": float(row.get("eeg_raw_event_span_sec", 0.0)),
                "eeg_analysis_span_sec": float(row.get("eeg_analysis_span_sec", eeg_diag["span_sec"])),
                "eeg_timeline_compression": float(row.get("eeg_timeline_compression", 0.0)),
                "eeg_fragment_start_sec": eeg_start,
                "eeg_fragment_end_sec": eeg_start + window_size,
                "music_fragment_start_sec": classical_start,
                "music_fragment_end_sec": classical_start + window_size,
                "eeg_wav_path": str(eeg_mid_out.with_suffix(".wav")),
                "classical_wav_path": str(cla_mid_out.with_suffix(".wav")),
                "comparison_wav_path": str(cmp_mid_out.with_suffix(".wav")),
            })
        return comp_rows

    # ------------------------------------------------------------------
    def _save_signal_snapshot(
        self,
        report_dir: Path,
        participant_id: str,
        variant_name: str,
        signal,
        fs: float,
        motifs: list,
        threshold_low_std: float,
        threshold_high_std: float,
        midi_path: str,
    ) -> None:
        """
        Сохраняет снимок EEG-сигнала и детектированных мотивов в report_dir/signal_snapshots/
        для последующей визуализации на вкладке «Преобразование».
        Сигнал прореживается до ~2500 точек чтобы JSON оставался лёгким.
        """
        import json as _json
        import numpy as _np

        report_dir = Path(report_dir)
        snap_dir = report_dir / "signal_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        sig = _np.asarray(signal, dtype=float).ravel()
        if sig.size == 0 or fs <= 0:
            return

        duration_sec = float(sig.size) / float(fs)
        signal_std = float(_np.std(sig)) if sig.size > 1 else 0.0
        signal_mean = float(_np.mean(sig))

        # Прореживание до MAX_POINTS точек, равномерно по времени.
        MAX_POINTS = 2500
        if sig.size > MAX_POINTS:
            idx = _np.linspace(0, sig.size - 1, MAX_POINTS).astype(int)
            sig_down = sig[idx]
            t_down = (idx / float(fs)).tolist()
        else:
            sig_down = sig
            t_down = (_np.arange(sig.size) / float(fs)).tolist()

        motifs_out = []
        for m in motifs or []:
            motifs_out.append({
                "onset_time": float(m.get("onset_time", 0.0)),
                "peak_time": float(m.get("peak_time", 0.0)),
                "duration": float(m.get("duration", 0.0)),
                "peak_amplitude": float(m.get("peak_amplitude", 0.0)),
                "rise_time": float(m.get("rise_time", 0.0)),
            })

        payload = {
            "participant_id": str(participant_id),
            "variant": str(variant_name),
            "fs": float(fs),
            "duration_sec": duration_sec,
            "n_samples": int(sig.size),
            "signal_mean": signal_mean,
            "signal_std": signal_std,
            "time": [float(x) for x in t_down],
            "signal": [float(x) for x in sig_down],
            "motifs": motifs_out,
            "motif_count": len(motifs_out),
            "threshold_low_std": float(threshold_low_std),
            "threshold_high_std": float(threshold_high_std),
            "midi_path": str(midi_path),
        }
        out_path = snap_dir / f"{participant_id}_{variant_name}.json"
        try:
            out_path.write_text(_json.dumps(payload), encoding="utf-8")
        except Exception as e:
            self._log(f"  snapshot: ошибка записи {out_path.name}: {e}")

    def _export_charts(self, results_df: pd.DataFrame, report_dir: Path):
        """Экспорт итоговых графиков (PNG) и CSV."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        report_dir = Path(report_dir)

        ranking_col = "music_match_score" if "music_match_score" in results_df.columns else "combined_similarity"

        # --- 1) Main score chart (top-K, music-first) ---
        try:
            top = results_df.sort_values(ranking_col, ascending=False).head(self.top_k).copy() \
                if ranking_col in results_df.columns else results_df.head(self.top_k).copy()
            if top.empty:
                return

            top["label"] = top.apply(
                lambda r: f"{r.get('participant_id','?')}/t{r.get('trial_idx',0)} "
                          f"→ {str(r.get('classical_piece',''))[:25]}",
                axis=1,
            )

            fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.5)))
            metrics = [ranking_col, "emotion_agreement_score", "feature_similarity_score"]
            avail = [m for m in metrics if m in top.columns]
            top_plot = top[avail + ["label"]].set_index("label")
            top_plot.plot.barh(ax=ax, width=0.7)
            ax.set_xlabel("Score")
            ax.set_title("Топ соответствий: music-first ranking + emotion evidence")
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            fig.savefig(str(report_dir / "music_match_chart.png"), dpi=150)
            plt.close(fig)
            self._log("Экспорт: music_match_chart.png")
        except Exception as e:
            self._log(f"Ошибка графика music match: {e}")

        # --- 2) Emotion distribution ---
        try:
            if "eeg_emotion" in results_df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                emo_counts = results_df["eeg_emotion"].value_counts()
                emo_counts.plot.bar(ax=ax, color=sns.color_palette("Set2"))
                ax.set_title("Распределение EEG эмоций")
                ax.set_ylabel("Кол-во")
                plt.tight_layout()
                fig.savefig(str(report_dir / "emotion_distribution.png"), dpi=150)
                plt.close(fig)
                self._log("Экспорт: emotion_distribution.png")
        except Exception as e:
            self._log(f"Ошибка графика emotion: {e}")

        # --- 3) Full EEG melody diagnostics ---
        try:
            self._export_melody_diagnostics(results_df, report_dir)
        except Exception as e:
            self._log(f"Ошибка melody diagnostics: {e}")

        # --- 4) Summary CSV ---
        try:
            summary = results_df.head(self.top_k)[
                [c for c in [
                    "participant_id", "trial_idx", "variant", "eeg_emotion",
                    "classical_piece", "classical_dataset", "classical_emotion",
                    "music_match_score", "combined_similarity", "emotion_agreement_score",
                    "feature_similarity_score", "fragment_alignment_score",
                ] if c in results_df.columns]
            ]
            summary.to_csv(report_dir / "top_matches_summary.csv", index=False)
            self._log("Экспорт: top_matches_summary.csv")
        except Exception as e:
            self._log(f"Ошибка экспорта CSV: {e}")

    def _export_melody_diagnostics(self, results_df: pd.DataFrame, report_dir: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from src.midi_utils import extract_note_events

        report_dir = Path(report_dir)
        ranking_col = "music_match_score" if "music_match_score" in results_df.columns else "combined_similarity"
        ranked = results_df.sort_values(ranking_col, ascending=False).reset_index(drop=True) \
            if ranking_col in results_df.columns else results_df.reset_index(drop=True)

        diagnostics_rows = []
        diag_dir = report_dir / "melody_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        seen_paths = set()

        for _, row in ranked.iterrows():
            midi_path = str(row.get("eeg_midi") or "").strip()
            if not midi_path or midi_path in seen_paths:
                continue
            seen_paths.add(midi_path)
            events = extract_note_events(midi_path)
            if not events:
                continue

            ordered = sorted(events, key=lambda ev: float(ev.get("onset", 0.0)))
            onsets = np.array([float(ev.get("onset", 0.0)) for ev in ordered], dtype=float)
            pitches = np.array([float(ev.get("pitch", 0.0)) for ev in ordered], dtype=float)
            durations = np.array([max(0.0, float(ev.get("duration", 0.0))) for ev in ordered], dtype=float)
            span_start = float(onsets[0])
            span_end = float(np.max(onsets + durations))
            span_sec = max(span_end - span_start, 1e-8)
            active_sec = float(np.sum(durations))
            silence_sec = 0.0
            for prev, curr in zip(ordered, ordered[1:]):
                prev_end = float(prev.get("onset", 0.0)) + float(prev.get("duration", 0.0))
                curr_onset = float(curr.get("onset", 0.0))
                silence_sec += max(0.0, curr_onset - prev_end)

            silence_ratio = float(min(max(silence_sec / span_sec, 0.0), 1.0))
            active_ratio = float(min(active_sec / span_sec, 1.0))
            participant = str(row.get("participant_id", "unknown"))
            variant = str(row.get("variant", "unknown"))
            image_path = diag_dir / f"{Path(midi_path).stem}.png"

            bins = np.linspace(float(np.min(onsets)), span_end, 9)
            hist, edges = np.histogram(onsets, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2.0

            fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2.2, 1.0]})
            axes[0].scatter(onsets, pitches, s=18, c="#2a6f97", alpha=0.85)
            axes[0].set_title(f"EEG melody timeline: {participant} / {variant}")
            axes[0].set_ylabel("Pitch")
            axes[0].grid(alpha=0.18)
            axes[1].bar(centers, hist, width=np.diff(edges), color="#c87d2a", alpha=0.85, align="center")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Notes/bin")
            axes[1].grid(alpha=0.18)
            fig.tight_layout()
            fig.savefig(image_path, dpi=150)
            plt.close(fig)

            diagnostics_rows.append({
                "participant_id": participant,
                "variant": variant,
                "eeg_midi": midi_path,
                "note_count": int(len(ordered)),
                "recording_duration_sec": float(row.get("eeg_recording_duration_sec", 0.0)),
                "raw_event_span_sec": float(row.get("eeg_raw_event_span_sec", 0.0)),
                "span_sec": float(span_sec),
                "active_ratio": active_ratio,
                "silence_ratio": silence_ratio,
                "timeline_compression": float(row.get("eeg_timeline_compression", 0.0)),
                "diagnostic_image": str(image_path),
            })

        if diagnostics_rows:
            diag_df = pd.DataFrame(diagnostics_rows)
            diag_df.to_csv(report_dir / "melody_diagnostics.csv", index=False)
            best_image = Path(diag_df.iloc[0]["diagnostic_image"])
            if best_image.exists():
                best_target = report_dir / "best_eeg_melody_timeline.png"
                best_target.write_bytes(best_image.read_bytes())
            self._log("Экспорт: melody_diagnostics.csv")
