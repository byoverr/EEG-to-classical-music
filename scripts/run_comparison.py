#!/usr/bin/env python3
"""
Главный скрипт для сравнения EEG-MIDI с классическими произведениями.

Этапы:
1. Загрузка DEAP данных (ЭЭГ + эмоциональные метки)
2. Загрузка MAESTRO (классические MIDI файлы)  
3. Преобразование ЭЭГ в несколько вариантов MIDI
4. Сравнение с помощью ComprehensiveMIDIComparator
5. Генерация HTML отчёта с топ результатами и графиками
"""
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Подавляем предупреждения до импорта других модулей
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', message='dtaidistance not installed')

import pandas as pd
import numpy as np
import shutil
import random

# Добавляем src в путь
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from src.config import (
    DEAP_DIR, MAESTRO_DIR, EMOPIA_DIR, REPORTS_DIR, BEST_MATCHES_DIR,
    DEAP_NUM_PARTICIPANTS, DEAP_NUM_TRIALS, DEAP_SAMPLE_RATE,
    COMPARISON_WINDOW_SIZE, COMPARISON_HOP_SIZE,
    EEG_THRESHOLD_LOW_STD, EEG_THRESHOLD_HIGH_STD,
    EEG_MIN_WAVE_DURATION, EEG_MAX_WAVE_DURATION, EEG_MIN_PEAK_DISTANCE,
    EEG_SCALE_KEY, SIMILARITY_WEIGHTS,
    HTML_FRAGMENT_DURATION, PLAYBACK_TEMPO_MULTIPLIER,
    TOP_N_MATCHES, USE_EMOPIA, EMOPIA_MAX_TRACKS, EMOTION_THRESHOLD,
    RUNS_DIR, DEFAULT_RUN_ID, USE_PSEUDO_LABELING,
    REUSE_EEG_MIDI, CLEAN_DATA_ON_RUN,
    DEFAULT_MAX_PARTICIPANTS, DEFAULT_MAX_TRIALS,
    DEFAULT_MAX_CLASSICAL, DEFAULT_TOP_K, DEFAULT_JOBS,
    DEFAULT_ONLY_EMOPIA, DEFAULT_BALANCED_EEG_EMOTIONS,
    DEFAULT_PER_EMOTION_TRIALS, DEFAULT_MATCH_EMOTIONS,
    EMOTION_MISMATCH_PENALTY, MAESTRO_PSEUDO_LABELS_PATH
)
from src.deap_loader import load_deap_participant_data, extract_eeg_from_deap, get_emotion_labels
from src.maestro_loader import get_maestro_midi_files, get_maestro_metadata
from src.emopia_loader import (
    get_emopia_midi_files, get_emopia_metadata, deap_to_emotion_quadrant,
    get_all_emotions, get_emopia_track_info
)
from src.track_features import (
    load_feature_cache, save_feature_cache, get_or_compute_features,
    features_to_vector
)
from src.eeg_preprocessing import prepare_signal_data, smooth_signal, pca_transform
from src.eeg_processing import detect_wave_motifs, map_motifs_to_adsr_sounds
from src.midi_utils import (
    create_midi_with_precise_timing,
    extract_melody_sequence,
    extract_melody_with_time,
    create_midi_from_notes,
    create_comparison_midi,
)
from src.MIDIComparator import ComprehensiveMIDIComparator
from src.html_generator import create_simple_comparison_html, create_comparison_html


def detect_events_robust(signal_array: np.ndarray, fs: float):
    """
    Робастная детекция мотивов, обёртка над detect_wave_motifs
    с параметрами из config.py.
    """
    signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)
    return detect_wave_motifs(
        signal_array,
        fs=fs,
        threshold1_std=EEG_THRESHOLD_LOW_STD,
        threshold2_std=EEG_THRESHOLD_HIGH_STD,
        min_duration=EEG_MIN_WAVE_DURATION,
        max_duration=EEG_MAX_WAVE_DURATION,
    )


def create_eeg_variants(signal_data: dict, output_dir: Path, 
                       participant_id: str, trial_idx: int,
                       reuse: bool = None) -> dict:
    """
    Создаёт несколько вариантов MIDI из ЭЭГ сигналов.
    
    Варианты:
    - original: из Fp1 канала (фронтальный, наиболее информативный)
    - smoothed: из сглаженного Fp1
    - pca: из первой главной компоненты
    
    Использует параметры из config.py для детекции волн.
    Если reuse=True, пропускает генерацию для уже существующих файлов.
    
    Возвращает словарь {variant_name: midi_path}
    """
    from scipy.signal import find_peaks, butter, filtfilt
    
    if reuse is None:
        reuse = REUSE_EEG_MIDI
    
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_paths = {}
    variants = {
        'original': signal_data['original'],
        'smoothed': signal_data['smoothed'],
        'pca': signal_data['pca']
    }
    
    for variant_name, signal_array in variants.items():
        # Проверяем, можно ли переиспользовать ранее созданный MIDI
        midi_filename = f"{participant_id}_trial{trial_idx:02d}_{variant_name}.mid"
        midi_path = output_dir / midi_filename
        if reuse and midi_path.exists() and midi_path.stat().st_size > 0:
            midi_paths[variant_name] = midi_path
            print(f"  ♻ {variant_name}: reused {midi_filename}")
            continue
        
        # Извлекаем 1D сигнал правильно
        if signal_array.ndim > 1:
            if variant_name == 'pca':
                # PCA: первая компонента
                analysis_signal = signal_array[0, :]
            else:
                # Original/Smoothed: используем Fp1 (канал 0) вместо среднего
                # Среднее даёт почти нулевой сигнал из-за противофазных каналов
                analysis_signal = signal_array[0, :]
        else:
            analysis_signal = signal_array
        
        # Очистка от NaN/Inf
        analysis_signal = np.nan_to_num(analysis_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Детектируем события с робастным методом
        motifs = detect_events_robust(analysis_signal, DEAP_SAMPLE_RATE)
        
        if not motifs:
            print(f"  ✗ {variant_name}: no events detected")
            continue
        
        # Преобразуем в музыкальные события
        music_events = map_motifs_to_adsr_sounds(motifs, scale_key=EEG_SCALE_KEY)
        
        if music_events:
            # Сохраняем MIDI
            create_midi_with_precise_timing(music_events, str(midi_path))
            midi_paths[variant_name] = midi_path
            print(f"  ✓ {variant_name}: {len(music_events)} events → {midi_filename}")
        else:
            print(f"  ✗ {variant_name}: no music events generated")
    
    return midi_paths


def _process_classical_file(args):
    """Обрабатывает один классический файл (для параллелизации)."""
    import music21 as m21
    from src.MIDIComparator import extract_window_features
    from src.config import COMPARISON_WINDOW_SIZE as _DEFAULT_WINDOW, COMPARISON_HOP_SIZE as _DEFAULT_HOP
    
    # Support both (name, path) and (name, path, window_size, hop_size) signatures
    if len(args) == 4:
        name, path, COMPARISON_WINDOW_SIZE, COMPARISON_HOP_SIZE = args
    else:
        name, path = args
        COMPARISON_WINDOW_SIZE = _DEFAULT_WINDOW
        COMPARISON_HOP_SIZE = _DEFAULT_HOP
    try:
        midi = m21.converter.parse(path)
        
        # Извлекаем окна
        total_duration = midi.duration.quarterLength
        windows = []
        start_time = 0.0
        window_id = 0
        
        while start_time + COMPARISON_WINDOW_SIZE <= total_duration:
            end_time = start_time + COMPARISON_WINDOW_SIZE
            window_stream = midi.flatten().getElementsByOffset(
                start_time, end_time,
                includeEndBoundary=False,
                mustFinishInSpan=False,
                mustBeginInSpan=True
            )
            features = extract_window_features(window_stream, start_time, end_time)
            if features and features.get('note_count', 0) >= 8:
                features['window_id'] = window_id
                features['start_time'] = start_time
                features['end_time'] = end_time
                features['source'] = name
                windows.append(features)
                window_id += 1
            start_time += COMPARISON_HOP_SIZE
        
        return name, windows, None  # None = no error
    except Exception as e:
        return name, [], str(e)


# Глобальный кэш для параллельной обработки
_classical_cache = {}
_classical_path_map = {}
_classical_meta_map = {}


def _init_worker(classical_cache, classical_path_map, classical_meta_map):
    """Инициализирует worker с кэшем классических окон."""
    global _classical_cache, _classical_path_map, _classical_meta_map
    _classical_cache = classical_cache
    _classical_path_map = classical_path_map
    _classical_meta_map = classical_meta_map


def _process_trial(args):
    """
    Обрабатывает один триал: создаёт EEG MIDI и сравнивает с классикой.
    Для параллелизации.
    """
    import music21 as m21
    from src.MIDIComparator import (
        extract_window_features, interval_similarity, pitch_class_similarity,
        contour_similarity_strict, dtw_distance, dynamic_range_similarity
    )
    from src.config import COMPARISON_WINDOW_SIZE as _DEFAULT_WINDOW, COMPARISON_HOP_SIZE as _DEFAULT_HOP, EEG_SCALE_KEY
    from src.eeg_preprocessing import prepare_signal_data
    from src.deap_loader import load_deap_participant_data, get_emotion_labels
    
    # Support both 4-tuple and 6-tuple signatures
    if len(args) == 6:
        participant_file, trial_idx, eeg_midi_dir, match_emotions, COMPARISON_WINDOW_SIZE, COMPARISON_HOP_SIZE = args
    else:
        participant_file, trial_idx, eeg_midi_dir, match_emotions = args
        COMPARISON_WINDOW_SIZE = _DEFAULT_WINDOW
        COMPARISON_HOP_SIZE = _DEFAULT_HOP
    participant_id = Path(participant_file).stem
    results = []
    
    try:
        data = load_deap_participant_data(str(participant_file))
        labels = get_emotion_labels(data, trial_idx)
        valence = labels.get('valence', 5.0)
        arousal = labels.get('arousal', 5.0)
        
        # Вычисляем эмоциональный квадрант для EEG
        eeg_emotion = deap_to_emotion_quadrant(valence, arousal, threshold=EMOTION_THRESHOLD)
        
        signal_data = prepare_signal_data(data, trial_idx)
        
        # Создаём варианты MIDI
        eeg_midi_paths = create_eeg_variants(
            signal_data, 
            eeg_midi_dir,
            participant_id, 
            trial_idx
        )
        
        if not eeg_midi_paths:
            return participant_id, trial_idx, [], "No MIDI variants"
        
        MIN_NOTES = 8
        
        for variant_name, eeg_midi_path in eeg_midi_paths.items():
            try:
                eeg_midi = m21.converter.parse(str(eeg_midi_path))
                total_duration = eeg_midi.duration.quarterLength
                eeg_windows = []
                start_time = 0.0
                window_id = 0
                
                while start_time + COMPARISON_WINDOW_SIZE <= total_duration:
                    end_time = start_time + COMPARISON_WINDOW_SIZE
                    window_stream = eeg_midi.flatten().getElementsByOffset(
                        start_time, end_time,
                        includeEndBoundary=False,
                        mustFinishInSpan=False,
                        mustBeginInSpan=True
                    )
                    features = extract_window_features(window_stream, start_time, end_time)
                    if features and features.get('note_count', 0) >= MIN_NOTES:
                        features['window_id'] = window_id
                        features['start_time'] = start_time
                        features['end_time'] = end_time
                        features['source'] = 'EEG'
                        eeg_windows.append(features)
                        window_id += 1
                    start_time += COMPARISON_HOP_SIZE
                
                if not eeg_windows:
                    continue
                
                # Сравниваем с классикой
                variant_results = []
                
                for classical_name, classical_wins in _classical_cache.items():
                    if not classical_wins:
                        continue
                    
                    # Извлекаем информацию о датасете из meta map
                    dataset_info = _classical_meta_map.get(classical_name, {})
                    if not dataset_info:
                        if '|' in classical_name:
                            parts = classical_name.split('|', 2)
                            dataset_info = {
                                'dataset': parts[0],
                                'track_id': parts[1]
                            }
                        else:
                            dataset_info = {'dataset': 'maestro'}
                    
                    valid_classical_wins = [w for w in classical_wins if w.get('note_count', 0) >= MIN_NOTES]
                    if not valid_classical_wins:
                        continue
                    
                    for eeg_win in eeg_windows:
                        best_score = -float('inf')
                        best_match_idx = 0
                        best_metrics = {}
                        
                        eeg_note_count = eeg_win.get('note_count', 0)
                        
                        def _ratio_sim(a: float, b: float) -> float:
                            if not np.isfinite(a) or not np.isfinite(b):
                                return 0.0
                            return float(min(a, b) / (max(a, b) + 1e-8))

                        def _dtw_sim(seq1: np.ndarray, seq2: np.ndarray) -> float:
                            if len(seq1) < 2 or len(seq2) < 2:
                                return 0.0
                            len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
                            if len_ratio < 0.3:
                                return 0.0
                            s1 = np.nan_to_num(seq1, nan=0.0, posinf=0.0, neginf=0.0)
                            s2 = np.nan_to_num(seq2, nan=0.0, posinf=0.0, neginf=0.0)
                            s1 = s1 / (np.mean(s1) + 1e-6)
                            s2 = s2 / (np.mean(s2) + 1e-6)
                            s1 = np.clip(s1, 0, 4)
                            s2 = np.clip(s2, 0, 4)
                            dist = dtw_distance(s1, s2)
                            return float(np.exp(-dist) * len_ratio)

                        for j, cla_win in enumerate(valid_classical_wins):
                            cla_note_count = cla_win.get('note_count', 0)
                            note_ratio = min(eeg_note_count, cla_note_count) / max(eeg_note_count, cla_note_count)
                            if note_ratio < 0.02:
                                continue
                            
                            eeg_intervals = eeg_win.get('intervals_raw', np.array([0]))
                            cla_intervals = cla_win.get('intervals_raw', np.array([0]))
                            interval_sim = interval_similarity(eeg_intervals, cla_intervals)
                            interval_simple = _ratio_sim(
                                float(np.mean(np.abs(eeg_intervals))) if len(eeg_intervals) > 0 else 0.0,
                                float(np.mean(np.abs(cla_intervals))) if len(cla_intervals) > 0 else 0.0
                            )
                            interval_sim = max(interval_sim, interval_simple)
                            
                            eeg_pc = eeg_win.get('pitch_class_hist', np.zeros(12))
                            cla_pc = cla_win.get('pitch_class_hist', np.zeros(12))
                            harmony_sim = pitch_class_similarity(eeg_pc, cla_pc)
                            
                            eeg_pitches = eeg_win.get('pitches_raw', np.array([60]))
                            cla_pitches = cla_win.get('pitches_raw', np.array([60]))
                            
                            eeg_std = float(np.std(eeg_pitches))
                            cla_std = float(np.std(cla_pitches))
                            eeg_range = float(np.ptp(eeg_pitches))
                            cla_range = float(np.ptp(cla_pitches))
                            
                            # Эмоционный фильтр при match_emotions
                            classical_emotion = dataset_info.get('emotion') or _classical_meta_map.get(classical_name, {}).get('emotion')
                            if match_emotions:
                                if not eeg_emotion or not classical_emotion:
                                    continue
                                if eeg_emotion != classical_emotion:
                                    continue

                            contour_sim = contour_similarity_strict(eeg_pitches, cla_pitches)
                            eeg_ioi = eeg_win.get('ioi_raw', np.array([0.0]))
                            cla_ioi = cla_win.get('ioi_raw', np.array([0.0]))
                            rhythm_sim = _dtw_sim(eeg_ioi, cla_ioi)
                            tempo_sim = _ratio_sim(
                                float(np.mean(eeg_ioi)) if len(eeg_ioi) > 0 else 0.0,
                                float(np.mean(cla_ioi)) if len(cla_ioi) > 0 else 0.0
                            )
                            rhythm_sim = max(rhythm_sim, tempo_sim)
                            density_sim = _ratio_sim(eeg_win.get('note_density', 0), cla_win.get('note_density', 0))
                            dynamic_sim = dynamic_range_similarity(eeg_pitches, cla_pitches)

                            sfi_diff = abs(eeg_win.get('sfi_pitch', 0) - cla_win.get('sfi_pitch', 0))
                            sfi_sim = float(np.exp(-sfi_diff))

                            combined_score = (
                                0.30 * contour_sim +
                                0.20 * interval_sim +
                                0.20 * rhythm_sim +
                                0.20 * harmony_sim +
                                0.10 * dynamic_sim
                            ) * note_ratio
                            if not np.isfinite(combined_score):
                                continue

                            # Soft emotion penalty (если есть эмоции у обеих сторон)
                            emotion_match = None
                            if eeg_emotion and classical_emotion:
                                emotion_match = (eeg_emotion == classical_emotion)
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_match_idx = j
                                best_metrics = {
                                    'contour_similarity': contour_sim,
                                    'interval_similarity': interval_sim,
                                    'rhythm_similarity': rhythm_sim,
                                    'density_similarity': density_sim,
                                    'dynamic_similarity': dynamic_sim,
                                    'eeg_pitch_std': eeg_std,
                                    'cla_pitch_std': cla_std,
                                    'eeg_pitch_range': eeg_range,
                                    'cla_pitch_range': cla_range,
                                    'harmony_similarity': harmony_sim,
                                    'sfi_similarity': sfi_sim,
                                    'eeg_note_count': eeg_note_count,
                                    'cla_note_count': cla_note_count,
                                    'emotion_match': emotion_match,
                                }
                        
                        if best_score < 0:
                            continue
                        
                        best_match = valid_classical_wins[best_match_idx]
                        meta = _classical_meta_map.get(classical_name, {})
                        variant_results.append({
                            'eeg_window_id': eeg_win['window_id'],
                            'eeg_start_time': eeg_win['start_time'],
                            'eeg_emotion': eeg_emotion,
                            'classical_piece': classical_name,
                            'classical_dataset': dataset_info.get('dataset', 'maestro'),
                            'classical_track_id': dataset_info.get('track_id', ''),
                            'classical_midi_path': _classical_path_map.get(classical_name, ''),
                            'classical_title': meta.get('title', None),
                            'classical_composer': meta.get('composer', None),
                            'classical_emotion': meta.get('emotion', None),
                            'classical_emotion_source': meta.get('emotion_source', None),
                            'classical_window_id': best_match['window_id'],
                            'classical_start_time': best_match['start_time'],
                            'combined_similarity': best_score,
                            **best_metrics
                        })
                
                if variant_results:
                    variant_df = pd.DataFrame(variant_results)
                    
                    # === EMOTION CHECK: не фильтруем, только сохраняем совпадение ===
                    # Строгое сравнение по эмоциям отключено; используем мягкий штраф (EMOTION_MISMATCH_PENALTY)
                    # === END EMOTION CHECK ===

                    # Балансировка top-K по датасетам: гарантируем присутствие EMOPIA
                    top_k = 5
                    if 'classical_dataset' in variant_df.columns:
                        emopia_df = variant_df[variant_df['classical_dataset'] == 'emopia']
                        maestro_df = variant_df[variant_df['classical_dataset'] == 'maestro']

                        if len(emopia_df) > 0:
                            emopia_k = max(1, round(top_k * 0.4))
                            maestro_k = max(0, top_k - emopia_k)
                            best_emopia = emopia_df.nlargest(emopia_k, 'combined_similarity')
                            best_maestro = maestro_df.nlargest(maestro_k, 'combined_similarity')
                            best = pd.concat([best_emopia, best_maestro], ignore_index=True)
                            best = best.nlargest(top_k, 'combined_similarity')
                        else:
                            best = variant_df.nlargest(top_k, 'combined_similarity')
                    else:
                        best = variant_df.nlargest(top_k, 'combined_similarity')

                    best = best.copy()
                    best['rank'] = range(1, len(best) + 1)
                    
                    for _, row in best.iterrows():
                        results.append({
                            'participant_id': participant_id,
                            'trial_idx': trial_idx,
                            'variant': variant_name,
                            'valence': valence,
                            'arousal': arousal,
                            'eeg_emotion': eeg_emotion,
                            'eeg_midi': str(eeg_midi_path),
                            'classical_piece': row['classical_piece'],
                            'classical_dataset': row.get('classical_dataset', 'maestro'),
                            'classical_track_id': row.get('classical_track_id', ''),
                            'classical_midi_path': row.get('classical_midi_path', ''),
                            'classical_title': row.get('classical_title', None),
                            'classical_composer': row.get('classical_composer', None),
                            'classical_emotion': row.get('classical_emotion', None),
                            'classical_emotion_source': row.get('classical_emotion_source', None),
                            'eeg_window_id': row['eeg_window_id'],
                            'eeg_start_time': row['eeg_start_time'],
                            'classical_window_id': row['classical_window_id'],
                            'classical_start_time': row['classical_start_time'],
                            'combined_similarity': row['combined_similarity'],
                            'contour_similarity': row['contour_similarity'],
                            'interval_similarity': row.get('interval_similarity', 0),
                            'harmony_similarity': row['harmony_similarity'],
                            'sfi_similarity': row.get('sfi_similarity', 0),
                            'eeg_note_count': row.get('eeg_note_count', 0),
                            'cla_note_count': row.get('cla_note_count', 0),
                            'eeg_pitch_std': row.get('eeg_pitch_std', 0),
                            'cla_pitch_std': row.get('cla_pitch_std', 0),
                            'eeg_pitch_range': row.get('eeg_pitch_range', 0),
                            'cla_pitch_range': row.get('cla_pitch_range', 0),
                            'rank': row['rank']
                        })
            except Exception as e:
                continue
        
        return participant_id, trial_idx, results, None
    except Exception as e:
        return participant_id, trial_idx, [], str(e)


def run_comparison(max_participants: int = 2, 
                   max_trials: int = 3,
                   max_classical: int = 10,
                   top_k: int = 10,
                   n_jobs: int = None,
                   only_emopia: bool = False,
                   balanced_eeg_emotions: bool = False,
                   per_emotion_trials: int = 3,
                   match_emotions: bool = False):
    """
    Основной пайплайн сравнения.
    
    Оптимизации:
    - Окна классических произведений вычисляются один раз и кэшируются
    - Параллельная обработка классических файлов
    - Только окна EEG пересчитываются для каждого варианта
    
    Параметры:
    - max_participants: сколько участников обработать
    - max_trials: сколько триалов на участника
    - max_classical: сколько классических произведений загружать
    - top_k: сколько лучших совпадений включить в отчёт
    - n_jobs: количество параллельных процессов (None = auto)
    - only_emopia: использовать только EMOPIA датасет (без MAESTRO)
    - balanced_eeg_emotions: сбалансировать EEG по 4 эмоциональным квадрантам
    - per_emotion_trials: сколько триалов брать на каждую эмоцию (для балансировки)
    - match_emotions: сравнивать только классические произведения с совпадающей эмоцией
    """
    import music21 as m21
    from src.MIDIComparator import extract_window_features
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import euclidean
    from scipy.stats import pearsonr
    
    print("=" * 60)
    print("EEG to Classical Music Comparison Pipeline")
    print("=" * 60)
    
    # Выводим конфигурацию запуска
    print("\nConfiguration:")
    print(f"  Emotion Threshold: {EMOTION_THRESHOLD}")
    print(f"  Match Emotions: {'✓ YES' if match_emotions else '✗ NO'}")

    # Директория текущего запуска
    run_dir = RUNS_DIR / DEFAULT_RUN_ID
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Загружаем классические произведения (MAESTRO + EMOPIA)
    print("\n[1/5] Загрузка датасетов...")
    
    maestro_files = []
    if not only_emopia:
        print("  [MAESTRO] Загрузка...")
        all_maestro_files = get_maestro_midi_files(str(MAESTRO_DIR), max_files=None)
        random.shuffle(all_maestro_files)
        maestro_files = all_maestro_files[:max_classical]
        print(f"  [MAESTRO] Загружено {len(maestro_files)} произведений (случайная выборка из {len(all_maestro_files)})")
    else:
        print("  [MAESTRO] Пропуск (only_emopia=True)")
    
    # EMOPIA (если включён)
    emopia_files = []
    if USE_EMOPIA or only_emopia:
        try:
            print("  [EMOPIA] Загрузка...")
            all_emopia_files = get_emopia_midi_files(str(EMOPIA_DIR), max_files=EMOPIA_MAX_TRACKS)
            random.shuffle(all_emopia_files)
            emopia_files = all_emopia_files[:max_classical]  # берём столько же, сколько MAESTRO
            print(f"  [EMOPIA] Загружено {len(emopia_files)} произведений (из {len(all_emopia_files)})")
        except Exception as e:
            print(f"  [EMOPIA] ОШИБКА загрузки: {e}")
            if only_emopia:
                print("  [EMOPIA] only_emopia=True, нет данных для сравнения")
            else:
                print(f"  [EMOPIA] Продолжаем только с MAESTRO")
    
    # Объединяем датасеты
    all_classical_files = maestro_files + emopia_files
    print(f"  Всего произведений для сравнения: {len(all_classical_files)}")

    if match_emotions and not only_emopia:
        # Проверяем, есть ли эмоции в MAESTRO метаданных
        maestro_has_emotions = False
        for path in maestro_files[:5]:
            meta = get_maestro_metadata(path)
            if meta.get('emotion'):
                maestro_has_emotions = True
                break
        if not maestro_has_emotions and not USE_PSEUDO_LABELING:
            print("  ⚠ Match Emotions enabled, но MAESTRO без эмоций. Рекомендуется --only-emopia или включить pseudo-labeling.")
    
    if not all_classical_files:
        print("ОШИБКА: Не найдены MIDI файлы")
        return
    
    
    # Создаём словарь {название: путь} с указанием источника
    classical_dict = {}
    classical_meta_map = {}
    for path in maestro_files:
        meta = get_maestro_metadata(Path(path).name)
        composer = meta.get('composer', 'Unknown')
        title = meta.get('title', Path(path).stem)[:30]
        track_id = meta.get('track_id', Path(path).stem)
        # Формат ключа: "dataset|track_id|display_name"
        name = f"maestro|{track_id}|{composer} - {title}"
        classical_dict[name] = path
        classical_meta_map[name] = {
            'dataset': 'maestro',
            'track_id': track_id,
            'composer': composer,
            'title': title,
            'emotion': meta.get('emotion'),
            'emotion_source': meta.get('emotion_source')
        }
    
    for path in emopia_files:
        # Извлекаем track_id из имени файла (например, Q1_xxx_0.mid → Q1_xxx_0)
        track_id = Path(path).stem
        meta = get_emopia_metadata(track_id)
        emotion = meta.get('emotion', 'Unknown')
        info = get_emopia_track_info(str(EMOPIA_DIR), track_id)
        title = info.get('title') or track_id
        uploader = info.get('uploader') or 'YouTube'
        clip_idx = info.get('clip_idx')
        clip_suffix = f" (clip {clip_idx})" if clip_idx is not None else ""
        # Формат: "emopia|track_id|Uploader - Title (clip n) [Emotion]"
        name = f"emopia|{track_id}|{uploader} - {title}{clip_suffix} [{emotion}]"
        classical_dict[name] = path
        classical_meta_map[name] = {
            'dataset': 'emopia',
            'track_id': track_id,
            'composer': uploader,
            'title': f"{title}{clip_suffix}",
            'emotion': emotion,
            'emotion_source': 'ground_truth'
        }

    # Опциональный pseudo-labeling для MAESTRO на основе ближайшего EMOPIA
    maestro_pseudo_emotions = {}
    if USE_PSEUDO_LABELING and emopia_files:
        print("\n[1b/5] Pseudo-labeling для MAESTRO по EMOPIA...")
        # 1) Пробуем загрузить заранее рассчитанные псевдо-метки
        if MAESTRO_PSEUDO_LABELS_PATH.exists():
            try:
                pseudo_df = pd.read_csv(MAESTRO_PSEUDO_LABELS_PATH)
                for _, row in pseudo_df.iterrows():
                    maestro_pseudo_emotions[str(row['track_id'])] = {
                        'emotion': row.get('emotion'),
                        'confidence': float(row.get('confidence', 0))
                    }
                print(f"  Загружены псевдо-метки MAESTRO: {len(maestro_pseudo_emotions)}")
            except Exception as e:
                print(f"  Не удалось загрузить псевдо-метки: {e}")

        # 2) Если нет файла — считаем на лету для выбранной подвыборки
        feature_cache_path = report_dir / "feature_cache.json"
        feature_cache = load_feature_cache(feature_cache_path)

        emopia_vectors = []
        emopia_emotions = []
        emopia_paths = []

        if not maestro_pseudo_emotions:
            for path in emopia_files:
                feats = get_or_compute_features(path, feature_cache)
                if feats is None:
                    continue
                emopia_vectors.append(features_to_vector(feats))
                emopia_paths.append(path)
                emopia_emotions.append(get_emopia_metadata(Path(path).stem).get('emotion', None))

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
                best_emotion = emopia_emotions[best_idx]
                best_dist = float(dists[best_idx])
                confidence = 1.0 / (1.0 + best_dist)
                maestro_pseudo_emotions[Path(path).stem] = {
                    'emotion': best_emotion,
                    'confidence': confidence
                }

        save_feature_cache(feature_cache_path, feature_cache)
    
    # 2. Предварительно вычисляем окна для всех классических произведений (ПАРАЛЛЕЛЬНО)
    print("\n[2/5] Предварительный расчёт окон классических произведений...")
    classical_windows_cache = {}
    classical_midis_cache = {}
    
    # Определяем количество процессов
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Параллельная обработка
    if n_jobs > 1 and len(classical_dict) > 3:
        print(f"  Используем {n_jobs} процессов...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_process_classical_file, (name, path)): name 
                      for name, path in classical_dict.items()}
            
            completed = 0
            for future in as_completed(futures):
                name, windows, error = future.result()
                completed += 1
                if error:
                    print(f"    [{completed}/{len(classical_dict)}] ОШИБКА {name[:40]}: {error}")
                else:
                    classical_windows_cache[name] = windows
                    if completed % 5 == 0 or completed == len(classical_dict):
                        print(f"    [{completed}/{len(classical_dict)}] Обработано")
    else:
        # Последовательная обработка для малого количества файлов
        for name, path in classical_dict.items():
            try:
                midi = m21.converter.parse(path)
                classical_midis_cache[name] = midi
                
                # Извлекаем окна
                total_duration = midi.duration.quarterLength
                windows = []
                start_time = 0.0
                window_id = 0
                
                while start_time + COMPARISON_WINDOW_SIZE <= total_duration:
                    end_time = start_time + COMPARISON_WINDOW_SIZE
                    window_stream = midi.flatten().getElementsByOffset(
                        start_time, end_time,
                        includeEndBoundary=False,
                        mustFinishInSpan=False,
                        mustBeginInSpan=True
                    )
                    features = extract_window_features(window_stream, start_time, end_time)
                    if features and features.get('note_count', 0) > 0:
                        features['window_id'] = window_id
                        features['start_time'] = start_time
                        features['end_time'] = end_time
                        features['source'] = name
                        windows.append(features)
                        window_id += 1
                    start_time += COMPARISON_HOP_SIZE
                
                classical_windows_cache[name] = windows
                print(f"    {name[:50]}: {len(windows)} окон")
            except Exception as e:
                print(f"    ОШИБКА {name}: {e}")
    
    print(f"  Всего кэшировано: {sum(len(w) for w in classical_windows_cache.values())} окон")
    
    # Копируем кэш в глобальную переменную для worker'ов
    global _classical_cache, _classical_path_map, _classical_meta_map
    _classical_cache = classical_windows_cache
    _classical_path_map = classical_dict
    _classical_meta_map = classical_meta_map
    
    # 3. Создаём директории для результатов
    eeg_midi_dir = BEST_MATCHES_DIR / "eeg_midi"
    if CLEAN_DATA_ON_RUN and eeg_midi_dir.exists():
        import shutil
        shutil.rmtree(eeg_midi_dir)
        print("  🗑  Очищена директория eeg_midi (CLEAN_DATA_ON_RUN=True)")
    eeg_midi_dir.mkdir(parents=True, exist_ok=True)
    
    if REUSE_EEG_MIDI:
        existing_midi = list(eeg_midi_dir.glob('*.mid'))
        if existing_midi:
            print(f"  ♻ REUSE_EEG_MIDI: найдено {len(existing_midi)} ранее созданных MIDI")
    
    # 4. Обрабатываем DEAP триалы (ПАРАЛЛЕЛЬНО)
    print(f"\n[3/5] Обработка DEAP триалов...")
    all_results = []
    
    participant_files = sorted(DEAP_DIR.glob('s*.dat'))[:max_participants]
    
    # Создаём список всех триалов для параллельной обработки
    trial_tasks = []
    if balanced_eeg_emotions:
        print(f"  Балансировка эмоций: по {per_emotion_trials} триала на эмоцию")
        emotion_counts = {e: 0 for e in get_all_emotions()}
        for participant_file in participant_files:
            data = load_deap_participant_data(str(participant_file))
            for trial_idx in range(40):
                labels = get_emotion_labels(data, trial_idx)
                valence = labels.get('valence', 5.0)
                arousal = labels.get('arousal', 5.0)
                emotion = deap_to_emotion_quadrant(valence, arousal, threshold=EMOTION_THRESHOLD)
                if emotion_counts[emotion] < per_emotion_trials:
                    trial_tasks.append((str(participant_file), trial_idx, eeg_midi_dir, match_emotions))
                    emotion_counts[emotion] += 1
                if all(c >= per_emotion_trials for c in emotion_counts.values()):
                    break
            if all(c >= per_emotion_trials for c in emotion_counts.values()):
                break
    else:
        for participant_file in participant_files:
            for trial_idx in range(min(max_trials, 40)):
                trial_tasks.append((str(participant_file), trial_idx, eeg_midi_dir, match_emotions))
    
    print(f"  Всего триалов: {len(trial_tasks)}")
    
    # Параллельная обработка триалов
    if n_jobs > 1 and len(trial_tasks) > 1:
        print(f"  Используем {n_jobs} процессов...")
        completed = 0
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(classical_windows_cache, classical_dict, classical_meta_map)) as executor:
            futures = {executor.submit(_process_trial, task): task for task in trial_tasks}
            
            for future in as_completed(futures):
                participant_id, trial_idx, results, error = future.result()
                completed += 1
                if error:
                    print(f"    [{completed}/{len(trial_tasks)}] {participant_id}/trial{trial_idx}: ОШИБКА {error}")
                else:
                    all_results.extend(results)
                    if results:
                        best = max(results, key=lambda x: x['combined_similarity'])
                        print(f"    [{completed}/{len(trial_tasks)}] {participant_id}/trial{trial_idx}: {best['classical_piece'][:30]} ({best['combined_similarity']:.3f})")
                    else:
                        print(f"    [{completed}/{len(trial_tasks)}] {participant_id}/trial{trial_idx}: нет совпадений")
    else:
        # Последовательная обработка
        for i, task in enumerate(trial_tasks):
            participant_id, trial_idx, results, error = _process_trial(task)
            if error:
                print(f"    [{i+1}/{len(trial_tasks)}] {participant_id}/trial{trial_idx}: ОШИБКА {error}")
            else:
                all_results.extend(results)
                if results:
                    best = max(results, key=lambda x: x['combined_similarity'])
                    print(f"    [{i+1}/{len(trial_tasks)}] {participant_id}/trial{trial_idx}: {best['classical_piece'][:30]} ({best['combined_similarity']:.3f})")
    
    # 5. Создаём итоговый DataFrame и сохраняем результаты
    print("\n[4/5] Формирование результатов...")
    
    if not all_results:
        print("ОШИБКА: Нет результатов для отчёта")
        return
    
    results_df = pd.DataFrame(all_results)

    # Добавляем информацию об эмоции для композиций (EMOPIA / pseudo-labeling)
    if 'classical_dataset' in results_df.columns:
        def _resolve_emotion(row):
            dataset = row.get('classical_dataset', 'maestro')
            track_id = row.get('classical_track_id', '')
            if dataset == 'emopia' and track_id:
                meta = get_emopia_metadata(track_id)
                return meta.get('emotion'), 'ground_truth'
            if dataset == 'maestro' and track_id:
                meta = get_maestro_metadata(row.get('classical_midi_path', track_id))
                if meta.get('emotion'):
                    return meta.get('emotion'), meta.get('emotion_source') or 'predicted'
                if USE_PSEUDO_LABELING and track_id:
                    pseudo = maestro_pseudo_emotions.get(track_id)
                    if pseudo:
                        return pseudo.get('emotion'), 'predicted'
            return None, None

        emotions = results_df.apply(lambda r: _resolve_emotion(r), axis=1)
        results_df['classical_emotion'] = emotions.apply(lambda x: x[0])
        results_df['classical_emotion_source'] = emotions.apply(lambda x: x[1])
    
    # Сортируем по combined_similarity (больше = лучше)
    results_df = results_df.sort_values('combined_similarity', ascending=False)
    
    # Сохраняем полные результаты в CSV
    results_csv_path = report_dir / "comparison_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"  Результаты сохранены: {results_csv_path}")
    
    # Берём топ-K
    top_results = results_df.head(top_k)
    
    print(f"  Всего результатов: {len(results_df)}")
    print(f"  Топ-{top_k} лучших совпадений (по combined_similarity):")
    
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        eeg_notes = row.get('eeg_note_count', 0)
        cla_notes = row.get('cla_note_count', 0)
        eeg_std = row.get('eeg_pitch_std', 0)
        cla_std = row.get('cla_pitch_std', 0)
        print(f"    {idx}. {row['participant_id']}/{row['trial_idx']} ({row['variant']}) → "
              f"{row['classical_piece'][:40]}")
        print(f"       Combined={row['combined_similarity']:.3f} "
              f"(contour={row['contour_similarity']:.2f}, interval={row.get('interval_similarity', 0):.2f}, "
              f"harmony={row['harmony_similarity']:.2f})")
        print(f"       Notes: {eeg_notes}/{cla_notes} | Variability: EEG={eeg_std:.1f}, Cla={cla_std:.1f} | "
              f"V={row['valence']:.1f}, A={row['arousal']:.1f}")
    
    # 5. Генерируем HTML
    print("\n[5/5] Генерация HTML отчёта...")
    
    html_path = report_dir / "index.html"
    
    # Prepare per-match MIDI fragments (EEG fragment, classical fragment, combined)
    matches_out_dir = BEST_MATCHES_DIR
    matches_out_dir.mkdir(parents=True, exist_ok=True)

    comp_rows = []
    window_size = HTML_FRAGMENT_DURATION  # Используем параметр из config

    for rank, (_, row) in enumerate(top_results.reset_index(drop=True).iterrows(), start=1):
        participant = row.get('participant_id', 'unknown')
        trial = int(row.get('trial_idx', 0))
        variant = row.get('variant', '')
        classical_name = row.get('classical_piece')
        classical_dataset = row.get('classical_dataset', 'maestro')
        classical_track_id = row.get('classical_track_id', '')
        eeg_emotion = row.get('eeg_emotion', 'Unknown')
        eeg_midi_path = Path(row.get('eeg_midi'))
        eeg_start = float(row.get('eeg_start_time', 0.0))
        classical_start = float(row.get('classical_start_time', 0.0))

        # Извлекаем имя для отображения (убираем префикс dataset|track_id|)
        if '|' in classical_name:
            parts = classical_name.split('|', 2)
            classical_display_name = parts[2] if len(parts) > 2 else classical_name
        else:
            classical_display_name = classical_name

        # Parse composer/title or use explicit meta
        composer = row.get('classical_composer')
        title = row.get('classical_title')
        if not composer or not title:
            try:
                if ' - ' in classical_display_name:
                    composer, title = classical_display_name.split(' - ', 1)
                elif ': ' in classical_display_name:
                    composer, title = classical_display_name.split(': ', 1)
                else:
                    composer = classical_dataset.upper()
                    title = classical_display_name
            except Exception:
                composer = 'Unknown'
                title = classical_display_name
        
        # Получаем эмоцию для EMOPIA треков / pseudo-labeling для MAESTRO
        classical_emotion = row.get('classical_emotion')
        classical_emotion_source = row.get('classical_emotion_source')
        if not classical_emotion:
            if classical_dataset == 'emopia' and classical_track_id:
                meta = get_emopia_metadata(classical_track_id)
                classical_emotion = meta.get('emotion')
                classical_emotion_source = 'ground_truth'
            elif classical_dataset == 'maestro' and classical_track_id and USE_PSEUDO_LABELING:
                pseudo = maestro_pseudo_emotions.get(classical_track_id)
                if pseudo:
                    classical_emotion = pseudo.get('emotion')
                    classical_emotion_source = 'predicted'

        # Безопасные имена файлов: убираем все проблемные символы
        def sanitize_filename(s, max_len=40):
            """Очищает строку для использования в имени файла."""
            # Заменяем проблемные символы
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
                s = s.replace(char, '_')
            # Убираем множественные подчеркивания
            while '__' in s:
                s = s.replace('__', '_')
            return s.strip('_')[:max_len]
        
        safe_title = sanitize_filename(title, 40)
        safe_comp = sanitize_filename(composer, 30)

        prefix = f"{rank:02d}_"
        variant_prefix = f"{variant}_" if variant else ""

        # EEG fragment: extract pitches around eeg_start from eeg_midi
        eeg_notes = extract_melody_with_time(str(eeg_midi_path)) if eeg_midi_path.exists() else []
        eeg_fragment_pitches = []
        
        # Пробуем извлечь ноты из указанного окна
        for p, t in eeg_notes:
            if t >= eeg_start and t < eeg_start + window_size:
                eeg_fragment_pitches.append(int(p))
        
        # Если ничего не нашли в окне, берём все ноты из начала файла (первые N секунд)
        if not eeg_fragment_pitches and eeg_notes:
            extended_window = window_size * 2  # расширяем окно до 16 секунд
            for p, t in eeg_notes:
                if t < extended_window:
                    eeg_fragment_pitches.append(int(p))
                if len(eeg_fragment_pitches) >= 50:  # лимит нот
                    break
        
        # Последний fallback: берём первые ноты без привязки ко времени
        if not eeg_fragment_pitches and eeg_notes:
            eeg_fragment_pitches = [int(p) for p, t in eeg_notes[:50]]

        # Classical fragment: extract from original classical midi
        classical_path = classical_dict.get(classical_name) if classical_dict else None

        cla_fragment_pitches = []
        if classical_path:
            cla_notes = extract_melody_with_time(str(classical_path))
            
            # Извлекаем ноты из окна
            for p, t in cla_notes:
                if t >= classical_start and t < classical_start + window_size:
                    cla_fragment_pitches.append(int(p))
            
            # Если ничего не нашли, расширяем окно
            if not cla_fragment_pitches and cla_notes:
                extended_window = window_size * 2
                for p, t in cla_notes:
                    if t >= classical_start and t < classical_start + extended_window:
                        cla_fragment_pitches.append(int(p))
                    if len(cla_fragment_pitches) >= 100:
                        break
            
            # Fallback: если classical_start за пределами файла, берём ноты с начала
            if not cla_fragment_pitches and cla_notes:
                # Берём ноты от начала файла (первые N секунд)
                for p, t in cla_notes:
                    if t < window_size * 2:
                        cla_fragment_pitches.append(int(p))
                    if len(cla_fragment_pitches) >= 100:
                        break
            
            # Последний fallback: просто первые 50 нот
            if not cla_fragment_pitches and cla_notes:
                cla_fragment_pitches = [int(p) for p, t in cla_notes[:50]]

        # === СИНХРОНИЗАЦИЯ ДЛИНЫ ФРАГМЕНТОВ ===
        # Для корректного сравнения оба фрагмента должны иметь одинаковое количество нот
        if eeg_fragment_pitches and cla_fragment_pitches:
            min_notes = min(len(eeg_fragment_pitches), len(cla_fragment_pitches))
            # Берём минимум, но не меньше 10 нот для осмысленного сравнения
            target_notes = max(min_notes, 10)
            eeg_fragment_pitches = eeg_fragment_pitches[:target_notes]
            cla_fragment_pitches = cla_fragment_pitches[:target_notes]
            print(f"    Synchronized: {target_notes} notes each")

        # Используем увеличенный темп для более быстрого воспроизведения
        playback_tempo = int(120 * PLAYBACK_TEMPO_MULTIPLIER)
        
        eeg_mid_name = f"{prefix}{variant_prefix}EEG_{safe_comp}_{safe_title}.mid"
        eeg_mid_out = matches_out_dir / eeg_mid_name
        if eeg_fragment_pitches:
            # Всегда перезаписываем чтобы обновить фрагменты
            create_midi_from_notes(eeg_fragment_pitches, str(eeg_mid_out), tempo_bpm=playback_tempo)
            print(f"    EEG fragment: {len(eeg_fragment_pitches)} notes → {eeg_mid_name}")
        elif eeg_midi_path.exists():
            # fallback: copy full EEG midi
            try:
                shutil.copy(str(eeg_midi_path), str(eeg_mid_out))
                print(f"    EEG fallback: copied full file → {eeg_mid_name}")
            except Exception as e:
                print(f"    EEG copy error: {e}")

        cla_mid_name = f"{prefix}{variant_prefix}Classical_{safe_comp}_{safe_title}.mid"
        cla_mid_out = matches_out_dir / cla_mid_name
        if cla_fragment_pitches:
            create_midi_from_notes(cla_fragment_pitches, str(cla_mid_out), tempo_bpm=playback_tempo)
            print(f"    Classical fragment: {len(cla_fragment_pitches)} notes → {cla_mid_name}")
        elif classical_path:
            try:
                shutil.copy(str(classical_path), str(cla_mid_out))
                print(f"    Classical fallback: copied full file → {cla_mid_name}")
            except Exception as e:
                print(f"    Classical copy error: {e}")

        # Comparison MIDI (both tracks)
        cmp_mid_name = f"{prefix}{variant_prefix}Comparison_{safe_comp}_{safe_title}.mid"
        cmp_mid_out = matches_out_dir / cmp_mid_name
        if eeg_fragment_pitches or cla_fragment_pitches:
            try:
                create_comparison_midi(eeg_fragment_pitches or [], cla_fragment_pitches or [], str(cmp_mid_out), tempo_bpm=playback_tempo)
                total_notes = len(eeg_fragment_pitches or []) + len(cla_fragment_pitches or [])
                print(f"    Comparison: {total_notes} total notes → {cmp_mid_name}")
            except Exception as e:
                print(f"    Comparison error: {e}")

        # Prepare row for HTML generator
        comp_rows.append({
            'file': cla_mid_out.name,
            'eeg_midi_path': str(eeg_mid_out),
            'classical_midi_path': str(cla_mid_out),
            'comparison_midi_path': str(cmp_mid_out),
            'composer': composer,
            'title': title,
            'variant': variant,
            'trial': f"Trial {trial}",
            'processing': variant,
            'eeg_valence': row.get('valence'),
            'eeg_arousal': row.get('arousal'),
            'eeg_emotion': eeg_emotion,
            'participant_id': participant,
            'classical_dataset': classical_dataset,
            'classical_emotion': classical_emotion,
            'classical_emotion_source': classical_emotion_source,
            'emotion_match': row.get('emotion_match', None),
            # Все метрики сходства
            'combined_similarity': float(row.get('combined_similarity', 0.0)),
            'contour_similarity': float(row.get('contour_similarity', 0.0)),
            'correlation_similarity': float(row.get('correlation_similarity', 0.0)),
            'harmony_similarity': float(row.get('harmony_similarity', 0.0)),
            'sfi_similarity': float(row.get('sfi_similarity', 0.0)),
            'stat_similarity': float(row.get('stat_similarity', 0.0)),
            # Для совместимости с HTML генератором
            'melodic_similarity': float(row.get('combined_similarity', 0.0)),
        })

    comp_df = pd.DataFrame(comp_rows)

    # Дедупликация карточек: оставляем лучший результат для каждого уникального совпадения
    dedup_cols = ['composer', 'title', 'eeg_emotion', 'trial', 'variant']
    if all(c in comp_df.columns for c in dedup_cols) and 'combined_similarity' in comp_df.columns:
        before = len(comp_df)
        comp_df = comp_df.sort_values('combined_similarity', ascending=False)
        comp_df = comp_df.drop_duplicates(subset=dedup_cols, keep='first')
        if len(comp_df) < before:
            print(f"  Дедупликация: {before} → {len(comp_df)} карточек")

    # Generate rich HTML with WAVs and similarity plots
    try:
        create_comparison_html(comp_df, saved_matches=None, output_path=str(html_path), convert_to_wav=True, media_dir=str(matches_out_dir))
    except Exception as e:
        print(f"Ошибка при создании расширенного HTML: {e}")
    
    print(f"\n✓ Отчёт сохранён: {html_path}")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG to Classical Music Comparison')
    parser.add_argument('--participants', type=int, default=DEFAULT_MAX_PARTICIPANTS, 
                        help='Количество участников DEAP')
    parser.add_argument('--trials', type=int, default=DEFAULT_MAX_TRIALS,
                        help='Количество триалов на участника')
    parser.add_argument('--classical', type=int, default=DEFAULT_MAX_CLASSICAL,
                        help='Количество классических произведений')
    parser.add_argument('--top', type=int, default=DEFAULT_TOP_K,
                        help='Количество лучших результатов в отчёте')
    parser.add_argument('--jobs', type=int, default=DEFAULT_JOBS,
                        help='Количество параллельных процессов (по умолчанию: авто)')
    parser.add_argument('--only-emopia', action='store_true', default=DEFAULT_ONLY_EMOPIA,
                        help='Сравнивать только с EMOPIA (без MAESTRO)')
    parser.add_argument('--balanced-eeg-emotions', action='store_true', default=DEFAULT_BALANCED_EEG_EMOTIONS,
                        help='Сбалансировать EEG по эмоциям (квадранты VA)')
    parser.add_argument('--per-emotion-trials', type=int, default=DEFAULT_PER_EMOTION_TRIALS,
                        help='Сколько EEG триалов брать на каждую эмоцию при балансировке')
    parser.add_argument('--match-emotions', action='store_true', default=DEFAULT_MATCH_EMOTIONS,
                        help='Сравнивать только с произведениями одной эмоции (EEG и classical должны совпадать)')
    parser.add_argument('--reuse-eeg-midi', action='store_true', default=REUSE_EEG_MIDI,
                        help='Переиспользовать ранее сгенерированные EEG MIDI файлы')
    parser.add_argument('--no-reuse-eeg-midi', dest='reuse_eeg_midi', action='store_false',
                        help='Всегда перегенерировать EEG MIDI')
    parser.add_argument('--clean', action='store_true', default=CLEAN_DATA_ON_RUN,
                        help='Очистить eeg_midi директорию перед запуском')
    
    args = parser.parse_args()
    
    # Устанавливаем конфиг из CLI
    import src.config as _cfg
    _cfg.REUSE_EEG_MIDI = args.reuse_eeg_midi
    _cfg.CLEAN_DATA_ON_RUN = args.clean
    
    run_comparison(
        max_participants=args.participants,
        max_trials=args.trials,
        max_classical=args.classical,
        top_k=args.top,
        n_jobs=args.jobs,
        only_emopia=args.only_emopia,
        balanced_eeg_emotions=args.balanced_eeg_emotions,
        per_emotion_trials=args.per_emotion_trials,
        match_emotions=args.match_emotions
    )
