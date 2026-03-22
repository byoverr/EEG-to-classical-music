"""
Полный модуль сравнения EEG-MIDI с классическими произведениями (MAESTRO dataset).
Объединяет методы Wu (2018) и Miranda (2010) с фрагментарным анализом.

Для дипломной работы: "Методы преобразования сигналов ЭЭГ в музыкальные структуры"

Автор: [Ваше имя]
Дата: 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import music21 as m21
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr, linregress, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from pathlib import Path

# DTW с fallback (тихо)
try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    # Warning только если не подавлен через warnings.filterwarnings
    warnings.warn("dtaidistance not installed. Using fallback DTW.", UserWarning)


# ============================================================================
# РАЗДЕЛ 1: SCALE-FREE ANALYSIS (Wu 2018)
# ============================================================================

def calculate_scale_free_index(pitches: np.ndarray, return_details: bool = False):
    """
    Scale-Free Index по методу Wu (2018).
    Основная метрика для сравнения ЭЭГ-музыки с классикой.
    
    Применяется Zipf's law: rank ~ frequency^(-α)
    Для музыки α ≈ 1.0 ("эстетичная" музыка, следует 1/f закону).
    
    Параметры:
    - pitches: массив MIDI питчей
    - return_details: вернуть ли детали фиттинга
    
    Возвращает:
    - scale_free_exponent: показатель степени (близость к 1.0)
    - r_squared: качество фиттинга (>0.8 хорошо)
    """
    if len(pitches) < 3:
        if return_details:
            return 0.0, 0.0, {}
        return 0.0, 0.0
    
    # Подсчет частот питчей
    pitch_counts = Counter(pitches)
    sorted_counts = sorted(pitch_counts.values(), reverse=True)
    
    if len(sorted_counts) < 3:
        if return_details:
            return 0.0, 0.0, {}
        return 0.0, 0.0
    
    # Zipf's law: log(rank) vs log(frequency)
    ranks = np.arange(1, len(sorted_counts) + 1)
    frequencies = np.array(sorted_counts)
    
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    
    # Линейная регрессия
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    
    scale_free_exponent = -slope
    r_squared = r_value ** 2
    
    if return_details:
        details = {
            'exponent': scale_free_exponent,
            'r_squared': r_squared,
            'p_value': p_value,
            'unique_pitches': len(sorted_counts),
            'total_notes': len(pitches),
            'log_ranks': log_ranks,
            'log_freqs': log_freqs,
            'slope': slope,
            'intercept': intercept
        }
        return scale_free_exponent, r_squared, details
    
    return scale_free_exponent, r_squared


def calculate_ioi_scale_free(ioi_sequence: np.ndarray):
    """Scale-free анализ для Inter-Onset Intervals (ритм)."""
    if len(ioi_sequence) < 3:
        return 0.0, 0.0
    
    ioi_rounded = np.round(ioi_sequence * 100) / 100  # Округление для группировки
    ioi_counts = Counter(ioi_rounded[ioi_rounded > 0])
    
    sorted_counts = sorted(ioi_counts.values(), reverse=True)
    
    if len(sorted_counts) < 3:
        return 0.0, 0.0
    
    ranks = np.arange(1, len(sorted_counts) + 1)
    frequencies = np.array(sorted_counts)
    
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    
    return -slope, r_value ** 2


# ============================================================================
# РАЗДЕЛ 2: CONSONANCE-DISSONANCE ANALYSIS (Wu 2018)
# ============================================================================

def calculate_consonance_fluctuation(pitches_over_time: List[List[int]]):
    """
    Анализ флуктуаций консонанса/диссонанса.
    Wu (2018) использовал для оценки "эстетичности" музыки.
    
    Возвращает:
    - mean_consonance: средний консонанс [0, 1]
    - consonance_fluctuation: стандартное отклонение
    - consonance_series: временной ряд
    """
    # Консонантные интервалы (в полутонах)
    consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12, 15, 16, 19, 20, 24}
    
    consonance_scores = []
    
    for pitches in pitches_over_time:
        if len(pitches) < 2:
            consonance_scores.append(0.5)
            continue
        
        pitches_unique = sorted(set(pitches))
        intervals = []
        
        for i in range(len(pitches_unique)):
            for j in range(i + 1, len(pitches_unique)):
                interval = abs(pitches_unique[j] - pitches_unique[i]) % 12
                intervals.append(interval)
        
        if intervals:
            consonant_count = sum(1 for iv in intervals if iv in consonant_intervals)
            consonance_scores.append(consonant_count / len(intervals))
        else:
            consonance_scores.append(0.5)
    
    consonance_series = np.array(consonance_scores)
    
    return {
        'mean_consonance': np.mean(consonance_series),
        'consonance_fluctuation': np.std(consonance_series),
        'consonance_series': consonance_series
    }


# ============================================================================
# РАЗДЕЛ 3: PITCH & TEMPORAL ANALYSIS
# ============================================================================

def analyze_pitch_distribution(pitches: np.ndarray):
    """Полный анализ распределения высот."""
    if len(pitches) == 0:
        return {}
    
    pitch_counts = Counter(pitches)
    total = sum(pitch_counts.values())
    probabilities = np.array([count / total for count in pitch_counts.values()])
    pitch_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return {
        'mean_pitch': np.mean(pitches),
        'std_pitch': np.std(pitches),
        'median_pitch': np.median(pitches),
        'pitch_range': np.ptp(pitches),
        'pitch_entropy': pitch_entropy,
        'unique_pitches': len(pitch_counts)
    }


def analyze_temporal_features(durations: np.ndarray, onsets: np.ndarray):
    """Анализ временных характеристик."""
    ioi = np.diff(onsets) if len(onsets) > 1 else np.array([0])
    
    return {
        'duration_mean': np.mean(durations),
        'duration_std': np.std(durations),
        'duration_median': np.median(durations),
        'ioi_mean': np.mean(ioi) if len(ioi) > 0 else 0,
        'ioi_std': np.std(ioi) if len(ioi) > 0 else 0,
        'ioi_cv': np.std(ioi) / (np.mean(ioi) + 1e-8) if len(ioi) > 0 else 0
    }


# ============================================================================
# РАЗДЕЛ 4: HARMONIC ANALYSIS
# ============================================================================

def compute_pitch_class_histogram(pitches: np.ndarray) -> np.ndarray:
    """Гистограмма pitch classes (0-11), нормализованная."""
    if len(pitches) == 0:
        return np.zeros(12)
    
    pitch_classes = pitches % 12
    hist = np.bincount(pitch_classes.astype(int), minlength=12)[:12]
    total = hist.sum()
    
    if total > 0:
        hist = hist / total
    
    return hist


def pitch_class_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Косинусное сходство гармонического содержания."""
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(hist1, hist2) / (norm1 * norm2)


# ============================================================================
# РАЗДЕЛ 5: MELODIC CONTOUR COMPARISON (IMPROVED)
# ============================================================================

# Минимальное количество нот для сравнения
MIN_NOTES_FOR_COMPARISON = 8

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """DTW расстояние между последовательностями."""
    if len(seq1) == 0 or len(seq2) == 0:
        return float('inf')
    
    if not HAS_DTW:
        # Простой fallback - интерполяция до одинаковой длины
        target_len = max(len(seq1), len(seq2))
        interp1 = np.interp(np.linspace(0, 1, target_len), 
                           np.linspace(0, 1, len(seq1)), seq1)
        interp2 = np.interp(np.linspace(0, 1, target_len), 
                           np.linspace(0, 1, len(seq2)), seq2)
        return np.sqrt(np.mean((interp1 - interp2)**2))
    
    try:
        return dtw.distance(seq1.astype(np.double), seq2.astype(np.double))
    except Exception:
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return float('inf')
        return np.sqrt(np.sum((seq1[:min_len] - seq2[:min_len])**2)) / min_len


def normalize_pitches(pitches: np.ndarray) -> np.ndarray:
    """Нормализация питчей к [0, 1]."""
    if len(pitches) == 0:
        return np.array([0.5])
    
    min_p, max_p = np.min(pitches), np.max(pitches)
    
    if max_p == min_p:
        return np.ones(len(pitches)) * 0.5
    
    return (pitches - min_p) / (max_p - min_p)


def interval_similarity(intervals1: np.ndarray, intervals2: np.ndarray) -> float:
    """
    Строгое сходство мелодических интервалов.
    Использует нормализованный DTW с учётом различий в длине.
    Штраф за большую разницу в длинах последовательностей.
    """
    if len(intervals1) < 2 or len(intervals2) < 2:
        return 0.0
    
    # Штраф за разницу в длине (чем больше разница, тем меньше сходство)
    len_ratio = min(len(intervals1), len(intervals2)) / max(len(intervals1), len(intervals2))
    if len_ratio < 0.3:  # слишком разные длины
        return 0.0
    
    # Нормализация интервалов к [-1, 1]
    max_interval = max(np.max(np.abs(intervals1)), np.max(np.abs(intervals2)), 1)
    norm1 = intervals1 / max_interval
    norm2 = intervals2 / max_interval
    
    # DTW расстояние
    dist = dtw_distance(norm1, norm2)
    
    # Нормализация: максимальное расстояние = 2 (от -1 до 1)
    # Среднее расстояние на элемент не должно превышать 2
    avg_dist = dist  # уже RMSE из улучшенного dtw_distance
    
    # Преобразование в сходство [0, 1]
    # При avg_dist = 0 -> sim = 1
    # При avg_dist = 1 -> sim ~ 0.37 (exp(-1))
    # При avg_dist = 2 -> sim ~ 0.14
    similarity = np.exp(-avg_dist)
    
    # Учитываем штраф за разницу в длине
    return similarity * len_ratio


def contour_similarity_strict(pitches1: np.ndarray, pitches2: np.ndarray) -> float:
    """
    Строгое сравнение мелодических контуров.
    Учитывает: вариативность, динамический диапазон, направление движения, форму.
    
    КЛЮЧЕВОЕ УЛУЧШЕНИЕ: монотонные последовательности (низкая вариативность)
    получают штраф, даже если направления совпадают.
    """
    if len(pitches1) < MIN_NOTES_FOR_COMPARISON or len(pitches2) < MIN_NOTES_FOR_COMPARISON:
        return 0.0
    
    # Штраф за разницу в длине
    len_ratio = min(len(pitches1), len(pitches2)) / max(len(pitches1), len(pitches2))
    if len_ratio < 0.3:
        return 0.0
    
    # === ВАРИАТИВНОСТЬ И ДИНАМИЧЕСКИЙ ДИАПАЗОН ===
    # Стандартное отклонение показывает "живость" мелодии
    std1 = np.std(pitches1)
    std2 = np.std(pitches2)
    range1 = np.ptp(pitches1)  # max - min
    range2 = np.ptp(pitches2)
    
    # Если одна из последовательностей монотонная (std < 2 полутонов) — штраф
    MIN_STD = 2.0  # минимум 2 полутона разброса для "живой" мелодии
    MIN_RANGE = 4.0  # минимум 4 полутона диапазона
    
    variability_penalty = 1.0
    if std1 < MIN_STD or std2 < MIN_STD:
        variability_penalty *= 0.5  # сильный штраф за монотонность
    if range1 < MIN_RANGE or range2 < MIN_RANGE:
        variability_penalty *= 0.7  # штраф за узкий диапазон
    
    # Сходство вариативности — обе последовательности должны быть "одинаково живыми"
    variability_ratio = min(std1, std2) / (max(std1, std2) + 1e-8)
    range_ratio = min(range1, range2) / (max(range1, range2) + 1e-8)
    
    # === НОРМАЛИЗАЦИЯ ДЛЯ СРАВНЕНИЯ ФОРМЫ ===
    norm1 = normalize_pitches(pitches1)
    norm2 = normalize_pitches(pitches2)
    
    # Интерполируем к одинаковой длине
    target_len = min(len(norm1), len(norm2))
    interp1 = np.interp(np.linspace(0, 1, target_len), 
                        np.linspace(0, 1, len(norm1)), norm1)
    interp2 = np.interp(np.linspace(0, 1, target_len), 
                        np.linspace(0, 1, len(norm2)), norm2)
    
    # === МЕТРИКИ СХОДСТВА ФОРМЫ ===
    
    # 1. Евклидово расстояние между нормализованными контурами
    eucl_dist = np.sqrt(np.mean((interp1 - interp2)**2))
    eucl_sim = np.exp(-eucl_dist * 4)  # усиленная чувствительность
    
    # 2. Направления движения (вверх/вниз/стоит)
    dir1 = np.sign(np.diff(interp1))
    dir2 = np.sign(np.diff(interp2))
    direction_match = np.mean(dir1 == dir2)
    
    # 3. Амплитуда движений — насколько похожи "скачки"
    jumps1 = np.abs(np.diff(interp1))
    jumps2 = np.abs(np.diff(interp2))
    jump_corr = 0.0
    if len(jumps1) > 2 and len(jumps2) > 2:
        try:
            # Интерполируем jumps к одной длине
            target_jump_len = min(len(jumps1), len(jumps2))
            j1 = np.interp(np.linspace(0, 1, target_jump_len), 
                          np.linspace(0, 1, len(jumps1)), jumps1)
            j2 = np.interp(np.linspace(0, 1, target_jump_len), 
                          np.linspace(0, 1, len(jumps2)), jumps2)
            corr, _ = pearsonr(j1, j2)
            jump_corr = max(0, (corr + 1) / 2)  # нормализуем к [0, 1]
        except:
            jump_corr = 0.5
    
    # 4. Корреляция контуров
    try:
        corr, _ = pearsonr(interp1, interp2)
        contour_corr = max(0, corr)
    except:
        contour_corr = 0.0
    
    # === КОМБИНИРОВАННАЯ МЕТРИКА ===
    # Вес на eucl_sim и jump_corr — это "воспринимаемое сходство"
    form_similarity = (
        0.30 * eucl_sim +        # близость по форме
        0.25 * direction_match + # совпадение направлений
        0.25 * jump_corr +       # сходство амплитуд скачков
        0.20 * contour_corr      # общая корреляция
    )
    
    # Учитываем сходство вариативности
    variability_sim = 0.5 * variability_ratio + 0.5 * range_ratio
    
    # Финальная оценка с учётом всех штрафов
    final_score = form_similarity * variability_sim * variability_penalty * len_ratio
    
    return final_score


def contour_correlation(pitches1: np.ndarray, pitches2: np.ndarray) -> float:
    """Корреляция нормализованных контуров."""
    if len(pitches1) < 3 or len(pitches2) < 3:
        return 0.0
    
    norm1 = normalize_pitches(pitches1)
    norm2 = normalize_pitches(pitches2)
    
    target_len = max(len(norm1), len(norm2))
    interp1 = np.interp(np.linspace(0, 1, target_len), 
                        np.linspace(0, 1, len(norm1)), norm1)
    interp2 = np.interp(np.linspace(0, 1, target_len), 
                        np.linspace(0, 1, len(norm2)), norm2)
    
    try:
        corr, _ = pearsonr(interp1, interp2)
        if np.isnan(corr):
            return 0.0
        return corr
    except Exception:
        return 0.0


# ============================================================================
# РАЗДЕЛ 5.5: ЭМОЦИОНАЛЬНЫЕ МЕТРИКИ СРАВНЕНИЯ
# ============================================================================

def compute_emotional_features(pitches: np.ndarray, durations: np.ndarray = None) -> dict:
    """
    Вычисляет эмоциональные характеристики мелодии.
    Основано на исследованиях музыкальной психологии:
    - Темп и ритм -> Arousal (возбуждение)
    - Высота и интервалы -> Valence (позитивность)
    - Динамический диапазон -> Intensity
    """
    if len(pitches) < 3:
        return {
            'arousal_index': 0.5,
            'valence_index': 0.5,
            'intensity_index': 0.5,
            'complexity_index': 0.5
        }
    
    intervals = np.diff(pitches)
    
    # AROUSAL (Возбуждение) - основано на:
    # - Размах интервалов (большие скачки = высокий arousal)
    # - Частота смен направления
    # - Стандартное отклонение
    interval_magnitude = np.mean(np.abs(intervals))
    direction_changes = np.sum(np.diff(np.sign(intervals)) != 0) / (len(intervals) - 1) if len(intervals) > 1 else 0
    pitch_std = np.std(pitches)
    
    # Нормализация к [0, 1]
    arousal_raw = (
        0.4 * min(interval_magnitude / 12, 1.0) +  # интервалы до октавы
        0.3 * direction_changes +                   # частота смен направления
        0.3 * min(pitch_std / 15, 1.0)             # разброс питчей
    )
    arousal_index = np.clip(arousal_raw, 0, 1)
    
    # VALENCE (Позитивность) - основано на:
    # - Общий тренд мелодии (восходящий = позитивный)
    # - Преобладание мажорных интервалов (3, 4, 7 полутонов)
    # - Соотношение положительных/отрицательных интервалов
    
    # Общий тренд
    if len(pitches) > 2:
        x = np.arange(len(pitches))
        slope, _, _, _, _ = linregress(x, pitches)
        trend = np.clip(slope / 5, -1, 1)  # нормализация
    else:
        trend = 0
    
    # Соотношение положительных интервалов
    pos_intervals = np.sum(intervals > 0) / len(intervals) if len(intervals) > 0 else 0.5
    
    # Мажорные интервалы ТОЛЬКО для восходящих движений
    # (3, 4, 5, 7 полутонов = большая терция, кварта, квинта, октава)
    major_intervals = [3, 4, 5, 7, 12]
    # Считаем только положительные (восходящие) мажорные интервалы
    pos_major_count = np.sum([(int(i) in major_intervals) for i in intervals if i > 0])
    neg_minor_count = np.sum([(abs(int(i)) in [1, 2, 6]) for i in intervals if i < 0])  # минорные нисходящие
    
    total_intervals = len(intervals)
    major_ratio = pos_major_count / total_intervals if total_intervals > 0 else 0
    minor_penalty = neg_minor_count / total_intervals if total_intervals > 0 else 0
    
    valence_raw = (
        0.40 * (trend + 1) / 2 +          # тренд [-1,1] -> [0,1] - главный фактор
        0.35 * pos_intervals +             # доля восходящих интервалов
        0.25 * major_ratio -               # доля мажорных восходящих
        0.10 * minor_penalty               # штраф за минорные нисходящие
    )
    valence_index = np.clip(valence_raw, 0, 1)
    
    # INTENSITY (Интенсивность) - основано на:
    # - Диапазон питчей
    # - Плотность нот (если есть durations)
    pitch_range = np.ptp(pitches)
    intensity_raw = min(pitch_range / 36, 1.0)  # 3 октавы = максимум
    intensity_index = np.clip(intensity_raw, 0, 1)
    
    # COMPLEXITY (Сложность) - основано на:
    # - Количество уникальных питчей
    # - Количество уникальных интервалов
    # - Энтропия питчей
    unique_pitches = len(np.unique(pitches))
    unique_intervals = len(np.unique(intervals))
    
    # Энтропия
    pitch_counts = np.bincount(pitches.astype(int) - int(np.min(pitches)))
    pitch_probs = pitch_counts / np.sum(pitch_counts)
    pitch_entropy = -np.sum(pitch_probs[pitch_probs > 0] * np.log2(pitch_probs[pitch_probs > 0]))
    max_entropy = np.log2(len(pitch_probs)) if len(pitch_probs) > 1 else 1
    normalized_entropy = pitch_entropy / max_entropy if max_entropy > 0 else 0
    
    complexity_raw = (
        0.4 * min(unique_pitches / 12, 1.0) +
        0.3 * min(unique_intervals / 10, 1.0) +
        0.3 * normalized_entropy
    )
    complexity_index = np.clip(complexity_raw, 0, 1)
    
    return {
        'arousal_index': float(arousal_index),
        'valence_index': float(valence_index),
        'intensity_index': float(intensity_index),
        'complexity_index': float(complexity_index)
    }


def emotional_similarity(pitches1: np.ndarray, pitches2: np.ndarray, 
                         durations1: np.ndarray = None, durations2: np.ndarray = None) -> dict:
    """
    Сравнивает эмоциональные характеристики двух мелодий.
    
    Возвращает:
    - emotional_distance: евклидово расстояние в 4D пространстве эмоций
    - emotional_similarity: нормализованное сходство [0, 1]
    - component_similarities: сходство по каждой компоненте
    """
    emo1 = compute_emotional_features(pitches1, durations1)
    emo2 = compute_emotional_features(pitches2, durations2)
    
    # Евклидово расстояние в 4D эмоциональном пространстве
    vec1 = np.array([emo1['arousal_index'], emo1['valence_index'], 
                     emo1['intensity_index'], emo1['complexity_index']])
    vec2 = np.array([emo2['arousal_index'], emo2['valence_index'], 
                     emo2['intensity_index'], emo2['complexity_index']])
    
    distance = np.sqrt(np.sum((vec1 - vec2)**2))
    # Максимальное расстояние = sqrt(4) = 2
    # Используем exp для более строгой трансформации
    similarity = np.exp(-distance * 2)  # Умножаем на 2 для более строгой оценки
    
    # Компонентные сходства (используем exp для строгости)
    component_sims = {
        'arousal_sim': np.exp(-abs(emo1['arousal_index'] - emo2['arousal_index']) * 3),
        'valence_sim': np.exp(-abs(emo1['valence_index'] - emo2['valence_index']) * 3),
        'intensity_sim': np.exp(-abs(emo1['intensity_index'] - emo2['intensity_index']) * 3),
        'complexity_sim': np.exp(-abs(emo1['complexity_index'] - emo2['complexity_index']) * 3),
    }
    
    return {
        'emotional_distance': float(distance),
        'emotional_similarity': float(np.clip(similarity, 0, 1)),
        **component_sims,
        'eeg_emotional': emo1,
        'classical_emotional': emo2
    }


def overall_trend_similarity(pitches1: np.ndarray, pitches2: np.ndarray) -> float:
    """
    Сравнивает ОБЩИЙ ТРЕНД мелодий (восходящий/нисходящий/плоский).
    
    Возвращает 1.0 если оба тренда одинаковы по направлению И силе.
    Возвращает 0.0 если тренды противоположные.
    """
    if len(pitches1) < 3 or len(pitches2) < 3:
        return 0.5
    
    # Тренд через линейную регрессию
    x1 = np.arange(len(pitches1))
    x2 = np.arange(len(pitches2))
    
    slope1, _, _, _, _ = linregress(x1, pitches1)
    slope2, _, _, _, _ = linregress(x2, pitches2)
    
    # Нормализация к [-1, 1] (более агрессивная)
    norm_slope1 = np.clip(slope1 / 2, -1, 1)
    norm_slope2 = np.clip(slope2 / 2, -1, 1)
    
    # Косинусное сходство для направления
    # Если оба положительные или оба отрицательные = высокое сходство
    # Если разные знаки = низкое сходство
    if abs(norm_slope1) < 0.1 and abs(norm_slope2) < 0.1:
        # Оба плоские
        return 1.0
    
    # Сходство направлений: используем знак
    sign_match = np.sign(norm_slope1) * np.sign(norm_slope2)  # 1 если совпадают, -1 если нет
    direction_sim = (sign_match + 1) / 2  # [0, 1]
    
    # Сходство величин
    magnitude_sim = 1 - abs(norm_slope1 - norm_slope2) / 2
    
    # Комбинированная оценка с штрафом за разные направления
    return float(direction_sim * magnitude_sim)



def dynamic_range_similarity(pitches1: np.ndarray, pitches2: np.ndarray) -> float:
    """
    Сравнивает динамический диапазон (range и std).
    """
    if len(pitches1) < 3 or len(pitches2) < 3:
        return 0.5
    
    range1, range2 = np.ptp(pitches1), np.ptp(pitches2)
    std1, std2 = np.std(pitches1), np.std(pitches2)
    
    # Соотношение диапазонов
    range_ratio = min(range1, range2) / (max(range1, range2) + 1e-8)
    std_ratio = min(std1, std2) / (max(std1, std2) + 1e-8)
    
    return float(0.5 * range_ratio + 0.5 * std_ratio)


# ============================================================================
# РАЗДЕЛ 6: WINDOW FEATURE EXTRACTION
# ============================================================================

def extract_window_features(window_stream, start_time: float, end_time: float):
    """
    Полное извлечение признаков окна с Scale-Free и Consonance анализом.
    """
    # Конвертируем в stream для эффективности
    if hasattr(window_stream, 'stream'):
        window_stream = window_stream.stream()
    notes = [n for n in window_stream.flatten().notes if hasattr(n, 'pitch')]
    
    if len(notes) == 0:
        return None
    
    # Базовые данные
    pitches = np.array([n.pitch.midi for n in notes], dtype=float)
    durations = np.array([n.quarterLength for n in notes], dtype=float)
    onsets = np.array([n.offset for n in notes], dtype=float)
    velocities = np.array([n.volume.velocity if hasattr(n, 'volume') and n.volume.velocity else 64 
                          for n in notes], dtype=float)
    
    # Производные данные
    ioi = np.diff(onsets) if len(onsets) > 1 else np.array([0])
    intervals = np.diff(pitches) if len(pitches) > 1 else np.array([0])
    pitch_class_hist = compute_pitch_class_histogram(pitches)
    
    # Scale-Free анализ
    sfi_pitch, sfi_r2 = calculate_scale_free_index(pitches)
    sfi_ioi, sfi_ioi_r2 = calculate_ioi_scale_free(ioi) if len(ioi) > 2 else (0.0, 0.0)
    
    # Consonance анализ
    time_points = np.arange(start_time, end_time, 0.25)
    pitches_over_time = []
    
    for t in time_points:
        active_pitches = [int(n.pitch.midi) for n in notes 
                         if n.offset <= t < (n.offset + n.quarterLength)]
        pitches_over_time.append(active_pitches)
    
    consonance_data = calculate_consonance_fluctuation(pitches_over_time)
    
    # Pitch & Temporal analysis
    pitch_stats = analyze_pitch_distribution(pitches)
    temporal_stats = analyze_temporal_features(durations, onsets)
    
    # Собираем все признаки
    features = {
        # Метаданные
        'note_count': len(notes),
        'duration': end_time - start_time,
        
        # Scale-Free (Wu 2018) - КЛЮЧЕВЫЕ МЕТРИКИ
        'sfi_pitch': sfi_pitch,
        'sfi_pitch_r2': sfi_r2,
        'sfi_rhythm': sfi_ioi,
        'sfi_rhythm_r2': sfi_ioi_r2,
        
        # Consonance (Wu 2018)
        'mean_consonance': consonance_data['mean_consonance'],
        'consonance_fluctuation': consonance_data['consonance_fluctuation'],
        
        # Pitch statistics
        **pitch_stats,
        
        # Temporal statistics
        **temporal_stats,
        
        # Velocity
        'velocity_mean': np.mean(velocities),
        'velocity_std': np.std(velocities),
        
        # Density
        'note_density': len(notes) / (end_time - start_time + 1e-8),
        
        # Интервалы
        'interval_mean_abs': np.mean(np.abs(intervals)) if len(intervals) > 0 else 0,
        'interval_std': np.std(intervals) if len(intervals) > 0 else 0,
        'interval_max': np.max(np.abs(intervals)) if len(intervals) > 0 else 0,
        
        # Сырые данные для детального сравнения
        'pitches_raw': pitches,
        'intervals_raw': intervals,
        'ioi_raw': ioi,
        'durations_raw': durations,
        'pitch_class_hist': pitch_class_hist,
        'consonance_series': consonance_data['consonance_series'],
        
        # Pitch class для обратной совместимости
        **{f'pc_{i}': pitch_class_hist[i] for i in range(12)}
    }
    
    return features


# ============================================================================
# РАЗДЕЛ 7: MAIN COMPARATOR CLASS
# ============================================================================

class ComprehensiveMIDIComparator:
    """
    Полный компаратор EEG-MIDI с классическими произведениями.
    Объединяет глобальный и фрагментарный (windowed) анализ.
    """
    
    def __init__(self, 
                 eeg_midi_path: str, 
                 classical_midi_paths: Dict[str, str],
                 window_size: float = 4.0,
                 hop_size: float = 2.0):
        """
        Параметры:
        - eeg_midi_path: путь к EEG-MIDI
        - classical_midi_paths: {'название': 'путь', ...}
        - window_size: размер окна в секундах (рекомендуется 4-8)
        - hop_size: шаг окна (рекомендуется window_size/2)
        """
        self.eeg_midi_path = eeg_midi_path
        self.classical_midi_paths = classical_midi_paths
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Загрузка MIDI
        print("Загрузка MIDI файлов...")
        self.eeg_midi = m21.converter.parse(eeg_midi_path)
        self.classical_midis = {
            name: m21.converter.parse(path)
            for name, path in classical_midi_paths.items()
        }
        
        # Кэш
        self.eeg_windows = None
        self.classical_windows = {}
        self.global_results = {}
    
    # === ГЛОБАЛЬНЫЙ АНАЛИЗ (весь файл) ===
    
    def analyze_global(self):
        """
        Глобальный анализ: весь MIDI файл целиком.
        Основная метрика - Scale-Free Index (Wu 2018).
        """
        print("\n" + "="*70)
        print("ГЛОБАЛЬНЫЙ АНАЛИЗ (весь файл целиком)")
        print("="*70)
        
        results = {}
        
        # EEG-MIDI
        print("\nАнализ EEG-MIDI...")
        eeg_notes = [n for n in self.eeg_midi.flatten().notes if hasattr(n, 'pitch')]
        eeg_pitches = np.array([n.pitch.midi for n in eeg_notes])
        eeg_durations = np.array([n.quarterLength for n in eeg_notes])
        eeg_onsets = np.array([n.offset for n in eeg_notes])
        
        sfi, r2, details = calculate_scale_free_index(eeg_pitches, return_details=True)
        pitch_stats = analyze_pitch_distribution(eeg_pitches)
        temporal_stats = analyze_temporal_features(eeg_durations, eeg_onsets)
        
        results['EEG'] = {
            'sfi': sfi,
            'sfi_r2': r2,
            'sfi_interpretation': self._interpret_sfi(sfi),
            **pitch_stats,
            **temporal_stats,
            'total_notes': len(eeg_notes),
            'total_duration': self.eeg_midi.duration.quarterLength,
            'sfi_details': details
        }
        
        print(f"  SFI: {sfi:.3f} (R²={r2:.3f}) - {self._interpret_sfi(sfi)}")
        
        # Классические произведения
        for name, midi in self.classical_midis.items():
            print(f"\nАнализ {name}...")
            notes = [n for n in midi.flatten().notes if hasattr(n, 'pitch')]
            pitches = np.array([n.pitch.midi for n in notes])
            durations = np.array([n.quarterLength for n in notes])
            onsets = np.array([n.offset for n in notes])
            
            sfi, r2, details = calculate_scale_free_index(pitches, return_details=True)
            pitch_stats = analyze_pitch_distribution(pitches)
            temporal_stats = analyze_temporal_features(durations, onsets)
            
            results[name] = {
                'sfi': sfi,
                'sfi_r2': r2,
                'sfi_interpretation': self._interpret_sfi(sfi),
                **pitch_stats,
                **temporal_stats,
                'total_notes': len(notes),
                'total_duration': midi.duration.quarterLength,
                'sfi_details': details
            }
            
            print(f"  SFI: {sfi:.3f} (R²={r2:.3f}) - {self._interpret_sfi(sfi)}")
        
        self.global_results = results
        return results
    
    def _interpret_sfi(self, sfi: float) -> str:
        """Интерпретация Scale-Free Index."""
        if 0.9 <= sfi <= 1.1:
            return "Highly Musical (1/f)"
        elif 0.8 <= sfi <= 1.2:
            return "Musical"
        elif 0.7 <= sfi <= 1.3:
            return "Moderately Musical"
        else:
            return "Non-musical"
    
    # === WINDOWED АНАЛИЗ ===
    
    def extract_windows(self, midi_stream, source_name: str = None):
        """Разбивает MIDI на окна."""
        total_duration = midi_stream.duration.quarterLength
        windows = []
        
        start_time = 0.0
        window_id = 0
        
        while start_time + self.window_size <= total_duration:
            end_time = start_time + self.window_size
            
            window_stream = midi_stream.flatten().getElementsByOffset(
                start_time, end_time,
                includeEndBoundary=False,
                mustFinishInSpan=False,
                mustBeginInSpan=True
            )
            
            features = extract_window_features(window_stream, start_time, end_time)
            
            if features and features.get('note_count', 0) >= MIN_NOTES_FOR_COMPARISON:
                features['window_id'] = window_id
                features['start_time'] = start_time
                features['end_time'] = end_time
                features['source'] = source_name
                windows.append(features)
                window_id += 1
            
            start_time += self.hop_size
        
        return windows
    
    def compute_window_similarities(self, use_melodic_metrics: bool = True):
        """
        Вычисляет сходство между окнами EEG и классики.
        Использует комбинированную метрику.
        """
        print("\n" + "="*70)
        print("ФРАГМЕНТАРНЫЙ АНАЛИЗ (windowed)")
        print("="*70)
        
        # Извлекаем окна
        if self.eeg_windows is None:
            print("\nИзвлечение окон из EEG-MIDI...")
            self.eeg_windows = self.extract_windows(self.eeg_midi, 'EEG')
            print(f"  Извлечено окон: {len(self.eeg_windows)}")
        
        if not self.classical_windows:
            print("\nИзвлечение окон из классических произведений...")
            for name, midi in self.classical_midis.items():
                self.classical_windows[name] = self.extract_windows(midi, name)
                print(f"  {name}: {len(self.classical_windows[name])} окон")
        
        # Определяем признаки для статистического сравнения
        sample = self.eeg_windows[0]
        exclude_keys = ['window_id', 'start_time', 'end_time', 'source', 'duration',
                       'pitches_raw', 'intervals_raw', 'ioi_raw', 'durations_raw',
                       'pitch_class_hist', 'consonance_series', 'sfi_details']
        
        feature_set = [k for k in sample.keys() 
                      if k not in exclude_keys and not isinstance(sample[k], (np.ndarray, dict))]
        
        # Нормализация признаков
        eeg_matrix = self._windows_to_matrix(self.eeg_windows, feature_set)
        scaler = StandardScaler()
        eeg_matrix_norm = scaler.fit_transform(eeg_matrix)
        
        # Сравнение с каждым произведением
        all_results = []
        
        for classical_name, classical_wins in self.classical_windows.items():
            print(f"\nСравнение с {classical_name}...")
            
            classical_matrix = self._windows_to_matrix(classical_wins, feature_set)
            classical_matrix_norm = scaler.transform(classical_matrix)
            
            for i, eeg_win in enumerate(self.eeg_windows):
                eeg_vec = eeg_matrix_norm[i]
                
                best_score = -float('inf')
                best_match_idx = 0
                best_metrics = {}
                
                for j, classical_win in enumerate(classical_wins):
                    classical_vec = classical_matrix_norm[j]
                    
                    # Статистическое расстояние
                    stat_dist = euclidean(eeg_vec, classical_vec)
                    stat_sim = 1 / (1 + stat_dist)
                    
                    if use_melodic_metrics:
                        # Мелодический контур
                        eeg_intervals = eeg_win.get('intervals_raw', np.array([0]))
                        cla_intervals = classical_win.get('intervals_raw', np.array([0]))
                        contour_sim = interval_similarity(eeg_intervals, cla_intervals)
                        
                        # Гармония
                        eeg_pc = eeg_win.get('pitch_class_hist', np.zeros(12))
                        cla_pc = classical_win.get('pitch_class_hist', np.zeros(12))
                        harmony_sim = pitch_class_similarity(eeg_pc, cla_pc)
                        
                        # Корреляция контуров
                        eeg_pitches = eeg_win.get('pitches_raw', np.array([60]))
                        cla_pitches = classical_win.get('pitches_raw', np.array([60]))
                        corr_sim = (contour_correlation(eeg_pitches, cla_pitches) + 1) / 2
                        
                        # SFI similarity
                        sfi_diff = abs(eeg_win.get('sfi_pitch', 0) - classical_win.get('sfi_pitch', 0))
                        sfi_sim = max(0, 1 - sfi_diff)  # Чем меньше разница, тем лучше
                        
                        # Комбинированная метрика с весами
                        combined_score = (
                            0.30 * contour_sim +      # Мелодический контур
                            0.25 * sfi_sim +          # Scale-free similarity
                            0.20 * corr_sim +         # Корреляция форм
                            0.15 * harmony_sim +      # Гармония
                            0.10 * stat_sim           # Статистика
                        )
                    else:
                        contour_sim = harmony_sim = corr_sim = sfi_sim = 0
                        combined_score = stat_sim
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match_idx = j
                        best_metrics = {
                            'stat_similarity': stat_sim,
                            'contour_similarity': contour_sim,
                            'harmony_similarity': harmony_sim,
                            'correlation_similarity': corr_sim,
                            'sfi_similarity': sfi_sim
                        }
                
                best_match = classical_wins[best_match_idx]
                
                all_results.append({
                    'eeg_window_id': eeg_win['window_id'],
                    'eeg_start_time': eeg_win['start_time'],
                    'eeg_sfi': eeg_win.get('sfi_pitch', 0),
                    
                    'classical_piece': classical_name,
                    'classical_window_id': best_match['window_id'],
                    'classical_start_time': best_match['start_time'],
                    'classical_sfi': best_match.get('sfi_pitch', 0),
                    
                    'combined_similarity': best_score,
                    'contour_similarity': best_metrics.get('contour_similarity', 0),
                    'harmony_similarity': best_metrics.get('harmony_similarity', 0),
                    'correlation_similarity': best_metrics.get('correlation_similarity', 0),
                    'sfi_similarity': best_metrics.get('sfi_similarity', 0),
                    'stat_similarity': best_metrics.get('stat_similarity', 0),
                })
        
        return pd.DataFrame(all_results)
    
    def _windows_to_matrix(self, windows: List[Dict], feature_set: List[str]):
        """Преобразует окна в матрицу признаков."""
        matrix = []
        for win in windows:
            row = []
            for feat in feature_set:
                val = win.get(feat, 0)
                if isinstance(val, (np.ndarray, dict)):
                    val = 0
                row.append(val)
            matrix.append(row)
        return np.array(matrix, dtype=float)
    
    # === СТАТИСТИЧЕСКИЙ АНАЛИЗ ===
    
    def statistical_comparison(self, feature_names: List[str] = None):
        """
        Статистическое сравнение EEG и классических произведений.
        Использует t-test, Mann-Whitney U, Cohen's d.
        """
        print("\n" + "="*70)
        print("СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ")
        print("="*70)
        
        if feature_names is None:
            feature_names = [
                'sfi_pitch', 'sfi_rhythm', 'mean_consonance', 'consonance_fluctuation',
                'mean_pitch', 'std_pitch', 'interval_mean_abs', 
                'ioi_mean', 'note_density', 'velocity_mean'
            ]
        
        all_results = []
        
        for classical_name, classical_wins in self.classical_windows.items():
            print(f"\nСравнение EEG с {classical_name}:")
            
            for feat in feature_names:
                eeg_vals = np.array([w[feat] for w in self.eeg_windows 
                                    if feat in w and not np.isnan(w[feat])])
                cla_vals = np.array([w[feat] for w in classical_wins 
                                    if feat in w and not np.isnan(w[feat])])
                
                if len(eeg_vals) < 3 or len(cla_vals) < 3:
                    continue
                
                # T-test
                t_stat, t_p = ttest_ind(eeg_vals, cla_vals)
                
                # Mann-Whitney U
                u_stat, u_p = mannwhitneyu(eeg_vals, cla_vals, alternative='two-sided')
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt((np.std(eeg_vals)**2 + np.std(cla_vals)**2) / 2)
                cohens_d = (np.mean(eeg_vals) - np.mean(cla_vals)) / (pooled_std + 1e-8)
                
                # Интерпретация effect size
                if abs(cohens_d) < 0.2:
                    effect = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect = "small"
                elif abs(cohens_d) < 0.8:
                    effect = "medium"
                else:
                    effect = "large"
                
                all_results.append({
                    'classical_piece': classical_name,
                    'feature': feat,
                    'eeg_mean': np.mean(eeg_vals),
                    'eeg_std': np.std(eeg_vals),
                    'classical_mean': np.mean(cla_vals),
                    'classical_std': np.std(cla_vals),
                    't_statistic': t_stat,
                    't_p_value': t_p,
                    'u_p_value': u_p,
                    'cohens_d': cohens_d,
                    'effect_size': effect,
                    'significant': u_p < 0.05
                })
                
                if u_p < 0.05:
                    print(f"  {feat}: p={u_p:.4f}, d={cohens_d:.3f} ({effect}) *")
        
        return pd.DataFrame(all_results)
    
    # === ВИЗУАЛИЗАЦИИ ===
    
    def plot_zipf_law(self, output_dir: str):
        """Визуализация Zipf's law для EEG и классики."""
        print("\nСоздание Zipf's law визуализаций...")
        
        eeg_notes = [n for n in self.eeg_midi.flatten().notes if hasattr(n, 'pitch')]
        eeg_pitches = np.array([n.pitch.midi for n in eeg_notes])
        
        for name, midi in self.classical_midis.items():
            notes = [n for n in midi.flatten().notes if hasattr(n, 'pitch')]
            classical_pitches = np.array([n.pitch.midi for n in notes])
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for i, (pitches, label, color) in enumerate([
                (eeg_pitches, 'EEG-MIDI', 'blue'),
                (classical_pitches, name, 'orange')
            ]):
                ax = axes[i]
                
                sfi, r2, details = calculate_scale_free_index(pitches, return_details=True)
                
                log_ranks = details['log_ranks']
                log_freqs = details['log_freqs']
                slope = details['slope']
                intercept = details['intercept']
                
                # Данные
                ax.scatter(log_ranks, log_freqs, alpha=0.6, s=50, color=color, label='Data')
                
                # Фит
                fit_line = slope * log_ranks + intercept
                ax.plot(log_ranks, fit_line, 'r--', linewidth=2, 
                       label=f'Fit: α={sfi:.3f}, R²={r2:.3f}')
                
                # Идеальная 1/f линия
                ideal_line = -1.0 * log_ranks + intercept
                ax.plot(log_ranks, ideal_line, 'g:', linewidth=2, alpha=0.7,
                       label='Ideal 1/f (α=1.0)')
                
                ax.set_xlabel('log(Rank)', fontsize=12)
                ax.set_ylabel('log(Frequency)', fontsize=12)
                ax.set_title(f"{label}\nSFI: {sfi:.3f} ({self._interpret_sfi(sfi)})", 
                           fontsize=13, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_name = name.replace(' ', '_').replace('/', '_')
            plt.savefig(f'{output_dir}/zipf_law_{safe_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_window_comparison(self, similarities_df: pd.DataFrame, output_dir: str):
        """Визуализация фрагментарного сравнения."""
        print("\nСоздание графиков фрагментарного сравнения...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Распределение combined similarity
        ax = axes[0, 0]
        similarities_df['combined_similarity'].hist(bins=50, ax=ax, edgecolor='black')
        ax.axvline(similarities_df['combined_similarity'].median(), 
                  color='r', linestyle='--', label='Median')
        ax.set_xlabel('Combined Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Распределение общего сходства')
        ax.legend()
        
        # 2. Box plot по произведениям
        ax = axes[0, 1]
        similarities_df.boxplot(column='combined_similarity', by='classical_piece', ax=ax)
        ax.set_xlabel('Classical Piece')
        ax.set_ylabel('Combined Similarity')
        ax.set_title('Сходство по произведениям')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
        
        # 3. SFI comparison
        ax = axes[0, 2]
        similarities_df.plot.scatter(x='eeg_sfi', y='classical_sfi', 
                                    c='combined_similarity', cmap='viridis',
                                    ax=ax, alpha=0.6)
        ax.plot([0, 2], [0, 2], 'r--', alpha=0.3, label='Perfect match')
        ax.set_xlabel('EEG SFI')
        ax.set_ylabel('Classical SFI')
        ax.set_title('SFI Comparison')
        ax.legend()
        
        # 4. Метрики сходства
        ax = axes[1, 0]
        metrics = ['contour_similarity', 'harmony_similarity', 'sfi_similarity', 'stat_similarity']
        means = [similarities_df[m].mean() for m in metrics]
        ax.bar(range(len(metrics)), means, color=['blue', 'orange', 'green', 'red'])
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['Contour', 'Harmony', 'SFI', 'Stat'], rotation=45, ha='right')
        ax.set_ylabel('Mean Similarity')
        ax.set_title('Средние значения метрик сходства')
        
        # 5. Best matches по произведениям
        ax = axes[1, 1]
        piece_counts = similarities_df.nlargest(50, 'combined_similarity')['classical_piece'].value_counts()
        piece_counts.plot(kind='barh', ax=ax)
        ax.set_xlabel('Count in Top-50 Matches')
        ax.set_ylabel('Classical Piece')
        ax.set_title('Лучшие совпадения по произведениям')
        
        # 6. Temporal distribution
        ax = axes[1, 2]
        for piece in similarities_df['classical_piece'].unique():
            piece_data = similarities_df[similarities_df['classical_piece'] == piece]
            best_10 = piece_data.nlargest(10, 'combined_similarity')
            ax.scatter(best_10['eeg_start_time'], best_10['classical_start_time'],
                      label=piece, alpha=0.6, s=50)
        ax.set_xlabel('EEG Time (s)')
        ax.set_ylabel('Classical Time (s)')
        ax.set_title('Временное соответствие топ-10')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Фрагментарное сравнение EEG-MIDI с классикой', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/window_comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # === ПОЛНЫЙ PIPELINE ===
    
    def run_full_comparison(self, output_dir: str = 'results'):
        """
        Полный pipeline сравнения для дипломной работы.
        
        Выполняет:
        1. Глобальный анализ (SFI всего файла)
        2. Windowed анализ (фрагменты)
        3. Статистические тесты
        4. Визуализации
        5. Сохранение всех результатов
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("ПОЛНОЕ СРАВНЕНИЕ EEG-MIDI С КЛАССИЧЕСКИМИ ПРОИЗВЕДЕНИЯМИ")
        print("="*70)
        print(f"\nРезультаты будут сохранены в: {output_dir}/")
        
        results = {}
        
        # 1. Глобальный анализ
        print("\n[1/5] Глобальный анализ...")
        global_results = self.analyze_global()
        
        # Сохраняем
        global_df = pd.DataFrame(global_results).T
        global_df.to_csv(f'{output_dir}/global_analysis.csv')
        results['global'] = global_results
        
        # 2. Zipf's law визуализация
        print("\n[2/5] Создание Zipf's law визуализаций...")
        self.plot_zipf_law(output_dir)
        
        # 3. Windowed анализ
        print("\n[3/5] Фрагментарный анализ...")
        similarities_df = self.compute_window_similarities(use_melodic_metrics=True)
        
        # Сохраняем
        similarities_df.to_csv(f'{output_dir}/window_similarities.csv', index=False)
        results['similarities'] = similarities_df
        
        # Лучшие совпадения
        best_overall = similarities_df.nlargest(20, 'combined_similarity')
        best_overall.to_csv(f'{output_dir}/best_matches_top20.csv', index=False)
        
        # По произведениям
        best_per_piece = pd.concat([
            similarities_df[similarities_df['classical_piece'] == piece].nlargest(10, 'combined_similarity')
            for piece in similarities_df['classical_piece'].unique()
        ])
        best_per_piece.to_csv(f'{output_dir}/best_matches_per_piece.csv', index=False)
        
        # 4. Статистические тесты
        print("\n[4/5] Статистическое сравнение...")
        stats_df = self.statistical_comparison()
        stats_df.to_csv(f'{output_dir}/statistical_tests.csv', index=False)
        results['statistics'] = stats_df
        
        # 5. Визуализации
        print("\n[5/5] Создание визуализаций...")
        self.plot_window_comparison(similarities_df, output_dir)
        
        # Финальный отчет
        print("\n" + "="*70)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("="*70)
        
        print("\n1. SCALE-FREE INDEX (Глобальный):")
        print(f"   EEG: {global_results['EEG']['sfi']:.3f} ({global_results['EEG']['sfi_interpretation']})")
        
        # Ближайшее по SFI произведение
        sfi_diffs = {name: abs(data['sfi'] - global_results['EEG']['sfi']) 
                     for name, data in global_results.items() if name != 'EEG'}
        closest = min(sfi_diffs.items(), key=lambda x: x[1])
        
        print(f"   Ближайшее произведение: {closest[0]}")
        print(f"   SFI разница: {closest[1]:.3f}")
        
        print("\n2. ЛУЧШИЕ СОВПАДЕНИЯ (Фрагментарный анализ):")
        print(f"   Топ-1: {best_overall.iloc[0]['classical_piece']}")
        print(f"   Combined Similarity: {best_overall.iloc[0]['combined_similarity']:.3f}")
        print(f"   Контур: {best_overall.iloc[0]['contour_similarity']:.3f}")
        print(f"   SFI: {best_overall.iloc[0]['sfi_similarity']:.3f}")
        
        print("\n3. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
        sig_features = stats_df[stats_df['significant'] == True]
        print(f"   Значимых различий: {len(sig_features)} из {len(stats_df)}")
        
        if len(sig_features) > 0:
            print("   Топ-3 по effect size:")
            top_effects = sig_features.nlargest(3, 'cohens_d', keep='all')
            for _, row in top_effects.iterrows():
                print(f"   - {row['feature']}: d={row['cohens_d']:.3f} ({row['effect_size']})")
        
        print("\n" + "="*70)
        print(f"✓ Все результаты сохранены в {output_dir}/")
        print("="*70)
        
        # Сохраняем summary в JSON
        summary = {
            'eeg_sfi': float(global_results['EEG']['sfi']),
            'eeg_sfi_r2': float(global_results['EEG']['sfi_r2']),
            'eeg_interpretation': global_results['EEG']['sfi_interpretation'],
            'closest_piece': closest[0],
            'sfi_difference': float(closest[1]),
            'best_match_piece': best_overall.iloc[0]['classical_piece'],
            'best_match_similarity': float(best_overall.iloc[0]['combined_similarity']),
            'significant_features': int(len(sig_features)),
            'total_windows_eeg': len(self.eeg_windows),
            'total_comparisons': len(similarities_df)
        }
        
        with open(f'{output_dir}/summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return results

    def find_best_matches(self, df: pd.DataFrame, top_k: int = 10, 
                          sort_by: str = 'combined_similarity') -> pd.DataFrame:
        """
        Находит лучшие совпадения из результатов сравнения.
        
        Параметры:
        - df: DataFrame с результатами compute_window_similarities
        - top_k: количество лучших результатов
        - sort_by: колонка для сортировки
        
        Возвращает DataFrame с добавленным рангом.
        """
        sorted_df = df.sort_values(sort_by, ascending=False).head(top_k).copy()
        sorted_df['rank'] = range(1, len(sorted_df) + 1)
        return sorted_df


# Alias для обратной совместимости
MIDIComparator = ComprehensiveMIDIComparator


