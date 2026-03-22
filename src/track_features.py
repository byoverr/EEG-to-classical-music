"""
Утилиты для извлечения трековых признаков из MIDI и кэширования.
Используется для EMOPIA анализа и pseudo-labeling.

v2: расширенный набор признаков (velocity, duration, mode, intervals,
    rhythm regularity, register) → 40-dim вектор.
    Быстрое извлечение через mido (без music21 для note parsing).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import mido
except ImportError:
    mido = None

from .MIDIComparator import (
    calculate_scale_free_index,
    calculate_consonance_fluctuation,
)


def load_feature_cache(cache_path: Path) -> Dict[str, dict]:
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_feature_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _extract_notes_mido(midi_path: str):
    """Быстрое извлечение нот через mido (в 10-50x быстрее music21)."""
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat or 480

    pitches = []
    onsets = []   # в quarter-length
    durations = []
    velocities = []

    # Собираем note_on / note_off для вычисления длительностей
    for track in mid.tracks:
        abs_tick = 0
        active_notes = {}  # pitch -> (start_tick, velocity)
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick, vel = active_notes.pop(msg.note)
                    onset_ql = start_tick / ticks_per_beat
                    dur_ql = (abs_tick - start_tick) / ticks_per_beat
                    if dur_ql <= 0:
                        dur_ql = 0.01
                    pitches.append(msg.note)
                    onsets.append(onset_ql)
                    durations.append(dur_ql)
                    velocities.append(vel)

    if not pitches:
        return None

    total_duration = max(onsets) + max(durations) if onsets else 0.0

    # Key detection: простой алгоритм Krumhansl-Schmuckler через pitch class histogram
    pc_counts = np.zeros(12)
    for p in pitches:
        pc_counts[p % 12] += 1

    # K-S profiles (major and minor)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1.0
    best_pc = 0
    best_mode = 1.0

    for shift in range(12):
        shifted = np.roll(pc_counts, -shift)
        if np.std(shifted) < 1e-8:
            continue
        major_corr = float(np.corrcoef(shifted, major_profile)[0, 1]) if np.std(major_profile) > 0 else 0
        minor_corr = float(np.corrcoef(shifted, minor_profile)[0, 1]) if np.std(minor_profile) > 0 else 0

        if major_corr > best_corr:
            best_corr = major_corr
            best_pc = shift
            best_mode = 1.0
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_pc = shift
            best_mode = 0.0

    return (np.array(pitches), np.array(onsets), np.array(durations),
            np.array(velocities), total_duration, best_pc, best_mode, best_corr)


def _pitch_class_hist(pitches: np.ndarray) -> np.ndarray:
    if len(pitches) == 0:
        return np.zeros(12)
    pc = pitches % 12
    hist = np.zeros(12)
    for p in pc:
        hist[int(p)] += 1
    hist = hist / (hist.sum() + 1e-10)
    return hist


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def extract_track_features(midi_path: str) -> Optional[dict]:
    """Извлекает расширенный набор признаков (v2) из MIDI файла. Использует mido для скорости."""
    try:
        result = _extract_notes_mido(midi_path)
        if result is None:
            return None
        pitches, onsets, durations, velocities, total_duration, key_pc, key_mode, key_corr = result

        if len(pitches) == 0:
            return None

        onsets_sorted = np.sort(onsets)
        if len(onsets_sorted) > 1:
            ioi = np.diff(onsets_sorted)
            ioi = ioi[ioi > 0]  # убираем нулевые (одновременные ноты)
            if len(ioi) == 0:
                ioi = np.array([0.0])
        else:
            ioi = np.array([0.0])

        # --- Pitch features ---
        note_density = float(len(pitches) / max(total_duration, 1e-6))
        pitch_mean = float(np.mean(pitches))
        pitch_std = float(np.std(pitches))
        pitch_range = float(np.ptp(pitches))

        # --- Velocity features (ключ для arousal!) ---
        vel_mean = float(np.mean(velocities))
        vel_std = float(np.std(velocities))
        vel_range = float(np.ptp(velocities))
        # Динамический контраст: процент нот с velocity > 90 (forte) vs < 50 (piano)
        vel_forte_ratio = float(np.mean(velocities > 90))
        vel_piano_ratio = float(np.mean(velocities < 50))

        # --- Duration features ---
        dur_mean = float(np.mean(durations))
        dur_std = float(np.std(durations))
        # Доля коротких нот (staccato proxy: < 0.25 quarterLength)
        staccato_ratio = float(np.mean(durations < 0.25)) if len(durations) > 0 else 0.0

        # --- IOI / Rhythm features ---
        ioi_mean = float(np.mean(ioi))
        ioi_std = float(np.std(ioi))
        # Ритмическая регулярность (коэффициент вариации IOI; низкий = ровный ритм)
        rhythm_regularity = float(ioi_std / (ioi_mean + 1e-6))

        # --- Interval features ---
        if len(pitches) > 1:
            # Берём интервалы последовательных нот (по onset order)
            sorted_idx = np.argsort(onsets)
            sorted_pitches = pitches[sorted_idx]
            intervals = np.diff(sorted_pitches)
            interval_mean = float(np.mean(np.abs(intervals)))
            interval_std = float(np.std(intervals))
            # Доля больших скачков (> 7 полутонов = квинта)
            leap_ratio = float(np.mean(np.abs(intervals) > 7))
        else:
            interval_mean = 0.0
            interval_std = 0.0
            leap_ratio = 0.0

        # --- Register distribution ---
        low_ratio = float(np.mean(pitches < 48))    # < C3
        mid_ratio = float(np.mean((pitches >= 48) & (pitches < 72)))  # C3-C5
        high_ratio = float(np.mean(pitches >= 72))   # >= C5

        # --- Pitch class & Harmony ---
        pc_hist = _pitch_class_hist(pitches)
        pc_entropy = _entropy(pc_hist)

        sfi_exp, sfi_r2 = calculate_scale_free_index(pitches)

        # Consonance
        bins = np.floor(onsets_sorted).astype(int)
        max_bin = int(bins.max()) if len(bins) > 0 else 0
        pitches_over_time = [[] for _ in range(max_bin + 1)]
        for p, b in zip(pitches, bins):
            if b >= 0:
                pitches_over_time[int(b)].append(int(p))
        consonance = calculate_consonance_fluctuation(pitches_over_time)

        tempo_proxy = float(1.0 / (ioi_mean + 1e-6)) if ioi_mean > 0 else 0.0

        return {
            # Pitch
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'note_density': note_density,
            # Velocity (NEW)
            'velocity_mean': vel_mean,
            'velocity_std': vel_std,
            'velocity_range': vel_range,
            'velocity_forte_ratio': vel_forte_ratio,
            'velocity_piano_ratio': vel_piano_ratio,
            # Duration (NEW)
            'duration_mean': dur_mean,
            'duration_std': dur_std,
            'staccato_ratio': staccato_ratio,
            # IOI / Rhythm
            'ioi_mean': ioi_mean,
            'ioi_std': ioi_std,
            'rhythm_regularity': rhythm_regularity,
            'tempo_proxy': tempo_proxy,
            # Intervals (NEW)
            'interval_mean': interval_mean,
            'interval_std': interval_std,
            'leap_ratio': leap_ratio,
            # Register (NEW)
            'register_low': low_ratio,
            'register_mid': mid_ratio,
            'register_high': high_ratio,
            # Key/Mode (NEW)
            'key_pitch_class': float(key_pc),
            'key_mode': key_mode,
            'key_correlation': key_corr,
            # Harmony
            'pitch_class_entropy': pc_entropy,
            'pitch_class_hist': pc_hist.tolist(),
            'sfi_pitch': float(sfi_exp),
            'consonance_mean': float(consonance.get('mean_consonance', 0.0)),
            'consonance_std': float(consonance.get('consonance_fluctuation', 0.0)),
            # Meta
            'note_count': int(len(pitches)),
            '_feature_version': 2,
        }
    except Exception:
        return None


def features_to_vector(features: dict) -> np.ndarray:
    """Конвертирует dict признаков в вектор. v2: 39-dim."""
    pc_hist = np.array(features.get('pitch_class_hist', [0.0] * 12), dtype=float)
    core = np.array([
        # Pitch (4)
        features.get('pitch_mean', 0.0),
        features.get('pitch_std', 0.0),
        features.get('pitch_range', 0.0),
        features.get('note_density', 0.0),
        # Velocity (5)
        features.get('velocity_mean', 64.0),
        features.get('velocity_std', 0.0),
        features.get('velocity_range', 0.0),
        features.get('velocity_forte_ratio', 0.0),
        features.get('velocity_piano_ratio', 0.0),
        # Duration (3)
        features.get('duration_mean', 0.5),
        features.get('duration_std', 0.0),
        features.get('staccato_ratio', 0.0),
        # IOI / Rhythm (4)
        features.get('ioi_mean', 0.0),
        features.get('ioi_std', 0.0),
        features.get('rhythm_regularity', 0.0),
        features.get('tempo_proxy', 0.0),
        # Intervals (3)
        features.get('interval_mean', 0.0),
        features.get('interval_std', 0.0),
        features.get('leap_ratio', 0.0),
        # Register (3)
        features.get('register_low', 0.0),
        features.get('register_mid', 1.0),
        features.get('register_high', 0.0),
        # Key/Mode (3)
        features.get('key_pitch_class', 0.0),
        features.get('key_mode', 1.0),
        features.get('key_correlation', 0.0),
        # Harmony (3)
        features.get('pitch_class_entropy', 0.0),
        features.get('sfi_pitch', 0.0),
        features.get('consonance_mean', 0.0),
        features.get('consonance_std', 0.0),
    ], dtype=float)
    return np.concatenate([core, pc_hist])  # 28 + 12 = 40 dim (was 11 + 12 = 23)  # noqa


def get_or_compute_features(midi_path: str, cache: Dict[str, dict]) -> Optional[dict]:
    key = str(midi_path)
    if key in cache:
        cached = cache[key]
        # Инвалидация старого кэша (v1 без velocity)
        if cached.get('_feature_version', 1) < 2:
            del cache[key]
        else:
            return cached
    feats = extract_track_features(midi_path)
    if feats is not None:
        cache[key] = feats
    return feats
