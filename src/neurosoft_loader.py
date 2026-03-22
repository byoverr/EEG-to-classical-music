"""
Загрузчик файлов Neurosoft .EEG → numpy-массивы для пайплайна EEG→MIDI.

Использует парсер из eeg_to_python_converter.py для чтения бинарного
формата Neurosoft, затем преобразует данные в numpy-массивы, совместимые
с eeg_preprocessing.prepare_signal_data().

Использование:
    from src.neurosoft_loader import load_neurosoft_eeg, prepare_neurosoft_signal_data

    info, signals = load_neurosoft_eeg("D0000001.EEG", eeg_only=True)
    signal_data = prepare_neurosoft_signal_data(signals, info['srate'])
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from .eeg_to_python_converter import parse_header, read_int24_le, BYTES_PER_SAMPLE


# Каналы, которые НЕ являются ЭЭГ (Neurosoft-специфичные)
_NON_EEG_CHANNELS = {"Bio1", "Bio2", "A1", "A2", "VSyn", "ASyn", "LABEL"}


def load_neurosoft_eeg(
    filepath: str,
    eeg_only: bool = True,
    max_seconds: Optional[float] = None,
) -> tuple[dict, np.ndarray]:
    """
    Загружает .EEG файл и возвращает метаданные + массив сигналов.

    Parameters
    ----------
    filepath : str
        Путь к .EEG файлу (Neurosoft Neuron-Spectrum).
    eeg_only : bool
        Если True — только ЭЭГ-каналы (без Bio, A1, A2, Sync, LABEL).
    max_seconds : float | None
        Ограничение длительности (None = весь файл).

    Returns
    -------
    info : dict
        Метаданные файла (srate, channels, study_name, duration_sec и т.д.)
        + ключ 'exported_channels' — список каналов в возвращённом массиве.
    signals : np.ndarray
        Массив shape (n_channels, n_samples) в микровольтах (µV).
    """
    filepath = str(filepath)
    info = parse_header(filepath)

    n_channels_total = info["n_channels"]
    srate = info["srate"]
    channels = info["channels"]
    data_offset = info["data_offset"]
    record_size = info["record_size"]
    n_samples = info["n_samples"]

    if max_seconds is not None:
        n_samples = min(n_samples, int(max_seconds * srate))

    # Выбираем каналы
    if eeg_only:
        export_channels = [ch for ch in channels if ch["name"] not in _NON_EEG_CHANNELS]
    else:
        export_channels = channels

    ch_indices = [ch["index"] for ch in export_channels]
    ch_cals = [ch["calibration"] for ch in export_channels]

    info["exported_channels"] = export_channels

    # Читаем все данные разом (быстрее чем побайтовый цикл)
    n_bytes_needed = n_samples * record_size
    with open(filepath, "rb") as f:
        f.seek(data_offset)
        raw = f.read(n_bytes_needed)

    actual_samples = len(raw) // record_size
    if actual_samples == 0:
        raise ValueError(f"Не удалось прочитать данные из {filepath}")
    if actual_samples < n_samples:
        n_samples = actual_samples

    # Быстрая распаковка через numpy (вместо Python-цикла)
    raw_arr = np.frombuffer(raw[: n_samples * record_size], dtype=np.uint8)
    raw_arr = raw_arr.reshape(n_samples, record_size)

    n_export = len(export_channels)
    out = np.empty((n_export, n_samples), dtype=np.float32)

    for ci, ch_idx in enumerate(ch_indices):
        byte_off = ch_idx * BYTES_PER_SAMPLE  # 3 bytes per channel
        b0 = raw_arr[:, byte_off].astype(np.int32)
        b1 = raw_arr[:, byte_off + 1].astype(np.int32)
        b2 = raw_arr[:, byte_off + 2].astype(np.int32)
        vals = b0 | (b1 << 8) | (b2 << 16)
        # Знаковое расширение 24→32 бит
        vals[vals >= 0x800000] -= 0x1000000
        out[ci, :] = vals.astype(np.float32) * ch_cals[ci]

    samples_read = n_samples

    # Обрезаем если прочитали меньше ожидаемого
    if samples_read < n_samples:
        out = out[:, :samples_read]
        info["n_samples"] = samples_read
        info["duration_sec"] = samples_read / srate

    return info, out


def prepare_neurosoft_signal_data(
    signals: np.ndarray,
    srate: int = 250,
) -> dict:
    """
    Аналог eeg_preprocessing.prepare_signal_data(), но для Neurosoft-данных.

    Принимает автономный numpy-массив (n_channels, n_samples) и
    возвращает словарь с тремя вариантами сигнала:
        original  — исходные каналы
        smoothed  — сглаженные
        pca       — первая главная компонента

    Parameters
    ----------
    signals : np.ndarray
        (n_channels, n_samples) — данные в µV.
    srate : int
        Частота дискретизации.

    Returns
    -------
    dict  с ключами 'original', 'smoothed', 'pca' — каждый ndarray.
    """
    from .eeg_preprocessing import smooth_signal, pca_transform

    smoothed = smooth_signal(signals, window_len=max(3, srate // 25))
    pca = pca_transform(signals, n_components=1)

    return {
        "original": signals,
        "smoothed": smoothed,
        "pca": pca,
    }


def get_neurosoft_file_summary(filepath: str) -> dict:
    """
    Краткая информация о .EEG файле (для отображения в GUI).
    Не загружает данные целиком — только заголовок.

    Returns
    -------
    dict с ключами: filename, study_name, date, time, device,
        n_channels, srate, duration_sec, n_samples, channel_names.
    """
    info = parse_header(str(filepath))
    return {
        "filepath": str(filepath),
        "filename": Path(filepath).name,
        "study_name": info.get("study_name", ""),
        "date": info.get("date", ""),
        "time": info.get("time", ""),
        "device": info.get("device", ""),
        "patient_birth": info.get("patient_birth", ""),
        "patient_sex": info.get("patient_sex", ""),
        "n_channels": info["n_channels"],
        "srate": info["srate"],
        "duration_sec": info["duration_sec"],
        "n_samples": info["n_samples"],
        "channel_names": [ch["name"] for ch in info["channels"]],
        "eeg_channels": [
            ch["name"] for ch in info["channels"]
            if ch["name"] not in _NON_EEG_CHANNELS
        ],
    }
