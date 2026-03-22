"""
Модуль для загрузки и обработки DEAP датасета
DEAP (Database for Emotion Analysis using Physiological signals)

Структура данных DEAP:
- 32 файла (s01.dat - s32.dat), по одному на участника
- data: (40, 40, 8064) — 40 триалов, 40 каналов, 8064 отсчётов
  - Каналы 0-31: ЭЭГ (используем)
  - Каналы 32-39: периферийные сигналы (не используем)
- labels: (40, 4) — Valence, Arousal, Dominance, Liking (1-9)
"""
import pickle
import numpy as np
from pathlib import Path

# Константы структуры DEAP
DEAP_EEG_CHANNELS = 32  # Первые 32 канала — ЭЭГ


def load_deap_participant_data(participant_file):
    """
    Загружает данные одного участника из DEAP датасета
    
    Параметры:
    - participant_file: путь к файлу .dat участника (например, s01.dat)
    
    Возвращает:
    - словарь с ключами: 'data', 'labels'
    """
    with open(participant_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    return data


def extract_eeg_from_deap(data_dict, trial_idx, channel_idx=None, eeg_only=True):
    """
    Извлекает ЭЭГ данные для конкретного триала
    
    Параметры:
    - data_dict: словарь данных участника
    - trial_idx: индекс триала (0-39)
    - channel_idx: индекс канала ЭЭГ (None = все EEG каналы)
    - eeg_only: если True, возвращает только 32 EEG канала (по умолчанию)
    
    Возвращает:
    - массив ЭЭГ данных (channels, time_points)
    """
    if 'data' not in data_dict:
        raise ValueError("Ключ 'data' не найден в данных")
    
    eeg_data = data_dict['data']
    
    # Формат DEAP: (trials, channels, time_points)
    if len(eeg_data.shape) == 3:
        if channel_idx is not None:
            return eeg_data[trial_idx, channel_idx, :]  # Один канал
        elif eeg_only:
            # Только первые 32 канала (EEG), исключаем периферийные
            return eeg_data[trial_idx, :DEAP_EEG_CHANNELS, :]
        else:
            return eeg_data[trial_idx, :, :]  # Все 40 каналов
    else:
        raise ValueError(f"Неожиданная форма данных: {eeg_data.shape}")


def get_emotion_labels(data_dict, trial_idx):
    """
    Получает эмоциональные метки для триала
    
    Параметры:
    - data_dict: словарь данных участника
    - trial_idx: индекс триала
    
    Возвращает:
    - словарь с ключами: 'valence', 'arousal', 'dominance', 'liking'
    """
    labels = {}
    
    if 'labels' in data_dict:
        labels_data = data_dict['labels']
        if len(labels_data.shape) == 2:
            labels['valence'] = labels_data[trial_idx, 0]
            labels['arousal'] = labels_data[trial_idx, 1]
            labels['dominance'] = labels_data[trial_idx, 2]
            labels['liking'] = labels_data[trial_idx, 3]
    
    return labels
