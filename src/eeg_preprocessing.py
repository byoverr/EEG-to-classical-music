"""
Модуль предобработки EEG сигналов из DEAP датасета.
Содержит функции для сглаживания, PCA-преобразования и подготовки данных.
"""
import numpy as np
from sklearn.decomposition import PCA
from .deap_loader import extract_eeg_from_deap

def smooth_signal(signal_array: np.ndarray, window_len: int = 5) -> np.ndarray:
    """
    Применяет сглаживание (скользящее среднее) к сигналу.
    
    Параметры:
    - signal_array: массив (channels, time) или (time,)
    - window_len: длина окна сглаживания
    
    Возвращает:
    - сглаженный массив той же формы
    """
    arr = np.array(signal_array)
    original_shape = arr.shape
    
    # Приводим к (time, channels) для итерации
    if arr.ndim == 1:
        # Если одномерный (time,), делаем (time, 1)
        arr = arr.reshape(-1, 1)
        transposed = False
    else:
        # Если (channels, time), транспонируем в (time, channels)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
            transposed = True
        else:
            transposed = False

    kernel = np.ones(window_len) / float(window_len)
    smoothed = np.zeros_like(arr)
    
    for ch in range(arr.shape[1]):
        # mode='same' сохраняет размер, но края могут быть искажены
        smoothed[:, ch] = np.convolve(arr[:, ch], kernel, mode='same')

    # Возвращаем к исходной форме
    if transposed:
        smoothed = smoothed.T
    if len(original_shape) == 1:
        smoothed = smoothed.flatten()
        
    return smoothed


def pca_transform(signal_array: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Применяет PCA (Principal Component Analysis) к сигналу.
    Используется для выделения главных компонент из многоканального EEG.
    
    Параметры:
    - signal_array: массив (channels, time)
    - n_components: количество компонент
    
    Возвращает:
    - массив (n_components, time)
    """
    arr = np.array(signal_array)
    
    # Sklearn PCA требует (n_samples, n_features). 
    # В нашем случае samples=time, features=channels.
    # Поэтому, если вход (channels, time), нужно транспонировать.
    
    if arr.ndim == 1:
        # Если 1 канал, PCA не имеет смысла для уменьшения размерности, 
        # но можно вернуть как есть или нормировать.
        return arr.reshape(1, -1)
        
    if arr.shape[0] < arr.shape[1]:
        # (channels, time) -> (time, channels)
        arr_t = arr.T
        transposed_input = True
    else:
        # Уже (time, channels), предполагаем что каналов меньше чем точек времени
        arr_t = arr
        transposed_input = False

    # Нормализация (стандартизация)
    mean = np.mean(arr_t, axis=0)
    std = np.std(arr_t, axis=0) + 1e-10 # избегаем деления на 0
    arr_norm = (arr_t - mean) / std

    pca = PCA(n_components=n_components)
    try:
        # fit_transform возвращает (time, n_components)
        comps = pca.fit_transform(arr_norm)
    except Exception as e:
        print(f"Ошибка PCA: {e}. Возвращаем среднее.")
        # Фолбек: просто среднее по каналам
        comps = np.mean(arr_t, axis=1).reshape(-1, 1)
        # Если нужно n_components > 1, дублируем (но это плохой кейс)
        if n_components > 1:
            comps = np.tile(comps, (1, n_components))

    # Возвращаем в формате (n_components, time)
    return comps.T


def prepare_signal_data(participant_data: dict, trial_idx: int) -> dict:
    """
    Собирает все варианты сигналов для заданного триала.
    Эта функция выполняет роль агрегатора для передачи данных в processing.
    
    Параметры:
    - participant_data: данные участника (из deap_loader)
    - trial_idx: индекс видео/триала (0-39)
    
    Возвращает словарь:
    {
        'original': np.array (32, 8064), # Исходные 32 канала (очищенные DEAP)
        'smoothed': np.array (32, 8064), # Сглаженные 32 канала
        'pca': np.array (1, 8064)        # 1 главная компонента
    }
    """
    # 1. Получаем исходные данные (уже 128Гц, 4-45Гц фильтр)
    # extract_eeg_from_deap возвращает (32, 8064)
    raw_eeg = extract_eeg_from_deap(participant_data, trial_idx, eeg_only=True)
    
    # 2. Создаем сглаженную версию
    # smooth_signal принимает (channels, time) и возвращает (channels, time)
    smoothed_eeg = smooth_signal(raw_eeg, window_len=10) # Чуть больше окно для заметного эффекта
    
    # 3. Создаем PCA версию (снижаем размерность до 1 главной компоненты)
    # pca_transform принимает (channels, time) и возвращает (n_components, time)
    pca_eeg = pca_transform(raw_eeg, n_components=1)
    
    return {
        'original': raw_eeg,
        'smoothed': smoothed_eeg,
        'pca': pca_eeg
    }
