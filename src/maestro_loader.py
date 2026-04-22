"""
Небольшой загрузчик MAESTRO dataset (CSV/JSON метаданные -> список MIDI файлов)
"""
import threading
from pathlib import Path
import pandas as pd
from typing import List, Optional, Dict, TypedDict


class TrackMeta(TypedDict):
    """Унифицированный формат метаданных трека для MAESTRO и EMOPIA."""
    track_id: str
    dataset: str  # "maestro" | "emopia"
    midi_path: str
    audio_path: Optional[str]
    title: Optional[str]
    composer: Optional[str]
    emotion: Optional[str]  # для EMOPIA: HVHA/HVLA/LVLA/LVHA; для MAESTRO: None или predicted
    emotion_source: Optional[str]  # "ground_truth" | "predicted" | None
    emotion_confidence: Optional[float]
    emotion_model: Optional[str]


# Глобальный кэш для метаданных MAESTRO + блокировка для thread-safety
_maestro_cache: Dict[str, TrackMeta] = {}
_maestro_cache_lock = threading.RLock()


def load_maestro_metadata(maestro_dir: str):
    """
    Загружает maestro-v3.0.0.csv (если есть) и возвращает DataFrame с полями,
    добавляя полный путь к midi файлу в колонку 'midi_path'.
    """
    # Обновляем кэш in-place; .clear() небезопасен при параллельной обработке
    maestro_dir = Path(maestro_dir)
    csv_path = maestro_dir / 'maestro-v3.0.0.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # В CSV часто есть колонка 'midi_filename' или 'midi_path'
    if 'midi_filename' in df.columns:
        df['midi_path'] = df['midi_filename'].apply(lambda p: str(maestro_dir / p))
    elif 'midi_path' in df.columns:
        df['midi_path'] = df['midi_path'].apply(lambda p: str(maestro_dir / p))
    else:
        # Попытаемся составить путь из year/track/filename
        if 'year' in df.columns and 'midi_filename' in df.columns:
            df['midi_path'] = df.apply(lambda r: str(maestro_dir / str(r['year']) / r['midi_filename']), axis=1)

    # Оставляем только существующие файлы
    df['midi_exists'] = df['midi_path'].apply(lambda p: Path(p).exists())
    
    # Создаём кэш метаданных в новом формате TrackMeta (thread-safe)
    for _, row in df.iterrows():
        filename = Path(row['midi_path']).name
        track_id = Path(row['midi_path']).stem
        entry = TrackMeta(
            track_id=track_id,
            dataset='maestro',
            midi_path=row['midi_path'],
            audio_path=row.get('audio_filename', None),
            composer=row.get('canonical_composer', ''),
            title=row.get('canonical_title', ''),
            emotion=row.get('emotion', None) if 'emotion' in df.columns else None,
            emotion_source=row.get('emotion_source', None) if 'emotion_source' in df.columns else None,
            emotion_confidence=row.get('emotion_confidence', None) if 'emotion_confidence' in df.columns else None,
            emotion_model=row.get('emotion_model', None) if 'emotion_model' in df.columns else None,
        )
        with _maestro_cache_lock:
            _maestro_cache[filename] = entry

    return df


def get_maestro_metadata(filename: str) -> TrackMeta:
    """
    Возвращает метаданные для MIDI файла из кэша.
    
    Параметры:
    - filename: имя файла (без пути или полный путь)
    
    Возвращает TrackMeta словарь с метаданными
    """
    # Нормализуем имя файла
    filename = Path(filename).name
    with _maestro_cache_lock:
        cached = _maestro_cache.get(filename)
    if cached is None:
        # Возвращаем пустую запись
        return TrackMeta(
            track_id=Path(filename).stem,
            dataset='maestro',
            midi_path='',
            audio_path=None,
            composer='',
            title='',
            emotion=None,
            emotion_source=None,
            emotion_confidence=None,
            emotion_model=None
        )
    return cached


def get_maestro_midi_files(maestro_dir: str, max_files: Optional[int] = None) -> List[str]:
    """
    Возвращает список существующих MIDI файлов из MAESTRO CSV.
    Также заполняет глобальный кэш метаданных.
    """
    df = load_maestro_metadata(maestro_dir)
    available = df[df['midi_exists']]
    paths = available['midi_path'].tolist()
    if max_files:
        paths = paths[:max_files]
    return paths
