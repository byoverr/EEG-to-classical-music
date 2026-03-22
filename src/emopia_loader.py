"""
Загрузчик датасета EMOPIA (Emotion in Music Emotion Piano) для интеграции с проектом.

EMOPIA содержит фортепианные MIDI с разметкой эмоций в 4 квадрантах VA-пространства:
- Q1 (HVHA): High Valence, High Arousal
- Q2 (HVLA): High Valence, Low Arousal  
- Q3 (LVLA): Low Valence, Low Arousal
- Q4 (LVHA): Low Valence, High Arousal

Структура датасета:
- midis/: MIDI файлы с префиксом Q1-Q4
- label.csv: метаданные (ID, 4Q, annotator)
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Tuple


class TrackMeta(TypedDict):
    """Унифицированный формат метаданных трека для MAESTRO и EMOPIA."""
    track_id: str
    dataset: str  # "maestro" | "emopia"
    midi_path: str
    audio_path: Optional[str]
    title: Optional[str]
    composer: Optional[str]
    emotion: Optional[str]  # для EMOPIA: HVHA/HVLA/LVLA/LVHA; для MAESTRO: None
    emotion_source: Optional[str]  # "ground_truth" | "predicted" | None


# Маппинг Q1-Q4 → читаемые эмоциональные квадранты
EMOPIA_QUADRANT_MAP = {
    'Q1': 'HVHA',  # High Valence, High Arousal
    'Q2': 'HVLA',  # High Valence, Low Arousal
    'Q3': 'LVLA',  # Low Valence, Low Arousal
    'Q4': 'LVHA',  # Low Valence, High Arousal
}


# Глобальный кэш метаданных EMOPIA
_emopia_cache: Dict[str, TrackMeta] = {}
_emopia_meta_cache: Dict[str, dict] = {}


def _parse_emopia_track_id(track_id: str) -> Tuple[str, Optional[int]]:
    """
    Парсит EMOPIA track_id: Q1_<youtube_id>_<clip>
    Возвращает (youtube_id, clip_index)
    """
    parts = track_id.split('_')
    if len(parts) >= 3:
        youtube_id = parts[1]
        try:
            clip_idx = int(parts[2])
        except Exception:
            clip_idx = None
        return youtube_id, clip_idx
    return track_id, None


def _load_youtube_metadata(emopia_dir: Path, track_id: str) -> dict:
    """Загружает metadata JSON для конкретного трека EMOPIA."""
    if track_id in _emopia_meta_cache:
        return _emopia_meta_cache[track_id]

    youtube_id, _ = _parse_emopia_track_id(track_id)
    # Metadata файлы содержат префикс квадранта Q1/Q2/Q3/Q4
    quadrant = track_id.split('_')[0]
    meta_path = emopia_dir / 'metadata' / f"{quadrant}_{youtube_id}.json"

    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _emopia_meta_cache[track_id] = data
                return data
        except Exception:
            pass

    _emopia_meta_cache[track_id] = {}
    return {}


def load_emopia_metadata(emopia_dir: str) -> pd.DataFrame:
    """
    Загружает метаданные EMOPIA из label.csv и добавляет пути к MIDI.
    
    Параметры:
    - emopia_dir: путь к корню датасета EMOPIA
    
    Возвращает:
    - DataFrame с колонками: ID, 4Q, emotion, midi_path, midi_exists
    """
    global _emopia_cache
    _emopia_cache.clear()
    
    emopia_dir = Path(emopia_dir)
    label_csv = emopia_dir / 'label.csv'
    
    if not label_csv.exists():
        raise FileNotFoundError(f"EMOPIA label.csv not found: {label_csv}")
    
    # Загружаем метаданные
    df = pd.read_csv(label_csv)
    
    # Извлекаем квадрант из ID (например, Q1_xxxxx_0 → Q1)
    df['quadrant'] = df['ID'].str.extract(r'^(Q\d)_')[0]
    
    # Мапим квадранты в читаемые эмоции
    df['emotion'] = df['quadrant'].map(EMOPIA_QUADRANT_MAP)
    
    # Строим путь к MIDI: midis/{ID}.mid
    midis_dir = emopia_dir / 'midis'
    df['midi_path'] = df['ID'].apply(lambda x: str(midis_dir / f"{x}.mid"))
    
    # Проверяем существование файлов
    df['midi_exists'] = df['midi_path'].apply(lambda p: Path(p).exists())
    
    # Заполняем кэш
    for _, row in df.iterrows():
        if row['midi_exists']:
            track_id = row['ID']
            meta = _load_youtube_metadata(emopia_dir, track_id)
            youtube_id, clip_idx = _parse_emopia_track_id(track_id)
            title = meta.get('title') or f"YouTube {youtube_id}"
            uploader = meta.get('uploader') or "YouTube"
            clip_suffix = f" (clip {clip_idx})" if clip_idx is not None else ""
            _emopia_cache[track_id] = TrackMeta(
                track_id=track_id,
                dataset='emopia',
                midi_path=row['midi_path'],
                audio_path=None,  # WAV генерируется по требованию
                title=f"{title}{clip_suffix}",
                composer=uploader,
                emotion=row['emotion'],
                emotion_source='ground_truth'
            )
    
    return df


def get_emopia_metadata(track_id: str) -> TrackMeta:
    """
    Возвращает метаданные для EMOPIA трека из кэша.
    
    Параметры:
    - track_id: ID трека (например, Q1_xxxxx_0)
    
    Возвращает:
    - TrackMeta словарь с метаданными
    """
    if track_id not in _emopia_cache:
        # Возвращаем пустую запись, если не найдено
        youtube_id, clip_idx = _parse_emopia_track_id(track_id)
        clip_suffix = f" (clip {clip_idx})" if clip_idx is not None else ""
        return TrackMeta(
            track_id=track_id,
            dataset='emopia',
            midi_path='',
            audio_path=None,
            title=f"YouTube {youtube_id}{clip_suffix}",
            composer="YouTube",
            emotion=None,
            emotion_source=None
        )
    return _emopia_cache[track_id]


def get_emopia_track_info(emopia_dir: str, track_id: str) -> dict:
    """
    Возвращает человекочитаемую информацию о треке EMOPIA.
    """
    emopia_dir = Path(emopia_dir)
    meta = _load_youtube_metadata(emopia_dir, track_id)
    youtube_id, clip_idx = _parse_emopia_track_id(track_id)
    return {
        'youtube_id': youtube_id,
        'clip_idx': clip_idx,
        'title': meta.get('title') or f"YouTube {youtube_id}",
        'uploader': meta.get('uploader') or "YouTube",
        'webpage_url': meta.get('webpage_url', None)
    }


def get_emopia_midi_files(emopia_dir: str, 
                           emotion_filter: Optional[List[str]] = None,
                           max_files: Optional[int] = None) -> List[str]:
    """
    Возвращает список путей к существующим MIDI файлам из EMOPIA.
    
    Параметры:
    - emopia_dir: путь к корню EMOPIA
    - emotion_filter: список эмоций для фильтрации (например, ['HVHA', 'LVLA'])
    - max_files: максимальное количество файлов (None = все)
    
    Возвращает:
    - список путей к MIDI файлам
    """
    df = load_emopia_metadata(emopia_dir)
    
    # Фильтруем только существующие файлы
    available = df[df['midi_exists']].copy()
    
    # Фильтруем по эмоциям, если указано
    if emotion_filter:
        available = available[available['emotion'].isin(emotion_filter)]
    
    paths = available['midi_path'].tolist()
    
    if max_files:
        paths = paths[:max_files]
    
    return paths


def get_emopia_tracks_by_emotion(emopia_dir: str) -> Dict[str, List[TrackMeta]]:
    """
    Группирует все EMOPIA треки по эмоциям.
    
    Параметры:
    - emopia_dir: путь к корню EMOPIA
    
    Возвращает:
    - словарь {emotion: [TrackMeta, ...]}
    """
    df = load_emopia_metadata(emopia_dir)
    available = df[df['midi_exists']]
    
    grouped = {}
    for emotion in ['HVHA', 'HVLA', 'LVLA', 'LVHA']:
        tracks = available[available['emotion'] == emotion]['ID'].tolist()
        grouped[emotion] = [get_emopia_metadata(tid) for tid in tracks]
    
    return grouped


def get_emotion_from_filename(filename: str) -> Optional[str]:
    """
    Извлекает эмоцию из имени EMOPIA файла.
    
    Параметры:
    - filename: имя файла (например, Q1_xxxxx_0.mid)
    
    Возвращает:
    - эмоция (HVHA/HVLA/LVLA/LVHA) или None
    """
    filename = Path(filename).stem
    
    # Извлекаем квадрант из имени
    for quadrant, emotion in EMOPIA_QUADRANT_MAP.items():
        if filename.startswith(quadrant + '_'):
            return emotion
    
    return None


def deap_to_emotion_quadrant(valence: float, arousal: float, 
                              threshold: float = 5.0) -> str:
    """
    Конвертирует оценки DEAP (valence, arousal) в квадранты EMOPIA.
    
    DEAP использует шкалу 1-9 для valence и arousal.
    Разделяем по порогу (по умолчанию 5.0):
    - valence >= threshold → High Valence
    - arousal >= threshold → High Arousal
    
    Параметры:
    - valence: оценка валентности (1-9)
    - arousal: оценка активации (1-9)
    - threshold: порог разделения (по умолчанию 5.0)
    
    Возвращает:
    - строку квадранта: HVHA/HVLA/LVHA/LVLA
    """
    high_v = valence >= threshold
    high_a = arousal >= threshold
    
    if high_v and high_a:
        return 'HVHA'
    elif high_v and not high_a:
        return 'HVLA'
    elif not high_v and high_a:
        return 'LVHA'
    else:
        return 'LVLA'


def get_all_emotions() -> List[str]:
    """Возвращает список всех эмоциональных квадрантов."""
    return ['HVHA', 'HVLA', 'LVLA', 'LVHA']
