"""
Конфигурация проекта EEG-to-Classical-Music.
Содержит только используемые константы.
"""
import os
from pathlib import Path

# =============================================================================
# ПУТИ К ДАННЫМ
# =============================================================================

# Корень проекта (автоопределение)
_config_dir = Path(__file__).parent
PROJECT_ROOT = _config_dir.parent

# Директории данных
DATA_DIR = PROJECT_ROOT / "data"
MAESTRO_DIR = DATA_DIR / "maestro-v3.0.0"
EMOPIA_DIR = DATA_DIR / "EMOPIA_1.0"
DEAP_DIR = DATA_DIR / "deap_dataset" / "data_preprocessed_python"
BEST_MATCHES_DIR = DATA_DIR / "best_matches"
REPORTS_DIR = DATA_DIR / "reports"
SOUNDFONT_PATH = DATA_DIR / "FluidR3_GM.sf2"

# Директория запусков (отчёты, плейлисты)
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_RUN_ID = "run_001"

# Псевдо-разметка MAESTRO (кэш)
MAESTRO_PSEUDO_LABELS_PATH = PROJECT_ROOT / "data" / "maestro_labeling" / "maestro_predictions.csv"

# =============================================================================
# ПАРАМЕТРЫ DEAP ДАТАСЕТА
# =============================================================================

# Частота дискретизации DEAP (фиксирована)
DEAP_SAMPLE_RATE = 128  # Hz

# Глобальный seed для воспроизводимости (random.shuffle, np.random, sklearn и т.п.)
RANDOM_SEED = 42

# Количество участников и триалов для обработки
DEAP_NUM_PARTICIPANTS = 3
DEAP_NUM_TRIALS = 5

# =============================================================================
# ПАРАМЕТРЫ СРАВНЕНИЯ
# =============================================================================

# Размер окна для sliding window сравнения (секунды)
COMPARISON_WINDOW_SIZE = 4.0

# Шаг окна (секунды)
COMPARISON_HOP_SIZE = 2.0

# Количество лучших совпадений для сохранения
TOP_N_MATCHES = 10

# =============================================================================
# ПАРАМЕТРЫ ЗАПУСКА ПО УМОЛЧАНИЮ (можно переопределить через CLI)
# =============================================================================

DEFAULT_MAX_PARTICIPANTS = DEAP_NUM_PARTICIPANTS
DEFAULT_MAX_TRIALS = DEAP_NUM_TRIALS
DEFAULT_MAX_CLASSICAL = 10
DEFAULT_TOP_K = TOP_N_MATCHES
DEFAULT_JOBS = None

DEFAULT_ONLY_EMOPIA = False
DEFAULT_BALANCED_EEG_EMOTIONS = False
DEFAULT_PER_EMOTION_TRIALS = 3
DEFAULT_MATCH_EMOTIONS = False
DEFAULT_INCLUDE_TOP_EMOTIONS = False

# Веса для комбинированной метрики сходства
SIMILARITY_WEIGHTS = {
    'contour': 0.25,
    'interval': 0.15,
    'harmony': 0.10,
    'emotional': 0.25,
    'trend': 0.15,
    'dynamic': 0.10
}

# =============================================================================
# ПАРАМЕТРЫ EEG ОБРАБОТКИ
# =============================================================================

# Порог для детекции волновых мотивов (в единицах std)
EEG_THRESHOLD_LOW_STD = 0.3
EEG_THRESHOLD_HIGH_STD = 0.8

# Минимальная/максимальная длительность волны (секунды)
EEG_MIN_WAVE_DURATION = 0.05
EEG_MAX_WAVE_DURATION = 2.0

# Минимальное расстояние между пиками (samples)
EEG_MIN_PEAK_DISTANCE = 10

# Тональность для EEG-MIDI (по умолчанию, если эмоция неизвестна)
EEG_SCALE_KEY = 'C_major_pentatonic'



# =============================================================================
# ПАРАМЕТРЫ ЭКСПОРТА И PLAYBACK
# =============================================================================

# Темп для воспроизведения MIDI (множитель)
PLAYBACK_TEMPO_MULTIPLIER = 1.2

# Длительность matched-фрагмента для GUI playback (секунды)
MATCH_FRAGMENT_DURATION = 8.0
HTML_FRAGMENT_DURATION = MATCH_FRAGMENT_DURATION  # алиас для совместимости

# =============================================================================
# ПАРАМЕТРЫ ЭМОЦИОНАЛЬНОЙ РАЗМЕТКИ
# =============================================================================

# Порог для маппинга DEAP оценок (1-9) в квадранты (high/low)
EMOTION_THRESHOLD = 5.0

# Источник классической музыки для сравнения: "both" | "emopia" | "maestro"
# - "both"    — MAESTRO + EMOPIA (по умолчанию)
# - "emopia"  — только EMOPIA (эмоционально размеченный)
# - "maestro" — только MAESTRO (псевдо-разметка)
DATASET_SOURCE = "both"

# Допустимые значения DATASET_SOURCE
DATASET_SOURCE_CHOICES = ("both", "emopia", "maestro")

# (устарело) Использовать ли EMOPIA — оставлено для обратной совместимости
USE_EMOPIA = True

# Максимальное количество треков из EMOPIA (None = все)
EMOPIA_MAX_TRACKS = None

# Использовать ли pseudo-labeling для MAESTRO
USE_PSEUDO_LABELING = True

# Переиспользовать ранее сгенерированные EEG MIDI (ускоряет повторный запуск)
REUSE_EEG_MIDI = True

# Очищать директорию eeg_midi перед запуском (если False и REUSE_EEG_MIDI=False — просто перезаписывает)
CLEAN_DATA_ON_RUN = False

# Квадранты эмоций (VALENCE-AROUSAL): высокая валентность = позитив, высокое возбуждение = энергия
# HVHA = High Valence, High Arousal (позитив + энергия, например радость, восторг)
# HVLA = High Valence, Low Arousal (позитив + спокойствие, например миролюбие, нежность)
# LVLA = Low Valence, Low Arousal (негатив + спокойствие, например печаль, грусть)
# LVHA = Low Valence, High Arousal (негатив + энергия, например гнев, интенсивность)
EMOTION_QUADRANTS = ['HVHA', 'HVLA', 'LVLA', 'LVHA']

# Штраф за несовпадение эмоций при сравнении (если --match-emotions не используется)
# Может быть полезным для weighted matching вместо strict filtering
EMOTION_MISMATCH_PENALTY = 0.2  # мягкий штраф за несовпадение эмоций
