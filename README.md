# EEG-to-Classical-Music

> **Дипломная работа СПбГЭТУ "ЛЭТИ"**  
> *"Методы преобразования сигналов ЭЭГ в музыкальные структуры и их сравнение с классическими произведениями"*

---

## 📋 Описание

Проект реализует полный pipeline для преобразования сигналов электроэнцефалограммы (ЭЭГ) в музыкальные MIDI-последовательности и их сравнения с классическими произведениями из датасета MAESTRO.

### Научная основа

Проект объединяет методы из трёх научных работ:

| Метод | Источник | Применение |
|-------|----------|------------|
| **Wave Motif Detection** | Destexhe & Foubert (2022) | Детекция волновых паттернов в ЭЭГ с двухпороговой системой |
| **ADSR Sonification** | Destexhe & Foubert (2022) | Маппинг параметров волн (амплитуда, длительность) на MIDI ноты |
| **Scale-Free Index** | Wu et al. (2018) | Анализ распределения питчей по закону Ципфа (Zipf's law) |
| **Contour Similarity** | Miranda (2010) | DTW-сравнение мелодических контуров |

---

## 🏗️ Архитектура

```
EEG-to-classical-music/
├── data/
│   ├── deap_dataset/          # DEAP EEG данные (32 участника × 40 триалов)
│   ├── maestro-v3.0.0/        # MAESTRO MIDI датасет (1200+ произведений)
│   ├── best_matches/          # Выходные MIDI фрагменты
│   ├── reports/               # HTML отчёты
│   └── FluidR3_GM.sf2         # SoundFont для синтеза аудио
│
├── src/
│   ├── eeg_processing.py      # Детекция волновых мотивов (Destexhe 2022)
│   ├── eeg_preprocessing.py   # Предобработка: сглаживание, PCA
│   ├── MIDIComparator.py      # Сравнение MIDI (Wu 2018 + Miranda 2010)
│   ├── midi_utils.py          # Утилиты для работы с MIDI
│   ├── deap_loader.py         # Загрузчик DEAP датасета
│   ├── maestro_loader.py      # Загрузчик MAESTRO датасета
│   ├── audio_converter.py     # Конвертация MIDI → WAV (FluidSynth)
│   ├── html_generator.py      # Генерация HTML отчётов
│   └── config.py              # Конфигурация путей
│
├── scripts/
│   └── run_comparison.py      # Главный скрипт запуска
│
└── notebooks/                 # Jupyter ноутбуки для экспериментов
```

---

## 🔬 Методология

### 1. Детекция волновых мотивов (Destexhe & Foubert, 2022)

```
EEG Signal
    │
    ▼
┌─────────────────────────────────────┐
│  Threshold 1 → Upward stroke        │
│  Peak detection → Amplitude         │
│  Threshold 2 → Downward stroke      │
└─────────────────────────────────────┘
    │
    ▼
Wave Motif Parameters:
  • onset_time
  • rise_time  
  • peak_amplitude
  • decay_time
  • duration
```

### 2. ADSR Маппинг на MIDI

| Параметр волны | MIDI атрибут |
|---------------|--------------|
| Peak amplitude | MIDI velocity (громкость) |
| Duration | Note duration (длительность) |
| Rise/Decay ratio | Attack/Release time |
| Frequency | Pitch (нота) |

### 3. Scale-Free Index (Wu, 2018)

Анализирует распределение частот нот по закону Ципфа:
```
rank ~ frequency^(-α)
```
- **α ≈ 1.0** — "эстетичная" музыка (1/f шум)
- **α < 0.5** — случайный шум
- **α > 1.5** — слишком предсказуемая структура

### 4. Комплексная метрика сходства

```python
combined_similarity = (
    0.30 × contour_similarity +    # DTW мелодического контура
    0.25 × sfi_similarity +        # Scale-Free Index
    0.20 × correlation +           # Корреляция Пирсона
    0.15 × harmony_similarity +    # Гармоническое сходство
    0.10 × statistical_similarity  # Статистические метрики
)
```

---

## ⚙️ Установка

### Требования
- Python 3.8+
- FluidSynth (для синтеза аудио)

### Шаги установки

```bash
# 1. Клонирование репозитория
git clone https://github.com/your-username/EEG-to-classical-music.git
cd EEG-to-classical-music

# 2. Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# или .venv\Scripts\activate  # Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка FluidSynth (macOS)
brew install fluid-synth

# 5. Скачивание датасетов
# DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
# MAESTRO: https://magenta.tensorflow.org/datasets/maestro
```

### Опциональные зависимости

```bash
# Для ускоренного DTW
pip install dtaidistance

# Для улучшенной визуализации
pip install seaborn
```

---

## 🚀 Использование

### Базовый запуск

```bash
python scripts/run_comparison.py
```

### Расширенные параметры

```bash
python scripts/run_comparison.py \
    --participants 3 \     # Количество участников DEAP (1-32)
    --trials 3 \           # Триалов на участника (1-40)  
    --classical 50 \       # Классических произведений (1-1200)
    --top 15               # Топ результатов в отчёте
```

### Пример вывода

```
============================================================
EEG to Classical Music Comparison Pipeline
============================================================

[1/4] Загрузка MAESTRO датасета...
  Загружено 50 классических произведений

[2/4] Обработка DEAP участников...
  Участник: s01
    Триал 0: Valence=1.3, Arousal=1.4
    ✓ original: 436 events → s01_trial00_original.mid
    ✓ smoothed: 393 events → s01_trial00_smoothed.mid
    ✓ pca: 431 events → s01_trial00_pca.mid

[3/4] Формирование результатов...
  Топ-15 лучших совпадений:
    1. s03/0 (pca) → Alexander Scriabin - Sonata No. 5
       Combined=0.861 (contour=1.00, sfi=0.98, harmony=0.94)

[4/4] Генерация HTML отчёта...
  ✓ Отчёт сохранён: runs/run_001/report/index.html
```

### Плейлисты по эмоциям (EEG → похожие композиции)

```bash
python scripts/export_emotion_playlists.py --convert-to-wav
```

Результат:
- runs/run_001/playlists/index.html
- runs/run_001/playlists/HVHA.html (и т.д.)

### Мини-отчёт по признакам эмоций

```bash
python scripts/emotion_feature_report.py
```

Результат:
- runs/run_001/report/emotion_feature_report.csv
- runs/run_001/report/emotion_feature_summary.md

---

## 📊 Выходные данные

### HTML отчёт

Интерактивный отчёт с:
- 🎵 Встроенными аудио-плеерами (EEG, Classical, Comparison)
- 📈 Метриками сходства
- 📋 Эмоциональными метками (Valence, Arousal)
- 🎹 Визуализацией фрагментов

### MIDI файлы

```
data/best_matches/
├── 01_pca_EEG_Alexander_Scriabin_Sonata_No._5.mid      # EEG фрагмент
├── 01_pca_Classical_Alexander_Scriabin_Sonata_No._5.mid # Классический фрагмент
├── 01_pca_Comparison_Alexander_Scriabin_Sonata_No._5.mid # Сравнение (2 дорожки)
└── ...
```

---

## 📚 Датасеты

### DEAP Dataset
- **Источник**: Queen Mary University of London
- **Содержимое**: 32 участника × 40 триалов × 32 EEG канала
- **Частота**: 512 Hz (downsampled до 128 Hz)
- **Метки**: Valence, Arousal, Dominance, Liking

### MAESTRO Dataset v3.0.0
- **Источник**: Google Magenta
- **Содержимое**: ~1200 классических MIDI файлов
- **Композиторы**: Chopin, Beethoven, Scriabin, Debussy, Bach и др.
- **Формат**: Aligned MIDI + Audio

### EMOPIA Dataset
- **Источник**: Emotion in Music (piano MIDI)
- **Содержимое**: MIDI клипы с метками эмоций
- **Метки**: 4 квадранта VA: HVHA, HVLA, LVHA, LVLA
- **Размещение**: data/EMOPIA_1.0
  - label.csv
  - midis/

### Псевдо-разметка MAESTRO (опционально)

По умолчанию выключено. Включить можно флагом в конфиге:

```python
# src/config.py
USE_PSEUDO_LABELING = True
```

В отчёте такие эмоции отмечаются как predicted.

---

## 🧪 Метрики качества

| Метрика | Описание | Диапазон |
|---------|----------|----------|
| **Combined** | Взвешенная комбинация всех метрик | 0-1 |
| **Contour** | DTW сходство мелодического контура | 0-1 |
| **SFI** | Scale-Free Index similarity | 0-1 |
| **Harmony** | Гармоническое сходство (интервалы) | 0-1 |
| **Correlation** | Корреляция Пирсона питчей | -1 to 1 |

---

## 📖 Ссылки

1. **Destexhe, A., & Foubert, B.** (2022). *Listening to the brain: Sonification of EEG, LFP and Spiking activity*. bioRxiv.

2. **Wu, B., et al.** (2018). *Modeling perceived musical emotion with Score-Free music features*. Frontiers in Psychology.

3. **Miranda, E. R.** (2010). *Brain-Computer Music Interface for Composition and Performance*. International Journal on Disability and Human Development.

---

## 📄 Лицензия

MIT License

---

## 👤 Автор

**Студент СПбГЭТУ "ЛЭТИ"**  
Факультет компьютерных технологий и информатики (ФКТИ)  
2025












































