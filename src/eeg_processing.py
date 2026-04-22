"""
Модуль обработки ЭЭГ и сонификации на основе волновых мотивов.
Точная реализация метода Destexhe and Foubert (2022).
"""
from __future__ import annotations

import numpy as np
from .midi_utils import create_midi_with_precise_timing
from . import config as _cfg


def _shift_motif(motif: dict, sample_offset: int, time_offset: float) -> dict:
    shifted = dict(motif)
    shifted["onset_time"] = float(shifted.get("onset_time", 0.0)) + time_offset
    shifted["peak_time"] = float(shifted.get("peak_time", 0.0)) + time_offset
    for key in ("_onset_idx", "_peak_idx", "_offset_idx"):
        if key in shifted:
            shifted[key] = int(shifted[key]) + sample_offset
    return shifted


def _deduplicate_motifs(motifs: list[dict], min_separation_sec: float = 0.08) -> list[dict]:
    if not motifs:
        return []

    ordered = sorted(
        motifs,
        key=lambda m: (
            float(m.get("onset_time", 0.0)),
            -abs(float(m.get("peak_amplitude", 0.0))),
            -float(m.get("duration", 0.0)),
        ),
    )
    deduped: list[dict] = []
    for motif in ordered:
        onset = float(motif.get("onset_time", 0.0))
        if deduped and abs(onset - float(deduped[-1].get("onset_time", 0.0))) < min_separation_sec:
            prev_amp = abs(float(deduped[-1].get("peak_amplitude", 0.0)))
            curr_amp = abs(float(motif.get("peak_amplitude", 0.0)))
            if curr_amp > prev_amp:
                deduped[-1] = motif
            continue
        deduped.append(motif)
    return deduped

def detect_wave_motifs(signal_array: np.ndarray,
                      fs: float | None = None,
                      threshold1_std: float | None = None,
                      threshold2_std: float | None = None,
                      min_duration: float | None = None,
                      max_duration: float | None = None):
    """
    Детекция волновых мотивов методом двух порогов (Destexhe & Foubert, 2022).
    
    Метод из статьи (Figure 1):
    1. Threshold 1 детектирует upward stroke (начало волны)
    2. Измеряется peak amplitude и peak time
    3. Threshold 2 детектирует downward stroke (конец волны)
    4. Извлекаются параметры: onset, rise time, amplitude, peak time, decay time
    
    Параметры:
    - signal_array: одномерный EEG сигнал
    - fs: частота дискретизации (250 Hz в статье)
    - threshold1_std: порог 1 для upstroke (в единицах std)
    - threshold2_std: порог 2 для downstroke (в единицах std)
    - min_duration: минимальная длительность волны (секунды)
    - max_duration: максимальная длительность волны (секунды)
    
    Возвращает:
    - список словарей с параметрами волн для ADSR маппинга
    """
    # Дефолты берём из config, чтобы не было рассинхрона между вызывающим кодом и конфигом
    if fs is None:
        fs = float(_cfg.DEAP_SAMPLE_RATE)
    if threshold1_std is None:
        threshold1_std = float(_cfg.EEG_THRESHOLD_LOW_STD)
    if threshold2_std is None:
        threshold2_std = float(_cfg.EEG_THRESHOLD_HIGH_STD)
    if min_duration is None:
        min_duration = float(_cfg.EEG_MIN_WAVE_DURATION)
    if max_duration is None:
        max_duration = float(_cfg.EEG_MAX_WAVE_DURATION)

    mean_sig = np.mean(signal_array)
    std_sig = np.std(signal_array)
    
    # Два независимых порога как в статье
    threshold1 = mean_sig + threshold1_std * std_sig  # Upstroke detection
    threshold2 = mean_sig + threshold2_std * std_sig  # Downstroke detection (может быть выше или ниже)
    
    n_samples = len(signal_array)
    motifs = []
    
    i = 0
    while i < n_samples - 1:
        # Шаг 1: Детекция upward stroke через threshold 1
        if signal_array[i] < threshold1 and signal_array[i+1] >= threshold1:
            onset_idx = i
            onset_time = onset_idx / fs
            
            # Шаг 2: Поиск пика (максимума) после onset
            j = i + 1
            peak_idx = j
            peak_amplitude = signal_array[j]
            
            found_peak = False
            
            while j < n_samples:
                # Обновляем пик если нашли больше значение
                if signal_array[j] > peak_amplitude:
                    peak_amplitude = signal_array[j]
                    peak_idx = j
                
                # Шаг 3: Детекция downward stroke 
                # Ищем когда сигнал упадет ниже threshold (может быть threshold1 или другой)
                # В статье используется второй порог, но логика - это обратное пересечение
                # Используем threshold1 для симметрии (волна возвращается к baseline)
                if j > onset_idx + 1 and signal_array[j-1] >= threshold1 and signal_array[j] < threshold1:
                    offset_idx = j
                    found_peak = True
                    break
                
                # Защита от бесконечных волн
                if (j - onset_idx) / fs > max_duration:
                    break
                    
                j += 1
            
            # Валидация и создание мотива
            if found_peak:
                duration_sec = (offset_idx - onset_idx) / fs
                peak_time = peak_idx / fs
                rise_time = (peak_idx - onset_idx) / fs
                decay_time = (offset_idx - peak_idx) / fs
                
                if min_duration <= duration_sec <= max_duration:
                    motif = {
                        # Временные параметры
                        'onset_time': onset_time,
                        'peak_time': peak_time,
                        'duration': duration_sec,
                        
                        # ADSR параметры (ключевые для синтеза!)
                        'attack': rise_time,      # Rise time -> Attack
                        'decay': decay_time,       # Decay time -> Decay
                        'peak_amplitude': peak_amplitude,
                        
                        # Дополнительные параметры
                        'rise_time': rise_time,
                        'decay_time': decay_time,
                        
                        # Индексы для отладки
                        '_onset_idx': onset_idx,
                        '_peak_idx': peak_idx,
                        '_offset_idx': offset_idx
                    }
                    motifs.append(motif)
                    i = offset_idx  # Переходим к концу волны
                    continue
            
            i += 1
        else:
            i += 1
    
    return motifs


def detect_wave_motifs_segmented(
    signal_array: np.ndarray,
    fs: float | None = None,
    threshold1_std: float | None = None,
    threshold2_std: float | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    segment_duration: float = 12.0,
    hop_duration: float = 6.0,
):
    """
    Локальная детекция мотивов по перекрывающимся сегментам.

    Это снимает проблему глобальных порогов на длинных записях:
    середина записи может иметь более слабую амплитуду, но все равно
    содержать валидные волновые структуры, которые теряются при одном
    глобальном std на весь файл.
    """
    if fs is None:
        fs = float(_cfg.DEAP_SAMPLE_RATE)
    if threshold1_std is None:
        threshold1_std = float(_cfg.EEG_THRESHOLD_LOW_STD)
    if threshold2_std is None:
        threshold2_std = float(_cfg.EEG_THRESHOLD_HIGH_STD)
    if min_duration is None:
        min_duration = float(_cfg.EEG_MIN_WAVE_DURATION)
    if max_duration is None:
        max_duration = float(_cfg.EEG_MAX_WAVE_DURATION)

    signal_array = np.asarray(signal_array, dtype=float)
    if signal_array.size == 0:
        return []

    segment_samples = max(int(segment_duration * fs), int(max_duration * fs * 3), 64)
    hop_samples = max(int(hop_duration * fs), int(segment_samples * 0.5), 1)
    if signal_array.size <= segment_samples:
        return detect_wave_motifs(
            signal_array,
            fs=fs,
            threshold1_std=threshold1_std,
            threshold2_std=threshold2_std,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    segment_starts = list(range(0, max(signal_array.size - segment_samples + 1, 1), hop_samples))
    tail_start = max(signal_array.size - segment_samples, 0)
    if not segment_starts or segment_starts[-1] != tail_start:
        segment_starts.append(tail_start)

    motifs: list[dict] = []
    for start in segment_starts:
        end = min(start + segment_samples, signal_array.size)
        segment = signal_array[start:end]
        seg_motifs = detect_wave_motifs(
            segment,
            fs=fs,
            threshold1_std=threshold1_std,
            threshold2_std=threshold2_std,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        time_offset = start / max(fs, 1e-8)
        for motif in seg_motifs:
            motifs.append(_shift_motif(motif, start, time_offset))

    min_sep = max(min_duration * 0.5, 0.05)
    return _deduplicate_motifs(motifs, min_separation_sec=min_sep)


def map_motifs_to_adsr_sounds(motifs: list, scale_key: str = 'C_major_pentatonic'):
    """
    Преобразует параметры мотивов в музыкальные события с ADSR envelope.
    Реализует маппинг из статьи (Section III.B, Figure 2-3).

    Маппинг из статьи:
    - Onset time (EEG) -> Onset time (sound)
    - Amplitude (EEG) -> Volume/Velocity (sound)
    - Rise time (EEG) -> Attack (ADSR)
    - Decay time (EEG) -> Decay (ADSR)
    - Duration (EEG) -> Duration (sound)

    Параметры:
    - motifs: список обнаруженных мотивов
    - scale_key: тональность для маппинга высоты

    Возвращает:
    - список событий для MIDI с ADSR параметрами
    """
    if not motifs:
        return []

    # =========================================================================
    # ГАММЫ: расширенный набор с музыкально-теоретическим обоснованием
    # =========================================================================
    # Каждая гамма покрывает 3 октавы (MIDI 48-84) для достаточного
    # мелодического диапазона. Выбор ступеней соответствует музыкальной
    # теории (интервальная структура сохранена).
    scales = {
        # Мажорная пентатоника (C-D-E-G-A): яркая, оптимистичная
        # Отсутствие IV и VII ступеней убирает все полутоновые напряжения
        'C_major_pentatonic': [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84],

        # Натуральный мажор (C-D-E-F-G-A-B): полная палитра, светлый характер
        'C_major': [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76],

        # Минорная пентатоника (A-C-D-E-G): грустная, но мягкая
        # Отсутствие II и VI ступеней — нет острых полутонов
        'A_minor_pentatonic': [45, 48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81],

        # Натуральный минор (A-B-C-D-E-F-G): классическая грусть
        'A_natural_minor': [45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72],

        # Гармонический минор (A-B-C-D-E-F-G#): напряжение, драматизм
        # Увеличенная секунда F-G# создаёт характерное «восточное» напряжение
        'A_harmonic_minor': [45, 47, 48, 50, 52, 53, 56, 57, 59, 60, 62, 64, 65, 68, 69, 71, 72],

        # Дорийский лад (D-E-F-G-A-B-C): мягкий минор, меланхолия без трагизма
        'D_dorian': [50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74],

        # Обратная совместимость
        'D_minor': [50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72],
    }

    scale = scales.get(scale_key, scales['C_major_pentatonic'])

    vel_min, vel_max = 50, 127
    vel_range = vel_max - vel_min

    # Нормализация параметров для более выразительного маппинга
    amplitudes = [m['peak_amplitude'] for m in motifs]
    durations = [m['duration'] for m in motifs]
    rise_times = [m['rise_time'] for m in motifs]
    min_amp = np.min(amplitudes)
    max_amp = np.max(amplitudes)
    amp_range = max_amp - min_amp if max_amp > min_amp else 1.0
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    dur_range = max_dur - min_dur if max_dur > min_dur else 1.0
    min_rise = np.min(rise_times)
    max_rise = np.max(rise_times)
    rise_range = max_rise - min_rise if max_rise > min_rise else 1.0

    events = []
    prev_pitch = None
    prev_norm_amp = None

    for idx, m in enumerate(motifs):
        # Нормализованная амплитуда [0, 1]
        norm_amp = (m['peak_amplitude'] - min_amp) / amp_range
        norm_dur = (m['duration'] - min_dur) / dur_range
        norm_rise = (m['rise_time'] - min_rise) / rise_range
        phrase_progress = idx / max(len(motifs) - 1, 1)

        # === МАППИНГ КАК В СТАТЬЕ ===

        # 1. Amplitude -> Velocity (Volume)
        # "the amplitude can be mapped to different parameters of the sound wave...
        # the amplitude of the EEG wave onto the amplitude of the sound wave (which reflects its volume)"
        velocity = int(vel_min + norm_amp * vel_range)

        # 2. EEG parameters -> Pitch.
        # Чтобы мелодия не схлопывалась в 2-3 одинаковые ноты,
        # используем не только амплитуду, но и длительность, rise-time и локальный contour.
        base_position = (
            0.45 * norm_amp
            + 0.25 * norm_dur
            + 0.20 * norm_rise
            + 0.10 * phrase_progress
        )
        pitch_idx = int(round(base_position * (len(scale) - 1)))

        if prev_norm_amp is not None:
            delta_amp = norm_amp - prev_norm_amp
            if delta_amp > 0.08:
                pitch_idx += 1
            elif delta_amp < -0.08:
                pitch_idx -= 1

        pitch_idx = max(0, min(len(scale) - 1, pitch_idx))
        pitch = scale[pitch_idx]

        # Избегаем длинных серий одинаковых нот.
        if prev_pitch is not None and pitch == prev_pitch:
            shift = 1 if (idx % 2 == 0) else -1
            alt_idx = max(0, min(len(scale) - 1, pitch_idx + shift))
            pitch = scale[alt_idx]

        # 3. ADSR параметры из EEG волны
        # "the attack and decay can be mapped to the rise and decay times of the EEG wave"
        attack = m['attack']
        decay = m['decay']

        # Sustain и Release из статьи:
        # "the tail of the sound wave (with sustain parameter) could be used
        # to let the sound last until the next sound comes in"
        sustain_level = 0.7  # 70% от пика
        release = min(0.15, decay * 0.5)  # Короткий release

        # 4. Duration mapping
        # "A natural conversion is the duration of the sound wave"
        duration = m['duration']

        event = {
            # Время
            'onset': m['onset_time'],
            'duration': duration,

            # Нота и громкость
            'pitch': pitch,
            'velocity': velocity,

            # ADSR envelope (ключевой элемент метода!)
            'attack': attack,
            'decay': decay,
            'sustain': sustain_level,
            'release': release,

            # Метаданные
            '_amplitude': m['peak_amplitude'],
            '_norm_amplitude': norm_amp,
            '_scale': scale_key,
        }

        events.append(event)
        prev_pitch = pitch
        prev_norm_amp = norm_amp

    return events


def compress_timed_events(
    events: list[dict],
    *,
    min_gap_sec: float = 0.08,
    max_gap_sec: float = 0.75,
    min_duration_sec: float = 0.10,
    max_duration_sec: float = 0.90,
):
    """
    Строит компактный analysis timeline для EEG-мелодии.

    Raw EEG onsets часто содержат большие паузы. Для playback и window-based
    musical matching нам нужен непрерывный мелодический поток, поэтому
    сохраняем порядок и относительную вариативность, но агрессивно сжимаем
    длинные интервалы между событиями.
    """
    if not events:
        return []

    compress_factor = 1.0

    ordered = sorted(events, key=lambda ev: float(ev.get("onset", 0.0)))
    compressed = []
    prev_raw_onset = None
    prev_new_onset = 0.0

    for idx, event in enumerate(ordered):
        raw_onset = float(event.get("onset", 0.0))
        raw_duration = max(min_duration_sec, float(event.get("duration", min_duration_sec)))
        duration = min(max_duration_sec, raw_duration)

        if idx == 0:
            new_onset = 0.0
        else:
            raw_gap = max(0.0, raw_onset - prev_raw_onset)
            compressed_gap = 0.06 + 0.35 * float(np.sqrt(raw_gap))
            compressed_gap = min(max_gap_sec, max(min_gap_sec, compressed_gap))
            new_onset = prev_new_onset + compressed_gap

        compressed.append({
            **event,
            "onset": float(new_onset),
            "duration": float(duration),
            "velocity": int(np.clip(int(event.get("velocity", 80)), 1, 127)),
        })

        prev_raw_onset = raw_onset
        prev_new_onset = new_onset

    return compressed


def process_eeg_to_midi(signal_data: dict, output_path: str,
                       fs: float | None = None,
                       threshold1: float | None = None,
                       threshold2: float | None = None,
                       scale: str = 'C_major_pentatonic'):
    """
    Главная функция: ЭЭГ -> MIDI через волновые мотивы.
    Полная реализация метода Destexhe & Foubert (2022).
    
    Процесс из статьи:
    1. Запись brain activity (EEG) - уже есть в signal_data
    2. Анализ сигнала: детекция и параметризация волн (detect_wave_motifs)
    3. Синтез звука: параметры волн -> ADSR envelope -> звуки (map_motifs_to_adsr_sounds)
    
    Параметры:
    - signal_data: {'original', 'smoothed', 'pca'} из preprocessing
    - output_path: путь к MIDI файлу
    - threshold1: порог 1 (в std) для upstroke
    - threshold2: порог 2 (в std) для downstroke
    - scale: музыкальная гамма
    
    Возвращает:
    - количество сгенерированных звуковых событий
    """
    # Шаг 1: Выбор сигнала для анализа
    # Статья работает с фронтальными отведениями (FP1, FP2) и дельта-волнами
    # Используем PCA (первая компонента) или сглаженный сигнал
    
    if 'pca' in signal_data and signal_data['pca'].size > 0:
        sig = signal_data['pca']
        analysis_signal = sig[0, :] if sig.ndim > 1 else sig
    elif 'smoothed' in signal_data and signal_data['smoothed'].size > 0:
        sig = signal_data['smoothed']
        analysis_signal = sig[0, :] if sig.ndim > 1 else sig
    else:
        raise ValueError("No suitable signal in signal_data (need 'pca' or 'smoothed')")

    if fs is None:
        fs = float(_cfg.DEAP_SAMPLE_RATE)

    # Шаг 2: Детекция волновых мотивов (Section III.A)
    motifs = detect_wave_motifs(
        analysis_signal,
        fs=fs,
        threshold1_std=threshold1,
        threshold2_std=threshold2,
        min_duration=_cfg.EEG_MIN_WAVE_DURATION,
        max_duration=_cfg.EEG_MAX_WAVE_DURATION,
    )

    # Если ничего не найдено — снижаем пороги, но всё равно требуем результата
    if not motifs:
        motifs = detect_wave_motifs(
            analysis_signal,
            fs=fs,
            threshold1_std=max(0.2, _cfg.EEG_THRESHOLD_LOW_STD * 0.6),
            threshold2_std=max(0.5, _cfg.EEG_THRESHOLD_HIGH_STD * 0.6),
            min_duration=max(0.05, _cfg.EEG_MIN_WAVE_DURATION * 0.5),
            max_duration=_cfg.EEG_MAX_WAVE_DURATION + 0.5,
        )

    if not motifs:
        # Не создаём пустой MIDI «молча» — сообщаем вызывающему, чтобы тот решил, что делать
        raise RuntimeError(
            f"Мотивы не обнаружены даже с пониженными порогами "
            f"(fs={fs} Hz, signal_len={len(analysis_signal)} samples). "
            "Проверьте качество сигнала/препроцессинг."
        )

    # Шаг 3: Преобразование в звуки с ADSR
    music_events = map_motifs_to_adsr_sounds(motifs, scale_key=scale)

    # Шаг 4: Сохранение в MIDI
    create_midi_with_precise_timing(music_events, output_path)

    return len(music_events)
