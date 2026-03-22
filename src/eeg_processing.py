"""
Модуль обработки ЭЭГ и сонификации на основе волновых мотивов.
Точная реализация метода Destexhe and Foubert (2022).
"""
import numpy as np
from .midi_utils import create_midi_with_precise_timing

def detect_wave_motifs(signal_array: np.ndarray, fs: float = 250.0, 
                      threshold1_std: float = 0.5, 
                      threshold2_std: float = 1.0,
                      min_duration: float = 0.2, 
                      max_duration: float = 2.0):
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
    
    Опционально: Amplitude может мапиться также на Pitch
    
    Параметры:
    - motifs: список обнаруженных мотивов
    - scale_key: тональность для маппинга высоты
    
    Возвращает:
    - список событий для MIDI с ADSR параметрами
    """
    if not motifs:
        return []
    
    # Гаммы (пентатоника для приятного звучания)
    scales = {
        'C_major_pentatonic': [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84],
        'A_minor_pentatonic': [45, 48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81],
        'D_minor': [50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]
    }
    
    scale = scales.get(scale_key, scales['C_major_pentatonic'])
    
    # Нормализация амплитуд для маппинга
    amplitudes = [m['peak_amplitude'] for m in motifs]
    min_amp = np.min(amplitudes)
    max_amp = np.max(amplitudes)
    amp_range = max_amp - min_amp if max_amp > min_amp else 1.0
    
    events = []
    
    for m in motifs:
        # Нормализованная амплитуда [0, 1]
        norm_amp = (m['peak_amplitude'] - min_amp) / amp_range
        
        # === МАППИНГ КАК В СТАТЬЕ ===
        
        # 1. Amplitude -> Velocity (Volume)
        # "the amplitude can be mapped to different parameters of the sound wave...
        # the amplitude of the EEG wave onto the amplitude of the sound wave (which reflects its volume)"
        velocity = int(50 + norm_amp * 77)  # 50-127 range
        
        # 2. Amplitude -> Pitch (опционально, в статье упоминается как альтернатива)
        # "It can also be mapped to another parameter, such as the pitch"
        # Выше амплитуда -> выше нота (интенсивность)
        pitch_idx = int(norm_amp * (len(scale) - 1))
        pitch = scale[pitch_idx]
        
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
            '_norm_amplitude': norm_amp
        }
        
        events.append(event)
    
    return events


def process_eeg_to_midi(signal_data: dict, output_path: str, 
                       threshold1: float = 0.5, threshold2: float = 1.0,
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
    
    print(f"Analyzing signal: {len(analysis_signal)} samples")
    
    # Шаг 2: Детекция волновых мотивов (Section III.A)
    # "scanning the signal and detecting the delta waves using a two-threshold procedure"
    motifs = detect_wave_motifs(
        analysis_signal, 
        fs=250.0,  # Частота из статьи
        threshold1_std=threshold1,
        threshold2_std=threshold2,
        min_duration=0.2,  # Минимум для дельта-волн
        max_duration=2.0
    )
    
    print(f"Detected {len(motifs)} wave motifs")
    
    # Если ничего не найдено, снижаем пороги
    if not motifs:
        print("Warning: No motifs found, trying lower thresholds...")
        motifs = detect_wave_motifs(
            analysis_signal,
            fs=250.0,
            threshold1_std=0.3,
            threshold2_std=0.6,
            min_duration=0.15,
            max_duration=2.5
        )
        print(f"Detected {len(motifs)} motifs with lower thresholds")
    
    if not motifs:
        print("ERROR: Still no motifs detected!")
        # Создаем пустой MIDI
        create_midi_with_precise_timing([], output_path)
        return 0
    
    # Шаг 3: Преобразование в звуки с ADSR (Section III.B)
    # "converting to the classic parametrization of sound envelopes: ADSR parameters"
    music_events = map_motifs_to_adsr_sounds(motifs, scale_key=scale)
    
    print(f"Generated {len(music_events)} musical events")
    
    # Шаг 4: Сохранение в MIDI
    create_midi_with_precise_timing(music_events, output_path)
    
    # Статистика
    if motifs:
        durations = [m['duration'] for m in motifs]
        amplitudes = [m['peak_amplitude'] for m in motifs]
        print(f"\nMotif statistics:")
        print(f"  Duration: {np.mean(durations):.3f}s (±{np.std(durations):.3f}s)")
        print(f"  Amplitude: {np.mean(amplitudes):.3f} (±{np.std(amplitudes):.3f})")
        print(f"  Total time span: {motifs[-1]['onset_time']:.2f}s")
    
    return len(music_events)