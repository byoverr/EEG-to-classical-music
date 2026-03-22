"""
Модуль для конвертации MIDI в WAV с поддержкой разных платформ
"""
import os
import platform
from pathlib import Path


def find_soundfont():
    """
    Находит доступный SoundFont файл для FluidSynth
    
    Возвращает:
    - путь к SoundFont файлу или None
    """
    system = platform.system()
    
    # Список возможных путей к SoundFont
    possible_paths = []
    
    if system == "Darwin":  # macOS
        possible_paths = [
            # Проверяем стандартные пути на Mac
            Path.home() / ".fluidsynth" / "default_sound_font.sf2",
            "/opt/homebrew/share/soundfonts/default.sf2",
            "/usr/local/share/soundfonts/default.sf2",
            "/usr/share/soundfonts/default.sf2",
            # Проверяем в проекте
            Path(__file__).parent.parent / "data" / "FluidR3_GM.sf2",
        ]
    elif system == "Linux":
        possible_paths = [
            "/usr/share/sounds/sf2/default.sf2",
            "/usr/share/soundfonts/default.sf2",
            Path(__file__).parent.parent / "data" / "FluidR3_GM.sf2",
        ]
    elif system == "Windows":
        possible_paths = [
            Path(__file__).parent.parent / "data" / "FluidR3_GM.sf2",
            "C:/Program Files/FluidSynth/soundfonts/default.sf2",
        ]
    
    # Проверяем каждый путь
    for path in possible_paths:
        p = Path(path)
        if p.exists() and p.is_file():
            return str(p)
    
    return None

import subprocess
import os

def midi_to_wav(midi_path, wav_path, soundfont_path, sample_rate=44100):
    
    # Убедитесь, что пути передаются как строки и они существуют
    midi_path_str = str(midi_path)
    wav_path_str = str(wav_path)
    soundfont_path_str = str(soundfont_path)

    if not os.path.exists(soundfont_path_str):
        print(f"Ошибка: Soundfont файл не найден по пути {soundfont_path_str}")
        return False
    if not os.path.exists(midi_path_str):
        print(f"Ошибка: MIDI файл не найден по пути {midi_path_str}")
        return False
        
    try:
        # !!! ПРАВИЛЬНЫЙ ПОРЯДОК АРГУМЕНТОВ !!!
        # Опции (-F, -r) идут ПЕРЕД позиционными аргументами (soundfont, midi)
        command = [
            'fluidsynth', 
            '-F', wav_path_str, 
            '-r', str(sample_rate),
            soundfont_path_str, 
            midi_path_str
        ]
        
        print(f"Выполнение команды: {' '.join(command)}")

        # Выполняем команду
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Команда выполнена успешно. Проверка файла...")
        
        # Проверяем, был ли создан файл
        if os.path.exists(wav_path_str) and os.path.getsize(wav_path_str) > 0:
            print(f"Файл {wav_path_str} успешно создан.")
            return True
        else:
            print(f"Ошибка: Файл {wav_path_str} пуст или не был создан.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения fluidsynth. STDOUT: {e.stdout}. STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return False






