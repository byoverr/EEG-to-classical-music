"""
Модуль для работы с MIDI файлами.
Содержит только используемые функции.
"""
import mido
from mido import MidiFile, MidiTrack, Message


def extract_melody_sequence(midi_path, max_notes=None):
    """
    Извлекает мелодическую последовательность из MIDI файла
    
    Возвращает:
    - список pitch values (MIDI ноты)
    """
    try:
        mid = MidiFile(midi_path)
        notes = []
        active_notes = {}
        
        for track in mid.tracks:
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time = active_notes[msg.note]
                        notes.append((msg.note, start_time))
                        del active_notes[msg.note]
        
        notes.sort(key=lambda x: x[1])
        
        if max_notes and len(notes) > max_notes:
            notes = notes[:max_notes]
        
        return [note[0] for note in notes]
    
    except Exception as e:
        print(f"Ошибка при чтении {midi_path}: {e}")
        return []


def extract_melody_with_time(midi_path, max_notes=None):
    """
    Извлекает мелодию с временными метками для sliding window сравнения
    
    Возвращает:
    - список кортежей (pitch, time_in_seconds)
    """
    try:
        mid = MidiFile(midi_path)
        notes = []
        
        tempo = 500000  # микросекунды на beat (120 BPM по умолчанию)
        ticks_per_beat = mid.ticks_per_beat
        
        for track in mid.tracks:
            current_time_ticks = 0
            active_notes = {}
            
            for msg in track:
                current_time_ticks += msg.time
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time_ticks
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_ticks = active_notes[msg.note]
                        time_in_seconds = mido.tick2second(start_ticks, ticks_per_beat, tempo)
                        notes.append((msg.note, time_in_seconds))
                        del active_notes[msg.note]
        
        notes.sort(key=lambda x: x[1])
        
        if max_notes and len(notes) > max_notes:
            notes = notes[:max_notes]
        
        return notes
    
    except Exception as e:
        print(f"Ошибка при чтении {midi_path}: {e}")
        return []


def create_midi_from_notes(notes, output_path, tempo_bpm=120, track_name="Melody"):
    """
    Создает MIDI файл из списка нот

    Параметры:
    - notes: список MIDI нот (pitch values)
    - output_path: путь для сохранения
    - tempo_bpm: темп
    - track_name: название трека
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tempo_microseconds = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
    track.append(mido.MetaMessage('track_name', name=track_name))

    ticks_per_beat = mid.ticks_per_beat
    note_duration_beats = 0.5
    duration_ticks = int(note_duration_beats * ticks_per_beat)

    for pitch in notes:
        track.append(Message('note_on', channel=0, note=pitch, velocity=80, time=0))
        track.append(Message('note_off', channel=0, note=pitch, velocity=80, time=duration_ticks))

    mid.save(output_path)
    return mid


def create_comparison_midi(eeg_fragment, classical_fragment, output_path, tempo_bpm=120):
    """
    Создает MIDI файл с двумя треками: ЭЭГ и классическое произведение

    Параметры:
    - eeg_fragment: фрагмент мелодии из ЭЭГ
    - classical_fragment: фрагмент мелодии из классического произведения
    - output_path: путь для сохранения
    - tempo_bpm: темп
    """
    mid = MidiFile()

    tempo_microseconds = mido.bpm2tempo(tempo_bpm)

    # Трек 1: ЭЭГ мелодия
    track_eeg = MidiTrack()
    mid.tracks.append(track_eeg)
    track_eeg.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
    track_eeg.append(mido.MetaMessage('track_name', name='EEG Melody'))

    ticks_per_beat = mid.ticks_per_beat
    note_duration_beats = 0.5
    duration_ticks = int(note_duration_beats * ticks_per_beat)

    for pitch in eeg_fragment:
        track_eeg.append(Message('note_on', channel=0, note=pitch, velocity=64, time=0))
        track_eeg.append(Message('note_off', channel=0, note=pitch, velocity=64, time=duration_ticks))

    # Трек 2: Классическая мелодия
    track_classical = MidiTrack()
    mid.tracks.append(track_classical)
    track_classical.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
    track_classical.append(mido.MetaMessage('track_name', name='Classical Melody'))

    for pitch in classical_fragment:
        track_classical.append(Message('note_on', channel=1, note=pitch, velocity=64, time=0))
        track_classical.append(Message('note_off', channel=1, note=pitch, velocity=64, time=duration_ticks))

    mid.save(output_path)
    return mid


def create_midi_with_precise_timing(events, output_path, tempo_bpm=120):
    """
    Creates a MIDI file from a list of timed events, preserving exact timing gaps.
    
    Parameters:
    - events: list of dicts with keys {'onset', 'duration', 'pitch', 'velocity'}
              'onset' and 'duration' are in seconds.
    - output_path: path to save the MIDI file.
    - tempo_bpm: tempo in BPM.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    tempo_microseconds = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))
    
    ticks_per_beat = mid.ticks_per_beat
    seconds_per_tick = mido.tick2second(1, ticks_per_beat, tempo_microseconds)
    
    events.sort(key=lambda x: x['onset'])
    
    # Linearize all Note On and Note Off messages
    midi_messages = []
    for event in events:
        onset_sec = event['onset']
        duration_sec = event['duration']
        pitch = int(event['pitch'])
        velocity = int(event['velocity'])
        
        midi_messages.append({
            'time_sec': onset_sec,
            'type': 'note_on',
            'note': pitch,
            'velocity': velocity
        })
        
        midi_messages.append({
            'time_sec': onset_sec + duration_sec,
            'type': 'note_off',
            'note': pitch,
            'velocity': 0
        })
        
    midi_messages.sort(key=lambda x: x['time_sec'])
    
    last_time_sec = 0
    
    for msg in midi_messages:
        current_time_sec = msg['time_sec']
        delta_seconds = max(0, current_time_sec - last_time_sec)
        delta_ticks = int(delta_seconds / seconds_per_tick)
        
        track.append(Message(msg['type'], note=msg['note'], velocity=msg['velocity'], time=delta_ticks))
        
        last_time_sec = current_time_sec
        
    mid.save(output_path)
    return mid
