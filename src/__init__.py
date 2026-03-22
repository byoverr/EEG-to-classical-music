"""
Модули для преобразования ЭЭГ в музыкальные структуры (сонификация).
Экспортирует только используемые функции.
"""

# EEG Processing
from .eeg_processing import (
    detect_wave_motifs,
    map_motifs_to_adsr_sounds,
    process_eeg_to_midi
)

# EEG Preprocessing
from .eeg_preprocessing import (
    smooth_signal,
    pca_transform,
    prepare_signal_data
)

# MIDI Utilities - только используемые
from .midi_utils import (
    extract_melody_sequence,
    extract_melody_with_time,
    create_midi_from_notes,
    create_comparison_midi,
    create_midi_with_precise_timing
)

# DEAP Dataset Loader - только используемые
from .deap_loader import (
    load_deap_participant_data,
    extract_eeg_from_deap,
    get_emotion_labels
)

# Maestro Dataset Loader - только используемые
from .maestro_loader import (
    get_maestro_metadata,
    get_maestro_midi_files
)

# HTML Generator - только используемые
from .html_generator import (
    create_comparison_html,
    create_simple_comparison_html
)

# MIDI Comparator
from .MIDIComparator import MIDIComparator, ComprehensiveMIDIComparator

__all__ = [
    # EEG Processing
    'detect_wave_motifs',
    'map_motifs_to_adsr_sounds',
    'process_eeg_to_midi',
    # EEG Preprocessing
    'smooth_signal',
    'pca_transform',
    'prepare_signal_data',
    # MIDI Utilities
    'extract_melody_sequence',
    'extract_melody_with_time',
    'create_midi_from_notes',
    'create_comparison_midi',
    'create_midi_with_precise_timing',
    # DEAP Loader
    'load_deap_participant_data',
    'extract_eeg_from_deap',
    'get_emotion_labels',
    # Maestro Loader
    'get_maestro_metadata',
    'get_maestro_midi_files',
    # HTML Generator
    'create_comparison_html',
    'create_simple_comparison_html',
    # MIDI Comparator
    'MIDIComparator',
    'ComprehensiveMIDIComparator',
]

