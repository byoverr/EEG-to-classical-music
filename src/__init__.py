"""
Public exports for the current EEG emotion-validation pipeline.
"""

from .deap_loader import extract_eeg_from_deap, get_emotion_labels, load_deap_participant_data
from .eeg_preprocessing import pca_transform, prepare_signal_data, smooth_signal
from .eeg_processing import (
    compress_timed_events,
    detect_wave_motifs,
    detect_wave_motifs_segmented,
    map_motifs_to_adsr_sounds,
    process_eeg_to_midi,
)
from .evaluation import add_hypothesis_scores, compute_hypothesis_metrics, save_hypothesis_artifacts
from .maestro_loader import get_maestro_metadata, get_maestro_midi_files
from .midi_utils import (
    create_comparison_midi,
    create_midi_from_notes,
    create_midi_with_precise_timing,
    extract_melody_sequence,
    extract_melody_with_time,
    extract_note_events,
)
from .neurosoft_loader import get_neurosoft_file_summary, load_neurosoft_eeg, prepare_neurosoft_signal_data

__all__ = [
    "add_hypothesis_scores",
    "compress_timed_events",
    "compute_hypothesis_metrics",
    "create_comparison_midi",
    "create_midi_from_notes",
    "create_midi_with_precise_timing",
    "detect_wave_motifs",
    "detect_wave_motifs_segmented",
    "extract_eeg_from_deap",
    "extract_melody_sequence",
    "extract_melody_with_time",
    "extract_note_events",
    "get_emotion_labels",
    "get_maestro_metadata",
    "get_maestro_midi_files",
    "get_neurosoft_file_summary",
    "load_deap_participant_data",
    "load_neurosoft_eeg",
    "map_motifs_to_adsr_sounds",
    "pca_transform",
    "prepare_neurosoft_signal_data",
    "prepare_signal_data",
    "process_eeg_to_midi",
    "save_hypothesis_artifacts",
    "smooth_signal",
]
