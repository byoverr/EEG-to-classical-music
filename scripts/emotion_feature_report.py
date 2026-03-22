#!/usr/bin/env python3
"""
Мини-отчет по признакам, отличающим эмоции.
Собирает фичи для EMOPIA (ground truth) и EEG-мелодий (DEAP → квадранты).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kruskal

from src.config import EMOPIA_DIR, RUNS_DIR, DEFAULT_RUN_ID
from src.emopia_loader import get_emopia_midi_files, get_emopia_metadata
from src.track_features import (
    load_feature_cache, save_feature_cache, get_or_compute_features
)


FEATURE_COLUMNS = [
    'pitch_mean', 'pitch_std', 'pitch_range',
    'note_density', 'ioi_mean', 'ioi_std',
    'consonance_mean', 'consonance_std',
    'pitch_class_entropy', 'tempo_proxy',
    'sfi_pitch'
]


def main():
    parser = argparse.ArgumentParser(description="Emotion feature report")
    default_report_dir = RUNS_DIR / DEFAULT_RUN_ID / 'report'
    parser.add_argument('--reports-dir', type=str, default=str(default_report_dir))
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = reports_dir / 'comparison_results.csv'
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    if 'eeg_emotion' not in df.columns:
        raise ValueError("comparison_results.csv missing 'eeg_emotion'")

    cache_path = reports_dir / 'feature_cache.json'
    cache = load_feature_cache(cache_path)

    rows = []

    # EMOPIA ground truth
    emopia_files = get_emopia_midi_files(str(EMOPIA_DIR))
    for path in emopia_files:
        feats = get_or_compute_features(path, cache)
        if feats is None:
            continue
        emotion = get_emopia_metadata(Path(path).stem).get('emotion', 'unknown')
        row = {'source': 'emopia', 'emotion': emotion, 'midi_path': str(path)}
        for col in FEATURE_COLUMNS:
            row[col] = feats.get(col, 0.0)
        rows.append(row)

    # EEG melodies (unique)
    eeg_unique = df[['eeg_midi', 'eeg_emotion']].drop_duplicates()
    for _, r in eeg_unique.iterrows():
        path = r['eeg_midi']
        feats = get_or_compute_features(path, cache)
        if feats is None:
            continue
        row = {'source': 'eeg', 'emotion': r['eeg_emotion'], 'midi_path': str(path)}
        for col in FEATURE_COLUMNS:
            row[col] = feats.get(col, 0.0)
        rows.append(row)

    save_feature_cache(cache_path, cache)

    features_df = pd.DataFrame(rows)
    csv_out = output_dir / 'emotion_feature_report.csv'
    features_df.to_csv(csv_out, index=False)

    # Statistical summary (Kruskal-Wallis across emotions)
    summary_rows = []
    for feature in FEATURE_COLUMNS:
        groups = []
        labels = []
        for emotion in ['HVHA', 'HVLA', 'LVHA', 'LVLA']:
            vals = features_df[features_df['emotion'] == emotion][feature].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(emotion)
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            summary_rows.append({'feature': feature, 'H': float(H), 'p_value': float(p)})

    summary_df = pd.DataFrame(summary_rows).sort_values('H', ascending=False)
    summary_csv = output_dir / 'emotion_feature_summary.csv'
    summary_df.to_csv(summary_csv, index=False)

    # Markdown summary
    md_lines = ["# Emotion Feature Summary", "", "Top distinguishing features (Kruskal-Wallis):", ""]
    for _, row in summary_df.head(8).iterrows():
        md_lines.append(f"- **{row['feature']}**: H={row['H']:.3f}, p={row['p_value']:.3e}")
    md_path = output_dir / 'emotion_feature_summary.md'
    md_path.write_text("\n".join(md_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
