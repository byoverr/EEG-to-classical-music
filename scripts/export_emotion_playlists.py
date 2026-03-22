#!/usr/bin/env python3
"""
Экспорт плейлистов по эмоциям для EEG мелодий и похожих композиций.
Создает HTML в runs/playlists и WAV-файлы для прослушивания.
"""
import argparse
import shutil
from pathlib import Path
import random
import pandas as pd

from src.config import RUNS_DIR, DEFAULT_RUN_ID
from src.audio_converter import midi_to_wav, find_soundfont


def sanitize_filename(s: str, max_len: int = 60) -> str:
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        s = s.replace(char, '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')[:max_len]


def prepare_audio_asset(midi_path: Path, output_dir: Path, prefix: str, convert: bool, soundfont: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(prefix + '_' + midi_path.stem)
    midi_out = output_dir / f"{safe_name}.mid"
    wav_out = output_dir / f"{safe_name}.wav"

    if not midi_out.exists():
        shutil.copy(str(midi_path), str(midi_out))

    if convert and soundfont and midi_out.exists() and not wav_out.exists():
        midi_to_wav(str(midi_out), str(wav_out), soundfont)

    return midi_out.name, wav_out.name if wav_out.exists() else None


def build_playlist_page(emotion: str, entries: list, output_path: Path):
    html = [
        "<!DOCTYPE html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>EEG Playlists - {emotion}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f5f7fa;color:#2c3e50;padding:20px}",
        ".card{background:#fff;border-radius:10px;padding:16px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}",
        ".title{font-weight:700;font-size:16px}",
        ".badge{display:inline-block;background:#ecf0f1;border-radius:6px;padding:2px 6px;font-size:11px;margin-left:6px}",
        ".row{margin-top:8px}",
        "audio{width:100%;height:32px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Плейлист эмоции {emotion}</h1>",
        "<a href='index.html'>← Назад к списку эмоций</a>",
    ]

    for entry in entries:
        html.append("<div class='card'>")
        html.append(f"<div class='title'>EEG мелодия: {entry['eeg_name']} <span class='badge'>{emotion}</span></div>")
        if entry.get('eeg_wav'):
            html.append(f"<div class='row'><audio controls><source src='{entry['eeg_wav']}' type='audio/wav'></audio></div>")
        html.append("<div class='row'><strong>Похожие композиции:</strong></div>")
        for match in entry['matches']:
            badge = f"{match['dataset'].upper()}"
            if match.get('emotion'):
                badge += f" · {match['emotion']}"
                if match.get('emotion_source') == 'predicted':
                    badge += " (pred)"
            html.append(f"<div class='row'>{match['title']} <span class='badge'>{badge}</span></div>")
            if match.get('wav'):
                html.append(f"<div class='row'><audio controls><source src='{match['wav']}' type='audio/wav'></audio></div>")
        html.append("</div>")

    html.append("</body></html>")
    output_path.write_text("\n".join(html), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Export emotion playlists")
    default_report_dir = RUNS_DIR / DEFAULT_RUN_ID / 'report'
    parser.add_argument('--reports-dir', type=str, default=str(default_report_dir))
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--n-eeg', type=int, default=4)
    parser.add_argument('--k-matches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--convert-to-wav', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (reports_dir.parent / 'playlists')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = reports_dir / 'comparison_results.csv'
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    if 'eeg_emotion' not in df.columns:
        raise ValueError("comparison_results.csv missing 'eeg_emotion'")

    soundfont = find_soundfont()
    can_convert = args.convert_to_wav and soundfont is not None

    emotions = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
    index_links = []

    for emotion in emotions:
        df_emotion = df[df['eeg_emotion'] == emotion].copy()
        if df_emotion.empty:
            continue

        # выбираем N EEG мелодий по лучшему совпадению
        best_per_eeg = df_emotion.sort_values('combined_similarity', ascending=False)
        best_per_eeg = best_per_eeg.groupby('eeg_midi', as_index=False).first()
        best_per_eeg = best_per_eeg.head(args.n_eeg)

        entries = []
        for _, eeg_row in best_per_eeg.iterrows():
            eeg_path = Path(eeg_row['eeg_midi'])
            eeg_mid, eeg_wav = prepare_audio_asset(eeg_path, output_dir, f"{emotion}_EEG", can_convert, soundfont)

            # топ-K матчей для этой EEG
            matches_df = df_emotion[df_emotion['eeg_midi'] == eeg_row['eeg_midi']]
            matches_df = matches_df.sort_values('combined_similarity', ascending=False).head(args.k_matches)

            matches = []
            for _, m in matches_df.iterrows():
                classical_path = Path(m.get('classical_midi_path', ''))
                if classical_path.exists():
                    mid_name, wav_name = prepare_audio_asset(
                        classical_path, output_dir, f"{emotion}_MATCH", can_convert, soundfont
                    )
                else:
                    mid_name, wav_name = None, None

                composer = m.get('classical_composer')
                title_val = m.get('classical_title')
                if composer and title_val:
                    title = f"{composer} - {title_val}"
                else:
                    title = m.get('classical_piece', 'Unknown')
                dataset = m.get('classical_dataset', 'unknown')
                matches.append({
                    'title': title,
                    'dataset': dataset,
                    'emotion': m.get('classical_emotion', None),
                    'emotion_source': m.get('classical_emotion_source', None),
                    'midi': mid_name,
                    'wav': wav_name
                })

            entries.append({
                'eeg_name': eeg_path.name,
                'eeg_midi': eeg_mid,
                'eeg_wav': eeg_wav,
                'matches': matches
            })

        emotion_page = output_dir / f"{emotion}.html"
        build_playlist_page(emotion, entries, emotion_page)
        index_links.append(f"<li><a href='{emotion}.html'>{emotion}</a></li>")

    index_html = "<html><head><meta charset='UTF-8'><title>Playlists</title></head><body>"
    index_html += "<h1>Плейлисты по эмоциям</h1><ul>" + "".join(index_links) + "</ul></body></html>"
    (output_dir / 'index.html').write_text(index_html, encoding='utf-8')


if __name__ == '__main__':
    main()
