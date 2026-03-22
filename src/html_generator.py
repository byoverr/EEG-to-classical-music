"""
Модуль для генерации HTML файлов с результатами сравнения
"""
import os
import json
from pathlib import Path
import mido
import numpy as np
from .audio_converter import midi_to_wav, find_soundfont


def analyze_midi_frequency_profile(midi_path, max_notes=100):
    """
    Анализирует частотный профиль MIDI файла.
    Возвращает распределение по октавам и временны́е характеристики.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
        pitches = []
        durations = []
        current_time = 0
        note_on_times = {}
        
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    pitches.append(msg.note)
                    note_on_times[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_on_times:
                        dur = current_time - note_on_times[msg.note]
                        durations.append(dur)
                        del note_on_times[msg.note]
                if len(pitches) >= max_notes:
                    break
            if len(pitches) >= max_notes:
                break
        
        if len(pitches) == 0:
            return None
        
        pitches = np.array(pitches[:max_notes])
        durations = np.array(durations[:len(pitches)]) if durations else np.ones(len(pitches))
        
        # Распределение по октавам (0-10)
        octaves = pitches // 12
        octave_hist = np.zeros(11)
        for o in octaves:
            if 0 <= o <= 10:
                octave_hist[o] += 1
        octave_hist = octave_hist / (octave_hist.sum() + 1e-10)
        
        # Статистика
        return {
            'mean_pitch': float(np.mean(pitches)),
            'std_pitch': float(np.std(pitches)),
            'min_pitch': int(np.min(pitches)),
            'max_pitch': int(np.max(pitches)),
            'pitch_range': int(np.max(pitches) - np.min(pitches)),
            'mean_duration': float(np.mean(durations)) if len(durations) > 0 else 0.5,
            'note_count': len(pitches),
            'octave_distribution': octave_hist.tolist(),
            'pitches': pitches.tolist()[:50]
        }
    except Exception:
        return None


def extract_pitches_from_midi(midi_path, max_notes=50):
    """Извлекает последовательность нот (питчей) из MIDI файла"""
    try:
        mid = mido.MidiFile(str(midi_path))
        pitches = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    pitches.append(msg.note)
                    if len(pitches) >= max_notes:
                        break
            if len(pitches) >= max_notes:
                break
        return pitches[:max_notes]
    except Exception:
        return []


def valence_arousal_to_color(value, min_val=1.0, max_val=9.0):
    """
    Конвертирует значение Valence/Arousal (1-9) в CSS цвет.
    Низкие значения → синий/холодный, высокие → красный/теплый
    """
    if value is None:
        return "#999", "#f5f5f5"  # серый для отсутствующих данных
    
    # Нормализуем к 0-1
    norm = (float(value) - min_val) / (max_val - min_val)
    norm = max(0, min(1, norm))
    
    # Градиент: синий (0) → зеленый (0.5) → красный (1)
    if norm < 0.5:
        # синий → зеленый
        r = int(52 + (46 - 52) * (norm * 2))
        g = int(152 + (204 - 152) * (norm * 2))
        b = int(219 + (113 - 219) * (norm * 2))
    else:
        # зеленый → красный
        r = int(46 + (231 - 46) * ((norm - 0.5) * 2))
        g = int(204 + (76 - 204) * ((norm - 0.5) * 2))
        b = int(113 + (60 - 113) * ((norm - 0.5) * 2))
    
    fg_color = f"rgb({r},{g},{b})"
    bg_color = f"rgba({r},{g},{b},0.12)"
    
    return fg_color, bg_color


def create_simple_comparison_html(results_df, output_path, media_dir=None, convert_to_wav=True):
    """
    Создает упрощённый HTML отчёт с результатами сравнения EEG-MIDI и классики.
    
    Параметры:
    - results_df: DataFrame с колонками:
        - participant_id, trial_idx, variant
        - valence, arousal
        - eeg_midi, classical_piece
        - euclidean_distance, cosine_similarity
    - output_path: путь для сохранения HTML файла
    - media_dir: директория с MIDI файлами
    - convert_to_wav: конвертировать ли MIDI в WAV
    """
    html_path = Path(output_path)
    reports_dir = html_path.parent
    matches_dir = Path(media_dir) if media_dir is not None else reports_dir
    
    try:
        media_rel_path = os.path.relpath(matches_dir, reports_dir)
    except ValueError:
        media_rel_path = str(matches_dir)
    
    soundfont_path = find_soundfont()
    can_convert = convert_to_wav and soundfont_path is not None
    
    total_matches = len(results_df)
    dataset_counts = {}
    eeg_emotion_counts = {}
    classical_emotion_counts = {}

    if 'classical_dataset' in results_df.columns:
        dataset_counts = results_df['classical_dataset'].fillna('unknown').value_counts().to_dict()
    if 'eeg_emotion' in results_df.columns:
        eeg_emotion_counts = results_df['eeg_emotion'].fillna('unknown').value_counts().to_dict()
    if 'classical_emotion' in results_df.columns:
        classical_emotion_counts = results_df['classical_emotion'].fillna('unknown').value_counts().to_dict()

    default_emotions = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
    emotions_from_data = []
    if 'eeg_emotion' in results_df.columns:
        emotions_from_data = [e for e in results_df['eeg_emotion'].dropna().unique().tolist() if e]
    all_emotions = default_emotions + [e for e in emotions_from_data if e not in default_emotions]

    def _compute_top_composers(df, emotion_col, composer_col, score_col, emotions):
        top_map = {}
        if emotion_col not in df.columns or composer_col not in df.columns:
            return {emotion: 'Unknown' for emotion in emotions}

        for emotion in emotions:
            emo_df = df[df[emotion_col] == emotion]
            if emo_df.empty:
                top_map[emotion] = 'Unknown'
                continue
            comp_series = emo_df[composer_col].fillna('Unknown')
            grouped = emo_df.assign(_composer=comp_series)
            counts = grouped['_composer'].value_counts()
            avg_scores = None
            max_scores = None
            if score_col in grouped.columns:
                avg_scores = grouped.groupby('_composer')[score_col].mean()
                max_scores = grouped.groupby('_composer')[score_col].max()

            candidates = []
            for composer, count in counts.items():
                avg_score = float(avg_scores.get(composer, 0.0)) if avg_scores is not None else 0.0
                max_score = float(max_scores.get(composer, 0.0)) if max_scores is not None else 0.0
                candidates.append((composer, int(count), avg_score, max_score))

            candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
            top_map[emotion] = candidates[0][0] if candidates else 'Unknown'

        return top_map

    composer_column = 'classical_composer' if 'classical_composer' in results_df.columns else 'composer'
    top_composers_full = _compute_top_composers(
        results_df,
        emotion_col='eeg_emotion',
        composer_col=composer_column,
        score_col='combined_similarity',
        emotions=all_emotions
    )

    top_composer_lines_html = ''.join([
        f"<div class=\"top-composer-line\" data-emotion=\"{emotion}\">{emotion} — {top_composers_full.get(emotion, 'Unknown')}</div>"
        for emotion in all_emotions
    ])

    top_composers_full_json = json.dumps(top_composers_full, ensure_ascii=False)
    all_emotions_json = json.dumps(all_emotions, ensure_ascii=False)

    default_emotions = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
    emotions_from_data = []
    if 'eeg_emotion' in results_df.columns:
        emotions_from_data = [e for e in results_df['eeg_emotion'].dropna().unique().tolist() if e]
    all_emotions = default_emotions + [e for e in emotions_from_data if e not in default_emotions]

    def _compute_top_composers(df, emotion_col, composer_col, score_col, emotions):
        top_map = {}
        if emotion_col not in df.columns or composer_col not in df.columns:
            return {emotion: 'Unknown' for emotion in emotions}

        for emotion in emotions:
            emo_df = df[df[emotion_col] == emotion]
            if emo_df.empty:
                top_map[emotion] = 'Unknown'
                continue
            comp_series = emo_df[composer_col].fillna('Unknown')
            grouped = emo_df.assign(_composer=comp_series)
            counts = grouped['_composer'].value_counts()
            avg_scores = None
            max_scores = None
            if score_col in grouped.columns:
                avg_scores = grouped.groupby('_composer')[score_col].mean()
                max_scores = grouped.groupby('_composer')[score_col].max()

            candidates = []
            for composer, count in counts.items():
                avg_score = float(avg_scores.get(composer, 0.0)) if avg_scores is not None else 0.0
                max_score = float(max_scores.get(composer, 0.0)) if max_scores is not None else 0.0
                candidates.append((composer, int(count), avg_score, max_score))

            candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
            top_map[emotion] = candidates[0][0] if candidates else 'Unknown'

        return top_map

    composer_column = 'classical_composer' if 'classical_composer' in results_df.columns else 'composer'
    top_composers_full = _compute_top_composers(
        results_df,
        emotion_col='eeg_emotion',
        composer_col=composer_column,
        score_col='combined_similarity',
        emotions=all_emotions
    )

    top_composer_lines_html = ''.join([
        f"<div class=\"top-composer-line\" data-emotion=\"{emotion}\">{emotion} — {top_composers_full.get(emotion, 'Unknown')}</div>"
        for emotion in all_emotions
    ])

    top_composers_full_json = json.dumps(top_composers_full, ensure_ascii=False)
    all_emotions_json = json.dumps(all_emotions, ensure_ascii=False)

    default_emotions = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
    emotions_from_data = []
    if 'eeg_emotion' in results_df.columns:
        emotions_from_data = [e for e in results_df['eeg_emotion'].dropna().unique().tolist() if e]
    all_emotions = default_emotions + [e for e in emotions_from_data if e not in default_emotions]

    def _compute_top_composers(df, emotion_col, composer_col, score_col, emotions):
        top_map = {}
        if emotion_col not in df.columns or composer_col not in df.columns:
            return {emotion: 'Unknown' for emotion in emotions}

        for emotion in emotions:
            emo_df = df[df[emotion_col] == emotion]
            if emo_df.empty:
                top_map[emotion] = 'Unknown'
                continue
            comp_series = emo_df[composer_col].fillna('Unknown')
            grouped = emo_df.assign(_composer=comp_series)
            counts = grouped['_composer'].value_counts()
            avg_scores = None
            max_scores = None
            if score_col in grouped.columns:
                avg_scores = grouped.groupby('_composer')[score_col].mean()
                max_scores = grouped.groupby('_composer')[score_col].max()

            candidates = []
            for composer, count in counts.items():
                avg_score = float(avg_scores.get(composer, 0.0)) if avg_scores is not None else 0.0
                max_score = float(max_scores.get(composer, 0.0)) if max_scores is not None else 0.0
                candidates.append((composer, int(count), avg_score, max_score))

            candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
            top_map[emotion] = candidates[0][0] if candidates else 'Unknown'

        return top_map

    composer_column = 'classical_composer' if 'classical_composer' in results_df.columns else 'composer'
    top_composers_full = _compute_top_composers(
        results_df,
        emotion_col='eeg_emotion',
        composer_col=composer_column,
        score_col='combined_similarity',
        emotions=all_emotions
    )

    top_composer_lines_html = ''.join([
        f"<div class=\"top-composer-line\" data-emotion=\"{emotion}\">{emotion} — {top_composers_full.get(emotion, 'Unknown')}</div>"
        for emotion in all_emotions
    ])

    top_composers_full_json = json.dumps(top_composers_full, ensure_ascii=False)
    all_emotions_json = json.dumps(all_emotions, ensure_ascii=False)

    html_content = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG to Classical Music - Comparison Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .subtitle { opacity: 0.9; font-size: 1.1rem; }
        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        .stat {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            color: white;
        }
        .stat-value { font-size: 1.5rem; font-weight: bold; }
        .stat-label { font-size: 0.8rem; opacity: 0.8; }
        .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .card {
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-5px); }
        .card-header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 15px 20px;
        }
        .rank { 
            display: inline-block;
            background: #e74c3c;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-weight: bold;
            margin-right: 10px;
        }
        .classical-piece { font-size: 0.9rem; opacity: 0.8; margin-top: 5px; }
        .card-body { padding: 20px; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }
        .metric {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value { font-size: 1.2rem; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 0.75rem; color: #7f8c8d; text-transform: uppercase; }
        .metric.good .metric-value { color: #27ae60; }
        .metric.emotion .metric-value { color: #9b59b6; }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            font-size: 0.9rem;
        }
        .info-label { color: #7f8c8d; }
        .info-value { font-weight: 500; color: #2c3e50; }
        .va-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .audio-section { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
        .audio-label { font-size: 0.85rem; color: #555; margin-bottom: 5px; }
        audio { width: 100%; height: 35px; }
        .midi-link {
            display: inline-block;
            margin-top: 8px;
            padding: 5px 12px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.8rem;
        }
        .midi-link:hover { background: #2980b9; }
        footer { text-align: center; color: white; opacity: 0.7; margin-top: 40px; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎵 EEG → Classical Music</h1>
            <p class="subtitle">Similarity Analysis Results</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">""" + str(len(results_df)) + """</div>
                    <div class="stat-label">Matches Found</div>
                </div>
                <div class="stat">
                    <div class="stat-value">""" + f"{results_df['cosine_similarity'].mean():.2f}" + """</div>
                    <div class="stat-label">Avg Similarity</div>
                </div>
            </div>
        </header>
        <div class="cards">
"""
    
    for idx, (_, row) in enumerate(results_df.iterrows(), 1):
        participant = row.get('participant_id', 'Unknown')
        trial = row.get('trial_idx', 0)
        variant = row.get('variant', '')
        valence = row.get('valence', 5.0)
        arousal = row.get('arousal', 5.0)
        eeg_midi = Path(row.get('eeg_midi', ''))
        classical = row.get('classical_piece', 'Unknown')
        distance = row.get('euclidean_distance', 0)
        cosine_sim = row.get('cosine_similarity', 0)
        
        # Цвета для V/A
        v_fg, v_bg = valence_arousal_to_color(valence)
        a_fg, a_bg = valence_arousal_to_color(arousal)
        
        # Относительный путь к MIDI
        if eeg_midi.exists():
            midi_rel = f"{media_rel_path}/{eeg_midi.name}"
            wav_path = eeg_midi.with_suffix('.wav')
            wav_rel = f"{media_rel_path}/{wav_path.name}"
            
            # Конвертируем в WAV если нужно
            if can_convert and not wav_path.exists():
                try:
                    from .audio_converter import midi_to_wav
                    midi_to_wav(str(eeg_midi), str(wav_path), soundfont_path)
                except:
                    pass
            
            audio_html = ""
            if wav_path.exists():
                audio_html = f'''
                <div class="audio-section">
                    <div class="audio-label">EEG-generated melody:</div>
                    <audio controls preload="metadata">
                        <source src="{wav_rel}" type="audio/wav">
                    </audio>
                    <a href="{midi_rel}" download class="midi-link">Download MIDI</a>
                </div>'''
            else:
                audio_html = f'''
                <div class="audio-section">
                    <a href="{midi_rel}" download class="midi-link">Download MIDI</a>
                </div>'''
        else:
            audio_html = ""
        
        card_html = f'''
        <div class="card">
            <div class="card-header">
                <span class="rank">{idx}</span>
                <strong>{participant.upper()} / Trial {trial}</strong>
                <span style="float:right;opacity:0.7">{variant}</span>
                <div class="classical-piece">→ {classical[:50]}{'...' if len(classical) > 50 else ''}</div>
            </div>
            <div class="card-body">
                <div class="metrics">
                    <div class="metric good">
                        <div class="metric-value">{cosine_sim:.3f}</div>
                        <div class="metric-label">Cosine Similarity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{distance:.2f}</div>
                        <div class="metric-label">Distance</div>
                    </div>
                    <div class="metric emotion">
                        <div class="metric-value">
                            <span class="va-badge" style="color:{v_fg};background:{v_bg}">V {valence:.1f}</span>
                        </div>
                        <div class="metric-label">Valence</div>
                    </div>
                    <div class="metric emotion">
                        <div class="metric-value">
                            <span class="va-badge" style="color:{a_fg};background:{a_bg}">A {arousal:.1f}</span>
                        </div>
                        <div class="metric-label">Arousal</div>
                    </div>
                </div>
                <div class="info-row">
                    <span class="info-label">EEG Window</span>
                    <span class="info-value">{row.get('eeg_window_id', '-')} @ {row.get('eeg_start_time', 0):.1f}s</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Classical Window</span>
                    <span class="info-value">{row.get('classical_window_id', '-')} @ {row.get('classical_start_time', 0):.1f}s</span>
                </div>
                {audio_html}
            </div>
        </div>
'''
        html_content += card_html
    
    html_content += """
        </div>
        <footer>
            Generated by EEG-to-Classical-Music Pipeline
        </footer>
    </div>
</body>
</html>"""
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report created: {html_path}")


def create_comparison_html(results_df, saved_matches, output_path, convert_to_wav=True, media_dir=None):
    """
    Создает HTML файл со встроенными WAV-плеерами и графиками сходства
    
    Параметры:
    - results_df: DataFrame с результатами сравнения
    - saved_matches: список сохраненных совпадений
    - output_path: путь для сохранения HTML файла
    - convert_to_wav: конвертировать ли MIDI в WAV автоматически
    """
    html_path = Path(output_path)
    reports_dir = html_path.parent
    matches_dir = Path(media_dir) if media_dir is not None else reports_dir
    
    # Вычисляем относительный путь от HTML к папке с медиа
    try:
        media_rel_path = os.path.relpath(matches_dir, reports_dir)
    except ValueError:
        media_rel_path = str(matches_dir)
    
    # Проверяем возможность конвертации
    soundfont_path = find_soundfont()
    CAN_CONVERT_TO_WAV = convert_to_wav and soundfont_path is not None
    
    if not CAN_CONVERT_TO_WAV and convert_to_wav:
        print("Внимание: SoundFont не найден — будут только ссылки на MIDI")

    total_matches = len(results_df)
    dataset_counts = {}
    eeg_emotion_counts = {}
    classical_emotion_counts = {}

    if 'classical_dataset' in results_df.columns:
        dataset_counts = results_df['classical_dataset'].fillna('unknown').value_counts().to_dict()
    if 'eeg_emotion' in results_df.columns:
        eeg_emotion_counts = results_df['eeg_emotion'].fillna('unknown').value_counts().to_dict()
    if 'classical_emotion' in results_df.columns:
        classical_emotion_counts = results_df['classical_emotion'].fillna('unknown').value_counts().to_dict()
    
    default_emotions = ['HVHA', 'HVLA', 'LVHA', 'LVLA']
    emotions_from_data = []
    if 'eeg_emotion' in results_df.columns:
        emotions_from_data = [e for e in results_df['eeg_emotion'].dropna().unique().tolist() if e]
    all_emotions = default_emotions + [e for e in emotions_from_data if e not in default_emotions]

    def _compute_top_composers(df, emotion_col, composer_col, score_col, emotions):
        top_map = {}
        if emotion_col not in df.columns or composer_col not in df.columns:
            return {emotion: 'Unknown' for emotion in emotions}

        for emotion in emotions:
            emo_df = df[df[emotion_col] == emotion]
            if emo_df.empty:
                top_map[emotion] = 'Unknown'
                continue
            comp_series = emo_df[composer_col].fillna('Unknown')
            grouped = emo_df.assign(_composer=comp_series)
            counts = grouped['_composer'].value_counts()
            avg_scores = None
            max_scores = None
            if score_col in grouped.columns:
                avg_scores = grouped.groupby('_composer')[score_col].mean()
                max_scores = grouped.groupby('_composer')[score_col].max()

            candidates = []
            for composer, count in counts.items():
                avg_score = float(avg_scores.get(composer, 0.0)) if avg_scores is not None else 0.0
                max_score = float(max_scores.get(composer, 0.0)) if max_scores is not None else 0.0
                candidates.append((composer, int(count), avg_score, max_score))

            candidates.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
            top_map[emotion] = candidates[0][0] if candidates else 'Unknown'

        return top_map

    composer_column = 'classical_composer' if 'classical_composer' in results_df.columns else 'composer'
    top_composers_full = _compute_top_composers(
        results_df,
        emotion_col='eeg_emotion',
        composer_col=composer_column,
        score_col='combined_similarity',
        emotions=all_emotions
    )

    top_composer_lines_html = ''.join([
        f"<div class=\"top-composer-line\" data-emotion=\"{emotion}\">{emotion} — {top_composers_full.get(emotion, 'Unknown')}</div>"
        for emotion in all_emotions
    ])

    top_composers_full_json = json.dumps(top_composers_full, ensure_ascii=False)
    all_emotions_json = json.dumps(all_emotions, ensure_ascii=False)

    html_content = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG to Classical Music - Similarity Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
            color: #2c3e50;
            min-height: 100vh;
            padding: 16px;
        }
        
        .container { max-width: 1600px; margin: 0 auto; padding: 0 16px; }
        
        header {
            text-align: center;
            margin-bottom: 24px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }
        
        h1 { font-size: 24px; font-weight: 700; color: #1a237e; margin-bottom: 6px; }
        .subtitle { font-size: 13px; color: #7f8c8d; }
        
        .controls-section {
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
            background: white;
            padding: 14px 18px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }

        .summary-section {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }

        .summary-card {
            background: white;
            padding: 12px 16px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            min-width: 220px;
        }

        .summary-title { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
        .summary-value { font-size: 20px; font-weight: 700; color: #1a237e; margin-top: 6px; }
        .summary-list { margin-top: 8px; font-size: 11px; color: #555; }
        .summary-list div { margin-top: 2px; }
        
        .tabs { display: flex; gap: 6px; flex-wrap: wrap; flex: 1; }
        
        .tab-btn {
            padding: 8px 14px;
            background: #ecf0f1;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            color: #2c3e50;
            transition: all 0.2s;
        }
        
        .tab-btn:hover { background: #d5dbdb; }
        .tab-btn.active { background: #3498db; color: white; }
        
        #searchInput {
            padding: 8px 14px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            min-width: 220px;
        }
        
        #searchInput:focus { outline: none; border-color: #3498db; }

        .filter-select {
            padding: 8px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 12px;
            color: #2c3e50;
            background: #fff;
        }
        
        .section { display: none; }
        .section.active { display: block; }
        
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            padding: 0 4px 4px 0;
        }
        
        .match-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            transition: all 0.2s;
            min-width: 0;
            width: 100%;
        }
        
        .match-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        
        .card-header {
            padding: 12px 14px;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
            min-width: 0;
        }
        
        .composer-name { font-size: 14px; font-weight: 700; color: #1a237e; }
        .track-name { font-size: 11px; color: #999; margin-top: 2px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        
        .card-meta { text-align: right; flex-shrink: 0; }
        .trial-info { font-size: 10px; color: #666; font-weight: 600; }
        .badge-row { display: flex; gap: 6px; justify-content: flex-end; margin-top: 4px; flex-wrap: wrap; }
        .badge {
            font-size: 9px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 6px;
            background: #ecf0f1;
            color: #2c3e50;
        }
        .badge.dataset { background: #dfe6e9; }
        .badge.emotion { background: #f5e6ff; color: #6c3483; }
        .badge.match-good { background: #e8f8f0; color: #1e8449; }
        .badge.match-bad { background: #fdecea; color: #b03a2e; }
        .badge.match-unknown { background: #f0f0f0; color: #666; }
        
        /* Valence/Arousal colored badges */
        .va-badges { display: flex; gap: 4px; margin-top: 3px; justify-content: flex-end; }
        .va-badge { 
            font-size: 9px; 
            font-weight: 700; 
            padding: 2px 5px; 
            border-radius: 3px; 
            font-family: monospace;
        }
        .va-badge .label { opacity: 0.7; font-weight: 400; }
        
        /* Metadata toggle section */
        .metadata-section {
            padding: 8px 14px;
            border-top: 1px solid #f0f0f0;
            background: #fafbfc;
        }
        
        .metadata-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 600;
            color: #666;
        }
        
        .metadata-toggle:hover { color: #3498db; }
        .metadata-toggle input { cursor: pointer; }
        
        .metadata-content { display: none; margin-top: 8px; }
        .metadata-content.open { display: block; }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
        }
        
        .meta-item {
            background: white;
            padding: 6px 8px;
            border-radius: 4px;
            border-left: 2px solid #9b59b6;
        }
        
        .meta-label { font-size: 9px; color: #999; text-transform: uppercase; }
        .meta-value { font-size: 11px; color: #333; font-weight: 500; margin-top: 1px; }
        
        .metrics-row {
            padding: 10px 14px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        
        .metric-item {
            background: #f8f9fa;
            padding: 6px 8px;
            border-radius: 6px;
            border-left: 2px solid #3498db;
            text-align: center;
        }
        
        .metric-label { font-size: 9px; color: #999; text-transform: uppercase; letter-spacing: 0.3px; }
        .metric-value { font-size: 13px; font-weight: 700; color: #3498db; margin-top: 2px; }
        
        .metric-item.emotion { border-left-color: #e74c3c; }
        .metric-item.emotion .metric-value { color: #e74c3c; }
        .metric-item.total { border-left-color: #27ae60; }
        .metric-item.total .metric-value { color: #27ae60; }
        
        .audio-section { padding: 12px 14px; border-top: 1px solid #f0f0f0; }
        
        .audio-row {
            margin-bottom: 10px;
        }
        
        .audio-row:last-child { margin-bottom: 0; }
        
        .audio-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        
        .audio-label { font-size: 11px; font-weight: 600; color: #555; }
        
        .audio-buttons { display: flex; gap: 6px; }
        
        audio { width: 100%; height: 32px; }
        
        .btn-sm {
            padding: 4px 8px;
            border: none;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            color: white;
            background: #3498db;
            flex-shrink: 0;
        }
        
        .btn-sm:hover { background: #2980b9; }
        .btn-dark { background: #555; }
        .btn-dark:hover { background: #333; }
        
        .graph-section {
            padding: 10px 14px;
            border-top: 1px solid #f0f0f0;
            background: #fafafa;
        }
        
        .graph-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 600;
            color: #666;
        }
        
        .graph-toggle:hover { color: #3498db; }
        .graph-toggle input { cursor: pointer; }
        
        .graph-container { display: none; margin-top: 10px; }
        .graph-container.open { display: block; }
        
        .chart-wrapper { background: white; border-radius: 6px; padding: 12px; height: 180px; }
        
        footer { text-align: center; padding: 20px; color: #999; font-size: 12px; margin-top: 30px; }
        
        .no-results { text-align: center; padding: 40px; color: #999; }
        
        @media (max-width: 720px) {
            .cards-grid { grid-template-columns: 1fr; }
            .match-card { max-width: 100%; }
            .metrics-row { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>EEG to Classical Music</h1>
            <p class="subtitle">Similarity Analysis Report</p>
        </header>

        <div class="summary-section">
            <div class="summary-card">
                <div class="summary-title">Total Matches</div>
                <div class="summary-value">""" + str(total_matches) + """</div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Dataset Sources</div>
                <div class="summary-list">
                    """ + ''.join([f"<div>{k.upper()}: {v}</div>" for k, v in dataset_counts.items()]) + """
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-title">EEG Emotions</div>
                <div class="summary-list">
                    """ + ''.join([f"<div>{k}: {v}</div>" for k, v in eeg_emotion_counts.items()]) + """
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-title">Reference Emotions</div>
                <div class="summary-list">
                    """ + ''.join([f"<div>{k}: {v}</div>" for k, v in classical_emotion_counts.items()]) + """
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-title">TOP COMPOSERS</div>
                <div class="summary-list" id="topComposersList">
                    """ + top_composer_lines_html + """
                </div>
            </div>
        </div>
        
        <div class="controls-section">
            <div class="tabs" id="tabContainer"></div>
            <input type="text" id="searchInput" placeholder="Search composer or title...">
            <select id="eegEmotionFilter" class="filter-select">
                <option value="all">EEG Emotion: All</option>
                <option value="HVHA">EEG Emotion: HVHA</option>
                <option value="HVLA">EEG Emotion: HVLA</option>
                <option value="LVHA">EEG Emotion: LVHA</option>
                <option value="LVLA">EEG Emotion: LVLA</option>
            </select>
            <select id="datasetFilter" class="filter-select">
                <option value="all">Dataset: All</option>
                <option value="maestro">Dataset: MAESTRO</option>
                <option value="emopia">Dataset: EMOPIA</option>
            </select>
        </div>
        
        <div id="sectionsContainer"></div>
        
        <footer>WAV files optimized for browser playback</footer>
    </div>"""
    
    def add_match(rank, row, prefix=""):
        """Генерирует карточку с информацией об одном совпадении"""
        safe_filename = row['file'].replace('.mid', '').replace('.midi', '')
        composer = row.get('composer', '')
        title = row.get('title', safe_filename)
        
        variant = row.get('variant', '')
        
        # Используем прямые пути к файлам если они переданы
        if 'eeg_midi_path' in row and row['eeg_midi_path']:
            eeg_mid = Path(row['eeg_midi_path'])
            cla_mid = Path(row['classical_midi_path'])
            cmp_mid = Path(row['comparison_midi_path'])
        else:
            # Fallback: формируем пути к файлам как раньше
            variant_prefix = f"{variant}_" if variant else ""
            base_name = f"{rank:02d}_{variant_prefix}"
            eeg_mid = matches_dir / f"{base_name}EEG_{composer}_{safe_filename}.mid"
            cla_mid = matches_dir / f"{base_name}Classical_{composer}_{safe_filename}.mid"
            cmp_mid = matches_dir / f"{base_name}Comparison_{composer}_{safe_filename}.mid"
        
        eeg_wav = eeg_mid.with_suffix('.wav')
        cla_wav = cla_mid.with_suffix('.wav')
        cmp_wav = cmp_mid.with_suffix('.wav')
        
        # Относительные пути для HTML
        eeg_mid_rel = f"{media_rel_path}/{eeg_mid.name}" if media_rel_path != "." else eeg_mid.name
        cla_mid_rel = f"{media_rel_path}/{cla_mid.name}" if media_rel_path != "." else cla_mid.name
        cmp_mid_rel = f"{media_rel_path}/{cmp_mid.name}" if media_rel_path != "." else cmp_mid.name
        eeg_wav_rel = f"{media_rel_path}/{eeg_wav.name}" if media_rel_path != "." else eeg_wav.name
        cla_wav_rel = f"{media_rel_path}/{cla_wav.name}" if media_rel_path != "." else cla_wav.name
        cmp_wav_rel = f"{media_rel_path}/{cmp_wav.name}" if media_rel_path != "." else cmp_wav.name
        
        if CAN_CONVERT_TO_WAV:
            for mid_file, wav_file in [(eeg_mid, eeg_wav), (cla_mid, cla_wav), (cmp_mid, cmp_wav)]:
                if mid_file.exists() and not wav_file.exists():
                    try:
                        midi_to_wav(str(mid_file), str(wav_file), soundfont_path)
                        print(f"Конвертировано → {wav_file.name}")
                    except Exception as e:
                        print(f"Не удалось конвертировать {mid_file.name}: {e}")
        
        def make_audio_row(label, wav_path, wav_rel, mid_rel):
            if wav_path.exists():
                return f'''<div class="audio-row">
                    <div class="audio-header">
                        <span class="audio-label">{label}</span>
                        <div class="audio-buttons">
                            <a href="{mid_rel}" download class="btn-sm">MIDI</a>
                            <a href="{wav_rel}" download class="btn-sm btn-dark">WAV</a>
                        </div>
                    </div>
                    <audio controls preload="metadata"><source src="{wav_rel}" type="audio/wav"></audio>
                </div>'''
            else:
                return f'''<div class="audio-row">
                    <div class="audio-header">
                        <span class="audio-label">{label}</span>
                        <div class="audio-buttons">
                            <a href="{mid_rel}" download class="btn-sm">MIDI</a>
                        </div>
                    </div>
                    <div style="color:#999;font-size:11px;padding:8px 0;">WAV not available</div>
                </div>'''
        
        # Извлекаем ноты для графика
        eeg_pitches = extract_pitches_from_midi(eeg_mid) if eeg_mid.exists() else []
        cla_pitches = extract_pitches_from_midi(cla_mid) if cla_mid.exists() else []
        
        edit_dist = row.get('edit_distance', 0)
        melodic_sim = row.get('melodic_similarity', 0)
        combined_sim = row.get('combined_similarity', 0)
        emotion_sim = row.get('emotion_similarity', 0)
        total_sim = row.get('total_similarity', 0)
        # New melodic metrics
        contour_sim = row.get('contour_similarity', 0)
        correlation_sim = row.get('correlation_similarity', 0)
        harmony_sim = row.get('harmony_similarity', 0)
        statistical_sim = row.get('statistical_similarity', 0)
        sfi_sim = row.get('sfi_similarity', 0)
        
        trial = row.get('trial', '')
        processing = row.get('processing', '')
        eeg_valence = row.get('eeg_valence', None)
        eeg_arousal = row.get('eeg_arousal', None)
        eeg_emotion = row.get('eeg_emotion', 'unknown')
        classical_dataset = row.get('classical_dataset', 'unknown')
        classical_emotion = row.get('classical_emotion', None)
        classical_emotion_source = row.get('classical_emotion_source', None)
        emotion_match = row.get('emotion_match', None)
        
        # Метаданные DEAP
        participant_id = row.get('participant_id', '')
        participant_age = row.get('participant_age', None)
        participant_gender = row.get('participant_gender', '')
        stimulus_artist = row.get('stimulus_artist', '')
        stimulus_title = row.get('stimulus_title', '')
        stimulus_tag = row.get('stimulus_tag', '')
        
        # Цветовая кодировка Valence/Arousal (шкала 1-9)
        va_html = ''
        if eeg_valence is not None or eeg_arousal is not None:
            v_fg, v_bg = valence_arousal_to_color(eeg_valence)
            a_fg, a_bg = valence_arousal_to_color(eeg_arousal)
            v_str = f"{eeg_valence:.1f}" if eeg_valence is not None else "?"
            a_str = f"{eeg_arousal:.1f}" if eeg_arousal is not None else "?"
            va_html = f'''<div class="va-badges">
                <span class="va-badge" style="color:{v_fg};background:{v_bg}"><span class="label">V</span> {v_str}</span>
                <span class="va-badge" style="color:{a_fg};background:{a_bg}"><span class="label">A</span> {a_str}</span>
            </div>'''
        
        # Метаданные секция (раскрывающаяся)
        metadata_html = ''
        has_meta = any([participant_id, stimulus_artist, stimulus_title])
        if has_meta:
            meta_id = f"meta_{prefix}_{rank}"
            meta_items = ''
            if participant_id:
                age_str = f", {participant_age}y" if participant_age else ""
                gender_str = f" ({participant_gender})" if participant_gender else ""
                meta_items += f'''<div class="meta-item"><div class="meta-label">Participant</div><div class="meta-value">{participant_id.upper()}{age_str}{gender_str}</div></div>'''
            if stimulus_artist:
                meta_items += f'''<div class="meta-item"><div class="meta-label">Stimulus Artist</div><div class="meta-value">{stimulus_artist}</div></div>'''
            if stimulus_title:
                meta_items += f'''<div class="meta-item"><div class="meta-label">Stimulus Title</div><div class="meta-value">{stimulus_title}</div></div>'''
            if stimulus_tag:
                meta_items += f'''<div class="meta-item"><div class="meta-label">Last.fm Tag</div><div class="meta-value">#{stimulus_tag}</div></div>'''
            
            metadata_html = f'''<div class="metadata-section">
                <label class="metadata-toggle">
                    <input type="checkbox" data-meta="{meta_id}"> DEAP Metadata
                </label>
                <div id="{meta_id}" class="metadata-content">
                    <div class="metadata-grid">{meta_items}</div>
                    <div style="font-size:9px;color:#aaa;margin-top:6px;">Valence/Arousal scale: 1 (low) — 9 (high)</div>
                </div>
            </div>'''
        
        # Компактные метрики - новые мелодические метрики
        metrics_html = f'''<div class="metrics-row">'''
        
        # Combined similarity - главная метрика
        if combined_sim > 0:
            metrics_html += f'''<div class="metric-item total"><div class="metric-label">Combined</div><div class="metric-value">{combined_sim:.3f}</div></div>'''
        
        # Contour (DTW-based melodic similarity)
        if contour_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">Contour</div><div class="metric-value">{contour_sim:.3f}</div></div>'''
        
        # SFI (Scale-Free Index similarity)
        if sfi_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">SFI</div><div class="metric-value">{sfi_sim:.3f}</div></div>'''
        
        # Harmony (pitch class distribution)
        if harmony_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">Harmony</div><div class="metric-value">{harmony_sim:.3f}</div></div>'''
        
        # Correlation (Pearson correlation of pitch contours)
        if correlation_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">Corr</div><div class="metric-value">{correlation_sim:.3f}</div></div>'''
        
        # Statistical similarity (mean/std based)
        if statistical_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">Stats</div><div class="metric-value">{statistical_sim:.3f}</div></div>'''
        
        # Legacy metrics fallback
        if combined_sim == 0 and melodic_sim > 0:
            metrics_html += f'''<div class="metric-item"><div class="metric-label">Melodic</div><div class="metric-value">{melodic_sim:.3f}</div></div>'''
        
        if emotion_sim > 0:
            metrics_html += f'''<div class="metric-item emotion"><div class="metric-label">Emotion</div><div class="metric-value">{emotion_sim:.3f}</div></div>'''
        
        if total_sim > 0:
            metrics_html += f'''<div class="metric-item total"><div class="metric-label">Total</div><div class="metric-value">{total_sim:.3f}</div></div>'''
        
        metrics_html += '</div>'
        
        # Аудио секция
        eeg_label = f"EEG ({eeg_emotion})" if eeg_emotion else "EEG"
        ref_label = f"Reference ({classical_emotion})" if classical_emotion else "Reference"
        audio_html = f'''<div class="audio-section">
            {make_audio_row(eeg_label, eeg_wav, eeg_wav_rel, eeg_mid_rel)}
            {make_audio_row(ref_label, cla_wav, cla_wav_rel, cla_mid_rel)}
            {make_audio_row("Combined", cmp_wav, cmp_wav_rel, cmp_mid_rel)}
        </div>'''
        
        # График питчей - уникальный ID для каждой карточки с данными питчей
        chart_id = f"chart_{prefix}_{rank}"
        eeg_pitches_json = json.dumps(eeg_pitches)
        cla_pitches_json = json.dumps(cla_pitches)
        
        has_data = len(eeg_pitches) > 0 and len(cla_pitches) > 0
        
        # Анализируем частотный профиль (распределение по октавам)
        eeg_profile = analyze_midi_frequency_profile(eeg_mid) if eeg_mid.exists() else None
        cla_profile = analyze_midi_frequency_profile(cla_mid) if cla_mid.exists() else None
        
        octave_chart_id = f"octave_{prefix}_{rank}"
        has_octave_data = eeg_profile is not None and cla_profile is not None
        
        # Показываем graph-section только если есть данные
        if has_data:
            graph_html = f'''<div class="graph-section">
                <label class="graph-toggle">
                    <input type="checkbox" data-chart="{chart_id}" data-eeg='{eeg_pitches_json}' data-classical='{cla_pitches_json}'>
                    🎵 Pitch Comparison
                </label>
                <div id="{chart_id}" class="graph-container">
                    <div class="chart-wrapper"><canvas id="canvas_{chart_id}"></canvas></div>
                </div>
            </div>'''
        else:
            graph_html = ''  # Нет данных для графика
        
        # Добавляем график распределения по октавам
        if has_octave_data:
            eeg_octaves = json.dumps(eeg_profile['octave_distribution'])
            cla_octaves = json.dumps(cla_profile['octave_distribution'])
            graph_html += f'''<div class="graph-section">
                <label class="graph-toggle">
                    <input type="checkbox" data-octave-chart="{octave_chart_id}" data-eeg-octaves='{eeg_octaves}' data-cla-octaves='{cla_octaves}'>
                    Frequency Distribution (Octaves)
                </label>
                <div id="{octave_chart_id}" class="graph-container">
                    <div class="chart-wrapper"><canvas id="canvas_{octave_chart_id}"></canvas></div>
                </div>
            </div>'''
        
        badge_html = f'''<div class="badge-row">
            <span class="badge dataset">{classical_dataset.upper()}</span>
            <span class="badge emotion">EEG {eeg_emotion}</span>
        '''
        if classical_emotion:
            if classical_emotion_source == 'predicted':
                badge_html += f'''<span class="badge emotion">{classical_emotion} (pred)</span>'''
            else:
                badge_html += f'''<span class="badge emotion">{classical_emotion}</span>'''
        if emotion_match is True:
            badge_html += f'''<span class="badge match-good">Emotion match</span>'''
        elif emotion_match is False:
            badge_html += f'''<span class="badge match-bad">Emotion mismatch</span>'''
        badge_html += '</div>'

        match_html = f'''<div class="match-card" data-composer="{composer}" data-title="{title}" data-eeg-emotion="{eeg_emotion}" data-dataset="{classical_dataset}" data-combined-score="{combined_sim}">
            <div class="card-header">
                <div>
                    <div class="composer-name">{composer}</div>
                    <div class="track-name" title="{title}">{title[:40]}{'...' if len(title) > 40 else ''}</div>
                </div>
                <div class="card-meta">
                    <div class="trial-info">{trial} · {processing}</div>
                    {va_html}
                    {badge_html}
                </div>
            </div>
            {metrics_html}
            {audio_html}
            {metadata_html}
            {graph_html}
        </div>'''
        
        return match_html
    
    # Дедупликация DataFrame для построения секций
    _dedup_cols = ['composer', 'title', 'eeg_emotion', 'trial', 'variant']
    _has_dedup = all(c in results_df.columns for c in _dedup_cols)
    if _has_dedup and 'combined_similarity' in results_df.columns:
        results_df = results_df.sort_values('combined_similarity', ascending=False)
        results_df = results_df.drop_duplicates(subset=_dedup_cols, keep='first')

    n_cards = len(results_df)
    metrics_sections = []
    if 'total_similarity' in results_df.columns:
        metrics_sections.append(('Total', 'Total Similarity', results_df.nlargest(n_cards, 'total_similarity'), 'T'))
    if 'combined_similarity' in results_df.columns:
        metrics_sections.append(('Combined', 'Combined Similarity', results_df.nlargest(n_cards, 'combined_similarity'), 'C'))
    if 'contour_similarity' in results_df.columns:
        metrics_sections.append(('Contour', 'Contour Similarity', results_df.nlargest(n_cards, 'contour_similarity'), 'Ct'))
    if 'sfi_similarity' in results_df.columns:
        metrics_sections.append(('SFI', 'Scale-Free Index', results_df.nlargest(n_cards, 'sfi_similarity'), 'S'))
    if 'harmony_similarity' in results_df.columns:
        metrics_sections.append(('Harmony', 'Harmony Similarity', results_df.nlargest(n_cards, 'harmony_similarity'), 'H'))
    # Fallback для старых метрик (Melodic пропускаем если есть Combined — они идентичны)
    if 'edit_distance' in results_df.columns:
        metrics_sections.append(('Edit', 'Edit Distance', results_df.nsmallest(n_cards, 'edit_distance'), ''))
    if 'melodic_similarity' in results_df.columns and 'combined_similarity' not in results_df.columns:
        metrics_sections.append(('Melodic', 'Melodic Similarity', results_df.nlargest(n_cards, 'melodic_similarity'), 'M'))

    # Дополнительные секции по датасетам
    if 'classical_dataset' in results_df.columns and 'combined_similarity' in results_df.columns:
        emopia_df = results_df[results_df['classical_dataset'] == 'emopia']
        maestro_df = results_df[results_df['classical_dataset'] == 'maestro']
        if len(emopia_df) > 0:
            metrics_sections.append(('EMOPIA', 'EMOPIA Top', emopia_df.nlargest(len(emopia_df), 'combined_similarity'), 'E'))
        if len(maestro_df) > 0:
            metrics_sections.append(('MAESTRO', 'MAESTRO Top', maestro_df.nlargest(len(maestro_df), 'combined_similarity'), 'Ma'))
    
    # Build tabs HTML
    tabs_html = ""
    for key, title, _, _ in metrics_sections:
        tabs_html += f'<button class="tab-btn" data-tab="{key}">{title}</button>'
    
    sections_html = ""
    for key, title, df_section, prefix in metrics_sections:
        cards_html = ""
        if len(df_section) == 0:
            cards_html = '<div class="no-results">No results found</div>'
        else:
            for idx, row in df_section.iterrows():
                rank = df_section.index.get_loc(idx) + 1
                cards_html += add_match(rank, row, prefix=prefix)
        
        sections_html += f'''<section class="section" id="section-{key}"><div class="cards-grid">{cards_html}</div></section>'''
    
    # Генерируем JavaScript отдельно - один раз для всех карточек
    html_content += f'''
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Вставляем контент
            document.getElementById('tabContainer').innerHTML = `{tabs_html}`;
            document.getElementById('sectionsContainer').innerHTML = `{sections_html}`;
            
            const tabs = document.querySelectorAll('.tab-btn');
            const sections = document.querySelectorAll('.section');
            
            // Табы
            tabs.forEach(tab => {{
                tab.addEventListener('click', () => {{
                    const tabKey = tab.getAttribute('data-tab');
                    tabs.forEach(t => t.classList.remove('active'));
                    sections.forEach(s => s.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById('section-' + tabKey).classList.add('active');
                }});
            }});
            
            // Первый таб активен
            if (tabs.length > 0) {{
                tabs[0].classList.add('active');
                sections[0].classList.add('active');
            }}
            
            const searchInput = document.getElementById('searchInput');
            const eegEmotionFilter = document.getElementById('eegEmotionFilter');
            const datasetFilter = document.getElementById('datasetFilter');
            const topComposersFull = {top_composers_full_json};
            const allEmotions = {all_emotions_json};

            function selectTopComposer(stats) {{
                const entries = Object.entries(stats).map(([composer, info]) => {{
                    const avg = info.count > 0 ? (info.total / info.count) : 0;
                    return {{ composer, count: info.count, avg, max: info.max }};
                }});
                entries.sort((a, b) => {{
                    if (b.count !== a.count) return b.count - a.count;
                    if (b.avg !== a.avg) return b.avg - a.avg;
                    if (b.max !== a.max) return b.max - a.max;
                    return a.composer.localeCompare(b.composer);
                }});
                return entries.length ? entries[0].composer : null;
            }}

            function computeTopComposerFromVisibleCards(emotion) {{
                const stats = {{}};
                document.querySelectorAll('.match-card').forEach(card => {{
                    if (card.style.display === 'none') return;
                    const cardEmotion = card.getAttribute('data-eeg-emotion') || '';
                    if (cardEmotion !== emotion) return;
                    const composer = card.getAttribute('data-composer') || 'Unknown';
                    const score = parseFloat(card.getAttribute('data-combined-score') || '0');
                    if (!stats[composer]) stats[composer] = {{ count: 0, total: 0, max: -Infinity }};
                    stats[composer].count += 1;
                    stats[composer].total += score;
                    stats[composer].max = Math.max(stats[composer].max, score);
                }});
                return selectTopComposer(stats);
            }}

            function updateTopComposers() {{
                const selectedEmotion = eegEmotionFilter.value;
                const lines = document.querySelectorAll('.top-composer-line');
                if (selectedEmotion === 'all') {{
                    lines.forEach(line => {{
                        const emotion = line.getAttribute('data-emotion');
                        const composer = topComposersFull[emotion] || 'Unknown';
                        line.textContent = `${{emotion}} — ${{composer}}`;
                        line.style.display = 'block';
                    }});
                    return;
                }}

                const fromFiltered = computeTopComposerFromVisibleCards(selectedEmotion);
                const fallback = topComposersFull[selectedEmotion] || 'Unknown';
                const finalComposer = fromFiltered || fallback || 'Unknown';

                lines.forEach(line => {{
                    const emotion = line.getAttribute('data-emotion');
                    if (emotion === selectedEmotion) {{
                        line.textContent = `${{emotion}} — ${{finalComposer}}`;
                        line.style.display = 'block';
                    }} else {{
                        line.style.display = 'none';
                    }}
                }});
            }}

            function applyFilters() {{
                const query = (searchInput.value || '').toLowerCase();
                const eegEmotion = eegEmotionFilter.value;
                const dataset = datasetFilter.value;

                document.querySelectorAll('.match-card').forEach(card => {{
                    const composer = (card.getAttribute('data-composer') || '').toLowerCase();
                    const title = (card.getAttribute('data-title') || '').toLowerCase();
                    const cardEmotion = (card.getAttribute('data-eeg-emotion') || 'unknown');
                    const cardDataset = (card.getAttribute('data-dataset') || 'unknown');

                    const matchesText = composer.includes(query) || title.includes(query);
                    const matchesEmotion = (eegEmotion === 'all') || (cardEmotion === eegEmotion);
                    const matchesDataset = (dataset === 'all') || (cardDataset === dataset);

                    card.style.display = (matchesText && matchesEmotion && matchesDataset) ? '' : 'none';
                }});
                updateTopComposers();
            }}

            // Поиск и фильтры
            searchInput.addEventListener('input', applyFilters);
            eegEmotionFilter.addEventListener('change', applyFilters);
            datasetFilter.addEventListener('change', applyFilters);
            updateTopComposers();
            
            // Графики - линейный pitch comparison
            const chartInstances = {{}};
            document.querySelectorAll('.graph-toggle input[type="checkbox"]').forEach(checkbox => {{
                checkbox.addEventListener('change', function() {{
                    const chartId = this.getAttribute('data-chart');
                    const container = document.getElementById(chartId);
                    
                    if (this.checked) {{
                        container.classList.add('open');
                        
                        if (!chartInstances[chartId]) {{
                            setTimeout(() => {{
                                const canvas = document.getElementById('canvas_' + chartId);
                                if (!canvas) return;
                                
                                let eegPitches = [];
                                let claPitches = [];
                                try {{
                                    eegPitches = JSON.parse(this.getAttribute('data-eeg') || '[]');
                                    claPitches = JSON.parse(this.getAttribute('data-classical') || '[]');
                                }} catch(e) {{ }}
                                
                                if (eegPitches.length === 0 && claPitches.length === 0) {{
                                    canvas.parentElement.innerHTML = '<div style="text-align:center;color:#999;padding:20px;">No pitch data</div>';
                                    return;
                                }}
                                
                                const maxLen = Math.max(eegPitches.length, claPitches.length);
                                const labels = Array.from({{length: maxLen}}, (_, i) => i + 1);
                                
                                chartInstances[chartId] = new Chart(canvas.getContext('2d'), {{
                                    type: 'line',
                                    data: {{
                                        labels: labels,
                                        datasets: [
                                            {{
                                                label: 'EEG Melody',
                                                data: eegPitches,
                                                borderColor: '#e74c3c',
                                                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                                borderWidth: 2,
                                                fill: false,
                                                tension: 0.3,
                                                pointRadius: 2
                                            }},
                                            {{
                                                label: 'Classical',
                                                data: claPitches,
                                                borderColor: '#3498db',
                                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                                borderWidth: 2,
                                                fill: false,
                                                tension: 0.3,
                                                pointRadius: 2
                                            }}
                                        ]
                                    }},
                                    options: {{
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        interaction: {{ intersect: false, mode: 'index' }},
                                        plugins: {{
                                            legend: {{
                                                display: true,
                                                position: 'top',
                                                labels: {{ boxWidth: 12, font: {{ size: 10 }} }}
                                            }}
                                        }},
                                        scales: {{
                                            x: {{
                                                display: true,
                                                title: {{ display: true, text: 'Note #', font: {{ size: 10 }} }},
                                                ticks: {{ font: {{ size: 9 }} }}
                                            }},
                                            y: {{
                                                display: true,
                                                title: {{ display: true, text: 'MIDI Pitch', font: {{ size: 10 }} }},
                                                ticks: {{ font: {{ size: 9 }} }}
                                            }}
                                        }}
                                    }}
                                }});
                            }}, 50);
                        }}
                    }} else {{
                        container.classList.remove('open');
                    }}
                }});
            }});
            
            // Metadata toggles
            document.querySelectorAll('.metadata-toggle input[type="checkbox"]').forEach(checkbox => {{
                checkbox.addEventListener('change', function() {{
                    const metaId = this.getAttribute('data-meta');
                    const container = document.getElementById(metaId);
                    if (this.checked) {{
                        container.classList.add('open');
                    }} else {{
                        container.classList.remove('open');
                    }}
                }});
            }});
            
            // Octave distribution chart toggles
            document.querySelectorAll('.graph-toggle input[data-octave-chart]').forEach(checkbox => {{
                checkbox.addEventListener('change', function() {{
                    const chartId = this.getAttribute('data-octave-chart');
                    const container = document.getElementById(chartId);
                    if (this.checked) {{
                        container.classList.add('open');
                        if (!chartInstances[chartId]) {{
                            setTimeout(() => {{
                                const canvas = document.getElementById('canvas_' + chartId);
                                if (!canvas) return;
                                
                                let eegOctaves = [];
                                let claOctaves = [];
                                try {{
                                    eegOctaves = JSON.parse(this.getAttribute('data-eeg-octaves') || '[]');
                                    claOctaves = JSON.parse(this.getAttribute('data-cla-octaves') || '[]');
                                }} catch(e) {{ }}
                                
                                // Октавы 0-10 (MIDI pitch / 12)
                                const labels = ['C-1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'];
                                
                                chartInstances[chartId] = new Chart(canvas.getContext('2d'), {{
                                    type: 'bar',
                                    data: {{
                                        labels: labels,
                                        datasets: [
                                            {{
                                                label: 'EEG MIDI',
                                                data: eegOctaves,
                                                backgroundColor: 'rgba(231, 76, 60, 0.7)',
                                                borderColor: '#e74c3c',
                                                borderWidth: 1
                                            }},
                                            {{
                                                label: 'Classical',
                                                data: claOctaves,
                                                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                                                borderColor: '#3498db',
                                                borderWidth: 1
                                            }}
                                        ]
                                    }},
                                    options: {{
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        plugins: {{
                                            legend: {{
                                                display: true,
                                                position: 'top',
                                                labels: {{ boxWidth: 12, font: {{ size: 10 }} }}
                                            }},
                                            title: {{
                                                display: true,
                                                text: 'Note Distribution by Octave',
                                                font: {{ size: 11 }}
                                            }}
                                        }},
                                        scales: {{
                                            x: {{
                                                display: true,
                                                title: {{ display: true, text: 'Octave', font: {{ size: 10 }} }},
                                                ticks: {{ font: {{ size: 9 }} }}
                                            }},
                                            y: {{
                                                display: true,
                                                title: {{ display: true, text: 'Proportion', font: {{ size: 10 }} }},
                                                ticks: {{ font: {{ size: 9 }} }},
                                                beginAtZero: true,
                                                max: 1.0
                                            }}
                                        }}
                                    }}
                                }});
                            }}, 50);
                        }}
                    }} else {{
                        container.classList.remove('open');
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>'''
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report created: {html_path}")
    print(f"Open file: {html_path}")

