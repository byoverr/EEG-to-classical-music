"""
Групповой анализ: этапы 3 и 4

Группирует результаты по эмоциям (квадрантам), вычисляет средние
музыкальные свойства мелодий для каждой группы и определяет,
какие классические произведения чаще всего подбираются каждой эмоции.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Ключевые музыкальные фичи для профилирования ─────────────────────
PROFILE_FEATURES = {
    # feature_column -> (human_label, description, higher_means)
    "velocity_mean": ("Velocity (Loudness)", "Average note loudness", "louder"),
    "velocity_std": ("Dynamic Range", "Variation in loudness", "more contrast"),
    "tempo_proxy": ("Tempo", "Notes per beat (speed)", "faster"),
    "pitch_mean": ("Pitch Height", "Average pitch register", "higher"),
    "pitch_range": ("Pitch Range", "Spread of pitch values", "wider range"),
    "note_density": ("Note Density", "Notes per second", "denser"),
    "rhythm_regularity": ("Rhythm Variability", "Irregularity of timing", "more variable"),
    "staccato_ratio": ("Staccato Ratio", "Proportion of short notes", "more staccato"),
    "interval_mean": ("Avg Interval Size", "Average melodic jump", "larger jumps"),
    "leap_ratio": ("Leap Ratio", "Proportion of large jumps (>7 semitones)", "more leaps"),
    "pitch_class_entropy": ("Tonal Diversity", "Number of different pitch classes used", "more diverse"),
    "consonance_mean": ("Consonance", "Harmonic consonance level", "more consonant"),
    "duration_mean": ("Note Duration", "Average note length", "longer notes"),
    "ioi_mean": ("Inter-onset Interval", "Time between notes", "more spaced"),
    "register_low": ("Low Register", "Proportion of low notes (<C3)", "more bass"),
    "register_high": ("High Register", "Proportion of high notes (>=C5)", "more treble"),
    "key_mode": ("Mode", "Major (1) vs Minor (0)", "more major"),
}

# Подмножество для основного радар-чарта (самые информативные)
RADAR_FEATURES = [
    "velocity_mean", "tempo_proxy", "pitch_mean", "pitch_range",
    "note_density", "rhythm_regularity", "staccato_ratio",
    "pitch_class_entropy", "consonance_mean", "key_mode",
]

EMOTION_LABELS = {
    "HVHA": "High Valence / High Arousal (Joy)",
    "HVLA": "High Valence / Low Arousal (Calm)",
    "LVLA": "Low Valence / Low Arousal (Sadness)",
    "LVHA": "Low Valence / High Arousal (Tension/Anger)",
}

EMOTION_COLORS = {
    "HVHA": "#e8453c",
    "HVLA": "#4285f4",
    "LVLA": "#9aa0a6",
    "LVHA": "#f9ab00",
}

EMOTION_ORDER = ["HVHA", "HVLA", "LVLA", "LVHA"]



def _parse_window_col(series: "pd.Series") -> list:
    """Parse a column of stringified lists into a list of np arrays."""
    import ast
    result = []
    for v in series:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            result.append(np.array([]))
            continue
        try:
            parsed = ast.literal_eval(str(v))
            result.append(np.array(parsed, dtype=float))
        except Exception:
            result.append(np.array([]))
    return result


def _derive_features_from_windows(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Derive track-level features from window columns when pre-computed
    feature columns (velocity_mean, tempo_proxy, etc.) are absent.

    Uses: classical_window_velocities, classical_window_pitches,
          classical_window_ioi, classical_window_durations
    """
    needed_raw = {
        "velocity_mean": ("classical_window_velocities", None),
        "velocity_std": ("classical_window_velocities", None),
        "pitch_mean": ("classical_window_pitches", None),
        "pitch_range": ("classical_window_pitches", None),
        "pitch_std": ("classical_window_pitches", None),
        "ioi_mean": ("classical_window_ioi", None),
        "ioi_std": ("classical_window_ioi", None),
        "tempo_proxy": ("classical_window_ioi", None),
        "rhythm_regularity": ("classical_window_ioi", None),
        "duration_mean": ("classical_window_durations", None),
        "staccato_ratio": ("classical_window_durations", None),
        "note_density": (None, None),
        "register_low": ("classical_window_pitches", None),
        "register_mid": ("classical_window_pitches", None),
        "register_high": ("classical_window_pitches", None),
    }

    out = df.copy()

    # Parse all needed window columns once
    parsed = {}
    for col in ["classical_window_velocities", "classical_window_pitches",
                "classical_window_ioi", "classical_window_durations"]:
        if col in df.columns:
            parsed[col] = _parse_window_col(df[col])

    if not parsed:
        return out

    n = len(df)

    def _agg(col_key, func):
        arrs = parsed.get(col_key)
        if arrs is None:
            return np.zeros(n)
        result = np.zeros(n)
        for i, arr in enumerate(arrs):
            if len(arr) > 0:
                result[i] = func(arr)
        return result

    vels = parsed.get("classical_window_velocities")
    pitches = parsed.get("classical_window_pitches")
    iois = parsed.get("classical_window_ioi")
    durs = parsed.get("classical_window_durations")

    # Velocity
    if vels is not None and "velocity_mean" not in out.columns:
        out["velocity_mean"] = [float(np.mean(a)) if len(a) > 0 else 0.0 for a in vels]
    if vels is not None and "velocity_std" not in out.columns:
        out["velocity_std"] = [float(np.std(a)) if len(a) > 1 else 0.0 for a in vels]

    # Pitch
    if pitches is not None:
        if "pitch_mean" not in out.columns:
            out["pitch_mean"] = [float(np.mean(a)) if len(a) > 0 else 0.0 for a in pitches]
        if "pitch_range" not in out.columns:
            out["pitch_range"] = [float(np.ptp(a)) if len(a) > 0 else 0.0 for a in pitches]
        if "pitch_std" not in out.columns:
            out["pitch_std"] = [float(np.std(a)) if len(a) > 1 else 0.0 for a in pitches]
        if "register_low" not in out.columns:
            out["register_low"] = [float(np.mean(a < 48)) if len(a) > 0 else 0.0 for a in pitches]
        if "register_mid" not in out.columns:
            out["register_mid"] = [float(np.mean((a >= 48) & (a < 72))) if len(a) > 0 else 0.0 for a in pitches]
        if "register_high" not in out.columns:
            out["register_high"] = [float(np.mean(a >= 72)) if len(a) > 0 else 0.0 for a in pitches]

    # IOI / Rhythm
    if iois is not None:
        pos_iois = [a[a > 0] if len(a) > 0 else np.array([]) for a in iois]
        if "ioi_mean" not in out.columns:
            out["ioi_mean"] = [float(np.mean(a)) if len(a) > 0 else 0.0 for a in pos_iois]
        if "ioi_std" not in out.columns:
            out["ioi_std"] = [float(np.std(a)) if len(a) > 1 else 0.0 for a in pos_iois]
        if "tempo_proxy" not in out.columns:
            out["tempo_proxy"] = [
                float(1.0 / (np.mean(a) + 1e-6)) if len(a) > 0 else 0.0
                for a in pos_iois
            ]
        if "rhythm_regularity" not in out.columns:
            out["rhythm_regularity"] = [
                float(np.std(a) / (np.mean(a) + 1e-6)) if len(a) > 1 else 0.0
                for a in pos_iois
            ]

    # Duration
    if durs is not None:
        if "duration_mean" not in out.columns:
            out["duration_mean"] = [float(np.mean(a)) if len(a) > 0 else 0.0 for a in durs]
        if "staccato_ratio" not in out.columns:
            out["staccato_ratio"] = [float(np.mean(a < 0.25)) if len(a) > 0 else 0.0 for a in durs]

    # Mode (major=1, minor=0) — эвристика: тоника = самая частая pitch-class,
    # мажор если большая терция встречается чаще малой
    if pitches is not None and "key_mode" not in out.columns:
        mode_vals = []
        for a in pitches:
            if len(a) < 3:
                mode_vals.append(0.5)
                continue
            pcs = np.asarray(a, dtype=int) % 12
            tonic = int(np.bincount(pcs, minlength=12).argmax())
            rel = (pcs - tonic) % 12
            n_maj3 = int(np.sum(rel == 4))
            n_min3 = int(np.sum(rel == 3))
            if n_maj3 + n_min3 == 0:
                mode_vals.append(0.5)
            else:
                mode_vals.append(1.0 if n_maj3 > n_min3 else 0.0)
        out["key_mode"] = mode_vals

    # Consonance — доля консонирующих интервалов между соседними нотами
    # Консонансы (в полутонах mod 12): 0, 3, 4, 5, 7, 8, 9
    if pitches is not None and "consonance_mean" not in out.columns:
        _consonant = {0, 3, 4, 5, 7, 8, 9}
        cons_vals = []
        for a in pitches:
            if len(a) < 2:
                cons_vals.append(0.0)
                continue
            intervals = np.abs(np.diff(np.asarray(a, dtype=int))) % 12
            if len(intervals) == 0:
                cons_vals.append(0.0)
            else:
                cons_vals.append(float(np.mean([int(i) in _consonant for i in intervals])))
        out["consonance_mean"] = cons_vals

    # Note density (notes per second of window span)
    if pitches is not None and "note_density" not in out.columns:
        if "classical_start_time" in df.columns and "ioi_mean" in out.columns:
            out["note_density"] = [
                float(len(p) / max(float(out["ioi_mean"].iloc[i]) * len(p), 1e-6))
                if len(p) > 0 else 0.0
                for i, p in enumerate(pitches)
            ]
        else:
            out["note_density"] = [float(len(p)) for p in pitches]

    return out

def compute_group_profiles(
    results_df: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict:
    """
    Группирует результаты по EEG-эмоциям и строит профиль для каждой группы.

    Returns dict with keys:
        - emotion_profiles: {emotion: {feature: mean_value, ...}}
        - emotion_profiles_std: {emotion: {feature: std_value, ...}}
        - top_works_by_emotion: {emotion: [{title, composer, count, avg_score}, ...]}
        - top_composers_by_emotion: {emotion: [{composer, count}, ...]}
        - n_per_emotion: {emotion: count}
        - feature_table: pd.DataFrame (emotion x features)
        - distinguishing_features: [(feature, f_stat, description), ...]
        - normalized_profiles: {emotion: {feature: 0..1 normalized value}}
    """
    if results_df.empty or "eeg_emotion" not in results_df.columns:
        return {}

    df = results_df.copy()

    # Берём лучший результат для каждого участника/триала
    group_cols = [c for c in ["participant_id", "trial_idx"] if c in df.columns]
    if not group_cols and "eeg_midi" in df.columns:
        group_cols = ["eeg_midi"]

    ranking_col = "music_match_score" if "music_match_score" in df.columns else "combined_similarity"
    if ranking_col in df.columns:
        df = df.sort_values(ranking_col, ascending=False)

    if group_cols:
        best_df = df.groupby(group_cols, dropna=False, as_index=False).head(1)
    else:
        best_df = df

    # Top-K для анализа произведений
    if group_cols:
        topk_df = df.groupby(group_cols, dropna=False, as_index=False).head(top_k)
    else:
        topk_df = df.head(top_k)

    emotions = sorted(best_df["eeg_emotion"].dropna().unique())
    # Derive features from window columns if pre-computed features are absent
    best_df = _derive_features_from_windows(best_df)
    topk_df = _derive_features_from_windows(topk_df)

    feature_cols = [f for f in PROFILE_FEATURES if f in best_df.columns]

    # ── 1. Средние профили по эмоциям ──
    emotion_profiles = {}
    emotion_profiles_std = {}
    n_per_emotion = {}

    for emo in emotions:
        mask = best_df["eeg_emotion"] == emo
        group = best_df.loc[mask, feature_cols]
        n_per_emotion[emo] = int(mask.sum())
        emotion_profiles[emo] = {}
        emotion_profiles_std[emo] = {}
        for f in feature_cols:
            vals = pd.to_numeric(group[f], errors="coerce").dropna()
            emotion_profiles[emo][f] = float(vals.mean()) if len(vals) > 0 else 0.0
            emotion_profiles_std[emo][f] = float(vals.std()) if len(vals) > 1 else 0.0

    # ── 2. Нормализованные профили (0-1) для радар-чарта ──
    normalized_profiles = {}
    if emotion_profiles:
        for f in feature_cols:
            all_vals = [emotion_profiles[e].get(f, 0.0) for e in emotions]
            fmin, fmax = min(all_vals), max(all_vals)
            rng = fmax - fmin if fmax > fmin else 1.0
            for emo in emotions:
                if emo not in normalized_profiles:
                    normalized_profiles[emo] = {}
                normalized_profiles[emo][f] = (emotion_profiles[emo].get(f, 0.0) - fmin) / rng

    # ── 3. Top works per emotion ──
    title_col = "title" if "title" in topk_df.columns else "classical_piece"
    composer_col = "composer" if "composer" in topk_df.columns else "classical_composer"

    top_works_by_emotion = {}
    top_composers_by_emotion = {}

    for emo in emotions:
        mask = topk_df["eeg_emotion"] == emo
        emo_df = topk_df[mask]

        if title_col in emo_df.columns:
            work_counts = emo_df[title_col].fillna("Unknown").value_counts()
            works = []
            for work, count in work_counts.head(5).items():
                work_mask = emo_df[title_col] == work
                avg_score = float(emo_df.loc[work_mask, ranking_col].mean()) if ranking_col in emo_df.columns else 0.0
                comp = "Unknown"
                if composer_col in emo_df.columns:
                    comp_vals = emo_df.loc[work_mask, composer_col].dropna()
                    comp = str(comp_vals.iloc[0]) if not comp_vals.empty else "Unknown"
                works.append({
                    "title": _clean_label(str(work)),
                    "composer": comp,
                    "count": int(count),
                    "avg_score": avg_score,
                })
            top_works_by_emotion[emo] = works

        if composer_col in emo_df.columns:
            comp_counts = emo_df[composer_col].fillna("Unknown").value_counts()
            top_composers_by_emotion[emo] = [
                {"composer": str(c), "count": int(n)}
                for c, n in comp_counts.head(3).items()
            ]

    # ── 4. Distinguishing features (ANOVA-like F-stat) ──
    distinguishing = []
    for f in feature_cols:
        groups = []
        for emo in emotions:
            mask = best_df["eeg_emotion"] == emo
            vals = pd.to_numeric(best_df.loc[mask, f], errors="coerce").dropna().values
            if len(vals) > 0:
                groups.append(vals)
        if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
            df_between = len(groups) - 1
            df_within = sum(len(g) for g in groups) - len(groups)
            if df_within > 0 and ss_within > 0:
                f_stat = (ss_between / df_between) / (ss_within / df_within)
            else:
                f_stat = 0.0
            label, desc, _ = PROFILE_FEATURES[f]
            distinguishing.append((f, label, f_stat, desc))

    distinguishing.sort(key=lambda x: x[2], reverse=True)

    # ── 5. Feature table (emotion × features) ──
    rows = []
    for emo in emotions:
        row = {"Emotion": emo, "N": n_per_emotion.get(emo, 0)}
        for f in feature_cols:
            label = PROFILE_FEATURES[f][0]
            val = emotion_profiles[emo].get(f, 0.0)
            row[label] = val
        rows.append(row)
    feature_table = pd.DataFrame(rows)

    return {
        "emotions": emotions,
        "emotion_profiles": emotion_profiles,
        "emotion_profiles_std": emotion_profiles_std,
        "normalized_profiles": normalized_profiles,
        "top_works_by_emotion": top_works_by_emotion,
        "top_composers_by_emotion": top_composers_by_emotion,
        "n_per_emotion": n_per_emotion,
        "feature_table": feature_table,
        "distinguishing_features": distinguishing,
    }


def save_group_analysis_artifacts(
    report_dir: Path,
    group_data: dict,
) -> list[str]:
    """Saves group analysis CSVs and charts. Returns list of generated file names."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    if not group_data:
        return generated

    # ── Save feature table CSV ──
    feature_table = group_data.get("feature_table")
    if feature_table is not None and not feature_table.empty:
        feature_table.to_csv(report_dir / "group_feature_profiles.csv", index=False)
        generated.append("group_feature_profiles.csv")

    # ── Save top works per emotion CSV ──
    works_rows = []
    for emo, works in group_data.get("top_works_by_emotion", {}).items():
        for w in works:
            works_rows.append({
                "eeg_emotion": emo,
                "title": w["title"],
                "composer": w["composer"],
                "count": w["count"],
                "avg_score": w["avg_score"],
            })
    if works_rows:
        pd.DataFrame(works_rows).to_csv(
            report_dir / "group_top_works.csv", index=False,
        )
        generated.append("group_top_works.csv")

    # ── Save distinguishing features CSV ──
    dist = group_data.get("distinguishing_features", [])
    if dist:
        pd.DataFrame(dist, columns=["feature_key", "feature_name", "f_statistic", "description"]).to_csv(
            report_dir / "group_distinguishing_features.csv", index=False,
        )
        generated.append("group_distinguishing_features.csv")

    # ── Generate charts ──
    chart_files = generate_group_charts(report_dir, group_data)
    generated.extend(chart_files)

    return generated


def generate_group_charts(report_dir: Path, group_data: dict) -> list[str]:
    """Generates matplotlib charts for group analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.ticker as ticker

    generated = []
    emotions = group_data.get("emotions", [])
    if not emotions:
        return generated

    profiles = group_data.get("emotion_profiles", {})
    norm_profiles = group_data.get("normalized_profiles", {})
    n_per = group_data.get("n_per_emotion", {})

    # Use consistent emotion ordering
    ordered_emotions = [e for e in EMOTION_ORDER if e in emotions]
    colors = [EMOTION_COLORS.get(e, "#999999") for e in ordered_emotions]

    # ── 1. Radar Chart ──
    try:
        radar_feats = [f for f in RADAR_FEATURES if f in norm_profiles.get(ordered_emotions[0], {})]
        if radar_feats and len(ordered_emotions) >= 2:
            labels = [PROFILE_FEATURES[f][0] for f in radar_feats]
            n_feats = len(radar_feats)
            angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            ax.set_facecolor("#fafbfc")
            fig.patch.set_facecolor("white")

            for emo, color in zip(ordered_emotions, colors):
                vals = [norm_profiles[emo].get(f, 0.0) for f in radar_feats]
                vals += vals[:1]
                ax.plot(angles, vals, "o-", linewidth=2.2, label=f"{emo} (n={n_per.get(emo, 0)})", color=color)
                ax.fill(angles, vals, alpha=0.12, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=9, fontweight="500")
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8, color="#888")
            ax.set_title("Emotion Group Feature Profiles", size=15, fontweight="bold", pad=24, color="#202124")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(str(report_dir / "group_radar_chart.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)
            generated.append("group_radar_chart.png")
    except Exception:
        pass

    # ── 2. Feature Comparison Bar Charts (top distinguishing features) ──
    try:
        dist_feats = group_data.get("distinguishing_features", [])
        show_feats = [d[0] for d in dist_feats[:6]]  # top 6 most distinguishing
        if show_feats and len(ordered_emotions) >= 2:
            n_feats = len(show_feats)
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.patch.set_facecolor("white")
            axes = axes.flatten()

            for idx, feat_key in enumerate(show_feats):
                ax = axes[idx]
                label = PROFILE_FEATURES[feat_key][0]
                vals = [profiles[e].get(feat_key, 0.0) for e in ordered_emotions]
                bars = ax.bar(
                    ordered_emotions, vals, color=colors,
                    edgecolor="white", linewidth=1.5, width=0.6,
                )
                ax.set_title(label, fontsize=11, fontweight="600", color="#202124")
                ax.tick_params(axis="x", labelsize=9)
                ax.tick_params(axis="y", labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

                # Add value labels on bars
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#444",
                    )

            # Hide empty subplots
            for idx in range(n_feats, 6):
                axes[idx].set_visible(False)

            fig.suptitle(
                "Most Distinguishing Features by Emotion",
                fontsize=14, fontweight="bold", color="#202124", y=1.01,
            )
            fig.tight_layout()
            fig.savefig(str(report_dir / "group_feature_bars.png"), dpi=160, bbox_inches="tight")
            plt.close(fig)
            generated.append("group_feature_bars.png")
    except Exception:
        pass

    # ── 3. Top Works Heatmap ──
    try:
        works_by_emo = group_data.get("top_works_by_emotion", {})
        if works_by_emo:
            # Collect all unique works across emotions
            all_works = set()
            for emo, works in works_by_emo.items():
                for w in works[:3]:
                    all_works.add(w["title"])
            all_works = sorted(all_works)

            if all_works and len(ordered_emotions) >= 2:
                matrix = np.zeros((len(ordered_emotions), len(all_works)))
                for i, emo in enumerate(ordered_emotions):
                    for w in works_by_emo.get(emo, []):
                        if w["title"] in all_works:
                            j = all_works.index(w["title"])
                            matrix[i, j] = w["count"]

                # Truncate long labels
                work_labels = [t[:35] + "..." if len(t) > 35 else t for t in all_works]

                fig, ax = plt.subplots(figsize=(max(8, len(all_works) * 1.2), 4))
                fig.patch.set_facecolor("white")

                import matplotlib.colors as mcolors
                cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#f8f9fa", "#1a73e8"])
                im = ax.imshow(matrix, aspect="auto", cmap=cmap)

                ax.set_xticks(range(len(work_labels)))
                ax.set_xticklabels(work_labels, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(len(ordered_emotions)))
                ax.set_yticklabels(ordered_emotions, fontsize=10, fontweight="600")

                for i in range(len(ordered_emotions)):
                    for j in range(len(all_works)):
                        val = int(matrix[i, j])
                        if val > 0:
                            ax.text(j, i, str(val), ha="center", va="center",
                                    fontsize=10, fontweight="bold",
                                    color="white" if val > matrix.max() * 0.5 else "#333")

                ax.set_title("Classical Works Selected per Emotion", fontsize=13,
                             fontweight="bold", color="#202124", pad=12)
                fig.colorbar(im, ax=ax, label="Selection count", shrink=0.8)
                fig.tight_layout()
                fig.savefig(str(report_dir / "group_works_heatmap.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
                generated.append("group_works_heatmap.png")
    except Exception:
        pass

    # ── 4. Feature Heatmap (normalized) ──
    try:
        if norm_profiles and len(ordered_emotions) >= 2:
            feat_keys = [f for f in PROFILE_FEATURES if f in norm_profiles.get(ordered_emotions[0], {})]
            if feat_keys:
                feat_labels = [PROFILE_FEATURES[f][0] for f in feat_keys]
                matrix = np.array([
                    [norm_profiles[e].get(f, 0.0) for f in feat_keys]
                    for e in ordered_emotions
                ])

                fig, ax = plt.subplots(figsize=(max(10, len(feat_keys) * 0.7), 4))
                fig.patch.set_facecolor("white")

                import matplotlib.colors as mcolors
                cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#4285f4", "#f8f9fa", "#e8453c"])
                im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

                ax.set_xticks(range(len(feat_labels)))
                ax.set_xticklabels(feat_labels, rotation=55, ha="right", fontsize=8)
                ax.set_yticks(range(len(ordered_emotions)))
                ax.set_yticklabels(
                    [f"{e} (n={n_per.get(e, 0)})" for e in ordered_emotions],
                    fontsize=10, fontweight="600",
                )

                for i in range(len(ordered_emotions)):
                    for j in range(len(feat_keys)):
                        val = matrix[i, j]
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="#333" if 0.3 < val < 0.7 else "white")

                ax.set_title("Feature Profile Heatmap (Normalized)", fontsize=13,
                             fontweight="bold", color="#202124", pad=12)
                fig.colorbar(im, ax=ax, label="Relative level (0=lowest, 1=highest)", shrink=0.8)
                fig.tight_layout()
                fig.savefig(str(report_dir / "group_feature_heatmap.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
                generated.append("group_feature_heatmap.png")
    except Exception:
        pass

    return generated


def _clean_label(text: str) -> str:
    text = text.strip()
    if "|" in text:
        text = text.split("|")[-1].strip()
    return text
