"""
Оценка гипотезы об эмоциональном соответствии EEG и музыки.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import numpy as np
import pandas as pd


DEFAULT_SCORE_WEIGHTS = {
    "emotion": 0.50,
    "feature": 0.35,
    "fragment": 0.15,
}

PRIMARY_RANKING_COLUMN = "music_match_score"

FEATURE_EVIDENCE_COLUMNS = [
    "contour_similarity",
    "interval_similarity",
    "harmony_similarity",
    "sfi_similarity",
    "stat_similarity",
    "rhythm_similarity",
    "dynamic_similarity",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _as_optional_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def _mean_existing(row: pd.Series, columns: Iterable[str]) -> float:
    values = [_safe_float(row.get(col), default=np.nan) for col in columns if col in row.index]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return 0.0
    return _clip01(float(np.mean(values)))


def add_hypothesis_scores(
    results_df: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Добавляет primary music ranking и отдельные гипотезные компоненты.
    """
    if results_df.empty:
        return results_df.copy()

    weights = dict(DEFAULT_SCORE_WEIGHTS if weights is None else weights)
    df = results_df.copy()

    if "emotion_match" not in df.columns:
        if {"eeg_emotion", "classical_emotion"}.issubset(df.columns):
            df["emotion_match"] = (
                df["eeg_emotion"].fillna("").astype(str)
                == df["classical_emotion"].fillna("").astype(str)
            )
        else:
            df["emotion_match"] = None

    music_scores = []
    emotion_scores = []
    feature_scores = []
    fragment_scores = []
    final_scores = []
    labels = []

    for _, row in df.iterrows():
        eeg_emotion = str(row.get("eeg_emotion") or "").strip()
        classical_emotion = str(row.get("classical_emotion") or "").strip()
        emotion_match = _as_optional_bool(row.get("emotion_match"))

        music_score = _clip01(_safe_float(row.get("combined_similarity"), 0.0))
        if emotion_match is True:
            emotion_score = 1.0
            label = "Совпадение"
        elif emotion_match is False:
            emotion_score = 0.0
            label = "Несовпадение"
        elif eeg_emotion and classical_emotion:
            emotion_score = 1.0 if eeg_emotion == classical_emotion else 0.0
            label = "Совпадение" if emotion_score >= 0.5 else "Несовпадение"
        else:
            emotion_score = 0.0
            label = "Неопределено"

        feature_score = _mean_existing(row, FEATURE_EVIDENCE_COLUMNS)
        fragment_score = music_score

        cemms = (
            weights["emotion"] * emotion_score
            + weights["feature"] * feature_score
            + weights["fragment"] * fragment_score
        )
        music_scores.append(music_score)
        emotion_scores.append(_clip01(emotion_score))
        feature_scores.append(_clip01(feature_score))
        fragment_scores.append(_clip01(fragment_score))
        final_scores.append(_clip01(cemms))
        labels.append(label)

    df[PRIMARY_RANKING_COLUMN] = music_scores
    df["emotion_agreement_score"] = emotion_scores
    df["feature_similarity_score"] = feature_scores
    df["fragment_alignment_score"] = fragment_scores
    df["cemms_score"] = final_scores
    df["match_label"] = labels
    sort_cols = [c for c in [PRIMARY_RANKING_COLUMN, "cemms_score"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return df


def compute_hypothesis_metrics(
    results_df: pd.DataFrame,
    *,
    top_k: int = 5,
    analysis_mode: str = "single",
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Считает ключевые метрики, confusion matrix, cohort summary и feature evidence.
    """
    if results_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = add_hypothesis_scores(results_df)

    group_cols = [c for c in ["participant_id", "trial_idx"] if c in df.columns]
    if not group_cols and "eeg_midi" in df.columns:
        group_cols = ["eeg_midi"]

    ranking_col = PRIMARY_RANKING_COLUMN if PRIMARY_RANKING_COLUMN in df.columns else "combined_similarity"
    df = df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
    best_by_group = (
        df.groupby(group_cols, dropna=False, as_index=False).head(1)
        if group_cols else df.head(1)
    )
    topk_by_group = (
        df.groupby(group_cols, dropna=False, as_index=False).head(top_k)
        if group_cols else df.head(top_k)
    )

    emotion_rows = best_by_group.dropna(subset=["eeg_emotion", "classical_emotion"], how="any")
    labels: list[str] = []
    confusion = pd.DataFrame()
    emotion_match_rate = 0.0
    macro_f1 = 0.0
    top_k_accuracy = 0.0

    if not emotion_rows.empty:
        y_true = emotion_rows["eeg_emotion"].astype(str)
        y_pred = emotion_rows["classical_emotion"].astype(str)
        labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
        emotion_match_rate = float((y_true == y_pred).mean())
        try:
            from sklearn.metrics import f1_score

            macro_f1 = float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            )
        except Exception:
            macro_f1 = 0.0

        confusion = pd.crosstab(
            y_true,
            y_pred,
            rownames=["EEG emotion"],
            colnames=["Music emotion"],
        ).reindex(index=labels, columns=labels, fill_value=0)

    if not topk_by_group.empty and "classical_emotion" in topk_by_group.columns:
        def _topk_hit(group: pd.DataFrame) -> bool:
            eeg_emo = group["eeg_emotion"].dropna().astype(str)
            if eeg_emo.empty:
                return False
            target = eeg_emo.iloc[0]
            pred = group["classical_emotion"].dropna().astype(str)
            return bool((pred == target).any())

        if group_cols:
            top_k_accuracy = float(
                topk_by_group.groupby(group_cols, dropna=False).apply(_topk_hit).mean()
            )

    cohort_rows = []
    comp_col = "title" if "title" in best_by_group.columns else "classical_piece"
    composer_col = "composer" if "composer" in best_by_group.columns else "classical_composer"
    if "eeg_emotion" in best_by_group.columns and comp_col in best_by_group.columns:
        for emotion, group in best_by_group.groupby("eeg_emotion"):
            title_counts = group[comp_col].fillna("Unknown").value_counts()
            composer_counts = (
                group[composer_col].fillna("Unknown").value_counts()
                if composer_col in group.columns else pd.Series(dtype=int)
            )
            if title_counts.empty:
                continue
            cohort_rows.append({
                "eeg_emotion": emotion,
                "top_work": title_counts.index[0],
                "top_composer": composer_counts.index[0] if not composer_counts.empty else "Unknown",
                "consistency": float(title_counts.iloc[0] / title_counts.sum()),
                "n_people": int(title_counts.sum()),
                "mean_music_match_score": float(group[ranking_col].mean()) if ranking_col in group.columns else 0.0,
                "mean_cemms_score": float(group["cemms_score"].mean()) if "cemms_score" in group.columns else 0.0,
            })
    cohort_df = pd.DataFrame(cohort_rows)

    feature_cols = [c for c in [
        "pitch_mean", "pitch_std", "pitch_range", "note_density",
        "ioi_mean", "rhythm_regularity", "consonance_mean",
        "pitch_class_entropy", "interval_mean", "interval_std",
        "velocity_mean", "velocity_std",
        "feature_similarity_score", "fragment_alignment_score", "cemms_score",
    ] if c in best_by_group.columns]
    feature_summary = pd.DataFrame()
    if feature_cols and "eeg_emotion" in best_by_group.columns:
        feature_summary = (
            best_by_group.groupby("eeg_emotion")[feature_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

    dominant_works = []
    if comp_col in best_by_group.columns:
        for work, count in best_by_group[comp_col].fillna("Unknown").value_counts().head(5).items():
            dominant_works.append({"work": work, "count": int(count)})
    dominant_composers = []
    if composer_col in best_by_group.columns:
        for comp, count in best_by_group[composer_col].fillna("Unknown").value_counts().head(5).items():
            dominant_composers.append({"composer": comp, "count": int(count)})

    metrics = {
        "analysis_mode": analysis_mode,
        "n_results": int(len(df)),
        "n_groups": int(len(best_by_group)),
        "primary_ranking": ranking_col,
        "mean_music_match_score": float(df[ranking_col].mean()) if ranking_col in df.columns else 0.0,
        "best_music_match_score": float(df[ranking_col].max()) if ranking_col in df.columns else 0.0,
        "emotion_match_rate": float(emotion_match_rate),
        "macro_f1": float(macro_f1),
        "top_k_accuracy": float(top_k_accuracy),
        "mean_cemms_score": float(df["cemms_score"].mean()) if "cemms_score" in df.columns else 0.0,
        "best_cemms_score": float(df["cemms_score"].max()) if "cemms_score" in df.columns else 0.0,
        "labels": labels,
        "top_works": dominant_works,
        "top_composers": dominant_composers,
        "group_consistency_mean": float(cohort_df["consistency"].mean()) if not cohort_df.empty else 0.0,
    }

    return metrics, confusion, cohort_df, feature_summary


def save_hypothesis_artifacts(
    report_dir: Path,
    metrics: dict,
    confusion: pd.DataFrame,
    cohort_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "hypothesis_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not confusion.empty:
        confusion.to_csv(report_dir / "confusion_matrix.csv")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6 + len(confusion.columns) * 0.35, 5.5))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title("Emotion Confusion Matrix")
            ax.set_xlabel("Music emotion")
            ax.set_ylabel("EEG emotion")
            fig.tight_layout()
            fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
            plt.close(fig)
        except Exception:
            pass

    if not cohort_df.empty:
        cohort_df.to_csv(report_dir / "cohort_emotion_summary.csv", index=False)

    if not feature_summary.empty:
        feature_summary.to_csv(report_dir / "emotion_feature_summary.csv", index=False)
