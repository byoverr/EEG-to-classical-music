#!/usr/bin/env python3
"""
Сравнение размеров окна для EEG -> classical music pipeline.

Скрипт нужен для экспериментального обоснования выбора окна, например:
4 секунды vs 8 секунд vs 12 секунд.

Для каждого окна запускается текущий pipeline, после чего считается сводка:
- mean/best music match
- emotion match rate
- macro-F1
- top-K emotion accuracy
- group consistency
- среднее количество EEG-окон на trial

Результаты сохраняются в runs/<experiment_name>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from scripts.run_comparison import run_comparison
from src.config import RUNS_DIR, DEFAULT_TOP_K
from src.evaluation import add_hypothesis_scores, compute_hypothesis_metrics


def _slug(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _mean_unique_windows(scored_df: pd.DataFrame) -> float:
    group_cols = [c for c in ["participant_id", "trial_idx", "variant"] if c in scored_df.columns]
    if not group_cols or "eeg_window_id" not in scored_df.columns or scored_df.empty:
        return 0.0
    grouped = (
        scored_df.groupby(group_cols, dropna=False)["eeg_window_id"]
        .nunique()
        .astype(float)
    )
    return float(grouped.mean()) if not grouped.empty else 0.0


def _mean_top1_music_match(scored_df: pd.DataFrame) -> float:
    if scored_df.empty or "music_match_score" not in scored_df.columns:
        return 0.0
    group_cols = [c for c in ["participant_id", "trial_idx"] if c in scored_df.columns]
    ranked = scored_df.sort_values("music_match_score", ascending=False)
    if group_cols:
        ranked = ranked.groupby(group_cols, dropna=False, as_index=False).head(1)
    else:
        ranked = ranked.head(1)
    return float(ranked["music_match_score"].mean()) if not ranked.empty else 0.0


def _selection_score(row: pd.Series) -> float:
    return float(
        0.40 * row.get("macro_f1", 0.0)
        + 0.25 * row.get("emotion_match_rate", 0.0)
        + 0.20 * row.get("top_k_accuracy", 0.0)
        + 0.15 * row.get("mean_top1_music_match", 0.0)
    )


def _save_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    x = summary_df["window_size"]

    axes[0].plot(x, summary_df["macro_f1"], marker="o", label="Macro-F1")
    axes[0].plot(x, summary_df["emotion_match_rate"], marker="o", label="Emotion Match Rate")
    axes[0].set_ylabel("Emotion")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, summary_df["mean_top1_music_match"], marker="o", color="#1b7f5f", label="Top-1 Music Match")
    axes[1].plot(x, summary_df["best_music_match_score"], marker="o", color="#0d5ea8", label="Best Music Match")
    axes[1].set_ylabel("Music")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(x, summary_df["top_k_accuracy"], marker="o", color="#a65300", label="Top-K Accuracy")
    axes[2].plot(x, summary_df["group_consistency_mean"], marker="o", color="#7a3db8", label="Group Consistency")
    axes[2].set_ylabel("Stability")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    axes[3].bar(x, summary_df["mean_eeg_windows_per_trial"], color="#555f77", width=0.8)
    axes[3].set_ylabel("Avg EEG windows")
    axes[3].set_xlabel("Window size (sec)")
    axes[3].grid(axis="y", alpha=0.25)

    fig.suptitle("Window Size Experiment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        output_path.write_text("# Window Size Experiment\n\nНет результатов.\n", encoding="utf-8")
        return

    best = summary_df.sort_values(
        ["selection_score", "macro_f1", "emotion_match_rate", "mean_top1_music_match"],
        ascending=False,
    ).iloc[0]

    table_df = summary_df.copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")

    headers = list(table_df.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    lines = [
        "# Window Size Experiment",
        "",
        "## Recommendation",
        "",
        f"- Recommended window: **{best['window_size']:.1f} sec**",
        f"- Hop size: **{best['hop_size']:.1f} sec**",
        f"- Selection score: **{best['selection_score']:.3f}**",
        "",
        "Selection score combines:",
        "- Macro-F1 (40%)",
        "- Emotion Match Rate (25%)",
        "- Top-K Accuracy (20%)",
        "- Mean Top-1 Music Match (15%)",
        "",
        "## Summary Table",
        "",
        *table_lines,
        "",
        "## How to use in thesis",
        "",
        "- Compare 4 s and 8 s on the same participant/trial subset.",
        "- Use Macro-F1 and Emotion Match Rate as primary evidence.",
        "- Use Mean Top-1 Music Match and Avg EEG windows per trial as supporting evidence.",
        "- If 8 s gives higher emotion metrics with stable music match, justify 8 s as the main operating window.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple window sizes for the EEG pipeline")
    parser.add_argument("--window-sizes", type=float, nargs="+", default=[4.0, 8.0],
                        help="Список размеров окна в секундах, например: 4 8 12")
    parser.add_argument("--hop-ratio", type=float, default=0.5,
                        help="Шаг окна как доля window_size (по умолчанию 0.5)")
    parser.add_argument("--participants", type=int, default=3,
                        help="Количество участников DEAP")
    parser.add_argument("--trials", type=int, default=5,
                        help="Количество триалов на участника")
    parser.add_argument("--classical", type=int, default=40,
                        help="Количество классических произведений")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K,
                        help="Top-K для оценки")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Количество процессов")
    parser.add_argument("--only-emopia", action="store_true", default=False,
                        help="Сравнивать только с EMOPIA")
    parser.add_argument("--balanced-eeg-emotions", action="store_true", default=False,
                        help="Брать EEG сбалансированно по эмоциям")
    parser.add_argument("--per-emotion-trials", type=int, default=3,
                        help="Сколько EEG триалов брать на каждую эмоцию")
    parser.add_argument("--match-emotions", action="store_true", default=False,
                        help="Фильтровать классику по совпадающей эмоции")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Имя директории в runs/ для сводки эксперимента")
    args = parser.parse_args()

    experiment_name = args.experiment_name or f"window_size_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = RUNS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for window_size in args.window_sizes:
        hop_size = max(0.5, round(window_size * args.hop_ratio, 3))
        run_id = f"{experiment_name}_w{_slug(window_size)}_h{_slug(hop_size)}"

        print("=" * 72)
        print(f"Window experiment: {window_size:.2f}s window / {hop_size:.2f}s hop")
        print("=" * 72)

        results_df = run_comparison(
            max_participants=args.participants,
            max_trials=args.trials,
            max_classical=args.classical,
            top_k=args.top,
            n_jobs=args.jobs,
            only_emopia=args.only_emopia,
            balanced_eeg_emotions=args.balanced_eeg_emotions,
            per_emotion_trials=args.per_emotion_trials,
            match_emotions=args.match_emotions,
            window_size=window_size,
            hop_size=hop_size,
            run_id=run_id,
        )

        if results_df is None or results_df.empty:
            summary_rows.append({
                "window_size": window_size,
                "hop_size": hop_size,
                "run_id": run_id,
                "n_results": 0,
                "n_groups": 0,
                "mean_music_match_score": 0.0,
                "best_music_match_score": 0.0,
                "mean_top1_music_match": 0.0,
                "emotion_match_rate": 0.0,
                "macro_f1": 0.0,
                "top_k_accuracy": 0.0,
                "group_consistency_mean": 0.0,
                "mean_eeg_windows_per_trial": 0.0,
                "selection_score": 0.0,
            })
            continue

        scored_df = add_hypothesis_scores(results_df)
        metrics, confusion, cohort_df, feature_summary = compute_hypothesis_metrics(
            scored_df,
            top_k=args.top,
            analysis_mode="window_experiment",
        )

        row = {
            "window_size": float(window_size),
            "hop_size": float(hop_size),
            "run_id": run_id,
            "n_results": int(metrics.get("n_results", len(scored_df))),
            "n_groups": int(metrics.get("n_groups", 0)),
            "mean_music_match_score": float(metrics.get("mean_music_match_score", 0.0)),
            "best_music_match_score": float(metrics.get("best_music_match_score", 0.0)),
            "mean_top1_music_match": _mean_top1_music_match(scored_df),
            "emotion_match_rate": float(metrics.get("emotion_match_rate", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "top_k_accuracy": float(metrics.get("top_k_accuracy", 0.0)),
            "group_consistency_mean": float(metrics.get("group_consistency_mean", 0.0)),
            "mean_eeg_windows_per_trial": _mean_unique_windows(scored_df),
        }
        row["selection_score"] = _selection_score(pd.Series(row))
        summary_rows.append(row)

        with open(experiment_dir / f"{run_id}_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "summary": row,
                    "pipeline_metrics": metrics,
                    "top_composers": metrics.get("top_composers", []),
                    "top_works": metrics.get("top_works", []),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

        if not confusion.empty:
            confusion.to_csv(experiment_dir / f"{run_id}_confusion_matrix.csv")
        if not cohort_df.empty:
            cohort_df.to_csv(experiment_dir / f"{run_id}_cohort_summary.csv", index=False)
        if not feature_summary.empty:
            feature_summary.to_csv(experiment_dir / f"{run_id}_feature_summary.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selection_score", "macro_f1", "emotion_match_rate", "mean_top1_music_match"],
        ascending=False,
    )
    summary_csv = experiment_dir / "window_size_summary.csv"
    summary_json = experiment_dir / "window_size_summary.json"
    summary_md = experiment_dir / "window_size_summary.md"
    summary_png = experiment_dir / "window_size_summary.png"

    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", force_ascii=False, indent=2)
    _save_markdown(summary_df, summary_md)
    _save_plot(summary_df.sort_values("window_size"), summary_png)

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {summary_md}")
    print(f"  {summary_png}")


if __name__ == "__main__":
    main()
