#!/usr/bin/env python3
"""
Эксперимент: тестирование разных размеров окон анализа EEG.

Запускает пайплайн с различными значениями COMPARISON_WINDOW_SIZE
(например, 10, 20, 30, 40, 50 секунд) и собирает метрики качества
для определения оптимального размера окна.

Запуск:
    python scripts/window_size_experiment.py
    python scripts/window_size_experiment.py --windows 5 10 15 20 30 --participants 2 --trials 3
    python scripts/window_size_experiment.py --output experiments/window_test

Это отдельный экспериментальный скрипт, НЕ входящий в десктопное приложение.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Добавляем корень проекта в путь
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))


def run_single_window_experiment(
    window_size: float,
    hop_size: float,
    max_participants: int,
    max_trials: int,
    max_classical: int,
    n_jobs: Optional[int],
    only_emopia: bool,
) -> dict:
    """
    Запускает один прогон пайплайна с заданным размером окна.

    Возвращает словарь с агрегированными метриками:
    - window_size, hop_size
    - mean/std/max/min combined_similarity
    - mean contour, interval, harmony, sfi similarity
    - total_results, elapsed_seconds
    """
    import src.config as cfg

    # Переопределяем размеры окна
    original_ws = cfg.COMPARISON_WINDOW_SIZE
    original_hs = cfg.COMPARISON_HOP_SIZE
    cfg.COMPARISON_WINDOW_SIZE = window_size
    cfg.COMPARISON_HOP_SIZE = hop_size

    from scripts.run_comparison import run_comparison

    t0 = time.time()
    try:
        results_df = run_comparison(
            max_participants=max_participants,
            max_trials=max_trials,
            max_classical=max_classical,
            top_k=50,  # больше для статистики
            n_jobs=n_jobs,
            only_emopia=only_emopia,
            match_emotions=False,
        )
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        cfg.COMPARISON_WINDOW_SIZE = original_ws
        cfg.COMPARISON_HOP_SIZE = original_hs
        return {
            "window_size": window_size,
            "hop_size": hop_size,
            "error": str(e),
        }
    finally:
        cfg.COMPARISON_WINDOW_SIZE = original_ws
        cfg.COMPARISON_HOP_SIZE = original_hs

    elapsed = time.time() - t0

    if results_df is None or results_df.empty:
        return {
            "window_size": window_size,
            "hop_size": hop_size,
            "total_results": 0,
            "elapsed_seconds": elapsed,
            "error": "No results",
        }

    # Агрегированные метрики
    metrics = {
        "window_size": window_size,
        "hop_size": hop_size,
        "total_results": len(results_df),
        "elapsed_seconds": round(elapsed, 1),
    }

    for col in [
        "combined_similarity",
        "contour_similarity",
        "interval_similarity",
        "harmony_similarity",
        "sfi_similarity",
    ]:
        if col in results_df.columns:
            vals = results_df[col].dropna()
            metrics[f"{col}_mean"] = round(float(vals.mean()), 5)
            metrics[f"{col}_std"] = round(float(vals.std()), 5)
            metrics[f"{col}_max"] = round(float(vals.max()), 5)
            metrics[f"{col}_min"] = round(float(vals.min()), 5)
            metrics[f"{col}_median"] = round(float(vals.median()), 5)

    # Дополнительные метрики качества
    if "eeg_note_count" in results_df.columns and "cla_note_count" in results_df.columns:
        note_ratios = results_df.apply(
            lambda r: min(r["eeg_note_count"], r["cla_note_count"])
            / max(r["eeg_note_count"], r["cla_note_count"], 1),
            axis=1,
        )
        metrics["note_ratio_mean"] = round(float(note_ratios.mean()), 5)

    if "eeg_emotion" in results_df.columns and "classical_emotion" in results_df.columns:
        matches = (results_df["eeg_emotion"] == results_df["classical_emotion"]).sum()
        total = len(results_df.dropna(subset=["eeg_emotion", "classical_emotion"]))
        metrics["emotion_match_rate"] = round(matches / max(total, 1), 4)

    return metrics


def plot_experiment_results(all_metrics: list[dict], output_dir: Path):
    """Строит графики сравнения метрик по размерам окон."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(all_metrics)

    if df.empty or "window_size" not in df.columns:
        print("Нет данных для визуализации")
        return

    window_sizes = df["window_size"].values

    # --- 1) Main metrics comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Влияние размера окна на метрики сходства", fontsize=16, fontweight="bold")

    metric_pairs = [
        ("combined_similarity_mean", "Combined Similarity (mean)"),
        ("contour_similarity_mean", "Contour Similarity (mean)"),
        ("harmony_similarity_mean", "Harmony Similarity (mean)"),
        ("interval_similarity_mean", "Interval Similarity (mean)"),
    ]

    for ax, (col, title) in zip(axes.flat, metric_pairs):
        if col not in df.columns:
            ax.set_title(f"{title} — нет данных")
            continue
        vals = df[col].values
        std_col = col.replace("_mean", "_std")
        stds = df[std_col].values if std_col in df.columns else None

        ax.plot(window_sizes, vals, "o-", linewidth=2, markersize=8, color="#1a73e8")
        if stds is not None:
            ax.fill_between(window_sizes, vals - stds, vals + stds, alpha=0.15, color="#1a73e8")
        ax.set_xlabel("Размер окна (сек)")
        ax.set_ylabel("Сходство")
        ax.set_title(title)
        ax.set_xticks(window_sizes)

        # Отметим лучшее значение
        best_idx = int(np.argmax(vals))
        ax.annotate(
            f"best: {vals[best_idx]:.4f}",
            xy=(window_sizes[best_idx], vals[best_idx]),
            xytext=(10, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red", fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(output_dir / "window_metrics_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  График: {output_dir / 'window_metrics_comparison.png'}")

    # --- 2) Execution time ---
    if "elapsed_seconds" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(
            [str(w) for w in window_sizes],
            df["elapsed_seconds"].values,
            color=sns.color_palette("Blues_d", len(window_sizes)),
        )
        ax2.set_xlabel("Размер окна (сек)")
        ax2.set_ylabel("Время (сек)")
        ax2.set_title("Время выполнения в зависимости от размера окна")
        plt.tight_layout()
        fig2.savefig(str(output_dir / "window_execution_time.png"), dpi=150)
        plt.close(fig2)
        print(f"  График: {output_dir / 'window_execution_time.png'}")

    # --- 3) Radar chart for best window ---
    try:
        radar_metrics = [
            "combined_similarity_mean",
            "contour_similarity_mean",
            "interval_similarity_mean",
            "harmony_similarity_mean",
        ]
        available = [m for m in radar_metrics if m in df.columns]
        if len(available) >= 3:
            fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
            angles += angles[:1]

            for _, row in df.iterrows():
                vals = [row[m] for m in available] + [row[available[0]]]
                label = f"window={row['window_size']}s"
                ax3.plot(angles, vals, "o-", linewidth=2, label=label)
                ax3.fill(angles, vals, alpha=0.05)

            labels = [m.replace("_similarity_mean", "").replace("_", " ").title() for m in available]
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(labels)
            ax3.set_title("Radar: метрики по размерам окон", fontsize=14, pad=20)
            ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
            plt.tight_layout()
            fig3.savefig(str(output_dir / "window_radar.png"), dpi=150)
            plt.close(fig3)
            print(f"  График: {output_dir / 'window_radar.png'}")
    except Exception as e:
        print(f"  Radar chart error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Эксперимент: тестирование размеров окон анализа EEG"
    )
    parser.add_argument(
        "--windows", nargs="+", type=float,
        default=[2.0, 4.0, 8.0, 10.0, 20.0, 30.0],
        help="Размеры окон для тестирования (в секундах)",
    )
    parser.add_argument(
        "--hop-ratio", type=float, default=0.5,
        help="Отношение hop/window (по умолчанию 0.5, т.е. 50%% перекрытие)",
    )
    parser.add_argument(
        "--participants", type=int, default=2,
        help="Количество участников DEAP",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Количество триалов на участника",
    )
    parser.add_argument(
        "--classical", type=int, default=10,
        help="Количество классических произведений",
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Количество процессов (None=авто)",
    )
    parser.add_argument(
        "--only-emopia", action="store_true",
        help="Только EMOPIA",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Директория для результатов (по умолчанию: runs/window_experiment)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path("runs") / "window_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    window_sizes = sorted(set(args.windows))
    print("=" * 60)
    print("Window Size Experiment")
    print("=" * 60)
    print(f"  Окна: {window_sizes}")
    print(f"  Hop ratio: {args.hop_ratio}")
    print(f"  Участники: {args.participants}, Триалы: {args.trials}")
    print(f"  Классических: {args.classical}")
    print(f"  Выход: {output_dir}")
    print("=" * 60)

    all_metrics = []
    for i, ws in enumerate(window_sizes, 1):
        hs = round(ws * args.hop_ratio, 2)
        print(f"\n{'─' * 50}")
        print(f"[{i}/{len(window_sizes)}] Window = {ws}s, Hop = {hs}s")
        print(f"{'─' * 50}")

        metrics = run_single_window_experiment(
            window_size=ws,
            hop_size=hs,
            max_participants=args.participants,
            max_trials=args.trials,
            max_classical=args.classical,
            n_jobs=args.jobs,
            only_emopia=args.only_emopia,
        )
        all_metrics.append(metrics)

        # Промежуточное сохранение
        interim_df = pd.DataFrame(all_metrics)
        interim_df.to_csv(output_dir / "experiment_results_interim.csv", index=False)

        # Print summary
        cs = metrics.get("combined_similarity_mean", "N/A")
        elapsed = metrics.get("elapsed_seconds", "N/A")
        total = metrics.get("total_results", 0)
        print(f"  → combined_mean={cs}, results={total}, time={elapsed}s")

    # Итоговые результаты
    print("\n" + "=" * 60)
    print("ИТОГИ ЭКСПЕРИМЕНТА")
    print("=" * 60)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "experiment_results.csv", index=False)
    print(f"\nCSV: {output_dir / 'experiment_results.csv'}")

    # JSON
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"JSON: {output_dir / 'experiment_results.json'}")

    # Таблица результатов
    display_cols = [
        "window_size", "hop_size", "total_results", "elapsed_seconds",
        "combined_similarity_mean", "combined_similarity_std",
        "contour_similarity_mean", "harmony_similarity_mean",
    ]
    avail_cols = [c for c in display_cols if c in results_df.columns]
    print(f"\n{results_df[avail_cols].to_string(index=False)}")

    # Лучший размер окна
    if "combined_similarity_mean" in results_df.columns:
        best_row = results_df.loc[results_df["combined_similarity_mean"].idxmax()]
        print(f"\n★ Лучший размер окна: {best_row['window_size']}s "
              f"(combined_mean = {best_row['combined_similarity_mean']:.5f})")

    # Графики
    print("\nГенерация графиков…")
    plot_experiment_results(all_metrics, output_dir)

    # Markdown отчёт
    md_path = output_dir / "experiment_report.md"
    with open(md_path, "w") as f:
        f.write("# Window Size Experiment Report\n\n")
        f.write(f"**Parameters:** participants={args.participants}, "
                f"trials={args.trials}, classical={args.classical}, "
                f"hop_ratio={args.hop_ratio}\n\n")
        f.write("## Results\n\n")
        f.write(results_df[avail_cols].to_markdown(index=False))
        f.write("\n\n")
        if "combined_similarity_mean" in results_df.columns:
            best_row = results_df.loc[results_df["combined_similarity_mean"].idxmax()]
            f.write(f"## Best Window Size\n\n")
            f.write(f"**{best_row['window_size']}s** — combined_similarity_mean = "
                    f"{best_row['combined_similarity_mean']:.5f}\n\n")
        f.write("## Charts\n\n")
        f.write("![Metrics Comparison](window_metrics_comparison.png)\n\n")
        f.write("![Execution Time](window_execution_time.png)\n\n")
        f.write("![Radar](window_radar.png)\n\n")
    print(f"Markdown: {md_path}")

    print("\n✓ Эксперимент завершён!")


if __name__ == "__main__":
    main()
