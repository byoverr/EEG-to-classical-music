#!/usr/bin/env python3
"""
Псевдо-разметка MAESTRO по эмоциям (HVHA/HVLA/LVHA/LVLA) — версия v4 (stacked).

Улучшения по сравнению с `pseudo_label_maestro.py`:
  * GroupKFold по youtube_id (предотвращает утечку между клипами одного видео).
  * StackingClassifier: HistGB + GB + ExtraTrees + RandomForest + LogReg → LogReg мета.
  * class_weight='balanced' там, где поддерживается.
  * Дополнительно StratifiedKFold — для сравнения и демонстрации утечки в ВКР.
  * Полный отчёт: per-class precision/recall/F1, balanced accuracy, Cohen's κ,
    confusion matrix, Markdown-отчёт пригодный для главы 3 ВКР.

Выход: `runs/maestro_labeling_<timestamp>/`
  * report.md, manifest.json, maestro_predictions.csv
  * cv_grouped_folds.csv, cv_stratified_folds.csv
  * per_class_metrics.csv, confusion_matrix_grouped.csv
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from src.config import MAESTRO_DIR, EMOPIA_DIR, RUNS_DIR
from src.maestro_loader import get_maestro_midi_files
from src.emopia_loader import get_emopia_midi_files, get_emopia_metadata
from src.track_features import (
    load_feature_cache,
    save_feature_cache,
    get_or_compute_features,
    features_to_vector,
)

MODEL_NAME = "stacked_v4_40dim"
EMOTION_ORDER = ("HVHA", "HVLA", "LVLA", "LVHA")


# ──────────────────────────────────────────────────────────────────────────
# Фичи
# ──────────────────────────────────────────────────────────────────────────

def build_features(midi_paths: list, cache_path: Path) -> Tuple[np.ndarray, list]:
    cache = load_feature_cache(cache_path)
    X, keys = [], []
    for p in midi_paths:
        feats = get_or_compute_features(str(p), cache)
        if feats is None:
            continue
        vec = features_to_vector(feats)
        if not np.isfinite(vec).all():
            continue
        X.append(vec)
        keys.append(str(p))
    save_feature_cache(cache_path, cache)
    if not X:
        return np.empty((0, 40), dtype=float), keys
    return np.asarray(X, dtype=float), keys


def _extract_youtube_id(midi_path: str) -> str:
    """EMOPIA: Q<n>_<youtube_id>_<clip>.mid → youtube_id."""
    stem = Path(midi_path).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[1]
    return stem


# ──────────────────────────────────────────────────────────────────────────
# Модель
# ──────────────────────────────────────────────────────────────────────────

def build_model(seed: int = 42) -> Pipeline:
    """Stacked ensemble: 5 базовых моделей + LogReg мета."""
    hgb = HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=None,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
    )
    gb = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    meta = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    stack = StackingClassifier(
        estimators=[
            ("hgb", hgb),
            ("gb", gb),
            ("et", et),
            ("rf", rf),
            ("lr", lr),
        ],
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", stack),
    ])


# ──────────────────────────────────────────────────────────────────────────
# CV
# ──────────────────────────────────────────────────────────────────────────

def _summarize(df: pd.DataFrame) -> dict:
    return {
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std()),
        "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(df["balanced_accuracy"].std()),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std()),
        "kappa_mean": float(df["kappa"].mean()),
        "kappa_std": float(df["kappa"].std()),
    }


def _fold_metrics(y_true, y_pred, fold_idx, n_train, n_val) -> dict:
    return {
        "fold": fold_idx,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def group_cv_report(X, y, groups, seed: int, n_splits: int = 5) -> dict:
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows, all_t, all_p = [], [], []
    for fold_idx, (tr, vl) in enumerate(splitter.split(X, y, groups), 1):
        model = build_model(seed=seed + fold_idx)
        model.fit(X[tr], y[tr])
        yp = model.predict(X[vl])
        fold_rows.append(_fold_metrics(y[vl], yp, fold_idx, len(tr), len(vl)))
        all_t.extend(y[vl].tolist()); all_p.extend(yp.tolist())
    df = pd.DataFrame(fold_rows)
    labels = [l for l in EMOTION_ORDER if l in set(all_t) | set(all_p)]
    cm = confusion_matrix(all_t, all_p, labels=labels).tolist()
    report = classification_report(all_t, all_p, labels=labels,
                                   zero_division=0, output_dict=True)
    return {
        "folds": df,
        "summary": _summarize(df),
        "labels": labels,
        "confusion": cm,
        "classification_report": report,
    }


def stratified_cv_report(X, y, seed: int, n_splits: int = 5) -> dict:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_idx, (tr, vl) in enumerate(splitter.split(X, y), 1):
        model = build_model(seed=seed + fold_idx)
        model.fit(X[tr], y[tr])
        yp = model.predict(X[vl])
        fold_rows.append(_fold_metrics(y[vl], yp, fold_idx, len(tr), len(vl)))
    df = pd.DataFrame(fold_rows)
    return {"folds": df, "summary": _summarize(df)}


# ──────────────────────────────────────────────────────────────────────────
# Markdown
# ──────────────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def save_markdown_report(report_dir, emopia_stats, stratified, grouped,
                         maestro_dist, maestro_conf, model_name):
    L = [
        "# Псевдо-разметка MAESTRO: отчёт (v4 stacked)",
        "",
        f"Модель: `{model_name}` (Stacking: HistGB + GB + ET + RF + LogReg → LogReg).",
        f"Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Обучающая выборка EMOPIA",
        "",
        f"- Треков с метками: **{emopia_stats['n_samples']}**",
        f"- Уникальных YouTube-роликов: **{emopia_stats['n_groups']}**",
        f"- Клипов на ролик (в среднем): **{emopia_stats['clips_per_group']:.1f}**",
        "",
        "Распределение классов EMOPIA:",
        "",
        "| Эмоция | Треков | Доля |",
        "| --- | --- | --- |",
    ]
    for emo, cnt in emopia_stats["class_counts"].items():
        share = cnt / emopia_stats["n_samples"]
        L.append(f"| {emo} | {cnt} | {_pct(share)} |")

    L += [
        "",
        "## 2. Кросс-валидация — Stratified vs GroupKFold",
        "",
        "- **StratifiedKFold** — случайное разбиение. Клипы одного YouTube-ролика могут попасть "
        "и в train, и в val одновременно → **утечка** и завышенные метрики.",
        "- **GroupKFold by youtube_id** — честное разбиение: все клипы одного ролика идут только "
        "в одну часть. Это и есть правильная оценка обобщения.",
        "",
        "Разница между схемами показывает масштаб утечки в предыдущей версии.",
        "",
        "| Схема | Accuracy | Balanced Acc | Macro-F1 | Cohen's κ |",
        "| --- | --- | --- | --- | --- |",
    ]
    if stratified.get("summary"):
        s = stratified["summary"]
        L.append(
            f"| Stratified (оптимистическая) | "
            f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} | "
            f"{s['balanced_accuracy_mean']:.3f} ± {s['balanced_accuracy_std']:.3f} | "
            f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} | "
            f"{s['kappa_mean']:.3f} ± {s['kappa_std']:.3f} |"
        )
    g = grouped["summary"]
    L.append(
        f"| **GroupKFold by youtube_id** | "
        f"**{g['accuracy_mean']:.3f} ± {g['accuracy_std']:.3f}** | "
        f"**{g['balanced_accuracy_mean']:.3f} ± {g['balanced_accuracy_std']:.3f}** | "
        f"**{g['macro_f1_mean']:.3f} ± {g['macro_f1_std']:.3f}** | "
        f"**{g['kappa_mean']:.3f} ± {g['kappa_std']:.3f}** |"
    )

    L += [
        "",
        "Именно цифры GroupKFold корректно отражают качество модели на новом материале.",
        "",
        "### 2.1 Confusion matrix (GroupKFold)",
        "",
    ]
    labels = grouped["labels"]
    cm = grouped["confusion"]
    L.append("| true \\ pred | " + " | ".join(labels) + " |")
    L.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for i, row in enumerate(cm):
        L.append(f"| **{labels[i]}** | " + " | ".join(str(v) for v in row) + " |")

    L += ["", "### 2.2 Per-class metrics (GroupKFold)", "",
          "| Эмоция | Precision | Recall | F1 | Support |",
          "| --- | --- | --- | --- | --- |"]
    cr = grouped["classification_report"]
    for lbl in labels:
        r = cr.get(lbl, {})
        L.append(
            f"| {lbl} | {r.get('precision', 0.0):.3f} | "
            f"{r.get('recall', 0.0):.3f} | {r.get('f1-score', 0.0):.3f} | "
            f"{int(r.get('support', 0))} |"
        )

    if maestro_dist:
        L += [
            "",
            "## 3. Предсказания на MAESTRO",
            "",
            f"Всего обработано треков: **{maestro_dist['n']}**",
            f"Средняя уверенность: **{maestro_conf['mean']:.3f}** (std {maestro_conf['std']:.3f})",
            f"Треков с confidence ≥ 0.6: **{_pct(maestro_conf['high_conf_ratio'])}**",
            "",
            "| Эмоция | Треков | Доля |",
            "| --- | --- | --- |",
        ]
        for emo, cnt in maestro_dist["classes"].items():
            share = cnt / maestro_dist["n"]
            L.append(f"| {emo} | {cnt} | {_pct(share)} |")

    L += [
        "",
        "## 4. Как интерпретировать",
        "",
        "- Разница Stratified ↔ GroupKFold = величина утечки в предыдущей оценке.",
        "- Balanced accuracy и macro-F1 предпочтительнее accuracy при несбалансированных классах.",
        "- Confusion matrix вскрывает системные ошибки модели (какие пары эмоций путаются).",
        "- Метки MAESTRO остаются **предсказанными** — их качество ожидается на уровне GroupKFold CV.",
        "",
    ]

    (report_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Improved MAESTRO pseudo-labeling (v4)")
    parser.add_argument("--maestro-dir", type=str, default=str(MAESTRO_DIR))
    parser.add_argument("--emopia-dir", type=str, default=str(EMOPIA_DIR))
    parser.add_argument("--max-maestro", type=int, default=None)
    parser.add_argument("--max-emopia", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=str(_project_root / "data" / "cache"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--write-back", action="store_true", default=True)
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    parser.add_argument("--skip-stratified", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or f"maestro_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir = RUNS_DIR / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Отчёт: {report_dir}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emopia_cache = cache_dir / "emopia_track_features.json"
    maestro_cache = cache_dir / "maestro_track_features.json"

    emopia_files = get_emopia_midi_files(args.emopia_dir, max_files=None)
    maestro_files = get_maestro_midi_files(args.maestro_dir, max_files=None)
    if args.max_emopia:
        emopia_files = emopia_files[:args.max_emopia]
    if args.max_maestro:
        maestro_files = maestro_files[:args.max_maestro]
    print(f"EMOPIA: {len(emopia_files)}  MAESTRO: {len(maestro_files)}")

    # 1) EMOPIA features + labels + groups
    print("→ Извлечение признаков EMOPIA...")
    X_all, keys_all = build_features(emopia_files, emopia_cache)
    y, groups, keep = [], [], []
    for i, key in enumerate(keys_all):
        emo = get_emopia_metadata(Path(key).stem).get("emotion")
        if emo is None or emo not in EMOTION_ORDER:
            continue
        y.append(emo)
        groups.append(_extract_youtube_id(key))
        keep.append(i)
    if len(keep) < 20:
        raise RuntimeError("Слишком мало размеченных EMOPIA файлов")
    X = X_all[keep]
    y = np.asarray(y)
    groups = np.asarray(groups)
    cls_unique, cls_counts = np.unique(y, return_counts=True)
    uniq_groups = np.unique(groups)
    print(f"  {len(y)} треков, {len(uniq_groups)} групп, классы: "
          f"{dict(zip(cls_unique.tolist(), cls_counts.tolist()))}")
    emopia_stats = {
        "n_samples": int(len(y)),
        "n_groups": int(len(uniq_groups)),
        "clips_per_group": float(len(y) / max(len(uniq_groups), 1)),
        "class_counts": {str(k): int(v) for k, v in zip(cls_unique, cls_counts)},
    }

    # 2) Stratified CV (compare)
    stratified_res = {"summary": {}}
    if not args.skip_stratified:
        print("→ StratifiedKFold CV (оптимистическая)...")
        stratified_res = stratified_cv_report(X, y, seed=args.seed, n_splits=5)
        s = stratified_res["summary"]
        print(f"  macro-F1: {s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} "
              f"bal-acc: {s['balanced_accuracy_mean']:.3f} κ: {s['kappa_mean']:.3f}")

    # 3) GroupKFold CV
    print("→ GroupKFold CV (честная)...")
    n_splits = min(5, len(uniq_groups))
    grouped_res = group_cv_report(X, y, groups, seed=args.seed, n_splits=n_splits)
    g = grouped_res["summary"]
    print(f"  macro-F1: {g['macro_f1_mean']:.3f} ± {g['macro_f1_std']:.3f} "
          f"bal-acc: {g['balanced_accuracy_mean']:.3f} κ: {g['kappa_mean']:.3f}")

    # Save CSVs
    if not args.skip_stratified:
        stratified_res["folds"].to_csv(report_dir / "cv_stratified_folds.csv", index=False)
    grouped_res["folds"].to_csv(report_dir / "cv_grouped_folds.csv", index=False)
    cm_df = pd.DataFrame(grouped_res["confusion"],
                         index=grouped_res["labels"],
                         columns=grouped_res["labels"])
    cm_df.to_csv(report_dir / "confusion_matrix_grouped.csv")
    per_class_rows = []
    for lbl in grouped_res["labels"]:
        r = grouped_res["classification_report"].get(lbl, {})
        per_class_rows.append({
            "emotion": lbl,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": int(r.get("support", 0)),
        })
    pd.DataFrame(per_class_rows).to_csv(report_dir / "per_class_metrics.csv", index=False)

    # 4) Финальная модель на всей EMOPIA
    print("→ Обучение финальной модели...")
    final_model = build_model(seed=args.seed)
    final_model.fit(X, y)

    # 5) Предсказания MAESTRO
    print("→ Признаки MAESTRO...")
    X_m, m_keys = build_features(maestro_files, maestro_cache)
    if len(m_keys) == 0:
        print("  MAESTRO features пусты — выход")
        return
    print(f"  {len(m_keys)} треков к предсказанию")
    print("→ Предсказание эмоций MAESTRO...")
    proba = final_model.predict_proba(X_m)
    pred = final_model.predict(X_m)
    conf = proba.max(axis=1)

    pu, pc = np.unique(pred, return_counts=True)
    maestro_dist = {"n": int(len(pred)),
                    "classes": {str(k): int(v) for k, v in zip(pu, pc)}}
    high_conf_ratio = float((conf >= 0.6).sum() / len(conf))
    maestro_conf = {
        "mean": float(conf.mean()),
        "std": float(conf.std()),
        "median": float(np.median(conf)),
        "p10": float(np.percentile(conf, 10)),
        "p90": float(np.percentile(conf, 90)),
        "high_conf_ratio": high_conf_ratio,
    }
    print(f"  dist: {maestro_dist['classes']} mean_conf: {maestro_conf['mean']:.3f} "
          f"high≥0.6: {_pct(high_conf_ratio)}")

    preds_df = pd.DataFrame({
        "track_id": [Path(p).stem for p in m_keys],
        "midi_path": m_keys,
        "emotion": pred,
        "emotion_confidence": conf,
        "emotion_source": "predicted",
        "emotion_model": MODEL_NAME,
    })
    preds_df.to_csv(report_dir / "maestro_predictions.csv", index=False)

    # 6) Manifest + markdown
    manifest = {
        "model_name": MODEL_NAME,
        "seed": args.seed,
        "emopia_stats": emopia_stats,
        "cv_stratified": stratified_res["summary"] if not args.skip_stratified else None,
        "cv_grouped_by_youtube_id": grouped_res["summary"],
        "cv_grouped_confusion": {
            "labels": grouped_res["labels"],
            "matrix": grouped_res["confusion"],
        },
        "maestro_distribution": maestro_dist,
        "maestro_confidence": maestro_conf,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_markdown_report(report_dir, emopia_stats,
                         stratified_res if not args.skip_stratified else {"summary": {}},
                         grouped_res, maestro_dist, maestro_conf, MODEL_NAME)

    # 7) Write-back to maestro-v3.0.0.csv
    if args.write_back:
        csv_path = Path(args.maestro_dir) / "maestro-v3.0.0.csv"
        if not csv_path.exists():
            print(f"  MAESTRO CSV не найден: {csv_path}")
        else:
            backup = csv_path.with_suffix(".csv.bak_v4")
            if not backup.exists():
                backup.write_bytes(csv_path.read_bytes())
                print(f"  backup: {backup.name}")
            df = pd.read_csv(csv_path)
            for col in ["emotion", "emotion_source", "emotion_confidence", "emotion_model"]:
                if col not in df.columns:
                    df[col] = ""
            tmap = {Path(r["midi_path"]).stem: r for _, r in preds_df.iterrows()}
            emotions, sources, confs, models = [], [], [], []
            hits = 0
            for _, row in df.iterrows():
                tid = Path(str(row["midi_filename"])).stem
                info = tmap.get(tid)
                if info is not None:
                    emotions.append(info["emotion"])
                    sources.append("predicted")
                    confs.append(float(info["emotion_confidence"]))
                    models.append(MODEL_NAME)
                    hits += 1
                else:
                    emotions.append(row.get("emotion", ""))
                    sources.append(row.get("emotion_source", ""))
                    confs.append(row.get("emotion_confidence", ""))
                    models.append(row.get("emotion_model", ""))
            df["emotion"] = emotions
            df["emotion_source"] = sources
            df["emotion_confidence"] = confs
            df["emotion_model"] = models
            df.to_csv(csv_path, index=False)
            print(f"  CSV обновлён: {hits} треков")

    print(f"\nГотово. Отчёт: {report_dir}")


if __name__ == "__main__":
    main()
