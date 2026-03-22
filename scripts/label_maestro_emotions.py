#!/usr/bin/env python3
"""
Разметка MAESTRO на 4 квадранта эмоций (HVHA/HVLA/LVHA/LVLA).

Алгоритм:
1) Берём EMOPIA как источник разметки.
2) Извлекаем набор устойчивых символических признаков (track_features).
3) Обучаем модель (RandomForest, balanced) и оцениваем на holdout.
4) Предсказываем эмоции для MAESTRO.
5) Записываем результат прямо в maestro-v3.0.0.csv (с резервной копией).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from src.config import MAESTRO_DIR, EMOPIA_DIR
from src.maestro_loader import get_maestro_midi_files
from src.emopia_loader import get_emopia_midi_files, get_emopia_metadata
from src.track_features import (
    load_feature_cache,
    save_feature_cache,
    get_or_compute_features,
    features_to_vector,
)

MODEL_NAME = "ensemble_v3_40dim"


def build_features(
    midi_paths: list[str],
    cache_path: Path,
) -> Tuple[np.ndarray, list[str], Dict[str, dict]]:
    cache = load_feature_cache(cache_path)
    X = []
    keys = []
    for p in midi_paths:
        feats = get_or_compute_features(p, cache)
        if feats is None:
            continue
        vec = features_to_vector(feats)
        if not np.isfinite(vec).all():
            continue
        X.append(vec)
        keys.append(p)
    save_feature_cache(cache_path, cache)
    return np.array(X, dtype=float), keys, cache


def main():
    parser = argparse.ArgumentParser(description="Label MAESTRO emotions from EMOPIA")
    parser.add_argument("--maestro-dir", type=str, default=str(MAESTRO_DIR))
    parser.add_argument("--emopia-dir", type=str, default=str(EMOPIA_DIR))
    parser.add_argument("--max-maestro", type=int, default=None)
    parser.add_argument("--max-emopia", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=str(_project_root / "data" / "cache"))
    parser.add_argument("--output", type=str, default=str(_project_root / "runs" / "run_001" / "report" / "maestro_emotion_labels.csv"))
    parser.add_argument("--write-back", action="store_true", default=True)
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emopia_cache = cache_dir / "emopia_track_features.json"
    maestro_cache = cache_dir / "maestro_track_features.json"

    emopia_files = get_emopia_midi_files(args.emopia_dir, max_files=None)
    maestro_files = get_maestro_midi_files(args.maestro_dir, max_files=None)

    if args.max_emopia:
        random.shuffle(emopia_files)
        emopia_files = emopia_files[:args.max_emopia]
    if args.max_maestro:
        random.shuffle(maestro_files)
        maestro_files = maestro_files[:args.max_maestro]

    print(f"EMOPIA files: {len(emopia_files)} | MAESTRO files: {len(maestro_files)}")

    # 1) Build EMOPIA training set
    X_emopia, emopia_keys, _ = build_features(emopia_files, emopia_cache)
    y_emopia = []
    valid_keys = []
    for p in emopia_keys:
        emotion = get_emopia_metadata(Path(p).stem).get("emotion")
        if emotion is None:
            continue
        y_emopia.append(emotion)
        valid_keys.append(p)

    if not y_emopia:
        raise RuntimeError("No EMOPIA labels found. Check EMOPIA metadata.")

    X_emopia = X_emopia[: len(y_emopia)]
    y_emopia = np.array(y_emopia)

    # 2) Train model + quick validation
    unique, counts = np.unique(y_emopia, return_counts=True)
    min_class = counts.min() if len(counts) > 0 else 0
    can_split = min_class >= 2 and len(y_emopia) >= 10

    gb = GradientBoostingClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=args.seed,
    )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight='balanced_subsample',
        min_samples_leaf=3, random_state=args.seed, n_jobs=-1,
    )
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(
            estimators=[('gb', gb), ('rf', rf)], voting='soft',
        )),
    ])

    if can_split:
        # 5-fold CV для честной оценки
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_acc = cross_val_score(model, X_emopia, y_emopia, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_emopia, y_emopia, cv=cv, scoring='f1_macro', n_jobs=-1)
        print(f"CV accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"CV macro_f1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_emopia, y_emopia, test_size=0.2,
            stratify=y_emopia, random_state=args.seed,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        print(f"Holdout: accuracy={acc:.3f}, macro_f1={f1:.3f}")
    else:
        model.fit(X_emopia, y_emopia)
        print("Validation skipped: not enough labeled samples per class")

    # 3) Predict MAESTRO
    X_maestro, maestro_keys, _ = build_features(maestro_files, maestro_cache)
    if len(maestro_keys) == 0:
        raise RuntimeError("No MAESTRO features computed.")

    proba = model.predict_proba(X_maestro)
    pred = model.predict(X_maestro)
    conf = proba.max(axis=1)

    rows = []
    for p, e, c in zip(maestro_keys, pred, conf):
        rows.append(
            {
                "track_id": Path(p).stem,
                "midi_path": str(p),
                "emotion": e,
                "emotion_source": "predicted",
                "emotion_confidence": float(c),
                "emotion_model": MODEL_NAME,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path} ({len(rows)} tracks)")

    if args.write_back:
        csv_path = Path(args.maestro_dir) / "maestro-v3.0.0.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"MAESTRO CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        for col in [
            "emotion",
            "emotion_source",
            "emotion_confidence",
            "emotion_model",
        ]:
            if col not in df.columns:
                df[col] = ""

        track_map = {r["track_id"]: r for r in rows}
        emotions = []
        sources = []
        confs = []
        models = []
        for _, row in df.iterrows():
            track_id = Path(row["midi_filename"]).stem
            info = track_map.get(track_id)
            if info:
                emotions.append(info["emotion"])
                sources.append(info["emotion_source"])
                confs.append(info["emotion_confidence"])
                models.append(info["emotion_model"])
            else:
                emotions.append(row.get("emotion", ""))
                sources.append(row.get("emotion_source", ""))
                confs.append(row.get("emotion_confidence", ""))
                models.append(row.get("emotion_model", ""))

        df["emotion"] = emotions
        df["emotion_source"] = sources
        df["emotion_confidence"] = confs
        df["emotion_model"] = models

        backup_path = csv_path.with_suffix(".csv.bak")
        if not backup_path.exists():
            csv_path.replace(backup_path)
            print(f"Backup created: {backup_path}")

        df.to_csv(csv_path, index=False)
        print(f"Updated MAESTRO metadata: {csv_path}")


if __name__ == "__main__":
    main()
