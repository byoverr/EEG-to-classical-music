#!/usr/bin/env python3
"""
Псевдо-разметка MAESTRO по эмоциям (HVHA/HVLA/LVHA/LVLA).

Модель: Ensemble (GradientBoosting + RandomForest) с soft voting.
Признаки: 40-мерный вектор из track_features v2
  (pitch, velocity, duration, IOI/rhythm, intervals, register, key/mode,
   harmony, pitch class histogram).
"""
import argparse
import sys
from pathlib import Path
import random
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from src.config import MAESTRO_DIR, EMOPIA_DIR, MAESTRO_PSEUDO_LABELS_PATH
from src.maestro_loader import get_maestro_midi_files
from src.emopia_loader import get_emopia_midi_files, get_emopia_metadata
from src.track_features import (
    load_feature_cache,
    save_feature_cache,
    get_or_compute_features,
    features_to_vector,
)

MODEL_NAME = "ensemble_v3_40dim"


def build_features(midi_paths: list, cache_path: Path):
    """Извлекает 40-мерные признаки из track_features v2 с кэшированием."""
    cache = load_feature_cache(cache_path)
    X = []
    keys = []
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
    return np.array(X, dtype=float) if X else np.array([]), keys, cache


def build_model(seed: int = 42):
    """Строит ансамбль GradientBoosting + RandomForest с StandardScaler."""
    gb = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft',
    )
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ensemble),
    ])


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label MAESTRO with EMOPIA emotions (Ensemble v3)")
    parser.add_argument('--maestro-dir', type=str, default=str(MAESTRO_DIR))
    parser.add_argument('--emopia-dir', type=str, default=str(EMOPIA_DIR))
    parser.add_argument('--output', type=str, default=str(MAESTRO_PSEUDO_LABELS_PATH))
    parser.add_argument('--max-maestro', type=int, default=None)
    parser.add_argument('--max-emopia', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--cache-dir', type=str, default=str(_project_root / "data" / "cache"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emopia_cache = cache_dir / "emopia_track_features.json"
    maestro_cache = cache_dir / "maestro_track_features.json"

    maestro_files = get_maestro_midi_files(args.maestro_dir, max_files=None)
    emopia_files = get_emopia_midi_files(args.emopia_dir, max_files=None)

    if args.max_maestro:
        random.shuffle(maestro_files)
        maestro_files = maestro_files[:args.max_maestro]
    if args.max_emopia:
        random.shuffle(emopia_files)
        emopia_files = emopia_files[:args.max_emopia]

    print(f"EMOPIA files: {len(emopia_files)} | MAESTRO files: {len(maestro_files)}")

    # 1) Build EMOPIA training set using rich v2 features (40-dim)
    X_emopia, emopia_keys, _ = build_features(emopia_files, emopia_cache)
    y_emopia = []
    valid_indices = []
    for i, p in enumerate(emopia_keys):
        emotion = get_emopia_metadata(Path(p).stem).get('emotion', None)
        if emotion is None:
            continue
        y_emopia.append(emotion)
        valid_indices.append(i)

    if not y_emopia:
        raise RuntimeError("No EMOPIA features computed — check EMOPIA metadata")

    X_emopia = X_emopia[valid_indices]
    y_emopia = np.array(y_emopia)

    unique, counts = np.unique(y_emopia, return_counts=True)
    print(f"\nEMOPIA training: {len(y_emopia)} samples")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt/len(y_emopia)*100:.1f}%)")

    # 2) Cross-validation (5-fold stratified)
    print(f"\n5-fold cross-validation...")
    model = build_model(args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_acc = cross_val_score(model, X_emopia, y_emopia, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_f1 = cross_val_score(model, X_emopia, y_emopia, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"  CV accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}  (per fold: {', '.join(f'{x:.3f}' for x in cv_acc)})")
    print(f"  CV macro_f1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}  (per fold: {', '.join(f'{x:.3f}' for x in cv_f1)})")

    # 3) Train final model on all EMOPIA data
    print(f"\nTraining final model on all {len(y_emopia)} EMOPIA samples...")
    model = build_model(args.seed)
    model.fit(X_emopia, y_emopia)

    # 4) Predict emotions for MAESTRO
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if args.resume and out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            existing_ids = set(existing_df['track_id'].astype(str).tolist())
            print(f"Resume: {len(existing_ids)} already labeled")
        except Exception:
            existing_ids = set()

    new_paths = [p for p in maestro_files if Path(p).stem not in existing_ids]
    if not new_paths:
        print("All MAESTRO files already labeled. Nothing to do.")
        return

    print(f"Predicting: {len(new_paths)} new MAESTRO files...")

    X_maestro, maestro_keys, _ = build_features(new_paths, maestro_cache)
    if len(maestro_keys) == 0:
        print("No MAESTRO features computed.")
        return

    proba = model.predict_proba(X_maestro)
    pred = model.predict(X_maestro)
    conf = proba.max(axis=1)

    # Статистика предсказаний
    pred_unique, pred_counts = np.unique(pred, return_counts=True)
    print(f"\nMAESTRO prediction distribution:")
    for cls, cnt in zip(pred_unique, pred_counts):
        print(f"  {cls}: {cnt} ({cnt/len(pred)*100:.1f}%)")
    print(f"  Mean confidence: {conf.mean():.3f}")

    rows = []
    batch_rows = []
    for p, e, c in zip(maestro_keys, pred, conf):
        row = {
            'track_id': Path(p).stem,
            'midi_path': str(p),
            'emotion': e,
            'confidence': float(c),
            'emotion_model': MODEL_NAME,
        }
        batch_rows.append(row)
        if len(batch_rows) >= args.save_every:
            pd.DataFrame(batch_rows).to_csv(
                out_path, index=False, mode='a',
                header=not out_path.exists()
            )
            rows.extend(batch_rows)
            batch_rows = []

    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(
            out_path, index=False, mode='a',
            header=not out_path.exists()
        )
        rows.extend(batch_rows)

    print(f"\nSaved: {out_path} (+{len(rows)} tracks)")
    print(f"Model: {MODEL_NAME} (GB-500 + RF-500 ensemble, 40-dim features)")


if __name__ == '__main__':
    main()
