"""
Менеджер истории сравнений (JSON-хранилище).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


_HISTORY_FILE = Path(__file__).resolve().parent.parent / "runs" / "history.json"


def _ensure_file():
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _HISTORY_FILE.exists():
        _HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history() -> list[dict]:
    """Возвращает список записей [{id, timestamp, label, eeg_files, params, report_dir, n_results, best_score}]."""
    _ensure_file()
    try:
        data = json.loads(_HISTORY_FILE.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_run(
    eeg_files: list[str],
    params: dict,
    report_dir: str,
    n_results: int,
    best_score: float,
    label: Optional[str] = None,
) -> dict:
    """Добавляет запись о прогоне и возвращает её."""
    history = load_history()
    run_id = max((r.get("id", 0) for r in history), default=0) + 1
    ts = datetime.now().isoformat(timespec="seconds")
    if not label:
        filenames = [Path(f).name for f in eeg_files[:3]]
        label = ", ".join(filenames) + (f" +{len(eeg_files)-3}" if len(eeg_files) > 3 else "")
        if not label:
            label = f"Запуск #{run_id}"
    entry = {
        "id": run_id,
        "timestamp": ts,
        "label": label,
        "eeg_files": eeg_files,
        "params": params,
        "report_dir": report_dir,
        "n_results": n_results,
        "best_score": round(best_score, 5),
    }
    history.insert(0, entry)
    # Храним до 50 последних
    history = history[:50]
    _ensure_file()
    _HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")
    return entry


def delete_run(run_id: int):
    """Удалить запись по ID."""
    history = load_history()
    history = [r for r in history if r.get("id") != run_id]
    _ensure_file()
    _HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")
