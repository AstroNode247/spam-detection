from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)
    return path


def load_pickle(file_path: str | Path) -> Any:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with path.open("rb") as file:
        return pickle.load(file)
