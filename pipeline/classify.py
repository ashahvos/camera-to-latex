"""Load the trained symbol classifier and run it on crops.

Two back ends share one interface:
  keras    models/<name>.keras  (the real model, needs TensorFlow)
  sklearn  models/<name>.joblib (a small MLP fallback: no TensorFlow needed)
Both come with models/<name>.json listing the classes in output order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from .symbols import SymbolTable, load_symbols


class Classifier:
    def __init__(self, backend: str, model, classes: list[str], info: dict | None = None):
        self.backend = backend
        self.model = model
        self.classes = list(classes)
        self.info = info or {}

    @classmethod
    def load(cls, name: str | None = None, models_dir: str | Path | None = None) -> "Classifier":
        name = name or config.MODEL_NAME
        models_dir = Path(models_dir or config.MODELS_DIR)
        info_path = models_dir / f"{name}.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} not found: train a model first (python -m model.train)")
        info = json.loads(info_path.read_text())
        classes = info["classes"]
        keras_path = models_dir / f"{name}.keras"
        joblib_path = models_dir / f"{name}.joblib"
        if keras_path.exists():
            import tensorflow as tf  # imported here so the rest works without TensorFlow
            model = tf.keras.models.load_model(keras_path)
            return cls("keras", model, classes, info)
        if joblib_path.exists():
            import joblib
            model = joblib.load(joblib_path)
            return cls("sklearn", model, classes, info)
        raise FileNotFoundError(f"no {keras_path.name} or {joblib_path.name} in {models_dir}")

    def check_table(self, table: SymbolTable) -> list[str]:
        """Problems between the model's classes and symbols.csv (empty = fine)."""
        problems = []
        if self.classes != table.classes:
            missing = [c for c in table.classes if c not in self.classes]
            extra = [c for c in self.classes if c not in table.classes]
            problems.append(
                f"model classes differ from symbols.csv (missing {missing[:5]}, extra {extra[:5]}); retrain")
        return problems

    def predict(self, crops) -> np.ndarray:
        """crops: (N, S, S) floats in [0, 1] -> (N, num_classes) probabilities."""
        X = np.asarray(crops, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        n = len(X)
        if n == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        if self.backend == "keras":
            p = self.model.predict(X[..., None], verbose=0, batch_size=256)
            return np.asarray(p, dtype=np.float32)
        feats = sklearn_features(X) if self.info.get("features") == "hog+pix" else X.reshape(n, -1)
        p = self.model.predict_proba(feats)
        full = np.zeros((n, len(self.classes)), np.float32)
        full[:, np.asarray(self.model.classes_, dtype=int)] = p
        return full


_HOG = None


def sklearn_features(X: np.ndarray) -> np.ndarray:
    """Features for the sklearn fallback: HOG (stroke directions) + a 16x16 thumbnail."""
    import cv2
    global _HOG
    if _HOG is None:
        _HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    X = np.asarray(X, dtype=np.float32)
    out = []
    for x in X:
        h = _HOG.compute(np.clip(x * 255, 0, 255).astype(np.uint8)).ravel()
        small = cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA).ravel()
        out.append(np.concatenate([h, small]))
    return np.asarray(out, dtype=np.float32)


def apply_predictions(symbols, probs: np.ndarray, classes: list[str]) -> None:
    """Store top class, confidence, margin and alternatives on each symbol."""
    for s, p in zip(symbols, probs):
        order = np.argsort(p)[::-1]
        s.probs = p
        s.cls = classes[order[0]]
        s.conf = float(p[order[0]])
        s.margin = float(p[order[0]] - (p[order[1]] if len(order) > 1 else 0.0))
        s.alternatives = [(classes[k], float(p[k])) for k in order[:3]]
        s.flagged = s.conf < config.CONF_MIN or s.margin < config.MARGIN_MIN


# Typical heights relative to the capital height, by height type.
HEIGHT_RATIO = {"cap": 1.0, "x": 0.68, "desc": 0.95, "tall": 1.4, "op": 0.6,
                "raised": 0.35, "dot": 0.15, "line": 0.06, "big": 1.5}


def refine_scale(symbols, table: SymbolTable | None = None, min_conf: float | None = None) -> float | None:
    """Better line scale once classes are known, or None if too few clues.

    Uses confident symbols whose class has a single height type: capitals and
    digits give the capital height directly; x-height letters give it through
    the usual x-height ratio.
    """
    table = table or load_symbols()
    min_conf = config.CONF_MIN if min_conf is None else min_conf
    caps, xs = [], []
    for s in symbols:
        if s.kind != "symbol" or s.cls is None or s.conf < min_conf:
            continue
        heights = table.class_heights(s.cls)
        if heights == {"cap"}:
            caps.append(s.box.h)
        elif heights == {"x"}:
            xs.append(s.box.h)
    # Scripts only ever make symbols smaller, so the upper quartile of the
    # capitals is used, and the x-height estimate wins when every capital on
    # the line is script-sized (x^2 + y^2 = r^2 has only small digits).
    from_caps = float(np.percentile(caps, 75)) if caps else None
    from_x = float(np.median(xs)) / HEIGHT_RATIO["x"] if xs else None
    if from_caps and from_x:
        return from_caps if from_caps >= 0.85 * from_x else from_x
    return from_caps or from_x
