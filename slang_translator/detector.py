"""Train a binary formality detector on the parallel corpus.

This is the routing piece you would need for auto formal↔slang. It is trained on
this project's pairs, so it learns 'corporate vs casual' more than Gen-Z slang.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import DETECTOR_PATH
from .preprocess import load_parallel
from .style import slang_score


def _xy(df: pd.DataFrame):
    texts = pd.concat([df["formal"], df["slang"]], ignore_index=True)
    # 1 = informal/slang-like, 0 = formal
    labels = pd.concat(
        [pd.Series([0] * len(df)), pd.Series([1] * len(df))],
        ignore_index=True,
    )
    return texts, labels


def train_detector(df: pd.DataFrame, seed: int = 42) -> dict:
    X, y = _xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(y_test, pred, target_names=["formal", "informal"], output_dict=True)
    DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, DETECTOR_PATH)
    return {
        "path": str(DETECTOR_PATH),
        "held_out_accuracy": report["accuracy"],
        "held_out_macro_f1": report["macro avg"]["f1-score"],
        "formal_f1": report["formal"]["f1-score"],
        "informal_f1": report["informal"]["f1-score"],
    }


class FormalityDetector:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DETECTOR_PATH
        self.model = joblib.load(self.path) if self.path.is_file() else None

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"label": "unknown", "informal_prob": 0.5, "slang_score": 0.0}
        slang = slang_score(text)
        if self.model is None:
            label = "informal" if slang >= 0.08 else "formal"
            return {"label": label, "informal_prob": min(1.0, slang * 4), "slang_score": slang}
        proba = float(self.model.predict_proba([text])[0][1])
        label = "informal" if proba >= 0.5 else "formal"
        return {"label": label, "informal_prob": proba, "slang_score": slang}
