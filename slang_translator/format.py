import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLEANED_CSV, FORMATTED_JSONL, RAW_CSV, TEST_CSV, TRAIN_JSONL, VAL_JSONL
from .preprocess import load_parallel
from .style import jaccard, lexical_slangify, render_llama_prompt, slang_score


def _strengthen_slang(formal: str, slang: str) -> str:
    """Pick the slangiest faithful candidate so SFT targets actually leave formal register."""
    candidates = [slang, lexical_slangify(slang), lexical_slangify(formal)]
    return max(candidates, key=lambda t: (slang_score(t), -jaccard(formal, t)))


def _keep_pair(formal: str, slang: str) -> bool:
    if not slang or formal.lower() == slang.lower():
        return False
    return True


def build_splits(seed: int = 42, test_size: float = 0.1, val_size: float = 0.1) -> dict:
    df = load_parallel(RAW_CSV)
    df["slang"] = [_strengthen_slang(f, s) for f, s in zip(df["formal"], df["slang"])]
    df = df[df.apply(lambda r: _keep_pair(r["formal"], r["slang"]), axis=1)].reset_index(drop=True)
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, random_state=seed)

    test.to_csv(TEST_CSV, index=False)
    _write_jsonl(train, TRAIN_JSONL, reverse=False)
    _write_jsonl(val, VAL_JSONL, reverse=False)
    # Combined file for older training scripts (train only — do not leak test).
    _write_jsonl(train, FORMATTED_JSONL, reverse=False)

    return {
        "kept": len(df),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "mean_slang_score_target": float(df["slang"].map(slang_score).mean()),
        "mean_slang_score_source": float(df["formal"].map(slang_score).mean()),
    }


def _write_jsonl(df: pd.DataFrame, path: Path, reverse: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            src, tgt = (row["slang"], row["formal"]) if reverse else (row["formal"], row["slang"])
            record = {
                "text": render_llama_prompt(src, reverse=reverse, target=tgt),
                "formal": row["formal"],
                "slang": row["slang"],
                "task": "slang_to_formal" if reverse else "formal_to_slang",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_bidirectional_jsonl(df: pd.DataFrame, path: Path) -> None:
    """For a future multi-task model: both directions in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            for reverse in (False, True):
                src, tgt = (row["slang"], row["formal"]) if reverse else (row["formal"], row["slang"])
                record = {
                    "text": render_llama_prompt(src, reverse=reverse, target=tgt),
                    "task": "slang_to_formal" if reverse else "formal_to_slang",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
