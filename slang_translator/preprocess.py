import re

import pandas as pd

_WS = re.compile(r"\s+")


def light_clean(text) -> str:
    """Keep slang intact: do not expand contractions or 'correct' spelling."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _WS.sub(" ", text).strip()
    return text


def load_parallel(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    formal_col = (
        cols.get("formal_text")
        or cols.get("formal_text_cleaned")
        or cols.get("formal")
        or cols.get("normal")
        or cols.get("plain english")
    )
    informal_col = (
        cols.get("informal_text")
        or cols.get("informal_text_cleaned")
        or cols.get("slang")
        or cols.get("gen_z")
        or cols.get("gen-z slang")
        or cols.get("genz slang")
    )
    if not formal_col or not informal_col:
        raise ValueError(f"Expected formal/slang columns in {path}, got {list(df.columns)}")
    out = pd.DataFrame(
        {
            "formal": df[formal_col].map(light_clean),
            "slang": df[informal_col].map(light_clean),
        }
    )
    out = out[(out["formal"].str.len() > 0) & (out["slang"].str.len() > 0)]
    out = out.drop_duplicates(subset=["formal", "slang"]).reset_index(drop=True)
    return out


def load_all_training_pairs() -> pd.DataFrame:
    """Office-casual pairs plus any Gen-Z CSVs under Dataa/genz/."""
    from .config import GENZ_CSVS, RAW_CSV

    frames = [load_parallel(RAW_CSV)]
    for path in GENZ_CSVS:
        if path.is_file():
            frames.append(load_parallel(path))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["formal", "slang"]).reset_index(drop=True)

