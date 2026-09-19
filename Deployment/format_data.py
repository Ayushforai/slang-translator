"""Regenerate Llama-format JSONL. Prefer: python -m slang_translator.cli prepare"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slang_translator.format import build_splits

if __name__ == "__main__":
    print(build_splits())
