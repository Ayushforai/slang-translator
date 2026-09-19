"""Local inference: formal English → slang."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slang_translator.engine import SlangEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="I would like to request your assistance.")
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args()
    engine = SlangEngine(require_adapter=not args.no_adapter)
    print(engine.generate(args.text))


if __name__ == "__main__":
    main()
