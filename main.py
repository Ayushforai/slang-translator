"""Run data prep. Training and the web app are separate commands (see README)."""

from slang_translator.format import build_splits


if __name__ == "__main__":
    print("--- Preparing formal → slang splits ---")
    print(build_splits())
    print("Next: python -m slang_translator.cli train-detector")
    print("Train: python Deployment/fine_tune.py")
    print("App:   python app.py")
