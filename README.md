---
title: Formal to Slang
emoji: 🗣️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Formal → Slang Rewriter

Rewrites **standard / formal English into modern slang**. It does **not** translate slang into formal English.

Live space: [ayushforai/slang-translator-web](https://huggingface.co/spaces/ayushforai/slang-translator-web)  
Adapter: [ayushforai/slang-translator-llama-1b](https://huggingface.co/ayushforai/slang-translator-llama-1b)

## Method

1. Parallel pairs: formal sentence ↔ slang/casual sentence.
2. Light cleaning that **keeps contractions and slang** on the target side.
3. Llama 3.2 instruct chat template (`Rewrite this in slang:`).
4. **QLoRA** SFT of `meta-llama/Llama-3.2-1B-Instruct` (LoRA rank 16 on attention + MLP).
5. Inference uses the chat template. If the model copies the input, a **lexical slang fallback** still shifts register.
6. Optional **TF-IDF + logistic regression** formality detector (routing for a future two-way model; the product always rewrites **toward slang**).

## Honest metrics (from the published adapter run)

Logged in `checkpoint-2106/trainer_state.json` (train set, not a test BLEU):

| | Start (step 10) | End (step 2100 / 2106) |
|---|---|---|
| Cross-entropy loss | 4.75 | 0.267 (−94.4%) |
| Predictive entropy | 2.80 | 0.273 |
| Token accuracy | 33.9% | 89.5% |

Adapter size on disk: **11.27M LoRA parameters** (0.91% of Llama 3.2 1B, ~1.24B). Training used 4-bit NF4 when a GPU was available. Do not report 20.9M or 1.3B unless you re-count `print_trainable_parameters()` on a new run.

After changing data or prompts, retrain, then report **test** BLEU and slang-score from:

```bash
python -m slang_translator.cli prepare
python -m slang_translator.cli eval-baselines
python Scriptss/evaluate.py --lexical-only
# after GPU train:
python Scriptss/evaluate.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
python -m slang_translator.cli prepare
python -m slang_translator.cli train-detector
```

Train (needs a Hugging Face token for Llama 3.2 and a GPU for QLoRA):

```bash
set HUGGINGFACE_HUB_TOKEN=...
python Deployment/fine_tune.py
```

Run the app:

```bash
python app.py
```

Open http://localhost:7860

## Project layout

- `slang_translator/` — preprocess, prompts, style score, detector, metrics, generation
- `Dataa/` — raw pairs, cleaned CSV, `train.jsonl` / `val.jsonl` / `test.csv`
- `Deployment/fine_tune.py` — QLoRA training on **train.jsonl only**
- `Scriptss/evaluate.py` — test-set BLEU, slang score, copy-rate
- `app.py` + `templates/index.html` — Flask UI

## Bidirectional (formal ↔ slang) — not in this product

To auto-detect register and convert **both** ways you would need:

1. **A router:** the TF-IDF detector already here, or a small classifier / LLM-as-judge. Mid-register sentences (“see you tomorrow”) are the failure mode.
2. **Two tasks in training:** same pairs reversed, with a different system prompt (`slang_to_formal`). Either one multi-task LoRA with a task prefix, or two adapters.
3. **Asymmetric data:** slang→formal is easier for Instruct models; formal→slang needs **much slangier** targets than office-casual paraphrases.
4. **Separate eval:** BLEU in the formal direction is meaningful; BLEU in the slang direction under-rewards valid slang that differs from the one reference.

Limitations: slang is many dialects; a wrong route makes text worse; reversing slang throws away tone; detectors trained on this CSV learn “corporate vs slightly casual,” not Twitter/Gen-Z.

## CV bullets you can defend

- Fine-tuned Llama 3.2 1B Instruct with QLoRA (rank 16, 11.3M trainable weights) for **formal → slang** rewriting.
- Train loss 4.75 → 0.267 (−94.4%) and entropy 2.80 → 0.273 over 2,106 steps; 89.5% train token accuracy (report as **training** dynamics).
- Flask + HTML/CSS app on Hugging Face Spaces; train/val/test split and BLEU/slang-score eval script.
