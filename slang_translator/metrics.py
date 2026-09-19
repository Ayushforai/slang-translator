import math
import re
from collections import Counter


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _ngrams(tokens: list[str], n: int):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    hyp_t, ref_t = _tok(hyp), _tok(ref)
    if not hyp_t:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        hng, rng = _ngrams(hyp_t, n), _ngrams(ref_t, n)
        if not hng:
            logs.append(math.log(1e-12))
            continue
        hc, rc = Counter(hng), Counter(rng)
        overlap = sum(min(hc[g], rc[g]) for g in hc)
        p = (overlap + 1) / (len(hng) + 1)
        logs.append(math.log(max(p, 1e-12)))
    geo = math.exp(sum(logs) / max_n)
    bp = 1.0 if len(hyp_t) > len(ref_t) else math.exp(1 - len(ref_t) / max(len(hyp_t), 1))
    return bp * geo * 100.0


def corpus_bleu(pairs: list[tuple[str, str]]) -> float:
    clip = [0] * 4
    tot = [0] * 4
    hyp_len = 0
    ref_len = 0
    for hyp, ref in pairs:
        ht, rt = _tok(hyp), _tok(ref)
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, 5):
            hng, rng = _ngrams(ht, n), _ngrams(rt, n)
            if not hng:
                continue
            hc, rc = Counter(hng), Counter(rng)
            clip[n - 1] += sum(min(hc[g], rc[g]) for g in hc)
            tot[n - 1] += len(hng)
    ps = [((clip[i] + 1) / (tot[i] + 1)) if tot[i] else 0.0 for i in range(4)]
    geo = math.exp(sum(math.log(max(p, 1e-12)) for p in ps) / 4)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * geo * 100.0
