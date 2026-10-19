"""Counting memories + learned weights: mix Mnemo with the LSTM on Penn Treebank.

Mnemo predicts bytes; the LSTM predicts words.  Mnemo's surprise for a word is
the sum of the surprise of its letters and the space that ends it (the line
break and the next line's leading space for <eos>), which makes the two
comparable word by word.  The mixture is linear,

    p(w) = lam * p_lstm(w) + (1 - lam) * p_mnemo(w),

with lam either tuned on the validation text or learned online on the test
stream itself (exponentiated-gradient on the log loss: no peeking ahead).
Mnemo's byte probabilities over in-vocabulary words sum to at most one, so
the mixed perplexity is an upper bound on the true one.

  ./build/mnemo bench --train data/ptb/ptb.train.txt --test data/ptb/ptb.test.txt \\
      --spaced --tb 22 --hb 24 --mb 22 --lex 4 --dump models/mnemo_ptb.test.online.f32
  python3 tools/hybrid_eval.py --valid-text data/ptb/ptb.valid.txt \\
      --valid-mnemo models/mnemo_ptb.valid.online.f32 --valid-lstm models/lstm_ptb.valid_logp.npy \\
      --text data/ptb/ptb.test.txt --mnemo models/mnemo_ptb.test.online.f32 \\
      --lstm models/lstm_ptb.test_logp.npy
"""
import argparse
import math

import numpy as np

LN2 = math.log(2.0)


def token_bits(text, byte_bits):
    """Per-token surprise (bits): words in reading order plus one <eos> per
    line, as in tools/train_lstm.py.  Whitespace after a word is charged to
    that word; the line break and the spaces after it to <eos>."""
    assert len(text) == len(byte_bits), (len(text), len(byte_bits))
    out, cur, pre, i, n = [], None, 0.0, 0, len(text)
    while i < n:
        c = text[i]
        if c == 10:                      # newline: <eos>
            out.append(float(byte_bits[i]))
            cur = len(out) - 1
            i += 1
        elif c in (32, 9, 13):
            if cur is None:
                pre += float(byte_bits[i])
            else:
                out[cur] += float(byte_bits[i])
            i += 1
        else:
            j = i
            while j < n and text[j] not in (32, 9, 10, 13):
                j += 1
            out.append(float(byte_bits[i:j].sum()) + pre)
            pre, cur, i = 0.0, len(out) - 1, j
    return np.array(out)


def load(text_path, mnemo_path, lstm_path):
    text = open(text_path, "rb").read()
    bits = np.fromfile(mnemo_path, dtype=np.float32).astype(np.float64)
    tb = token_bits(text, bits)
    lp = np.load(lstm_path).astype(np.float64)        # natural log, tokens 1..N-1
    assert len(lp) == len(tb) - 1, (len(lp), len(tb))
    ln_m = -tb[1:] * LN2                               # same tokens as the LSTM
    return ln_m, lp, tb


def mix_ppl(ln_m, ln_l, lam):
    lam = np.clip(lam, 1e-6, 1 - 1e-6)
    ln_p = np.logaddexp(np.log(lam) + ln_l, np.log1p(-lam) + ln_m)
    return math.exp(-ln_p.mean())


def tune(ln_m, ln_l):
    grid = np.linspace(0.0, 1.0, 101)
    ppl = [mix_ppl(ln_m, ln_l, g) for g in grid]
    k = int(np.argmin(ppl))
    lo, hi = grid[max(0, k - 1)], grid[min(100, k + 1)]
    for _ in range(40):                                # golden-section refinement
        a, b = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if mix_ppl(ln_m, ln_l, a) < mix_ppl(ln_m, ln_l, b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def online(ln_m, ln_l, lr=0.02, lam0=0.5):
    """Learn the weight while reading: exponentiated gradient on -ln p."""
    w = np.log([lam0, 1 - lam0])
    total = 0.0
    for a, b in zip(ln_l, ln_m):
        q = np.exp(w - np.logaddexp(w[0], w[1]))
        lp = np.logaddexp(math.log(q[0]) + a, math.log(q[1]) + b)
        total -= lp
        r = np.exp(np.array([a, b]) - lp)              # p_i / p_mix
        w += lr * r
    return math.exp(total / len(ln_m))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text", required=True)
    ap.add_argument("--mnemo", required=True, help="per-byte bits (float32) from mnemo --dump")
    ap.add_argument("--lstm", required=True, help="per-token natural log-probs (.npy)")
    ap.add_argument("--valid-text")
    ap.add_argument("--valid-mnemo")
    ap.add_argument("--valid-lstm")
    ap.add_argument("--lam", type=float, help="fixed LSTM weight (else tuned on validation)")
    a = ap.parse_args()

    ln_m, ln_l, tb = load(a.text, a.mnemo, a.lstm)
    n = len(ln_m)
    print(f"{a.text}: {len(tb)} tokens ({n} scored by both models)")
    print(f"  Mnemo alone    ppl {math.exp(-ln_m.mean()):8.2f}   "
          f"(all tokens: {2 ** (tb.sum() / len(tb)):.2f})")
    print(f"  LSTM alone     ppl {math.exp(-ln_l.mean()):8.2f}")
    lam = a.lam
    if lam is None and a.valid_text:
        vm, vl, _ = load(a.valid_text, a.valid_mnemo, a.valid_lstm)
        lam = tune(vm, vl)
        print(f"  weight tuned on {a.valid_text}: lam = {lam:.3f} "
              f"(valid ppl {mix_ppl(vm, vl, lam):.2f})")
    if lam is not None:
        print(f"  mixture        ppl {mix_ppl(ln_m, ln_l, lam):8.2f}   (lam = {lam:.3f})")
    print(f"  mixture online ppl {online(ln_m, ln_l):8.2f}   (weight learned while reading)")
    best = tune(ln_m, ln_l)
    print(f"  (oracle lam = {best:.3f} would give {mix_ppl(ln_m, ln_l, best):.2f})")


if __name__ == "__main__":
    main()
