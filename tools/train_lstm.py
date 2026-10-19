"""Train the small LSTM language model (mnemo/neural.py) on a word-level corpus.

  python3 tools/train_lstm.py --train data/ptb/ptb.train.txt --valid data/ptb/ptb.valid.txt \
      --test data/ptb/ptb.test.txt --out models/lstm_ptb

Saves the best model (by validation perplexity) to OUT.npz, a resumable
checkpoint to OUT.ckpt.npz, a log to OUT.log and, at the end, the per-token
natural-log probabilities of the test set to OUT.test_logp.npy (used by
tools/hybrid_eval.py).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnemo.neural import LSTMLM, read_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--bptt", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="for timing tests")
    a = ap.parse_args()

    train, vocab = read_tokens(a.train)
    valid, _ = read_tokens(a.valid, vocab)
    words = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    logf = open(a.out + ".log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ck = a.out + ".ckpt.npz"
    state = {"epoch": 0, "lr": a.lr, "best": float("inf")}
    m = LSTMLM(len(vocab), a.hidden, a.layers, a.dropout, seed=a.seed)
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        for k in m.p:
            m.p[k] = z[k].astype(np.float32)
        state = json.loads(str(z["_state"]))
        log(f"resumed at epoch {state['epoch']} lr {state['lr']:.4f} best {state['best']:.2f}")
    log(f"vocab {len(vocab)}, train {len(train)} tokens, {m.n_params / 1e6:.2f}M parameters, "
        f"hidden {a.hidden} x {a.layers}, dropout {a.dropout}")
    if a.max_batches:
        cut = a.batch * a.bptt * a.max_batches + 1
        t0 = time.time()
        ppl = m.run_epoch(train[:cut], a.batch, a.bptt, a.lr, train=True)
        dt = time.time() - t0
        log(f"timing: {a.max_batches} batches in {dt:.1f} s -> {dt / a.max_batches * 1000:.0f} "
            f"ms/batch, epoch estimate {dt / a.max_batches * len(train) / (a.batch * a.bptt) / 60:.1f} "
            f"min (ppl {ppl:.1f})")
        return
    while state["epoch"] < a.epochs and state["lr"] > 1e-3:
        t0 = time.time()
        tr = m.run_epoch(train, a.batch, a.bptt, state["lr"], train=True, log=log)
        va = m.run_epoch(valid, a.batch, a.bptt, train=False)
        state["epoch"] += 1
        prev = state["best"]
        improved = va < prev
        log(f"epoch {state['epoch']} lr {state['lr']:.4f} train ppl {tr:.2f} valid ppl {va:.2f} "
            f"({(time.time() - t0) / 60:.1f} min){' *' if improved else ''}")
        if improved:
            state["best"] = va
            m.save(a.out + ".npz", words)
        if va > prev * 0.99:          # less than 1% better than the best so far: decay
            state["lr"] *= a.decay
        np.savez(ck, **m.p, _state=np.array(json.dumps(state)), allow_pickle=True)
    best, _ = LSTMLM.load(a.out + ".npz")
    for name, path in (("valid", a.valid), ("test", a.test)):
        if not path:
            continue
        ids, _ = read_tokens(path, vocab)
        lp = best.token_logprobs(ids)
        np.save(f"{a.out}.{name}_logp.npy", lp)
        log(f"{name} ppl {math.exp(-lp.mean()):.2f} over {len(lp)} predicted tokens")


if __name__ == "__main__":
    main()
