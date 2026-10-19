"""A small weight-based intuition: a word-level LSTM language model in numpy.

This is the one place where Mnemo learns weights by gradient descent.  It is
deliberately small (a few million parameters, trainable on a laptop CPU in
about an hour) and is used as one more expert next to the counting memories:
neural networks generalise across similar contexts, count memories remember
exactly and adapt instantly, and their errors are different, so a mixture of
the two is better than either (Mikolov et al. 2011: RNNLM 124.7 + KN5+cache
-> 97.5 on Penn Treebank).

Architecture (Zaremba, Sutskever & Vinyals 2014; Press & Wolf 2017; Gal &
Ghahramani 2016):
  tied input/output embeddings, 2-layer LSTM, variational dropout on the
  non-recurrent connections, truncated BPTT, SGD with gradient clipping.

No framework: forward and backward passes are written out by hand and checked
numerically in tests/test_neural.py.
"""
import math
import time

import numpy as np

F = np.float32


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


class LSTMLM:
    def __init__(self, vocab, hidden=320, layers=2, dropout=0.4, init=0.05, seed=0):
        self.V, self.H, self.L = vocab, hidden, layers
        self.dropout = dropout
        rng = np.random.default_rng(seed)
        u = lambda *s: rng.uniform(-init, init, s).astype(F)  # noqa: E731
        self.p = {"E": u(vocab, hidden), "bo": np.zeros(vocab, F)}
        for l in range(layers):
            self.p[f"Wx{l}"] = u(hidden, 4 * hidden)
            self.p[f"Wh{l}"] = u(hidden, 4 * hidden)
            b = np.zeros(4 * hidden, F)
            b[hidden:2 * hidden] = 1.0  # forget-gate bias: remember by default
            self.p[f"b{l}"] = b
        self.rng = rng

    @property
    def n_params(self):
        return sum(v.size for v in self.p.values())

    def zero_state(self, batch):
        dt = self.p["E"].dtype
        return [(np.zeros((batch, self.H), dt), np.zeros((batch, self.H), dt))
                for _ in range(self.L)]

    # ------------------------------------------------------------------ core
    def _mask(self, shape, train):
        if not train or self.dropout <= 0:
            return None
        keep = 1.0 - self.dropout
        dt = self.p["E"].dtype
        return ((self.rng.random(shape) < keep) / keep).astype(dt)

    def forward(self, x, y, state, train=True, need_grad=True):
        """x, y: (T, B) int arrays. Returns (loss summed over time and averaged
        over the batch, grads or None, new state, per-token log-probs)."""
        p, H = self.p, self.H
        dt = p["E"].dtype
        T, B = x.shape
        cache = {}
        inp = p["E"][x]                                   # (T, B, H)
        m = self._mask((1, B, H), train)
        if m is not None:
            inp = inp * m
        cache["m_in"] = m
        new_state = []
        for l in range(self.L):
            h, c = state[l]
            Zx = (inp.reshape(T * B, H) @ p[f"Wx{l}"]).reshape(T, B, 4 * H) + p[f"b{l}"]
            Wh = p[f"Wh{l}"]
            hs = np.empty((T, B, H), dt)
            cs = np.empty((T, B, H), dt)
            gates = np.empty((T, B, 4 * H), dt)
            hprev = np.empty((T, B, H), dt)
            cprev = np.empty((T, B, H), dt)
            for t in range(T):
                hprev[t], cprev[t] = h, c
                z = Zx[t] + h @ Wh
                i = sigmoid(z[:, :H])
                f = sigmoid(z[:, H:2 * H])
                o = sigmoid(z[:, 2 * H:3 * H])
                g = np.tanh(z[:, 3 * H:])
                c = f * c + i * g
                h = o * np.tanh(c)
                gates[t, :, :H], gates[t, :, H:2 * H] = i, f
                gates[t, :, 2 * H:3 * H], gates[t, :, 3 * H:] = o, g
                hs[t], cs[t] = h, c
            new_state.append((h, c))
            cache[l] = (inp, hs, cs, gates, hprev, cprev)
            m = self._mask((1, B, H), train)
            inp = hs * m if m is not None else hs
            cache[f"m{l}"] = m
        out = inp.reshape(T * B, H)
        logits = out @ p["E"].T + p["bo"]
        logits -= logits.max(1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(1, keepdims=True)           # now probabilities
        yy = y.reshape(-1)
        py = logits[np.arange(T * B), yy]
        logp = np.log(np.maximum(py, 1e-30))
        loss = -logp.sum() / B
        if not need_grad:
            return loss, None, new_state, logp.reshape(T, B)

        # ------------------------------------------------------------ backward
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dlog = logits
        dlog[np.arange(T * B), yy] -= 1.0
        dlog /= B
        g["bo"] = dlog.sum(0)
        g["E"] += dlog.T @ out
        dinp = (dlog @ p["E"]).reshape(T, B, H)
        for l in reversed(range(self.L)):
            m = cache[f"m{l}"]
            if m is not None:
                dinp = dinp * m
            xin, hs, cs, gates, hprev, cprev = cache[l]
            Wh = p[f"Wh{l}"]
            dZ = np.empty((T, B, 4 * H), dt)
            dh_next = np.zeros((B, H), dt)
            dc_next = np.zeros((B, H), dt)
            for t in reversed(range(T)):
                i, f = gates[t, :, :H], gates[t, :, H:2 * H]
                o, gg = gates[t, :, 2 * H:3 * H], gates[t, :, 3 * H:]
                tc = np.tanh(cs[t])
                dh = dinp[t] + dh_next
                do = dh * tc
                dc = dh * o * (1.0 - tc * tc) + dc_next
                di = dc * gg
                dgg = dc * i
                df = dc * cprev[t]
                dc_next = dc * f
                dz = dZ[t]
                dz[:, :H] = di * i * (1.0 - i)
                dz[:, H:2 * H] = df * f * (1.0 - f)
                dz[:, 2 * H:3 * H] = do * o * (1.0 - o)
                dz[:, 3 * H:] = dgg * (1.0 - gg * gg)
                dh_next = dz @ Wh.T
            dZf = dZ.reshape(T * B, 4 * H)
            g[f"Wh{l}"] += hprev.reshape(T * B, H).T @ dZf
            g[f"Wx{l}"] += xin.reshape(T * B, H).T @ dZf
            g[f"b{l}"] += dZf.sum(0)
            dinp = (dZf @ p[f"Wx{l}"].T).reshape(T, B, H)
        m = cache["m_in"]
        if m is not None:
            dinp = dinp * m
        np.add.at(g["E"], x.reshape(-1), dinp.reshape(T * B, H))
        return loss, g, new_state, logp.reshape(T, B)

    # ------------------------------------------------------------------ training
    def sgd(self, grads, lr, clip=5.0):
        norm = math.sqrt(sum(float((v * v).sum()) for v in grads.values()))
        scale = lr * (clip / norm if norm > clip else 1.0)
        for k, v in grads.items():
            self.p[k] -= scale * v
        return norm

    def run_epoch(self, data, batch=20, bptt=35, lr=1.0, train=True, log=None):
        """data: 1-D int array. Returns perplexity (and trains if train)."""
        cols = len(data) // batch
        d = data[: cols * batch].reshape(batch, cols).T     # (cols, batch)
        state = self.zero_state(batch)
        nll, n, t0 = 0.0, 0, time.time()
        steps = (cols - 1) // bptt
        for k, s in enumerate(range(0, cols - 1, bptt)):
            T = min(bptt, cols - 1 - s)
            x, y = d[s:s + T], d[s + 1:s + 1 + T]
            loss, grads, state, _ = self.forward(x, y, state, train=train, need_grad=train)
            if train:
                self.sgd(grads, lr)
            nll += loss * batch
            n += T * batch
            if log and train and k % 200 == 0:
                log(f"  step {k}/{steps} ppl {math.exp(nll / n):.1f} "
                    f"({n / (time.time() - t0):.0f} tok/s)")
        return math.exp(nll / n)

    def token_logprobs(self, data, batch=20, bptt=35):
        """Natural-log probability of every token data[1:] given its history,
        computed in `batch` parallel columns (each column starts fresh)."""
        cols = (len(data) - 1) // batch
        n = cols * batch
        x_all = np.asarray(data[:n]).reshape(batch, cols).T
        y_all = np.asarray(data[1:n + 1]).reshape(batch, cols).T
        state = self.zero_state(batch)
        out = np.empty((cols, batch), self.p["E"].dtype)
        for s in range(0, cols, bptt):
            T = min(bptt, cols - s)
            _, _, state, lp = self.forward(x_all[s:s + T], y_all[s:s + T], state, train=False,
                                           need_grad=False)
            out[s:s + T] = lp
        lp = out.T.reshape(-1)            # back to stream order for positions 1..n
        rest = len(data) - 1 - n          # tail tokens: score them with a fresh state
        if rest > 0:
            tail = np.asarray(data[n:])
            st = self.zero_state(1)
            _, _, _, tl = self.forward(tail[:-1, None], tail[1:, None], st, train=False,
                                       need_grad=False)
            lp = np.concatenate([lp, tl[:, 0]])
        return lp

    # ------------------------------------------------------------------ io
    def save(self, path, vocab_words=None):
        extra = {"_meta": np.array([self.V, self.H, self.L], np.int64)}
        if vocab_words is not None:
            extra["_vocab"] = np.array(vocab_words, dtype=object)
        np.savez(path, **{k: v.astype(np.float16) for k, v in self.p.items()}, **extra,
                 allow_pickle=True)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        V, H, L = (int(v) for v in z["_meta"])
        m = cls(V, H, L, dropout=0.0)
        for k in m.p:
            m.p[k] = z[k].astype(F)
        vocab = [str(w) for w in z["_vocab"]] if "_vocab" in z else None
        return m, vocab


def read_tokens(path, vocab=None, eos="<eos>"):
    """Whitespace tokens with an end-of-sentence token per line."""
    toks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks.extend(line.split())
            toks.append(eos)
    if vocab is None:
        words = sorted(set(toks))
        vocab = {w: i for i, w in enumerate(words)}
    unk = vocab.get("<unk>", 0)
    return np.array([vocab.get(t, unk) for t in toks], np.int64), vocab
