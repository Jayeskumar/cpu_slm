"""Distributional semantic memory.

Word meaning is learned from co-occurrence alone, the way Latent Semantic
Analysis models human vocabulary acquisition (Landauer & Dumais 1997): count
which words appear near which, re-weight the counts by positive pointwise
mutual information (with the 0.75 context smoothing that makes count models
match word2vec, Levy, Goldberg & Dagan 2015), and compress the matrix with a
truncated SVD computed by randomized range finding.  No gradient descent.

The result supports similarity, analogy, concept clustering (spherical
k-means), and binary hypervector codes (sign random projection, so similar
meanings get similar hypervectors) for the hyperdimensional memory.
"""
import re

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9']+|[^\x00-\x7f]+")
_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def tokenize(text):
    """Word tokens as the cortex sees them (ASCII letters lower-cased)."""
    return [t.translate(_LOWER) for t in _TOKEN.findall(text)]


class Semantic:
    def __init__(self, dim=200, window=4, max_words=12000, max_contexts=3000, min_count=3,
                 seed=0):
        self.dim, self.window = dim, window
        self.max_words, self.max_contexts, self.min_count = max_words, max_contexts, min_count
        self.seed = seed
        self.words = []
        self.index = {}
        self.counts = None
        self.E = None

    # -- learning ("sleep consolidation": a batch pass over experience) --------
    def fit(self, text):
        toks = tokenize(text) if isinstance(text, str) else list(text)
        uniq, inv, cnt = np.unique(np.array(toks, dtype=object), return_inverse=True,
                                   return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        keep = order[cnt[order] >= self.min_count][: self.max_words]
        rank = np.full(len(uniq), -1, dtype=np.int64)
        rank[keep] = np.arange(len(keep))
        ids = rank[inv]
        self.words = [str(uniq[i]) for i in keep]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = cnt[keep].astype(np.int64)
        V, C = len(self.words), min(self.max_contexts, len(self.words))
        if V < 2:
            raise ValueError("not enough text to learn word meanings")

        # co-occurrence counts with harmonic distance weighting, in row blocks
        M = np.zeros((V, C), dtype=np.float32)
        pairs_t, pairs_c, pairs_w = [], [], []
        for d in range(1, self.window + 1):
            a, b = ids[:-d], ids[d:]
            for t, c in ((a, b), (b, a)):
                ok = (t >= 0) & (c >= 0) & (c < C)
                pairs_t.append(t[ok])
                pairs_c.append(c[ok])
                pairs_w.append(np.full(int(ok.sum()), 1.0 / d, dtype=np.float32))
        pt = np.concatenate(pairs_t)
        pc = np.concatenate(pairs_c)
        pw = np.concatenate(pairs_w)
        block = max(1, (1 << 24) // C)
        for lo in range(0, V, block):
            sel = (pt >= lo) & (pt < lo + block)
            rows = min(block, V - lo)
            flat = (pt[sel] - lo) * C + pc[sel]
            M[lo:lo + rows] = np.bincount(flat, weights=pw[sel], minlength=rows * C).reshape(
                rows, C)

        # positive PMI with smoothed context distribution
        total = M.sum()
        row = M.sum(1, keepdims=True)
        col = M.sum(0) ** 0.75
        col = col / col.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(M / total) - np.log(row / total) - np.log(col)[None, :]
        P = np.where(M > 0, np.maximum(pmi, 0), 0).astype(np.float32)

        # randomized truncated SVD (Halko, Martinsson & Tropp 2011)
        k = min(self.dim, C - 1, V - 1)
        rng = np.random.default_rng(self.seed)
        Y = P @ rng.standard_normal((C, k + 20)).astype(np.float32)
        for _ in range(2):
            Y = P @ (P.T @ Y)
        Q, _ = np.linalg.qr(Y)
        Ub, S, _ = np.linalg.svd(Q.T @ P, full_matrices=False)
        E = (Q @ Ub[:, :k]) * np.sqrt(S[:k])[None, :]
        norm = np.linalg.norm(E, axis=1, keepdims=True)
        self.E = (E / np.maximum(norm, 1e-9)).astype(np.float32)
        self.dim = k
        return self

    # -- queries ---------------------------------------------------------------
    def __contains__(self, w):
        return w in self.index

    def vector(self, w):
        i = self.index.get(w)
        return None if i is None else self.E[i]

    def similarity(self, a, b):
        va, vb = self.vector(a), self.vector(b)
        return None if va is None or vb is None else float(va @ vb)

    def similar(self, w, k=10):
        v = self.vector(w) if isinstance(w, str) else np.asarray(w, dtype=np.float32)
        if v is None:
            return []
        s = self.E @ v
        out = []
        numeric = isinstance(w, str) and any(ch.isdigit() for ch in w)
        for i in np.argsort(-s):
            if isinstance(w, str) and self.words[i] == w:
                continue
            if not numeric and any(ch.isdigit() for ch in self.words[i]):
                continue
            out.append((self.words[i], float(s[i])))
            if len(out) >= k:
                break
        return out

    def analogy(self, a, b, c, k=5):
        """a is to b as c is to ? (3CosMul, Levy & Goldberg 2014)."""
        va, vb, vc = self.vector(a), self.vector(b), self.vector(c)
        if va is None or vb is None or vc is None:
            return []
        sa, sb, sc = ((self.E @ v + 1) / 2 for v in (va, vb, vc))
        score = sb * sc / (sa + 1e-3)
        numeric = any(w.isdigit() for w in (a, b, c))
        out = []
        for i in np.argsort(-score):
            w = self.words[i]
            if w in (a, b, c) or (not numeric and any(ch.isdigit() for ch in w)):
                continue
            out.append((self.words[i], float(score[i])))
            if len(out) >= k:
                break
        return out

    def concepts(self, k=256, iters=20, min_count=5):
        """Spherical k-means over word vectors -> {word: class id in 1..k}."""
        use = np.nonzero(self.counts >= min_count)[0]
        X = self.E[use]
        k = max(1, min(k, len(use)))
        rng = np.random.default_rng(self.seed + 1)
        # frequency-weighted farthest-point initialisation (k-means++ style)
        cent = [X[0]]
        best = X @ cent[0]
        for _ in range(1, k):
            d = np.maximum(1 - best, 0)
            p = d / d.sum() if d.sum() > 0 else None
            j = rng.choice(len(X), p=p)
            cent.append(X[j])
            best = np.maximum(best, X @ X[j])
        Cm = np.stack(cent)
        for _ in range(iters):
            lab = np.argmax(X @ Cm.T, axis=1)
            for j in range(k):
                m = X[lab == j]
                if len(m):
                    c = m.sum(0)
                    Cm[j] = c / max(np.linalg.norm(c), 1e-9)
        lab = np.argmax(X @ Cm.T, axis=1)
        self.centroids = Cm
        return {self.words[i]: int(lab[n]) + 1 for n, i in enumerate(use)}

    def hypervectors(self, dim=2048, seed=0):
        """Sign random projection of the meaning vectors: words with similar
        meanings receive similar bipolar hypervectors."""
        R = np.random.default_rng(seed + 7).standard_normal((self.E.shape[1], dim)).astype(
            np.float32)
        H = np.sign(self.E @ R).astype(np.int8)
        H[H == 0] = 1
        return H

    # -- persistence -------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, E=self.E, counts=self.counts,
                            words=np.array(self.words, dtype=object), dim=self.dim,
                            window=self.window, allow_pickle=True)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        s = cls(dim=int(z["dim"]), window=int(z["window"]))
        s.E, s.counts = z["E"], z["counts"]
        s.words = [str(w) for w in z["words"]]
        s.index = {w: i for i, w in enumerate(s.words)}
        return s
