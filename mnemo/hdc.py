"""Hyperdimensional computing (vector symbolic architecture, MAP flavour).

Symbols are random bipolar vectors in a high-dimensional space; in such a
space random vectors are almost orthogonal, so one fixed-width vector can hold
a superposition of many bound pairs and still be decoded by similarity
(Kanerva 2009; Gayler 2003).

  bind(a, b)    elementwise product: dissimilar to both inputs, invertible
                (bind is its own inverse for bipolar vectors)
  bundle(vs)    sign of the sum: similar to every input (a set / superposition)
  permute(a, k) cyclic shift: encodes order and position
"""
import hashlib

import numpy as np


class HyperSpace:
    def __init__(self, dim=2048, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._symbols = {}
        self._tie = self._random_vector("__tie__")

    # -- atoms ---------------------------------------------------------------
    def _random_vector(self, name):
        h = hashlib.blake2b(f"{self.seed}:{name}".encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "little"))
        return (rng.integers(0, 2, self.dim, dtype=np.int8) * 2 - 1).astype(np.int8)

    def symbol(self, name):
        """The (stable) random hypervector of a symbol name."""
        v = self._symbols.get(name)
        if v is None:
            v = self._symbols[name] = self._random_vector(name)
        return v

    def set_symbol(self, name, vec):
        """Use a given vector for a name (e.g. a semantic code)."""
        self._symbols[name] = np.asarray(vec, dtype=np.int8)

    # -- algebra ---------------------------------------------------------------
    @staticmethod
    def bind(a, b):
        return (a * b).astype(np.int8)

    unbind = bind

    def bundle(self, vectors, weights=None):
        vs = list(vectors)
        if not vs:
            return np.zeros(self.dim, dtype=np.int8)
        acc = np.zeros(self.dim, dtype=np.float64)
        for i, v in enumerate(vs):
            acc += v * (1.0 if weights is None else weights[i])
        out = np.sign(acc)
        out[out == 0] = self._tie[out == 0]
        return out.astype(np.int8)

    @staticmethod
    def permute(a, k=1):
        return np.roll(a, k)

    def sim(self, a, b):
        return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / self.dim

    # -- structures --------------------------------------------------------------
    def record(self, pairs):
        """A role-filler record: bundle of bind(role, filler)."""
        return self.bundle(self.bind(self.symbol(r) if isinstance(r, str) else r,
                                     self.symbol(f) if isinstance(f, str) else f)
                           for r, f in pairs)

    def sequence(self, items):
        return self.bundle(self.permute(self.symbol(x) if isinstance(x, str) else x, i)
                           for i, x in enumerate(items))

    def cleanup(self, v, names=None, k=1):
        """Nearest known symbols to v: [(name, similarity)]."""
        names = list(self._symbols) if names is None else list(names)
        names = [n for n in names if not n.startswith("__")]
        if not names:
            return []
        M = np.stack([self.symbol(n) for n in names]).astype(np.int32)
        s = M @ v.astype(np.int32) / self.dim
        order = np.argsort(-s)[:k]
        return [(names[i], float(s[i])) for i in order]

    def analogy(self, source, target, cue, names=None, k=1):
        """Kanerva's "What is the dollar of Mexico?": with records source and
        target, map the cue's role in source onto target.  The mapping vector
        bind(source, target) carries every filler of one record to its
        counterpart in the other."""
        mapping = self.bind(source, target)
        q = self.bind(self.symbol(cue) if isinstance(cue, str) else cue, mapping)
        return self.cleanup(q, names, k)
