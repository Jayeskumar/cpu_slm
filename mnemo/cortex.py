"""Python face of the C predictive cortex (src/cortex.c, src/lexicon.c)."""
import ctypes
import math
import os
import random

from . import _lib


def _bytes(text):
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


class Cortex:
    """Online byte-level predictive memory.

    It learns from every byte it observes (one pass, local update rules) and
    can predict, score, compress-in-principle and generate text.

    Parameters mirror ``mn_config`` in ``src/mnemo.h``: ``table_bits`` sizes
    the context table (2**bits buckets of 16 bytes), ``hist_bits`` the episodic
    history buffer, ``match_bits`` the recall index, ``lex_order`` the word
    n-gram order of the mental lexicon (0 disables it) and ``spaced`` selects
    whitespace tokenization for pre-tokenized corpora.
    """

    def __init__(self, table_bits=22, hist_bits=24, match_bits=22, lex_order=4, spaced=False,
                 class_bits=0, _handle=None):
        self._lib = _lib.lib()
        if _handle is not None:
            self._m = _handle
        else:
            cfg = _lib.Config()
            self._lib.mn_default_config(ctypes.byref(cfg))
            cfg.table_bits, cfg.hist_bits, cfg.match_bits = table_bits, hist_bits, match_bits
            cfg.class_bits = class_bits
            cfg.lex_order, cfg.lex_spaced = lex_order, int(bool(spaced))
            self._m = self._lib.mn_new(ctypes.byref(cfg))
        if not self._m:
            raise MemoryError("could not create the cortex")
        self.bytes_seen = 0

    # -- lifecycle -------------------------------------------------------
    def close(self):
        if getattr(self, "_m", None):
            self._lib.mn_free(self._m)
            self._m = None

    def __del__(self):
        self.close()

    def save(self, path):
        if self._lib.mn_save(self._m, os.fsencode(path)) != 0:
            raise OSError(f"cannot save model to {path}")

    @classmethod
    def load(cls, path):
        h = _lib.lib().mn_load(os.fsencode(path))
        if not h:
            raise OSError(f"cannot load model from {path}")
        return cls(_handle=h)

    @property
    def memory_bytes(self):
        return self._lib.mn_memory(self._m)

    # -- learning and perception -------------------------------------------
    @property
    def learning(self):
        return bool(self._lib.mn_get_learning(self._m))

    @learning.setter
    def learning(self, on):
        self._lib.mn_set_learning(self._m, int(bool(on)))

    def observe(self, text, learn=None):
        """Feed text; returns its information content in bits (surprise)."""
        b = _bytes(text)
        if learn is not None:
            old = self.learning
            self.learning = learn
        bits = self._lib.mn_observe(self._m, b, len(b))
        if learn is not None:
            self.learning = old
        self.bytes_seen += len(b)
        return bits

    def learn(self, text):
        return self.observe(text, learn=True)

    def learn_file(self, path, chunk=1 << 20):
        bits = 0.0
        n = 0
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                bits += self.observe(b, learn=True)
                n += len(b)
        return bits, n

    # -- imagination: evaluate without leaving a trace -------------------------
    def surprise(self, text):
        """Bits needed to encode ``text`` as the next input; nothing changes."""
        snap = self._lib.mn_snapshot(self._m)
        try:
            b = _bytes(text)
            return self._lib.mn_observe(self._m, b, len(b))
        finally:
            self._lib.mn_restore(self._m, snap)

    def choose(self, options, prefix=""):
        """Rank continuations of ``prefix`` by plausibility: returns
        [(bits per byte, total bits, option)], most plausible first."""
        outer = self._lib.mn_snapshot(self._m)
        try:
            if prefix:
                b = _bytes(prefix)
                self._lib.mn_observe(self._m, b, len(b))
            scored = []
            for o in options:
                bits = self.surprise(o)
                scored.append((bits / max(1, len(_bytes(o))), bits, o))
        finally:
            self._lib.mn_restore(self._m, outer)
        return sorted(scored)

    # -- prediction and generation ---------------------------------------------
    def next_byte_distribution(self):
        out = (ctypes.c_double * 256)()
        self._lib.mn_next_byte_dist(self._m, out)
        return list(out)

    def next_words(self, k=10):
        """Most expected next tokens according to the mental lexicon."""
        lex = self._lib.mn_lexicon(self._m)
        if not lex:
            return []
        ids = (ctypes.c_uint32 * k)()
        probs = (ctypes.c_float * k)()
        n = self._lib.lex_top_next(lex, ids, probs, k)
        out = []
        for i in range(n):
            ln = ctypes.c_int()
            p = self._lib.lex_word(lex, ids[i], ctypes.byref(ln))
            out.append((ctypes.string_at(p, ln.value).decode("utf-8", "replace"), probs[i]))
        return out

    def generate(self, prompt="", n=300, temperature=0.7, seed=None, stop=None, learn=False,
                 keep=False):
        """Continue ``prompt``. With keep=False the cortex is left exactly as
        before (generation happens in imagination)."""
        snap = None if keep else self._lib.mn_snapshot(self._m)
        old = self.learning
        try:
            self.learning = learn
            if prompt:
                b = _bytes(prompt)
                self._lib.mn_observe(self._m, b, len(b))
            buf = ctypes.create_string_buffer(n)
            stop_b = -1 if stop is None else _bytes(stop)[0]
            seed = random.getrandbits(63) if seed is None else seed
            k = self._lib.mn_generate(self._m, buf, n, float(temperature), seed, stop_b)
            return buf.raw[:k].decode("utf-8", "replace")
        finally:
            if snap is not None:
                self._lib.mn_restore(self._m, snap)
            else:
                self.learning = old

    # -- concepts ----------------------------------------------------------------
    def word_hash(self, word):
        b = _bytes(word)
        return self._lib.mn_word_hash(b, len(b))

    def set_concepts(self, classes):
        """Attach concept classes (word -> int 1..65535) used as contexts."""
        for w, c in classes.items():
            self._lib.mn_set_class(self._m, self.word_hash(w), int(c))

    # -- lexicon access ----------------------------------------------------------
    def vocabulary(self):
        lex = self._lib.mn_lexicon(self._m)
        if not lex:
            return []
        n = self._lib.lex_num_words(lex)
        words = []
        ln = ctypes.c_int()
        for i in range(n):
            p = self._lib.lex_word(lex, i, ctypes.byref(ln))
            words.append((ctypes.string_at(p, ln.value).decode("utf-8", "replace"),
                          self._lib.lex_count(lex, i)))
        return words

    @staticmethod
    def bits_to_ppl(bits, tokens):
        return 2.0 ** (bits / max(1, tokens))

    @staticmethod
    def entropy_bits(dist):
        return -sum(p * math.log2(p) for p in dist if p > 0)
