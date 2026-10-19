"""Mnemo: a tiny, CPU-only, non-neural mind.

Components (each a classic idea from cognitive science or compression,
implemented with counting, hashing, hypervectors and logic -- no gradient
descent, no GPU):

* ``Cortex``    -- online predictive memory over bytes (context mixing with a
                  mental lexicon, priming and episodic recall), written in C.
* ``Semantic``  -- distributional semantic memory (PPMI + truncated SVD, the
                  LSA family) with hyperdimensional codes and concept classes.
* ``Knowledge`` -- declarative memory of facts with evidence-based truth values
                  (NARS-style) and ACT-R activation-based retrieval.
* ``World``     -- episodic event memory that tracks where things are.
* ``Mind``      -- the integrated agent: reads, remembers, reasons, answers,
                  explains and talks.
"""
import importlib

_EXPORTS = {
    "Cortex": "cortex",
    "HyperSpace": "hdc",
    "Semantic": "semantic",
    "Knowledge": "knowledge",
    "Truth": "knowledge",
    "World": "world",
    "Mind": "mind",
}
__all__ = list(_EXPORTS)
__version__ = "0.1.0"


def __getattr__(name):  # lazy imports keep `import mnemo` instant
    if name in _EXPORTS:
        return getattr(importlib.import_module("." + _EXPORTS[name], __name__), name)
    raise AttributeError(name)
