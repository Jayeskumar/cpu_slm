import os
import subprocess
import tempfile

from mnemo.cortex import Cortex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = ("The quick brown fox jumps over the lazy dog. " * 3 +
        "A journey of a thousand miles begins with a single step. "
        "To be or not to be, that is the question. " * 2) * 20


def small():
    return Cortex(table_bits=16, hist_bits=16, match_bits=14, lex_order=3)


def test_compress_roundtrip_is_lossless():
    """The predictor is a valid causal model: arithmetic coding with it
    decodes back to the exact input."""
    exe = os.path.join(ROOT, "build", "mnemo")
    if not os.path.exists(exe):
        subprocess.run(["make", "-C", ROOT], check=True, capture_output=True)
    with tempfile.TemporaryDirectory() as d:
        src, arc, out = (os.path.join(d, n) for n in ("in.txt", "a.mnz", "out.txt"))
        with open(src, "w") as f:
            f.write(TEXT)
        r = subprocess.run([exe, "compress", src, arc, "--tb", "16", "--hb", "16", "--mb", "14"],
                           check=True, capture_output=True, text=True)
        subprocess.run([exe, "decompress", arc, out], check=True, capture_output=True)
        assert open(out).read() == TEXT
        assert os.path.getsize(arc) < len(TEXT) / 4, r.stdout


def test_learning_reduces_surprise():
    c = small()
    first = c.learn(TEXT[:2000])
    again = c.surprise(TEXT[:2000])
    assert again < first * 0.5


def test_imagination_leaves_no_trace():
    c = small()
    c.learn(TEXT)
    c.learning = False
    b1 = c.surprise("the lazy dog")
    c.generate("To be", n=50, seed=3)
    c.surprise("completely different words here")
    b2 = c.surprise("the lazy dog")
    assert abs(b1 - b2) < 1e-9


def test_choose_prefers_plausible_continuation():
    c = small()
    c.learn(TEXT)
    ranked = c.choose([" the question.", " zqx vwk."], prefix="that is")
    assert ranked[0][2] == " the question."


def test_save_load_roundtrip():
    c = small()
    c.learn(TEXT)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.bin")
        c.save(p)
        c2 = Cortex.load(p)
        assert abs(c.surprise("the lazy dog") - c2.surprise("the lazy dog")) < 1e-9


def test_lexicon_predicts_next_words():
    c = small()
    c.learn(TEXT)
    c.learning = False
    c.observe("To be or not to ")
    words = [w for w, _ in c.next_words(3)]
    assert "be" in words
