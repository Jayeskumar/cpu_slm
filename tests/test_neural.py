import numpy as np

from mnemo.neural import LSTMLM


def test_gradients_match_finite_differences():
    m = LSTMLM(vocab=7, hidden=4, layers=2, dropout=0.0, init=0.5, seed=3)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 7, (3, 2))
    y = rng.integers(0, 7, (3, 2))
    st = m.zero_state(2)
    _, g, _, _ = m.forward(x, y, st, train=False)
    eps = 1e-6
    for k in ["E", "bo", "Wx0", "Wh0", "b0", "Wx1", "Wh1", "b1"]:
        flat = m.p[k].reshape(-1)
        for idx in rng.choice(flat.size, 4, replace=False):
            old = flat[idx]
            flat[idx] = old + eps
            lp, _, _, _ = m.forward(x, y, st, train=False, need_grad=False)
            flat[idx] = old - eps
            lm, _, _, _ = m.forward(x, y, st, train=False, need_grad=False)
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = g[k].reshape(-1)[idx]
            assert abs(num - ana) < 1e-5 * max(1.0, abs(num)), (k, idx, num, ana)


def test_learns_a_repeating_sequence():
    seq = np.array([0, 1, 2, 3, 4, 5] * 300)
    m = LSTMLM(vocab=6, hidden=16, layers=1, dropout=0.0, init=0.1, seed=1)
    first = m.run_epoch(seq, batch=4, bptt=10, lr=1.0)
    for _ in range(4):
        last = m.run_epoch(seq, batch=4, bptt=10, lr=1.0)
    assert last < 1.2 < first


def test_token_logprobs_cover_every_token_once():
    seq = np.array([0, 1, 2, 3, 4, 5] * 50 + [0, 1, 2])     # does not divide into columns
    m = LSTMLM(vocab=6, hidden=8, layers=1, dropout=0.0, seed=2)
    lp = m.token_logprobs(seq, batch=4, bptt=7)
    assert lp.shape == (len(seq) - 1,) and np.all(lp < 0)
    one = m.token_logprobs(seq, batch=1, bptt=1000)            # a single stream: same tokens
    assert one.shape == lp.shape
