import numpy as np

from mnemo.hdc import HyperSpace


def test_bind_is_invertible_and_dissimilar():
    h = HyperSpace(4096)
    a, b = h.symbol("a"), h.symbol("b")
    c = h.bind(a, b)
    assert abs(h.sim(c, a)) < 0.1
    assert np.array_equal(h.unbind(c, a), b)


def test_bundle_is_similar_to_members():
    h = HyperSpace(4096)
    vs = [h.symbol(x) for x in "abcde"]
    s = h.bundle(vs)
    assert all(h.sim(s, v) > 0.2 for v in vs)
    assert abs(h.sim(s, h.symbol("z"))) < 0.1


def test_record_query_and_dollar_of_mexico():
    h = HyperSpace(4096)
    usa = h.record([("name", "usa"), ("capital", "washington"), ("currency", "dollar")])
    mex = h.record([("name", "mexico"), ("capital", "mexico city"), ("currency", "peso")])
    # what is the capital in the USA record?
    got = h.cleanup(h.unbind(usa, h.symbol("capital")), ["washington", "dollar", "usa"])
    assert got[0][0] == "washington"
    best = h.analogy(usa, mex, "dollar", names=["mexico", "mexico city", "peso"])
    assert best[0][0] == "peso"
