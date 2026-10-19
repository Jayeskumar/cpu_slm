from mnemo.knowledge import Knowledge, Truth


def close(a, b, eps=0.01):
    return abs(a - b) < eps


def test_nars_truth_functions():
    t = Truth(1.0, 0.9)
    d = Truth.deduction(t, t)
    assert close(d.f, 1.0) and close(d.c, 0.81)
    i = Truth.induction(t, t)
    assert close(i.f, 1.0) and close(i.c, 0.45, 0.01)
    r = i.revise(i)
    assert close(r.c, 0.62, 0.01)


def kb():
    k = Knowledge()
    k.display.update({"tweety": "Tweety", "opus": "Opus"})
    k.tell("tweety", "isa", "bird")
    k.tell("bird", "can", "fly", generic=True)
    k.tell("penguin", "isa", "bird", generic=True)
    k.tell("penguin", "can", "fly", positive=False, generic=True)
    k.tell("opus", "isa", "penguin")
    return k


def test_default_reasoning_with_exception():
    k = kb()
    a = k.ask("tweety", "can", "fly")
    assert a.truth.f > 0.5 and a.truth.c > 0.7
    b = k.ask("opus", "can", "fly")
    assert b.truth.f < 0.5 and b.truth.c > 0.7
    assert "penguins cannot fly" in k.explain(b)


def test_transitivity_and_asymmetry():
    k = Knowledge()
    k.tell("box", "bigger_than", "chocolate")
    k.tell("chest", "bigger_than", "box")
    assert k.ask("chest", "bigger_than", "chocolate").truth.f > 0.5
    assert k.ask("chocolate", "smaller_than", "chest").truth.f > 0.5
    assert k.ask("chocolate", "bigger_than", "chest").truth.f < 0.5


def test_induction_from_examples():
    k = Knowledge()
    for x in ("robin", "sparrow", "lark"):
        k.tell(x, "isa", "bird", generic=True)
        k.tell(x, "can", "sing", generic=True)
    a = k.ask("bird", "can", "sing")
    assert a.truth.f > 0.5 and 0.5 < a.truth.c < 0.8
    assert a.how == "induction"


def test_activation_decays_with_time():
    k = Knowledge()
    f = k.tell("sky", "is", "blue")
    a0 = k.activation(f)
    for _ in range(50):
        k.tick()
    assert k.activation(f) < a0
