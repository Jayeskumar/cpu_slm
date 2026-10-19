import random

from mnemo.semantic import Semantic


def corpus(n=6000, seed=0):
    r = random.Random(seed)
    animals = "cat dog horse cow sheep goat".split()
    fruits = "apple pear plum cherry grape lemon".split()
    animal_ctx = "the {} ran across the field and ate grass near the barn".split()
    fruit_ctx = "she picked a ripe {} from the tree and ate it sweet juicy".split()
    out = []
    for _ in range(n):
        if r.random() < 0.5:
            w, ctx = r.choice(animals), animal_ctx
        else:
            w, ctx = r.choice(fruits), fruit_ctx
        sent = [x if x != "{}" else w for x in ctx]
        k = r.randint(0, len(sent) - 4)
        out.extend(sent[k:k + 6] + [w])
    return " ".join(out), animals, fruits


def test_similar_words_share_contexts():
    text, animals, fruits = corpus()
    s = Semantic(dim=20, window=3, min_count=2).fit(text)
    near = [w for w, _ in s.similar("cat", 4)]
    assert sum(w in animals for w in near) >= 3
    near = [w for w, _ in s.similar("pear", 4)]
    assert sum(w in fruits for w in near) >= 3


def test_concepts_cluster_categories():
    text, animals, fruits = corpus()
    s = Semantic(dim=20, window=3, min_count=2).fit(text)
    cl = s.concepts(k=8, min_count=2)
    assert len({cl[a] for a in animals} & {cl[f] for f in fruits}) == 0
