from mnemo.mind import Mind

import story_tasks  # noqa: E402  (tests/ is on sys.path under pytest and run_tests)


def talk(m, script):
    out = []
    for line in script:
        out.append(m.hear(line))
    return out


def test_dialogue_reasoning():
    m = Mind(learn_language=False)
    r = talk(m, ["Tweety is a bird.", "Birds can fly.", "Penguins are birds.",
                 "Penguins can't fly.", "Opus is a penguin.", "Can Tweety fly?",
                 "Can Opus fly?", "Why?"])
    assert r[5].startswith("Yes")
    assert r[6].startswith("No")
    assert "penguin" in r[7]


def test_episodic_tracking():
    m = Mind(learn_language=False)
    r = talk(m, ["Mary went to the kitchen.", "Mary picked up the apple.",
                 "Mary went to the garden.", "Where is the apple?", "She dropped the apple.",
                 "Mary went to the office.", "Where is the apple?", "What is Mary carrying?"])
    assert "garden" in r[3]
    assert "garden" in r[6]
    assert r[7].startswith("Nothing")


def test_hyperdimensional_analogy():
    m = Mind(learn_language=False)
    talk(m, ["The currency of the USA is the dollar.", "The capital of the USA is Washington.",
             "The currency of Mexico is the peso.", "The capital of Mexico is Mexico City."])
    assert "peso" in m.hear("What is the dollar of Mexico?").lower()


def test_story_tasks_all_correct():
    res, _ = story_tasks.run(n=20, seed=7)
    assert all(acc == 1.0 for acc in res.values()), res
