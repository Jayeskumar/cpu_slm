from mnemo.language import Event, Fact, Query, WorkingMemory, parse_sentence, singular


def p(s, wm=None):
    return parse_sentence(s, wm or WorkingMemory()).items


def test_singular():
    assert [singular(w) for w in ["birds", "wolves", "mice", "boxes", "ponies", "glass"]] == \
        ["bird", "wolf", "mouse", "box", "pony", "glass"]


def test_facts():
    assert p("Penguins can't fly.") == [Fact("penguin", "can", "fly", False, generic=True)]
    assert p("Tweety is a bird.") == [Fact("tweety", "isa", "bird")]
    assert p("A dog has four legs.")[0].qty == 4
    assert p("The kitchen is north of the hallway.")[0].r == "north_of"


def test_events_and_pronouns():
    wm = WorkingMemory()
    assert p("Daniel went to the garden.", wm) == [Event("move", "daniel", place="garden")]
    assert p("He picked up the milk.", wm) == [Event("take", "daniel", obj="milk")]
    wm2 = WorkingMemory()
    p("Mary and Sandra went to the office.", wm2)
    assert len(p("Then they went to the garden.", wm2)) == 2


def test_questions():
    assert p("Can Opus fly?") == [Query("yesno", "opus", "can", "fly")]
    assert p("Where is the football?")[0].kind == "where"
    assert p("What color is the sky?")[0].extra == "color"
    assert p("How do you go from the den to the kitchen?")[0].kind == "path"
