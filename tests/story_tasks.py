"""Story-understanding tasks in the style of bAbI (Weston et al. 2015).

The stories are generated here with our own templates and vocabulary; this is
NOT the official bAbI dataset (it could not be downloaded in this
environment), so the scores are a functional test of the episodic memory and
reasoner, not a comparison with published bAbI numbers.

  python3 tests/story_tasks.py [stories_per_task]
"""
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnemo.mind import Mind  # noqa: E402

PEOPLE = ["Mary", "John", "Sandra", "Daniel"]
MORE_PEOPLE = ["Fred", "Bill", "Jeff", "Julie"]
PLACES = ["bathroom", "hallway", "kitchen", "garden", "office", "bedroom"]
OBJECTS = ["football", "apple", "milk"]
MOVE = ["went to", "moved to", "journeyed to", "travelled to", "went back to"]
TAKE = ["picked up", "got", "grabbed", "took"]
DROP = ["dropped", "discarded", "put down", "left"]


def move(p, pl, r):
    return f"{p} {r.choice(MOVE)} the {pl}."


def t1(r):
    lines, loc = [], {}
    for _ in range(r.randint(2, 6)):
        p, pl = r.choice(PEOPLE), r.choice(PLACES)
        lines.append(move(p, pl, r))
        loc[p] = pl
    p = r.choice(list(loc))
    return lines, f"Where is {p}?", loc[p]


def t2(r):
    lines, loc, held = [], {}, {}
    for p in r.sample(PEOPLE, 2):
        pl = r.choice(PLACES)
        lines.append(move(p, pl, r))
        loc[p] = pl
    p = r.choice(list(loc))
    o = r.choice(OBJECTS)
    lines.append(f"{p} {r.choice(TAKE)} the {o} there.")
    for _ in range(r.randint(1, 3)):
        q = r.choice(list(loc))
        pl = r.choice(PLACES)
        lines.append(move(q, pl, r))
        loc[q] = pl
    return lines, f"Where is the {o}?", loc[p]


def t3(r):
    p, o = r.choice(PEOPLE), r.choice(OBJECTS)
    a, b, c = r.sample(PLACES, 3)
    other = r.choice([x for x in PEOPLE if x != p])
    lines = [move(p, a, r), f"{p} {r.choice(TAKE)} the {o} there.", move(p, b, r),
             move(other, r.choice(PLACES), r), move(p, c, r)]
    return lines, f"Where was the {o} before the {c}?", b


def t4(r):
    a, b, c = r.sample(PLACES, 3)
    d1, d2 = r.sample(["north", "south", "east", "west"], 2)
    lines = [f"The {a} is {d1} of the {b}.", f"The {c} is {d2} of the {b}."]
    return lines, f"What is {d1} of the {b}?", a


def t5(r):
    g, h = r.sample(MORE_PEOPLE, 2)
    o = r.choice(OBJECTS)
    lines = [f"{g} {r.choice(TAKE)} the {o}.", f"{g} gave the {o} to {h}."]
    kind = r.randint(0, 2)
    if kind == 0:
        return lines, f"Who gave the {o} to {h}?", g.lower()
    if kind == 1:
        return lines, f"What did {g} give to {h}?", o
    return lines, f"Who received the {o}?", h.lower()


def t6(r):
    lines, ans = t1(r)[0], None
    p = r.choice(PEOPLE)
    pl = r.choice(PLACES)
    lines.append(move(p, pl, r))
    q = r.choice(PLACES)
    return lines, f"Is {p} in the {q}?", "yes" if q == pl else "no"


def t7(r):
    p = r.choice(PEOPLE)
    lines, held = [move(p, r.choice(PLACES), r)], set()
    for _ in range(r.randint(1, 4)):
        o = r.choice(OBJECTS)
        if o in held and r.random() < 0.5:
            lines.append(f"{p} {r.choice(DROP)} the {o}.")
            held.discard(o)
        elif o not in held:
            lines.append(f"{p} {r.choice(TAKE)} the {o}.")
            held.add(o)
    n = len(held)
    return lines, f"How many objects is {p} carrying?", ["none", "one", "two", "three"][n]


def t8(r):
    lines, q, _ = t7(r)
    p = q.split()[4]
    held = set()
    for ln in lines:
        m = re.match(rf"{p} (picked up|got|grabbed|took) the (\w+)", ln)
        if m:
            held.add(m.group(2))
        m = re.match(rf"{p} (dropped|discarded|put down|left) the (\w+)", ln)
        if m:
            held.discard(m.group(2))
    return lines, f"What is {p} carrying?", ",".join(sorted(held)) or "nothing"


def t9(r):
    p, a = r.choice(PEOPLE), r.choice(PLACES)
    if r.random() < 0.5:
        return [move(p, a, r), f"{p} is no longer in the {a}."], f"Is {p} in the {a}?", "no"
    b = r.choice([x for x in PLACES if x != a])
    return [f"{p} is no longer in the {b}.", f"{p} is in the {a}."], f"Is {p} in the {a}?", "yes"


def t10(r):
    p = r.choice(MORE_PEOPLE)
    a, b, c = r.sample(PLACES, 3)
    lines = [f"{p} is either in the {a} or the {b}."]
    q = r.choice([a, b, c])
    return lines, f"Is {p} in the {q}?", "maybe" if q in (a, b) else "no"


def t11(r):
    p = r.choice(["Daniel", "John"])
    a, b = r.sample(PLACES, 2)
    lines = [move(r.choice(["Mary", "Sandra"]), r.choice(PLACES), r), move(p, a, r),
             f"Afterwards he {r.choice(MOVE)} the {b}."]
    return lines, f"Where is {p}?", b


def t12(r):
    p, q = r.sample(PEOPLE, 2)
    a = r.choice(PLACES)
    lines = [f"{p} and {q} {r.choice(MOVE)} the {a}."]
    return lines, f"Where is {r.choice([p, q])}?", a


def t13(r):
    p, q = r.sample(PEOPLE, 2)
    a, b = r.sample(PLACES, 2)
    lines = [f"{p} and {q} {r.choice(MOVE)} the {a}.", f"Then they {r.choice(MOVE)} the {b}."]
    return lines, f"Where is {r.choice([p, q])}?", b


ANIMALS = [("sheep", "sheep"), ("wolf", "wolves"), ("cat", "cats"), ("mouse", "mice")]
NAMES_ANIMAL = ["Gertrude", "Winona", "Jessica", "Emily"]


def t15(r):
    fears = {"sheep": "wolf", "wolf": "cat", "cat": "mouse", "mouse": "sheep"}
    plur = dict(ANIMALS)
    lines = []
    for a, b in fears.items():
        lines.append(f"{plur[a].capitalize()} are afraid of {plur[b]}.")
    names = r.sample(NAMES_ANIMAL, 3)
    kinds = {}
    for n in names:
        k = r.choice(list(fears))
        kinds[n] = k
        lines.append(f"{n} is a {k}.")
    n = r.choice(names)
    return lines, f"What is {n} afraid of?", fears[kinds[n]]


def t16(r):
    kinds = ["swan", "lion", "frog", "rhino"]
    colors = r.sample(["white", "yellow", "green", "gray"], 4)
    color = dict(zip(kinds, colors))
    names = r.sample(["Lily", "Bernhard", "Greg", "Brian", "Julius"], 5)
    lines = []
    for n in names[:-1]:
        k = r.choice(kinds)
        lines += [f"{n} is a {k}.", f"{n} is {color[k]}."]
    seen = [ln.split()[-1].rstrip(".") for ln in lines if " is a " in ln]
    k = r.choice(seen)
    lines.append(f"{names[-1]} is a {k}.")
    return lines, f"What color is {names[-1]}?", color[k]


def t17(r):
    shapes = ["red square", "blue square", "triangle", "pink rectangle"]
    a, b, c = r.sample(shapes, 3)
    rels = [("to the left of", (-1, 0)), ("to the right of", (1, 0)), ("above", (0, 1)),
            ("below", (0, -1))]
    (w1, d1), (w2, d2) = r.choice(rels), r.choice(rels)
    pos = {b: (0, 0)}
    pos[a] = (d1[0], d1[1])
    pos[c] = (pos[a][0] + d2[0], pos[a][1] + d2[1])
    lines = [f"The {a} is {w1} the {b}.", f"The {c} is {w2} the {a}."]
    qw, qd = r.choice(rels)
    dx, dy = pos[c][0] - pos[b][0], pos[c][1] - pos[b][1]
    truth = {"to the left of": dx < 0, "to the right of": dx > 0, "above": dy > 0,
             "below": dy < 0}[qw]
    return lines, f"Is the {c} {qw} the {b}?", "yes" if truth else "no"


def t18(r):
    things = ["chocolate", "box", "suitcase", "chest", "container"]
    a, b, c = r.sample(things, 3)
    lines = [f"The {b} is bigger than the {a}.", f"The {c} is bigger than the {b}."]
    if r.random() < 0.5:
        return lines, f"Does the {a} fit in the {c}?", "yes"
    return lines, f"Is the {a} bigger than the {c}?", "no"


def t19(r):
    places = r.sample(PLACES, 4)
    dirs = ["north", "south", "east", "west"]
    opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
    lines, links = [], []
    step = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    pos = [(0, 0)]
    for i in range(3):  # a self-avoiding walk on a grid, so the map is consistent
        free = [d for d in dirs if (pos[-1][0] + step[d][0], pos[-1][1] + step[d][1]) not in pos]
        d = r.choice(free)
        pos.append((pos[-1][0] + step[d][0], pos[-1][1] + step[d][1]))
        lines.append(f"The {places[i + 1]} is {d} of the {places[i]}.")
        links.append(d)
    r.shuffle(lines)
    if r.random() < 0.5:
        return lines, f"How do you go from the {places[0]} to the {places[3]}?", ",".join(links)
    back = [opp[d] for d in reversed(links)]
    return lines, f"How do you go from the {places[3]} to the {places[0]}?", ",".join(back)


def t20(r):
    motive = {"hungry": "kitchen", "thirsty": "kitchen", "tired": "bedroom", "bored": "garden"}
    names = r.sample(["Antoine", "Sumit", "Yann", "Jason", "John", "Fred"], 5)
    lines = []
    for n in names[:-1]:
        st = r.choice(list(motive))
        lines += [f"{n} is {st}.", f"{n} went to the {motive[st]}."]
    st = r.choice([ln.split()[-1].rstrip(".") for ln in lines if " is " in ln])
    if r.random() < 0.5:
        lines.append(f"{names[-1]} is {st}.")
        return lines, f"Where will {names[-1]} go?", motive[st]
    who = lines[0].split()[0]
    place = lines[1].split()[-1].rstrip(".")
    return lines, f"Why did {who} go to the {place}?", lines[0].split()[-1].rstrip(".")


TASKS = {1: ("single supporting fact", t1), 2: ("two supporting facts", t2),
         3: ("three supporting facts", t3), 4: ("two argument relations", t4),
         5: ("three argument relations", t5), 6: ("yes/no questions", t6),
         7: ("counting", t7), 8: ("lists/sets", t8), 9: ("simple negation", t9),
         10: ("indefinite knowledge", t10), 11: ("basic coreference", t11),
         12: ("conjunction", t12), 13: ("compound coreference", t13),
         15: ("basic deduction", t15), 16: ("basic induction", t16),
         17: ("positional reasoning", t17), 18: ("size reasoning", t18),
         19: ("path finding", t19), 20: ("agent's motivations", t20)}

NUMS = {"none": "0", "nothing": "0", "zero": "0", "one": "1", "two": "2", "three": "3"}


def correct(reply, gold):
    rep = reply.lower()
    if gold in ("yes", "no", "maybe"):
        first = re.findall(r"[a-z]+", rep)[:1]
        return bool(first) and first[0] == gold
    if gold in NUMS:
        first = re.findall(r"[a-z]+", rep)[:1]
        return bool(first) and NUMS.get(first[0], first[0]) == NUMS[gold]
    if "," in gold or gold == "nothing":
        if gold == "nothing":
            return rep.startswith("nothing")
        if all(d in ("north", "south", "east", "west") for d in gold.split(",")):
            return re.findall(r"north|south|east|west", rep) == gold.split(",")
        return sorted(set(re.findall(r"[a-z]+", rep)) & set(OBJECTS)) == gold.split(",")
    if gold in ("north", "south", "east", "west") or "," in gold:
        return re.findall(r"north|south|east|west", rep) == gold.split(",")
    words = set(re.findall(r"[a-z]+", rep))
    singular = {w[:-1] for w in words if w.endswith("s")} | {"wolf" for w in words if w == "wolves"}
    singular |= {"mouse" for w in words if w == "mice"}
    return gold in words or gold in singular


def run(n=100, seed=1, verbose=False):
    r = random.Random(seed)
    results = {}
    t0 = time.time()
    for tid, (name, gen) in TASKS.items():
        ok = 0
        for _ in range(n):
            lines, q, gold = gen(r)
            m = Mind(learn_language=False)
            for ln in lines:
                m.hear(ln)
            reply = m.hear(q)
            good = correct(reply, gold)
            ok += good
            if verbose and not good:
                print(f"[task {tid}] {' '.join(lines)} | {q} -> {reply!r} (gold {gold})")
        results[tid] = ok / n
    dt = time.time() - t0
    return results, dt


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    res, dt = run(n, verbose="-v" in sys.argv)
    for tid, acc in res.items():
        print(f"task {tid:2d} {TASKS[tid][0]:26s} {acc * 100:6.1f}%")
    total = sum(res.values()) / len(res)
    print(f"mean accuracy {total * 100:.1f}% over {len(res)} tasks, {n} stories each, "
          f"{dt:.1f} s total ({dt / (n * len(res)) * 1000:.1f} ms per story)")
