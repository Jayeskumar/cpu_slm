"""Declarative memory and reasoning.

Facts are (subject, relation, object) statements carrying a NARS truth value
(Wang 2006/2013): frequency f = w+/w (how often the statement held in the
evidence) and confidence c = w/(w+k) (how much evidence there is, k = 1).
Uncertain conclusions are drawn with NARS truth functions:

  deduction  M->P <f1,c1>, S->M <f2,c2>  |-  S->P <f1 f2, f1 f2 c1 c2>
  induction  M->P <f1,c1>, M->S <f2,c2>  |-  S->P <f1, w2c(f2 c1 c2)>
  abduction  P->M <f1,c1>, S->M <f2,c2>  |-  S->P <f2, w2c(f1 c1 c2)>
  revision   pools independent evidence:  w = w1 + w2
  choice     among competing conclusions, the more confident one wins

so "birds fly, penguins are birds, penguins don't fly" needs no special
exception machinery: the direct, more specific belief is more confident than
the one inherited through an extra link.

Retrieval follows ACT-R: every use of a fact leaves a trace, and its
base-level activation B = ln(sum_j t_j^-d), d = 0.5, ranks what comes to mind
(Anderson & Schooler 1991).
"""
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .language import NUMBER_WORDS, plural, third_person

K = 1.0  # evidential horizon


def w2c(w):
    return w / (w + K)


def c2w(c):
    return K * c / (1.0 - c) if c < 1 else 1e9


@dataclass
class Truth:
    f: float = 0.5
    c: float = 0.0

    @property
    def expectation(self):
        return self.c * (self.f - 0.5) + 0.5

    def revise(self, other):
        w1, w2 = c2w(self.c), c2w(other.c)
        if w1 + w2 == 0:
            return Truth()
        return Truth((w1 * self.f + w2 * other.f) / (w1 + w2), min(0.99, w2c(w1 + w2)))

    @staticmethod
    def deduction(a, b):
        """a = S->M, b = M->P.  Polarity of the property premise is carried
        through (deducing on the negated statement when b is negative)."""
        if b.f >= 0.5:
            return Truth(a.f * b.f, a.f * b.f * a.c * b.c)
        g = 1 - b.f
        return Truth(1 - a.f * g, a.f * g * a.c * b.c)

    @staticmethod
    def induction(m_p, m_s):
        """From an instance M: M->P and M->S give S->P."""
        return Truth(m_p.f, w2c(m_s.f * m_p.c * m_s.c))

    @staticmethod
    def abduction(p_m, s_m):
        return Truth(s_m.f, w2c(p_m.f * p_m.c * s_m.c))

    def __repr__(self):
        return f"<{self.f:.2f}, {self.c:.2f}>"


TRANSITIVE = {
    "isa", "in", "part_of", "bigger_than", "smaller_than", "taller_than", "shorter_than",
    "older_than", "younger_than", "heavier_than", "lighter_than", "faster_than",
    "slower_than", "stronger_than", "weaker_than", "longer_than", "hotter_than",
    "colder_than", "richer_than", "north_of", "south_of", "east_of", "west_of", "left_of",
    "right_of", "above", "below", "ancestor_of",
}
INVERSE = {}
for a, b in [("bigger_than", "smaller_than"), ("taller_than", "shorter_than"),
             ("older_than", "younger_than"), ("heavier_than", "lighter_than"),
             ("faster_than", "slower_than"), ("stronger_than", "weaker_than"),
             ("hotter_than", "colder_than"), ("north_of", "south_of"), ("east_of", "west_of"),
             ("left_of", "right_of"), ("above", "below"), ("parent_of", "child_of")]:
    INVERSE[a], INVERSE[b] = b, a
INVERSE["near"] = "near"
NOT_INHERITED = TRANSITIVE | {"near", "on", "parent_of", "child_of"}
ASYMMETRIC = (TRANSITIVE - {"isa", "in", "part_of"}) | {"parent_of", "child_of"}


def inheritable(r):
    return r not in NOT_INHERITED and (not r.endswith("_of") or r in ("afraid_of", "made_of"))


@dataclass
class Fact:
    s: str
    r: str
    o: str
    truth: Truth
    qty: int = None
    source: str = "told"
    uses: list = field(default_factory=list)


@dataclass
class Answer:
    truth: Truth
    premises: list           # facts (or derived Answers) used
    how: str = "direct"      # direct, deduction, transitivity, inheritance, induction, ...
    value: str = None
    stmt: tuple = None       # (s, r, o) of a derived statement

    @property
    def known(self):
        return self.truth.c > 0.05


NONE = Answer(Truth(0.5, 0.0), [], "unknown")


class Knowledge:
    def __init__(self, semantic=None, decay=0.5):
        self.facts = {}
        self.by_s = defaultdict(set)
        self.by_o = defaultdict(set)
        self.clock = 0
        self.decay = decay
        self.semantic = semantic
        self.display = {}
        self.categories = set()

    # -- memory --------------------------------------------------------------
    def tick(self):
        self.clock += 1

    def tell(self, s, r, o, positive=True, conf=0.9, qty=None, source="told", replace=False,
             generic=False):
        key = (s, r, o)
        t = Truth(1.0 if positive else 0.0, conf)
        old = self.facts.get(key)
        if old is not None and not replace and old.source == source == "told" and \
                (old.truth.f >= 0.5) == positive:
            old.truth = old.truth.revise(t)
            old.uses.append(self.clock)
            if qty is not None:
                old.qty = qty
            return old
        fact = Fact(s, r, o, t, qty, source, [self.clock])
        self.facts[key] = fact
        self.by_s[s].add(key)
        self.by_o[o].add(key)
        if generic:
            self.categories.add(s)
        if r == "isa":
            self.categories.add(o)
        return fact

    def forget(self, s, r, o):
        key = (s, r, o)
        if key in self.facts:
            del self.facts[key]
            self.by_s[s].discard(key)
            self.by_o[o].discard(key)

    def get(self, s, r, o):
        return self.facts.get((s, r, o))

    def out(self, s, r=None):
        return [self.facts[k] for k in self.by_s.get(s, ()) if r is None or k[1] == r]

    def inn(self, o, r=None):
        return [self.facts[k] for k in self.by_o.get(o, ()) if r is None or k[1] == r]

    def activation(self, fact):
        """ACT-R base-level activation of a fact."""
        now = self.clock + 1
        return math.log(sum((now - t + 0.5) ** -self.decay for t in fact.uses))

    def use(self, *facts):
        for f in facts:
            if isinstance(f, Fact):
                f.uses.append(self.clock)

    def concepts(self):
        return set(self.by_s) | set(self.by_o)

    # -- reasoning -------------------------------------------------------------
    def categories_of(self, s, depth=4):
        """All categories of s with the truth of membership: [(m, Answer)]."""
        out, frontier, seen = [], [(s, Truth(1.0, 1.0), [])], {s}
        for _ in range(depth):
            nxt = []
            for x, tx, prem in frontier:
                for f in self.out(x, "isa"):
                    if f.truth.f < 0.5 or f.o in seen:
                        continue
                    t = f.truth if not prem else Truth.deduction(tx, f.truth)
                    seen.add(f.o)
                    p = prem + [f]
                    out.append((f.o, Answer(t, p, "transitivity" if prem else "direct")))
                    nxt.append((f.o, t, p))
            frontier = nxt
        return out

    def instances_of(self, m):
        return [f for f in self.inn(m, "isa") if f.truth.f >= 0.5]

    def ask(self, s, r, o, depth=3, _seen=None):
        """Best-supported answer to 'does s r o hold?' (NARS choice rule)."""
        seen = set() if _seen is None else _seen
        if (s, r, o) in seen or depth < 0:
            return NONE
        seen = seen | {(s, r, o)}
        cands = []
        f = self.get(s, r, o)
        if f:
            cands.append(Answer(f.truth, [f], "direct"))
        if r in INVERSE:
            g = self.get(o, INVERSE[r], s)
            if g:
                cands.append(Answer(g.truth, [g], "inverse"))
        if r == "can":
            g = self.get(s, "does", o)
            if g and g.truth.f >= 0.5:
                cands.append(Answer(Truth(1.0, g.truth.c * 0.95), [g], "does-implies-can"))
        if r == "does":
            g = self.get(s, "can", o)
            if g and g.truth.f < 0.5:
                cands.append(Answer(g.truth, [g], "cannot-implies-does-not"))
        if r in TRANSITIVE and depth > 0:
            for e in self._edges(s, r):
                if e.truth.f < 0.5 or e.o == o:
                    continue
                sub = self.ask(e.o, r, o, depth - 1, seen)
                if sub.known and sub.truth.f >= 0.5:
                    cands.append(Answer(Truth.deduction(e.truth, sub.truth), [e] + sub.premises,
                                        "transitivity"))
        if inheritable(r) or r == "isa":
            for m, mem in self.categories_of(s):
                if r == "isa":
                    if m == o and mem.how != "direct":
                        cands.append(Answer(mem.truth, mem.premises, "transitivity"))
                    continue
                g = self.get(m, r, o)
                if g:
                    cands.append(Answer(Truth.deduction(mem.truth, g.truth), mem.premises + [g],
                                        "inheritance"))
                elif depth > 0:
                    ind = self._induce(m, r, o, exclude=s)
                    if ind.known:
                        cands.append(Answer(Truth.deduction(mem.truth, ind.truth),
                                            mem.premises + [ind], "inheritance+induction"))
        if not cands and r == "is" and depth > 0:
            for g in self.out(o, "opposite"):
                op = self.ask(s, "is", g.o, depth - 1, seen)
                if op.known and op.truth.f >= 0.5:
                    cands.append(Answer(Truth(0.0, op.truth.c * 0.9), op.premises + [g],
                                        "opposites"))
        if not cands and r in ASYMMETRIC and depth > 0:
            rev = self.ask(o, r, s, depth - 1, seen)
            if rev.known and rev.truth.f >= 0.5:
                cands.append(Answer(Truth(0.0, rev.truth.c * 0.95), rev.premises, "asymmetry"))
        if not cands and inheritable(r):
            ind = self._induce(s, r, o)
            if ind.known:
                cands.append(ind)
        if not cands and r == "isa":
            ab = self._abduce(s, o)
            if ab.known:
                cands.append(ab)
        if not cands and self.semantic is not None and inheritable(r):
            an = self._analogy(s, r, o)
            if an.known:
                cands.append(an)
        if not cands:
            return NONE
        return max(cands, key=lambda a: (a.truth.c, a.how == "direct"))

    def _edges(self, s, r):
        out = self.out(s, r)
        if r in INVERSE:
            inv = INVERSE[r]
            for g in self.inn(s, inv):
                out.append(Fact(s, r, g.s, g.truth, source="inverse", uses=g.uses))
        return out

    def _induce(self, m, r, o, exclude=None):
        """Generalise from known instances of category m (NARS induction +
        revision): 'robins fly, sparrows fly, both are birds' -> birds fly."""
        acc, prem = None, []
        for inst in self.instances_of(m):
            x = inst.s
            if x == exclude:
                continue
            g = self.get(x, r, o)
            if g is None:
                continue
            t = Truth.induction(g.truth, inst.truth)
            acc = t if acc is None else acc.revise(t)
            prem += [inst, g]
        if acc is None:
            return NONE
        return Answer(acc, prem, "induction", stmt=(m, r, o))

    def _abduce(self, s, o):
        """'Tweety flies, birds fly' -> maybe Tweety is a bird (weak)."""
        acc, prem = None, []
        for g in self.out(o):
            if g.r == "isa" or not inheritable(g.r):
                continue
            h = self.get(s, g.r, g.o)
            if h is None or (h.truth.f >= 0.5) != (g.truth.f >= 0.5):
                continue
            t = Truth.abduction(g.truth, h.truth)
            acc = t if acc is None else acc.revise(t)
            prem += [h, g]
        return NONE if acc is None else Answer(acc, prem, "abduction")

    def _analogy(self, s, r, o, k=5, min_sim=0.45):
        """Similar concepts (semantic memory) probably share properties."""
        if s not in self.semantic:
            return NONE
        best = NONE
        for y, sim in self.semantic.similar(s, k):
            if sim < min_sim:
                break
            g = self.get(y, r, o)
            if g:
                t = Truth(g.truth.f, g.truth.c * sim * 0.5)
                if t.c > best.truth.c:
                    best = Answer(t, [g], f"analogy with {y} (similarity {sim:.2f})")
        return best

    def objects(self, s, r, positive=True):
        """All o with s r o, inherited and induced ones included."""
        found = {}

        def add(o, ans):
            if o not in found or ans.truth.c > found[o].truth.c:
                found[o] = ans

        for f in self.out(s, r):
            add(f.o, Answer(f.truth, [f], "direct", f.o))
        if r in INVERSE:
            for g in self.inn(s, INVERSE[r]):
                add(g.s, Answer(g.truth, [g], "inverse", g.s))
        if inheritable(r):
            for m, mem in self.categories_of(s):
                for g in self.out(m, r):
                    add(g.o, Answer(Truth.deduction(mem.truth, g.truth), mem.premises + [g],
                                    "inheritance", g.o))
                for inst in self.instances_of(m):
                    if inst.s == s:
                        continue
                    for g in self.out(inst.s, r):
                        ind = self._induce(m, r, g.o, exclude=s)
                        if ind.known and g.o not in found:
                            add(g.o, Answer(Truth.deduction(mem.truth, ind.truth),
                                            mem.premises + [ind], "inheritance+induction", g.o))
        if r in TRANSITIVE:
            for o in list(found):
                for g in self.objects(o, r) if r != "isa" else []:
                    if g.value not in found and g.value != s:
                        add(g.value, Answer(Truth.deduction(found[o].truth, g.truth),
                                            found[o].premises + g.premises, "transitivity",
                                            g.value))
        # the more specific evidence decides each candidate (choice rule)
        out = []
        for o, ans in found.items():
            best = self.ask(s, r, o) if inheritable(r) else ans
            if not best.known:
                best = ans
            best.value = o
            if (best.truth.f >= 0.5) == positive and best.truth.c > 0.05:
                out.append(best)
        out.sort(key=lambda a: (-a.truth.expectation, -a.truth.c))
        return out

    def subjects(self, r, o):
        found = {}
        for f in self.inn(o, r):
            if f.truth.f >= 0.5:
                found[f.s] = Answer(f.truth, [f], "direct", f.s)
        if r in INVERSE:
            for g in self.out(o, INVERSE[r]):
                if g.truth.f >= 0.5 and g.o not in found:
                    found[g.o] = Answer(g.truth, [g], "inverse", g.o)
        return sorted(found.values(), key=lambda a: -a.truth.c)

    # -- language --------------------------------------------------------------
    def is_proper(self, x):
        return x in self.display

    def name(self, x, generic=False):
        """Display form: proper names as given, categories in the plural."""
        if x in self.display:
            return self.display[x]
        return plural(x) if generic and not self._is_mass(x) else x

    @staticmethod
    def _is_mass(x):
        return x in ("water", "milk", "rice", "music", "information", "money", "fish", "sheep",
                     "food", "bread", "gold", "silver", "air", "fire", "sand", "snow", "rain")

    def ref(self, x):
        """A definite reference: 'Paris', 'the kitchen'."""
        if x == "user":
            return "you"
        if x == "mnemo":
            return "me"
        return self.display[x] if x in self.display else "the " + x

    def article(self, x):
        if x in self.display:
            return self.display[x]
        return ("an " if x[:1] in "aeiou" else "a ") + x

    def generic(self, x):
        return x in self.categories and x not in self.display

    def say(self, f, generic=None):
        """Render a fact (or derived answer) as English."""
        if isinstance(f, Answer):
            return self.say_answer(f)
        gen = self.generic(f.s) if generic is None else generic
        subj = self.name(f.s, True) if gen else self.ref(f.s)
        if f.s == "user":
            subj, gen = "you", True
        pos = f.truth.f >= 0.5
        be = "are" if gen else "is"
        do = "do" if gen else "does"
        neg = "" if pos else "not "
        r, o = f.r, f.o
        if r == "isa":
            obj = self.name(o, True) if gen else self.article(o)
            return f"{subj} {be} {neg}{obj}"
        if r == "can":
            return f"{subj} {'can' if pos else 'cannot'} {o}"
        if r == "does":
            return f"{subj} {o if gen else third_person(o)}" if pos else f"{subj} {do} not {o}"
        if r == "has":
            if not pos:
                return f"{subj} {do} not have {self.name(o, True)}"
            have = "have" if gen else "has"
            if f.qty == -1:
                return f"{subj} {have} {self.name(o, True)}"
            if f.qty is not None:
                q = NUMBER_WORDS.get(f.qty, str(f.qty))
                return f"{subj} {have} {q} {self.name(o, f.qty != 1)}"
            return f"{subj} {have} {self.article(o)}"
        if r == "is":
            return f"{subj} {be} {neg}{o}"
        if r in ("in", "on", "near", "above", "below"):
            return f"{subj} {be} {neg}{r} {self.ref(o)}"
        if r.endswith("_than"):
            return f"{subj} {be} {neg}{r[:-5]} than {self.ref(o)}"
        if r in ("north_of", "south_of", "east_of", "west_of"):
            return f"{subj} {be} {neg}{r[:-3]} of {self.ref(o)}"
        if r in ("left_of", "right_of"):
            return f"{subj} {be} {neg}to the {r[:-3]} of {self.ref(o)}"
        if r == "afraid_of":
            return f"{subj} {be} {neg}afraid of {self.name(o, f.qty == -1)}"
        if r == "made_of":
            return f"{subj} {be} {neg}made of {self.name(o)}"
        if r == "opposite":
            return f"{o} is the opposite of {f.s}"
        if r.endswith("_of"):
            return f"{subj} {be} {neg}the {r[:-3]} of {self.name(o)}"
        if pos:
            return f"{subj} {r if gen else third_person(r)} {self.name(o, f.qty == -1)}"
        return f"{subj} {do} not {r} {self.name(o, f.qty == -1)}"

    def say_answer(self, a):
        if a.how.startswith("induction"):
            insts = [p for p in a.premises if isinstance(p, Fact) and p.r == "isa"]
            names = [self.name(p.s, self.generic(p.s)) for p in insts]
            if a.stmt:
                m, r, o = a.stmt
                txt = self.say(Fact(m, r, o, Truth(round(a.truth.f), a.truth.c)), generic=True)
                return f"judging from {self._list(names)}, {txt}"
            return f"judging from {self._list(names)}"
        if a.how.startswith("analogy"):
            return f"(by {a.how}: " + "; ".join(self.say(p) for p in a.premises) + ")"
        return ", and ".join(self.say(p) for p in a.premises)

    @staticmethod
    def _list(names):
        names = list(dict.fromkeys(names))
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def explain(self, a):
        if not a.premises:
            return "I have no evidence either way."
        if a.how == "induction" and a.stmt:
            return self._cap_first(self.say_answer(a)) + "."
        parts = []
        for p in a.premises:
            parts.append(self.say(p))
        text = ", and ".join(parts)
        if a.how == "asymmetry":
            text = "it is the other way round: " + text
        if a.how.startswith("abduction"):
            text = "it shares properties with them: " + text
        return self._cap_first(text) + "."

    @staticmethod
    def _cap_first(t):
        return t[:1].upper() + t[1:]
