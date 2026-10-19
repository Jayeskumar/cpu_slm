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
from collections import defaultdict
from dataclasses import dataclass, field

from .language import NUMBER_WORDS, plural, third_person

K = 1.0  # evidential horizon


def w2c(w):
    return w / (w + K)


def c2w(c):
    return K * c / (1.0 - c) if c < 1 else 1e9


@dataclass(slots=True)
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
INVERSE["opposite"] = "opposite"
NOT_INHERITED = TRANSITIVE | {"near", "on", "parent_of", "child_of", "opposite"}
ASYMMETRIC = (TRANSITIVE - {"isa", "in", "part_of"}) | {"parent_of", "child_of"}


def inheritable(r):
    return r not in NOT_INHERITED and (not r.endswith("_of") or r in ("afraid_of", "made_of"))


@dataclass(slots=True)
class Fact:
    s: str
    r: str
    o: str
    truth: Truth
    qty: int = None
    source: str = "told"
    uses: list = field(default_factory=list)
    generic: bool = None     # said of a kind ('dogs bark') or of one thing ('the box ...')


@dataclass(slots=True)
class Answer:
    truth: Truth
    premises: list           # facts (or derived Answers) used
    how: str = "direct"      # direct, deduction, transitivity, inheritance, induction, ...
    value: str = None
    stmt: tuple = None       # (s, r, o) of a derived statement
    dist: int = 0            # isa links between the subject and the source category

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
        self.definitions = {}   # concept -> definition (WordNet gloss)
        self.senses = {}        # word -> its concepts, the dominant sense first
        self.aliases = {}       # word -> dominant concept when keyed differently ('metal')
        self.proper = {}        # word -> proper-noun concept ('China' vs 'china')
        self.words = {}         # concept -> other words for it
        self.surface = {}       # concept -> the word people use for it ('country', not 'state')
        self.primed = {}        # word -> the sense recently chosen for it (one sense per discourse)
        self.mass = set()       # mass nouns ('gold', 'water'): 'gold is', not 'gold are'
        self.stuff = set()      # substances and foods: a bare 'Chocolate is ...' is not a name
        self.taxa = set()       # animal and plant classes (disjoint siblings)
        self.authority = {"wordnet", "pack", "innate"}  # curated sources
        self.loaded = set()     # bulk sources already fed ('wordnet', 'pack:general')

    def canon(self, x, proper=False):
        """The concept a word names: the sense chosen for it lately, else its
        dominant sense ('metal' -> 'metallic element'); a capitalised word
        prefers a proper-noun sense, and a capitalised common noun is a name
        ('Lily' the cat is not the lily)."""
        if x in self.primed:
            return self.primed[x]
        if proper:
            if x in self.proper:
                return self.proper[x]
            if (x in self.definitions or x in self.aliases) and x not in self.display:
                return x + "#0"      # a name, not the dictionary word it spells
        return self.aliases.get(x, x)

    def other_senses(self, word, concept):
        return [k for k in self.senses.get(word, ()) if k != concept]

    def curated(self, x):
        """Does x's place in the taxonomy come from a curated source?"""
        return any(f.source in self.authority for f in self.out(x, "isa"))

    # -- memory --------------------------------------------------------------
    def tick(self):
        self.clock += 1

    def tell(self, s, r, o, positive=True, conf=0.9, qty=None, source="told", replace=False,
             generic=None):
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
        fact = Fact(s, r, o, t, qty, source, [self.clock], generic)
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
    def categories_of(self, s, depth=24):
        """All categories of s, nearest first, with the truth of membership:
        [(m, Answer)]; Answer.dist is the number of isa links."""
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
                    out.append((f.o, Answer(t, p, "transitivity" if prem else "direct",
                                            dist=len(p))))
                    nxt.append((f.o, t, p))
            frontier = nxt
        return out

    def reach(self, s, r, depth=12):
        """Everything s is r-related to through a transitive relation, nearest
        first: {o: Answer with the chain of premises}."""
        out, frontier, seen = {}, [(s, None, [])], {s}
        for _ in range(depth):
            nxt = []
            for x, tx, prem in frontier:
                for e in self._edges(x, r):
                    if e.truth.f < 0.5 or e.o in seen:
                        continue
                    t = e.truth if tx is None else Truth.deduction(tx, e.truth)
                    seen.add(e.o)
                    p = prem + [e]
                    out[e.o] = Answer(t, p, "transitivity" if prem else "direct", e.o,
                                      dist=len(p) - 1)
                    nxt.append((e.o, t, p))
            if not nxt:
                break
            frontier = nxt
        return out

    def instances_of(self, m, limit=None):
        out = []
        for f in self.inn(m, "isa"):
            if f.truth.f >= 0.5:
                out.append(f)
                if limit and len(out) >= limit:
                    break
        return out

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
        if r in TRANSITIVE and r != "isa":
            hit = self.reach(s, r).get(o)
            if hit is not None and hit.how == "transitivity":
                cands.append(hit)
        if inheritable(r) or r == "isa":
            for m, mem in self.categories_of(s):
                if r == "isa":
                    if m == o and mem.how != "direct":   # a told exception beats the chain
                        cands.append(Answer(mem.truth, mem.premises, "transitivity",
                                            dist=mem.dist))
                    continue
                g = self.get(m, r, o)
                if g:
                    cands.append(Answer(Truth.deduction(mem.truth, g.truth), mem.premises + [g],
                                        "inheritance", dist=mem.dist))
                elif depth > 0 and mem.dist <= 3:
                    ind = self._induce(m, r, o, exclude=s)
                    if ind.known and self._enough(ind):
                        cands.append(Answer(Truth.deduction(mem.truth, ind.truth),
                                            mem.premises + [ind], "inheritance+induction",
                                            dist=mem.dist))
        if not cands and r == "is" and depth > 0:
            for g in self.out(o, "opposite"):
                op = self.ask(s, "is", g.o, depth - 1, seen)
                if op.known and op.truth.f >= 0.5:
                    cands.append(Answer(Truth(0.0, op.truth.c * 0.9), op.premises + [g],
                                        "opposites"))
        if not cands and r in ASYMMETRIC:
            rev = self.reach(o, r).get(s)
            if rev is not None:
                cands.append(Answer(Truth(0.0, rev.truth.c * 0.95), rev.premises, "asymmetry"))
        if not cands and inheritable(r):
            ind = self._induce(s, r, o)
            if ind.known and self._enough(ind):
                cands.append(ind)
        if not cands and r == "isa":
            neg = self._disjoint(s, o)
            if neg.known:
                cands.append(neg)
        if not cands and r == "in":
            neg = self._elsewhere(s, o)
            if neg.known:
                cands.append(neg)
        if not cands and r == "isa" and not self.curated(s):
            ab = self._abduce(s, o)
            if ab.known:
                cands.append(ab)
        if not cands and self.semantic is not None and inheritable(r):
            an = self._analogy(s, r, o)
            if an.known:
                cands.append(an)
        if not cands:
            return NONE
        return self._choose(cands)

    @staticmethod
    def _choose(cands):
        """Choice between competing conclusions: what is said about the thing
        itself wins; otherwise the most specific category that has an opinion
        wins (inferential distance), then the more confident conclusion."""
        own = [a for a in cands if a.dist == 0]
        if own:
            return max(own, key=lambda a: (a.truth.c, a.how == "direct"))
        stated = [a for a in cands if a.how == "inheritance"]
        cands = stated or cands      # what is stated about a kind beats a guess from examples
        dmin = min(a.dist for a in cands)
        return max((a for a in cands if a.dist == dmin), key=lambda a: a.truth.c)

    _TOO_GENERAL = {"entity", "physical entity", "abstraction", "object", "whole",
                    "living thing", "matter", "causal agent", "thing"}

    def _disjoint(self, s, o):
        """Closed world for curated taxonomies: s and o sit on different
        branches under a shared parent p that o is close to ('bats are
        mammals; mammals and birds are different kinds of vertebrate'), so s
        is (probably) not an o.  stmt = (s's branch, o, p)."""
        cats = self.categories_of(s)
        if not cats or o in {m for m, _ in cats} or not (
                self.curated(s) or any(self.curated(f.o) for f in self.out(s, "isa"))):
            return NONE       # closed world only where the taxonomy is curated
        up_o = {o: (0, [])}
        for m, a in self.categories_of(o, depth=2):
            up_o[m] = (a.dist, a.premises)
        if len(up_o) == 1 or s in up_o:
            return NONE
        mine = {m for m, _ in cats}
        above_o = None
        for p, a in cats:
            if a.dist > 8:
                break
            if p not in up_o or p in self._TOO_GENERAL:
                continue
            chain = a.premises
            branch = chain[-2].o if len(chain) > 1 else s
            o_chain = up_o[p][1]
            o_branch = o_chain[-2].o if len(o_chain) > 1 else o
            if above_o is None:
                above_o = {m for m, _ in self.categories_of(o)}
            if branch == o_branch or branch in above_o or o_branch in mine:
                continue          # o is itself under s's branch: look lower
            # biological siblings are disjoint by construction; other WordNet
            # siblings ('chemical', 'mineral') often overlap
            sure = p == "organism" or (branch in self.taxa and o_branch in self.taxa) or \
                not self.taxa
            return Answer(Truth(0.0, 0.7 if sure else 0.45),
                          chain[:-1] if len(chain) > 1 else [], "taxonomy", stmt=(branch, o, p))
        return NONE

    _REGIONS = ("continent", "country#2", "country", "state#4")

    def _elsewhere(self, s, o):
        """'Is Tokyo in Europe?': Tokyo is in Asia, and continents (like
        countries) do not overlap, so no -- unless s is known to be in o."""
        kind = next((k for k in self._REGIONS if self.ask(o, "isa", k).truth.f >= 0.5
                     and self.ask(o, "isa", k).known), None)
        if kind is None:
            return NONE
        for x, a in self.reach(s, "in").items():
            if x != o and self.get(x, "isa", kind) is not None:
                return Answer(Truth(0.0, 0.7), a.premises, "elsewhere", stmt=(x, o, kind))
        return NONE

    def _edges(self, s, r):
        out = self.out(s, r)
        if r in INVERSE:
            inv = INVERSE[r]
            for g in self.inn(s, inv):
                out.append(Fact(s, r, g.s, g.truth, source="inverse", uses=g.uses,
                                generic=getattr(g, "generic", None)))
        return out

    def _induce(self, m, r, o, exclude=None):
        """Generalise from known instances of category m (NARS induction +
        revision): 'robins fly, sparrows fly, both are birds' -> birds fly."""
        acc, prem = None, []
        for inst in self.instances_of(m, limit=400):
            x = inst.s
            if x == exclude:
                continue
            g = self.get(x, r, o)
            if g is None or g.source == "wordnet":
                continue
            t = Truth.induction(g.truth, inst.truth)
            acc = t if acc is None else acc.revise(t)
            prem += [inst, g]
        if acc is None:
            return NONE
        return Answer(acc, prem, "induction", stmt=(m, r, o))

    def _enough(self, ind):
        """One example the user gave ('Lily is a swan; Lily is white') is
        enough to generalise from; reference data needs two ('salt is made of
        chlorine' says nothing about other compounds)."""
        facts = ind.premises[1::2]
        return len(facts) >= 2 or all(f.source not in self.authority for f in facts)

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
                                    "inheritance", g.o, dist=mem.dist))
                if mem.dist > 2:
                    continue
                for inst in self.instances_of(m, limit=200):
                    if inst.s == s:
                        continue
                    for g in self.out(inst.s, r):
                        ind = self._induce(m, r, g.o, exclude=s)
                        if ind.known and g.o not in found and self._enough(ind):
                            add(g.o, Answer(Truth.deduction(mem.truth, ind.truth),
                                            mem.premises + [ind], "inheritance+induction", g.o))
        if r in TRANSITIVE and r != "isa":
            for o, ans in self.reach(s, r).items():
                if o not in found and o != s:
                    add(o, ans)
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

    @staticmethod
    def bare(x):
        """'gold#3' -> 'gold': the word for a concept that is not its word's
        dominant sense."""
        i = x.find("#")
        return x if i < 0 else x[:i]

    def word(self, x):
        return self.surface.get(x) or self.bare(x)

    def name(self, x, generic=False):
        """Display form: proper names as given, categories in the plural."""
        if x in self.display:
            return self.display[x]
        w = self.word(x)
        return plural(w) if generic and not self._is_mass(x) else w

    _MASS = {"water", "milk", "rice", "music", "information", "money", "fish", "sheep", "food",
             "bread", "gold", "silver", "air", "fire", "sand", "snow", "rain", "grass", "sugar",
             "honey", "salt", "ice", "coal", "wood", "paper", "glass", "iron", "copper", "steel",
             "blood", "wine", "beer", "tea", "coffee", "butter", "cheese", "flour", "oil",
             "steam", "smoke", "oxygen", "hydrogen", "nitrogen", "meat", "sunlight", "electricity"}

    def _is_mass(self, x):
        return x in self.mass or self.bare(x) in self._MASS

    def many(self, x):
        """Is x spoken of in the plural ('dogs are', but 'gold is')?"""
        return self.generic(x) and not self._is_mass(x)

    def ref(self, x):
        """A definite reference: 'Paris', 'the kitchen'."""
        if x == "user":
            return "you"
        if x == "mnemo":
            return "me"
        return self.display[x] if x in self.display else "the " + self.word(x)

    def article(self, x):
        if x in self.display:
            return self.display[x]
        w = self.word(x)
        return ("an " if w[:1] in "aeiou" else "a ") + w

    def generic(self, x):
        return x in self.categories and x not in self.display

    def say(self, f, generic=None):
        """Render a fact (or derived answer) as English."""
        if isinstance(f, Answer):
            return self.say_answer(f)
        subj, pred = self._parts(f, generic)
        return pred if subj is None else f"{subj} {pred}"

    def _obj(self, o, gen=False, qty=None, kind=False):
        """An object phrase: 'Paris', 'meat', 'mice', 'the sun'; kind: after
        a generic subject, the category ('dogs are bigger than cats')."""
        if o in self.display:
            return self.display[o]
        if self._is_mass(o):
            return self.word(o)
        if qty == -1 or (kind and gen and self.generic(o)):
            return self.name(o, True)
        if qty == 1:
            return self.article(o)
        if qty is None and gen and not kind:
            return self.word(o)               # 'pandas eat bamboo'
        return "the " + self.word(o)

    def _parts(self, f, generic=None):
        """(subject, predicate) of a fact; subject None when the predicate is
        a whole clause ('sad is the opposite of happy')."""
        if generic is None:
            generic = getattr(f, "generic", None)
        gen = self.generic(f.s) if generic is None else generic
        subj = self.name(f.s, True) if gen else self.ref(f.s)
        if f.s == "user":
            subj, gen = "you", True
        elif gen and self._is_mass(f.s):
            gen = False           # 'gold is', 'water is made of'
        pos = f.truth.f >= 0.5
        be = "are" if gen else "is"
        do = "do" if gen else "does"
        neg = "" if pos else "not "
        r, o = f.r, f.o
        if r == "isa":
            return subj, f"{be} {neg}{self.name(o, True) if gen else self.article(o)}"
        if r == "can":
            return subj, f"{'can' if pos else 'cannot'} {o}"
        if r == "does":
            return subj, (o if gen else third_person(o)) if pos else f"{do} not {o}"
        if r == "has":
            if not pos:
                return subj, f"{do} not have {self.name(o, True)}"
            have = "have" if gen else "has"
            if f.qty == -1:
                return subj, f"{have} {self.name(o, True)}"
            if f.qty is not None and f.qty > 0:
                q = NUMBER_WORDS.get(f.qty, str(f.qty))
                return subj, f"{have} {q} {self.name(o, f.qty != 1)}"
            return subj, f"{have} {self.article(o)}"
        if r == "is":
            return subj, f"{be} {neg}{o}"
        if r in ("in", "on", "near", "above", "below"):
            return subj, f"{be} {neg}{r} {self.ref(o) if not self._is_mass(o) else self.word(o)}"
        if r.endswith("_than"):
            return subj, f"{be} {neg}{r[:-5]} than {self._obj(o, gen, kind=True)}"
        if r in ("north_of", "south_of", "east_of", "west_of"):
            return subj, f"{be} {neg}{r[:-3]} of {self.ref(o)}"
        if r in ("left_of", "right_of"):
            return subj, f"{be} {neg}to the {r[:-3]} of {self.ref(o)}"
        if r == "afraid_of":
            return subj, f"{be} {neg}afraid of {self.name(o, f.qty == -1)}"
        if r == "made_of":
            return subj, f"{be} {neg}made of {self.name(o)}"
        if r == "opposite":
            return None, f"{self.word(o)} is the opposite of {self.word(f.s)}"
        if r.endswith("_of"):
            return subj, f"{be} {neg}the {r[:-3]} of {self.name(o)}"
        verb = r if gen else third_person(r)
        if pos:
            return subj, f"{verb} {self._obj(o, gen, f.qty)}"
        return subj, f"{do} not {r} {self._obj(o, gen, f.qty)}"

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
        return self._render(a.premises)

    def _render(self, premises, brief=False):
        """Premises as English; a run of is-a links reads as one chain, and
        so does a run of the same relation ('bigger than X, which is ...')."""
        parts, i = [], 0
        while i < len(premises):
            p = premises[i]
            if isinstance(p, Fact) and p.r in TRANSITIVE and p.r != "isa":
                j = i
                while j + 1 < len(premises) and isinstance(premises[j + 1], Fact) and \
                        premises[j + 1].r == p.r and premises[j + 1].s == premises[j].o:
                    j += 1
                if j > i:
                    text = self.say(p)
                    for q in premises[i + 1:j + 1]:
                        text += ", which " + self._parts(q)[1]
                    parts.append(text)
                    i = j + 1
                    continue
            j = i
            while self._isa(premises[j]) and j + 1 < len(premises) and \
                    self._isa(premises[j + 1]) and premises[j + 1].s == premises[j].o:
                j += 1
            if self._isa(premises[i]) and (j > i or brief):
                parts.append(self._chain(premises[i:j + 1], short=brief and j + 1 < len(premises),
                                         elide=brief))
            else:
                parts.append(self.say(premises[i]))
            i = j + 1
        return ", and ".join(parts)

    @staticmethod
    def _isa(p):
        return isinstance(p, Fact) and p.r == "isa" and p.truth.f >= 0.5

    def _chain(self, run, short=False, elide=False):
        """'dogs are canines, carnivores, ... and so animals'; short: just
        'dogs are animals' (the details are kept for 'Why?')."""
        if len(run) <= 2 and not short:
            return ", and ".join(self.say(f) for f in run)
        s = run[0].s
        gen = self.generic(s)
        subj = self.name(s, True) if gen else self.ref(s)
        gen = gen and not self._is_mass(s)          # 'gold is a noble metal'
        classes = [run[-1].o] if short else [f.o for f in run]
        terms = [self.name(c, True) if gen else self.article(c) for c in classes]
        if elide and len(terms) > 5:
            terms = terms[:2] + ["\u2026"] + terms[-2:]
        be = "are" if gen else "is"
        if len(terms) == 1:
            return f"{subj} {be} {terms[0]}"
        return f"{subj} {be} " + ", ".join(terms[:-1]) + " and so " + terms[-1]

    @staticmethod
    def _list(names):
        names = list(dict.fromkeys(names))
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def explain(self, a, brief=False):
        """The evidence behind an answer; brief keeps a reply short."""
        if a.how == "taxonomy" and a.stmt:
            branch, o, p = a.stmt
            lead = ""
            if a.premises:
                lead = (self._chain(a.premises, short=True) if brief
                        else self._render(a.premises)) + "; "
            kinds = " and ".join(self.display.get(x) or
                                 (self.word(x) if self.bare(x) in self._MASS else plural(self.word(x)))
                                 for x in (branch, o))
            return self._cap_first(f"{lead}{kinds} are different kinds of {self.name(p)}") + "."
        if a.how == "elsewhere" and a.stmt:
            x, o, kind = a.stmt
            text = self._render(a.premises, brief)
            return self._cap_first(f"{text}, and {self.ref(x)} is not part of {self.ref(o)}") + "."
        if not a.premises:
            return "I have no evidence either way."
        if a.how == "induction" and a.stmt:
            return self._cap_first(self.say_answer(a)) + "."
        text = self._render(a.premises, brief)
        if a.how == "asymmetry":
            text = "it is the other way round: " + text
        if a.how.startswith("abduction"):
            text = "it shares properties with them: " + text
        return self._cap_first(text) + "."

    @staticmethod
    def _cap_first(t):
        return t[:1].upper() + t[1:]
