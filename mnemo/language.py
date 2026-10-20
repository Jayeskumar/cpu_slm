"""From simple English to meaning, and back.

A small rule-based reader: sentences are normalised (contractions expanded,
determiners removed, nouns singularised) and matched against patterns for
statements (facts), events (things happening in the world), questions and
commands.  Pronouns are resolved against a working memory of recently
mentioned entities (capacity ~4, Cowan 2001).

This is deliberately transparent: every parse can be inspected, and the
coverage limits are explicit (see ``PATTERNS`` below).
"""
import functools
import re
from collections import deque
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# word forms

_IRREGULAR_PLURALS = {
    "men": "man", "women": "woman", "children": "child", "people": "person", "mice": "mouse",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "oxen": "ox", "wolves": "wolf",
    "knives": "knife", "wives": "wife", "lives": "life", "leaves": "leaf", "halves": "half",
    "loaves": "loaf", "thieves": "thief", "shelves": "shelf", "calves": "calf",
    "sheep": "sheep", "fish": "fish", "deer": "deer", "moose": "moose", "cacti": "cactus",
    "fungi": "fungus", "data": "data", "criteria": "criterion", "phenomena": "phenomenon",
    "dice": "die", "lice": "louse", "elves": "elf", "scarves": "scarf", "hooves": "hoof",
    "emus": "emu", "kiwis": "kiwi", "gnus": "gnu", "menus": "menu", "taxis": "taxi",
    "skis": "ski", "tutus": "tutu", "safaris": "safari", "tsunamis": "tsunami",
}
# -ses plurals of words that end in -s ('buses'); other -ses plurals drop only
# the s ('horses', 'houses', 'roses')
_S_STEMS = {"bus", "gas", "virus", "bonus", "campus", "cactus", "octopus", "status", "focus",
            "chorus", "circus", "genius", "census", "walrus", "platypus", "atlas", "canvas",
            "alias", "bias", "iris", "lens", "plus", "christmas", "fungus", "hippopotamus"}
_NOT_PLURAL = {
    "is", "was", "has", "does", "this", "his", "hers", "its", "yes", "us", "bus", "gas", "plus",
    "news", "physics", "mathematics", "always", "perhaps", "species", "series", "chaos",
    "glass", "grass", "class", "mass", "boss", "moss", "loss", "less", "kiss", "dress", "chess",
    "lens", "atlas", "iris", "virus", "cactus", "octopus", "status", "campus", "bonus",
    "analysis", "basis", "thesis", "crisis", "axis", "tennis", "christmas", "texas", "paris",
    "james", "charles", "thomas", "jesus", "venus", "mars", "zeus", "athens", "wales",
    "famous", "nervous", "dangerous", "delicious", "curious", "various", "serious", "previous",
    "sometimes", "across", "afterwards", "towards", "besides", "unless", "nowhere", "gas",
}
# names and pluralia tantum that end in -s but are one thing (extended by the
# WordNet importer with every such lemma)
KEEP_PLURAL = {w.strip() for w in """netherlands|united states|united states of america|
philippines|bahamas|brussels|honduras|laos|seychelles|comoros|barbados|maldives|andes|alps|
himalayas|pyrenees|rockies|azores|marshall islands|solomon islands|united arab emirates|
falkland islands|faroe islands|scissors|trousers|jeans|pants|clothes|shorts|pliers|tongs|
binoculars|goggles""".split("|")}

_IRREGULAR_VERBS = {
    "went": "go", "goes": "go", "gone": "go", "came": "come", "ran": "run", "ate": "eat",
    "eaten": "eat", "took": "take", "taken": "take", "gave": "give", "given": "give",
    "got": "get", "made": "make", "saw": "see", "seen": "see", "knew": "know", "known": "know",
    "thought": "think", "brought": "bring", "bought": "buy", "caught": "catch",
    "taught": "teach", "flew": "fly", "flies": "fly", "drank": "drink", "swam": "swim",
    "sang": "sing", "wrote": "write", "written": "write", "fell": "fall", "felt": "feel",
    "kept": "keep", "left": "leave", "lost": "lose", "met": "meet", "paid": "pay",
    "said": "say", "sold": "sell", "sent": "send", "sat": "sit", "slept": "sleep",
    "spoke": "speak", "stood": "stand", "told": "tell", "won": "win", "wore": "wear",
    "built": "build", "began": "begin", "broke": "break", "chose": "choose", "drove": "drive",
    "found": "find", "grew": "grow", "held": "hold", "hid": "hide", "led": "lead",
"lit": "light", "put": "put", "read": "read", "rode": "ride", "rose": "rise",
    "shook": "shake", "shot": "shoot", "stole": "steal", "struck": "strike", "threw": "throw",
    "understood": "understand", "woke": "wake", "has": "have", "had": "have", "does": "do",
    "did": "do", "was": "be", "were": "be", "is": "be", "are": "be", "am": "be",
}


def singular(word):
    """Singular form of an English noun (rules plus exceptions)."""
    w = word.lower()
    if w in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[w]
    if len(w) <= 3 or w in _NOT_PLURAL or w.endswith(("ss", "us", "is", "ous")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes") and len(w) > 4:
        return w[:-2]
    if w.endswith(("ches", "shes", "xes", "zes", "sses")):
        return w[:-2]
    if w.endswith("ses") and len(w) > 4:
        stem = w[:-2]
        return stem if stem in _S_STEMS or stem in _NOT_PLURAL else w[:-1]
    if w.endswith("s"):
        return w[:-1]
    return w


def is_plural(word):
    return singular(word) != word.lower()


_SAME_PLURAL = {"sheep", "fish", "deer", "moose", "salmon", "trout", "cod", "bison", "squid",
                "shrimp", "elk", "swine", "aircraft", "offspring", "species", "series"}
_O_ES = {"tomato", "potato", "hero", "echo", "torpedo", "veto", "mosquito", "volcano",
         "buffalo", "domino", "embargo", "tornado", "mango", "flamingo", "cargo", "motto"}


def plural(word):
    head, sep, last = word.rpartition(" ")      # 'eucalyptus leaf' -> 'eucalyptus leaves'
    for p, s in _IRREGULAR_PLURALS.items():
        if s == last and p != s:
            return head + sep + p
    if word in _SAME_PLURAL or word.split()[-1] in _SAME_PLURAL:
        return word
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")) or word.split()[-1] in _O_ES:
        return word + "es"
    return word + "s"


def verb_lemma(v):
    v = v.lower()
    if v in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[v]
    if v.endswith("ies") and len(v) > 4:
        return v[:-3] + "y"
    if v.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return v[:-2]
    if v.endswith("s") and not v.endswith("ss") and len(v) > 3:
        return v[:-1]
    if v.endswith("ed") and len(v) > 4:
        base = v[:-2]
        if base.endswith(base[-1] * 2) and base[-1] not in "ls":
            return base[:-1]
        if base.endswith("i"):
            return base[:-1] + "y"
        return base if not base.endswith(("v", "c", "u")) else base + "e"
    return v


def third_person(v):
    if v == "be":
        return "is"
    if v == "have":
        return "has"
    if v in ("do", "go"):
        return v + "es"
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    return v + "s"


_NUMBERS = {
    "zero": 0, "no": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "hundred": 100, "thousand": 1000, "many": None, "several": None,
}
NUMBER_WORDS = {v: k for k, v in _NUMBERS.items() if isinstance(v, int) and k not in ("a", "an", "no")}

_DETERMINERS = {"a", "an", "the", "some", "all", "every", "each", "any", "most", "many",
                "this", "that", "these", "those", "my", "your", "our", "its"}

FEMALE = set("""mary sandra julie emily sarah anna lisa kate jane susan alice lily emma olivia
sophia mia ella grace chloe amy lucy ruth helen laura linda maria nancy karen betty diana eve
jessica jennifer elizabeth barbara margaret patricia gertrude winona yann""".split())
MALE = set("""john daniel bill fred jeff tom bob jason antoine sumit yann bernhard brian greg
julius peter paul james mark david michael robert william richard joseph charles george henry
jack harry adam sam ben max luke noah ryan tweety opus""".split()) - {"tweety", "opus", "yann"}

# ---------------------------------------------------------------------------
# parse results


@dataclass
class Fact:
    s: str
    r: str
    o: str
    positive: bool = True
    qty: int = None
    generic: bool = False


@dataclass
class Event:
    kind: str           # move, take, drop, give, be_at, not_at, either_at, feel
    agent: str
    obj: str = None
    place: object = None
    recipient: str = None


@dataclass
class Query:
    kind: str           # yesno, objects, subjects, describe, where, where_before, carrying,
    #                     count_carrying, who_has, who_gave, what_gave, attr, why, similar,
    #                     analogy, path, how_many, why_go, will_go, hdc_analogy
    s: str = None
    r: str = None
    o: str = None
    extra: object = None


@dataclass
class Command:
    kind: str           # generate, forget, stats, explain
    text: str = ""


@dataclass
class Parse:
    items: list = field(default_factory=list)
    sentence: str = ""


# ---------------------------------------------------------------------------
# normalisation

_CONTRACTIONS = [
    (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon't\b", "will not"),
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"), (r"\bdidn't\b", "did not"),
    (r"\bisn't\b", "is not"), (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"), (r"\bhasn't\b", "has not"), (r"\bhaven't\b", "have not"),
    (r"\bcouldn't\b", "could not"), (r"\bshouldn't\b", "should not"), (r"\bit's\b", "it is"),
    (r"\bwhat's\b", "what is"), (r"\bwhere's\b", "where is"), (r"\bwho's\b", "who is"),
    (r"\bthat's\b", "that is"), (r"\bhe's\b", "he is"), (r"\bshe's\b", "she is"),
    (r"\bi'm\b", "i am"), (r"\byou're\b", "you are"), (r"\bthey're\b", "they are"),
]
_DISCOURSE = re.compile(
    r"^(then|after that|afterwards|following that|next|and then|later|meanwhile|finally|"
    r"also|so|well|now|ok|okay|hey|please|yesterday|this morning|this afternoon|"
    r"this evening|today)\s*,?\s+")


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize(sentence):
    s = sentence.strip()
    question = s.endswith("?")
    s = s.rstrip(".!?").strip()
    s = s.replace("’", "'")
    low = s.lower()
    for pat, rep in _CONTRACTIONS:
        low = re.sub(pat, rep, low)
    low = re.sub(r"[,;:\"()]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    while True:
        m = _DISCOURSE.match(low)
        if not m:
            break
        low = low[m.end():]
    return low, question


_NAMES = set()   # names capitalised in the sentence being parsed ('Buenos Aires')


def _whole(phrase):
    """A name or plural-looking singular that must not be singularised."""
    return phrase in KEEP_PLURAL or phrase in _NAMES


def noun(np_text):
    """Canonical concept name of a noun phrase: determiners dropped, head
    noun singular.  'the big Red Apples' -> 'big red apple'."""
    words = [w for w in np_text.lower().split() if w not in _DETERMINERS]
    if not words or _whole(" ".join(words)):
        return " ".join(words)
    words[-1] = singular(words[-1])
    return " ".join(words)


def np_is_plural(np_text):
    words = np_text.lower().split()
    if not words or words[0] in ("a", "an"):
        return False
    content = [w for w in words if w not in _DETERMINERS]
    if content == words and words[-1] in ("fish", "sheep", "deer", "moose"):
        return True               # bare 'fish can swim' is about fish in general
    return not _whole(" ".join(content)) and is_plural(words[-1])


def quantity(np_text):
    """'four legs' -> (4, 'leg'); 'a tail' -> (1, 'tail'); 'legs' -> (None, 'leg')."""
    words = np_text.lower().split()
    if words and words[0] in _NUMBERS and len(words) > 1:
        n = _NUMBERS[words[0]]
        return n, noun(" ".join(words[1:]))
    if words and words[0].isdigit() and len(words) > 1:
        return int(words[0]), noun(" ".join(words[1:]))
    if np_is_plural(np_text):
        return -1, noun(np_text)
    return None, noun(np_text)


# ---------------------------------------------------------------------------
# working memory for reference resolution


class WorkingMemory:
    """Recently mentioned entities, most recent first (bounded like human
    working memory)."""

    def __init__(self, capacity=4):
        self.items = deque(maxlen=capacity)
        self.groups = deque(maxlen=2)
        self.gender = {}

    def mention(self, entity, gender=None):
        if not entity:
            return
        if gender:
            self.gender[entity] = gender
        if entity in self.items:
            self.items.remove(entity)
        self.items.appendleft(entity)

    def mention_group(self, entities):
        self.groups.appendleft(list(entities))
        for e in entities:
            self.mention(e)

    def guess_gender(self, e):
        if e in self.gender:
            return self.gender[e]
        if e in FEMALE:
            return "f"
        if e in MALE:
            return "m"
        return None

    def resolve(self, pronoun):
        p = pronoun.lower()
        if p in ("they", "them", "their"):
            return self.groups[0] if self.groups else (list(self.items)[:1] or None)
        want = {"he": "m", "him": "m", "his": "m", "she": "f", "her": "f", "hers": "f"}.get(p)
        for e in self.items:
            g = self.guess_gender(e)
            if p in ("it", "its"):
                if g is None:
                    return [e]
            elif want is None or g == want or g is None:
                return [e]
        return None


_PRONOUNS = r"\b(he|she|it|they|him|her|them)\b"


def resolve_pronouns(low, wm, keep=()):
    """Replace pronouns by their referents (a group becomes 'x and y');
    keep: words that are not pronouns here ('the symbol He')."""
    def sub(m):
        if m.group(1) in keep:
            return m.group(0)
        ref = wm.resolve(m.group(1))
        if not ref:
            return m.group(0)
        return " and ".join(ref)
    return re.sub(_PRONOUNS, sub, low)


# ---------------------------------------------------------------------------
# patterns

NP = r"(?P<{}>[a-z0-9' -]+?)"
_MOVE = (r"(?:went|go|goes|moved|move|moves|travelled|traveled|travels|travel|journeyed|journeys|"
         r"journey|ran|runs|run|walked|walks|walk|came|comes|come|returned|returns|headed|"
         r"heads|rushed|hurried|drove|flew)(?: back)?")
_TAKE = r"(?:picked up|picks up|pick up|got|gets|get|grabbed|grabs|grab|took|takes|take)"
_DROP = (r"(?:dropped|drops|drop|discarded|discards|discard|put down|puts down|left|leaves|"
         r"leave|set down)")
_GIVE = r"(?:gave|gives|give|handed|hands|hand|passed|passes|pass)"
DIRECTIONS = {"north": "north_of", "south": "south_of", "east": "east_of", "west": "west_of"}
SPATIAL = {
    "in": "in", "at": "in", "inside": "in", "within": "in", "on": "on", "on top of": "on",
    "under": "below", "below": "below", "beneath": "below", "above": "above", "over": "above",
    "near": "near", "next to": "near", "beside": "near", "to the left of": "left_of",
    "left of": "left_of", "to the right of": "right_of", "right of": "right_of",
    "north of": "north_of", "south of": "south_of", "east of": "east_of", "west of": "west_of",
    "part of": "part_of",
}
_SPATIAL_RE = "|".join(sorted((re.escape(k) for k in SPATIAL), key=len, reverse=True))
COMPARATIVES = {
    "bigger": "bigger_than", "larger": "bigger_than", "smaller": "smaller_than",
    "taller": "taller_than", "shorter": "shorter_than", "older": "older_than",
    "younger": "younger_than", "heavier": "heavier_than", "lighter": "lighter_than",
    "faster": "faster_than", "slower": "slower_than", "stronger": "stronger_than",
    "weaker": "weaker_than", "longer": "longer_than", "hotter": "hotter_than",
    "colder": "colder_than", "richer": "richer_than",
}
_COMP_RE = "|".join(COMPARATIVES)
ATTRIBUTE_KINDS = ("color", "colour", "size", "shape", "taste", "temperature", "texture",
                   "material", "mood", "age")


def _s(name):
    return NP.format(name)


@functools.lru_cache(maxsize=None)
def _rx(p):
    return re.compile("^" + p + "$").match


def _split_and(text):
    parts = [p.strip() for p in re.split(r"\s+and\s+|\s*,\s*", text) if p.strip()]
    return parts


def parse_sentence(sentence, wm=None, names=()):
    """Parse one sentence into a list of Fact / Event / Query / Command.
    names: lower-cased names that were capitalised in the sentence."""
    global _NAMES
    _NAMES = set(names)
    try:
        return _parse_sentence(sentence, wm)
    finally:
        _NAMES = set()


def _parse_sentence(sentence, wm=None):
    wm = wm or WorkingMemory()
    raw = sentence.strip()
    low, question = normalize(raw)
    if not low:
        return Parse([], raw)
    gen = re.match(r"^(?:complete|continue|write|imagine|generate)\s*:?\s+(.+)$", low, re.S)
    if gen:
        text = re.sub(r"^(?:complete|continue|write|imagine|generate)\s*:?\s+", "", raw,
                      flags=re.I)
        return Parse([Command("generate", text)], raw)
    if low in ("why", "explain", "how do you know", "how do you know that"):
        return Parse([Query("why")], raw)
    if low in ("stats", "statistics", "status", "what do you know", "memory"):
        return Parse([Command("stats")], raw)
    # a capitalised 'He' inside a sentence is a name or a symbol (helium), not 'he'
    keep = {w.lower() for w in re.findall(r"(?<=[a-z,] )(He|She|It|They|Him|Her|We|Us)\b", raw)}
    low = resolve_pronouns(low, wm, keep)
    items = (_parse_question(low, wm) if question or _looks_like_question(low)
             else _parse_statement(low, wm))
    return Parse(items or [], raw)


def _looks_like_question(low):
    if re.match(r"^(tell (?:me|user)|describe|explain|list|name|define|give|show|mention)\b",
                low):
        return True
    return bool(re.match(r"^(who|what|where|when|why|how|which|is|are|can|could|does|do|did|"
                         r"was|were|has|have)\b", low))


# -- statements ------------------------------------------------------------

def _parse_statement(low, wm):
    m = re.match(r"^(.+?) and (.+?) (" + _MOVE + r"|" + _TAKE + r"|" + _DROP + r") (.+)$", low)
    if m and " and " not in m.group(1):
        agents = [noun(m.group(1)), noun(m.group(2))]
        wm.mention_group(agents)
        out = []
        for a in agents:
            out += _parse_statement(f"{a} {m.group(3)} {m.group(4)}", wm)
        return out

    # lives: 'Tom was born in 1930', 'Sue was born in Paris', 'Tom died in 2001'
    m = _rx(_s("s") + r" (?:was|were) born in (?:the year )?(?P<y>\d{1,4}(?: bc)?)")(low)
    if m:
        return [Fact(_subject(m["s"], wm), "born", m["y"].replace(" bc", " BC"))]
    m = _rx(_s("s") + r" (?:died|passed away) in (?:the year )?(?P<y>\d{1,4}(?: bc)?)")(low)
    if m:
        return [Fact(_subject(m["s"], wm), "died", m["y"].replace(" bc", " BC"))]
    m = _rx(_s("s") + r" (?:was|were) born in " + _s("p"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "born_in", noun(m["p"]))]
    m = _rx(_s("s") + r" (?:is|was|are|were|comes|came) from " + _s("p"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "from", noun(m["p"]))]

    # world events
    m = _rx(_s("a") + r" " + _MOVE + r" (?:to|into|towards|toward) " + _s("p"))(low)
    if m and _agent_ok(m["a"]):
        a, p = noun(m["a"]), noun(m["p"])
        wm.mention(a)
        return [Event("move", a, place=p)]
    m = _rx(_s("a") + r" " + _TAKE + r" " + _s("o") + r"(?: there| up)?")(low)
    if m and _agent_ok(m["a"]) and not m["o"].startswith(("to ", "into ")):
        a, o = noun(m["a"]), noun(m["o"])
        wm.mention(a)
        return [Event("take", a, obj=o)]
    m = _rx(_s("a") + r" " + _GIVE + r" " + _s("o") + r" to " + _s("r"))(low)
    if m and _agent_ok(m["a"]):
        a, o, r = noun(m["a"]), noun(m["o"]), noun(m["r"])
        wm.mention(a)
        wm.mention(r)
        return [Event("give", a, obj=o, recipient=r)]
    m = _rx(_s("a") + r" " + _DROP + r" " + _s("o") + r"(?: there| down)?")(low)
    if m and _agent_ok(m["a"]) and not re.match(r"^(to|for|from|of)\b", m["o"]):
        a, o = noun(m["a"]), noun(m["o"])
        wm.mention(a)
        return [Event("drop", a, obj=o)]
    m = _rx(_s("a") + r" (?:is|are|was) no longer (?:in|at) " + _s("p"))(low)
    if m:
        a = noun(m["a"])
        wm.mention(a)
        return [Event("not_at", a, place=noun(m["p"]))]
    m = _rx(_s("a") + r" (?:is|are|was) either (?:in|at) " + _s("p") + r" or (?:in |at )?" +
            _s("q"))(low)
    if m:
        a = noun(m["a"])
        wm.mention(a)
        return [Event("either_at", a, place=[noun(m["p"]), noun(m["q"])])]

    # negation markers
    m = _rx(_s("s") + r" (?:is|are|was|were) not (?P<c>" + _COMP_RE + r") than " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), COMPARATIVES[m["c"]], noun(m["o"]), False,
                     generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) not " + _s("o"))(low)
    if m:
        return _fact_is(m["s"], m["o"], wm, positive=False)
    m = _rx(_s("s") + r" (?:can|could|will) not " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        return [Fact(s, "can", _vp(m["o"]), False, generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:can|could) " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        return [Fact(s, "can", _vp(m["o"]), True, generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:does|do|did) not (?:have|own) " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        q, o = quantity(m["o"])
        return [Fact(s, "has", o, False, q)]
    m = _rx(_s("s") + r" (?:has|have|had) (?:no|not|not any) " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        return [Fact(s, "has", noun(m["o"]), False)]
    m = _rx(_s("s") + r" (?:has|have|had|owns|own) " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        q, o = quantity(m["o"])
        return [Fact(s, "has", o, True, q, generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:does|do|did) not " + r"(?P<v>[a-z]+) " + _s("o"))(low)
    if m:
        s = _subject(m["s"], wm)
        return [Fact(s, verb_lemma(m["v"]), noun(m["o"]), False)]

    # copula
    m = _rx(_s("s") + r" (?:is|are|was|were) (?:a |an )?(?:kind|type|sort) of " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "isa", noun(m["o"]), True, generic=True)]
    m = _rx(_s("s") + r" (?:is|are|was|were) (?P<c>" + _COMP_RE + r") than " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), COMPARATIVES[m["c"]], noun(m["o"]),
                     generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) (?P<d>north|south|east|west) of " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), DIRECTIONS[m["d"]], noun(m["o"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) (?P<p>" + _SPATIAL_RE + r") " + _s("o"))(low)
    if m:
        s, o, rel = _subject(m["s"], wm), noun(m["o"]), SPATIAL[m["p"]]
        out = [Fact(s, rel, o)]
        if rel == "in":
            out.append(Event("be_at", s, place=o))
        return out
    m = _rx(_s("s") + r" (?:fits|fit) (?:inside|in|into) " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "smaller_than", noun(m["o"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) (?:afraid|scared|frightened) of " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "afraid_of", noun(m["o"]), True,
                     -1 if np_is_plural(m["o"]) else None, generic=np_is_plural(m["s"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) made of " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), "made_of", noun(m["o"]))]
    m = _rx(r"the (?P<r>[a-z]+) of " + _s("o") + r" (?:is|are|was|were) " + _s("s"))(low)
    if m and m["r"] not in ("kind", "type", "sort"):
        return [Fact(noun(m["s"]), m["r"] + "_of", _subject(m["o"], wm))]
    m = _rx(_s("s") + r" (?:is|are|was|were) the (?P<r>[a-z]+) of " + _s("o"))(low)
    if m:
        return [Fact(_subject(m["s"], wm), m["r"] + "_of", noun(m["o"]))]
    m = _rx(_s("o") + r"'s (?P<r>[a-z]+) (?:is|was) " + _s("s"))(low)
    if m:
        return [Fact(noun(m["s"]), m["r"] + "_of", noun(m["o"]))]
    m = _rx(_s("s") + r" (?:is|are|was|were) (?P<o>.+)")(low)
    if m:
        return _fact_is(m["s"], m["o"], wm, positive=True)

    # intransitive: "dogs bark", "the baby sleeps"
    words = low.split()
    if 2 <= len(words) <= 4 and words[-1] not in _DETERMINERS and \
            verb_lemma(words[-2]) not in VERBS:     # not 'people need sleep'
        v = verb_lemma(words[-1])
        if (v in VERBS or v in INTRANSITIVE) and words[-1] != v or v in INTRANSITIVE:
            s_np = " ".join(words[:-1])
            return [Fact(_subject(s_np, wm), "does", v, True, generic=np_is_plural(s_np))]

    # generic subject-verb-object
    m = _rx(_s("s") + r" (?:is |are |was |were )?(?P<v>[a-z]+) (?P<p>" + _SPATIAL_RE + r") " +
            _s("o"))(low)
    if m and (m["v"] in _POSTURE or verb_lemma(m["v"]) in _POSTURE):
        s, o, rel = _subject(m["s"], wm), noun(m["o"]), SPATIAL[m["p"]]
        out = [Fact(s, rel, o, generic=_generic_subject(m["s"], m["v"]))]
        if rel == "in":
            out.append(Event("be_at", s, place=o))
        return out
    svo = _split_svo(low)
    if svo:
        s_np, v, o_np = svo
        s = _subject(s_np, wm)
        if v in ("live", "stay", "reside") and o_np.startswith(("in ", "at ")):
            return [Fact(s, "in", noun(o_np[3:]))]
        if v in ("feel", "become", "get", "seem") and o_np in FEELINGS:
            return [Fact(s, "is", o_np), Event("feel", s, obj=o_np)]
        det = o_np.split()[0]
        qty = -1 if np_is_plural(o_np) else 0 if det == "the" else 1 if det in ("a", "an") \
            else None    # kept for rendering: 'orbits the sun', 'eats bamboo', 'likes an apple'
        return [Fact(s, v, noun(o_np), True, qty,
                     generic=_generic_subject(s_np, low.split()[len(s_np.split())]))]
    return []


def _generic_subject(s_np, verb_word):
    """Plural nouns, or bare nouns with a base-form verb ('fish live')."""
    if np_is_plural(s_np):
        return True
    words = s_np.split()
    return (len(words) == 1 and words[0] not in _DETERMINERS and verb_word == verb_lemma(verb_word)
            and verb_word not in ("was", "is", "has", "did"))


INTRANSITIVE = set("""bark meow purr fly swim run walk sleep sing dance laugh cry smile shine
glow grow melt freeze burn float sink jump climb crawl hop roar howl hiss buzz bloom rain snow
breathe hibernate migrate""".split())

VERBS = set("""like love hate fear want need know see drink eat play read write sing live own
chase bite kill help visit build make use wear carry hold teach study speak watch hunt catch
grow produce contain cause prefer enjoy admire respect trust marry follow lead rule govern
invent discover paint compose drive fly swim climb attack protect feed bring sell buy
find meet call meet resemble support oppose defeat beat win lose sleep cook bake collect
play guard serve visit inhabit cross border orbit conduct attract pump lay spin store
absorb reflect carry pollinate digest""".split())


def _split_svo(low):
    """Split 'the old man likes red apples' into ('the old man', 'like',
    'red apples') using a verb vocabulary and third-person/past forms."""
    words = low.split()
    for i in range(1, len(words) - 1):
        w = words[i]
        if w in _DETERMINERS or w in ("and", "or", "of", "to", "in", "on", "at", "not"):
            continue
        lem = verb_lemma(w)
        known = lem in VERBS or w in _IRREGULAR_VERBS
        third = w.endswith("s") and not w.endswith("ss") and words[i - 1] not in _DETERMINERS
        if known or (third and i == len(words) - 2 and len(words) >= 3 and i <= 3):
            if lem in ("be", "have", "do"):
                return None
            return " ".join(words[:i]), lem, " ".join(words[i + 1:])
    return None


FEELINGS = {"hungry", "thirsty", "tired", "bored", "sleepy", "cold", "hot", "sick", "dirty",
            "sad", "happy", "angry", "scared", "lonely", "wet"}


def _split_subject_verb(text):
    """'a penguin swim' -> ('a penguin', 'swim'); prefers a split whose
    predicate starts with a known verb."""
    words = text.split()
    first = None
    for i in range(1, len(words)):
        subj = [w for w in words[:i] if w not in _DETERMINERS]
        if not subj or words[i - 1] in _DETERMINERS:
            continue
        if first is None:
            first = (" ".join(words[:i]), " ".join(words[i:]))
        lem = verb_lemma(words[i])
        if lem in VERBS or lem in INTRANSITIVE or words[i] in _IRREGULAR_VERBS:
            return " ".join(words[:i]), " ".join(words[i:])
    return first


_POSTURE = {"sit", "stand", "lie", "lay", "sleep", "stay", "wait", "hide", "rest", "live", "sat",
            "stood", "lying", "sitting", "standing", "sleeping", "hiding", "resting", "waiting"}


def _agent_ok(a):
    """An event's agent is a plain noun phrase (no verb or preposition)."""
    return not re.search(r"\b(is|are|was|were|to|of|be|been|not)\b", a)


def _vp(text):
    """Verb phrase after 'can': lemma of the verb plus its object."""
    words = text.split()
    if not words:
        return text
    words[0] = verb_lemma(words[0])
    return " ".join(w for w in words if w not in _DETERMINERS)


def _subject(np_text, wm):
    s = noun(np_text)
    wm.mention(s)
    return s


def _fact_is(s_np, o_np, wm, positive=True):
    s = _subject(s_np, wm)
    generic = np_is_plural(s_np)
    o_words = o_np.split()
    if not o_words:
        return []
    if o_words[0] in ("a", "an"):
        return [Fact(s, "isa", noun(o_np), positive, generic=generic)]
    if o_words[0] == "the":
        return [Fact(s, "isa", noun(o_np), positive, generic=generic)]
    if np_is_plural(o_np) and generic:
        return [Fact(s, "isa", noun(o_np), positive, generic=True)]
    o = " ".join(w for w in o_words if w not in ("very", "quite", "really", "so"))
    out = [Fact(s, "is", o, positive, generic=generic)]
    if positive and o in FEELINGS:
        out.append(Event("feel", s, obj=o))
    return out


# -- questions -------------------------------------------------------------

_NOT_A_KIND = {"do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
               "would", "should", "kind", "type", "sort", "one", "time", "else", "about",
               "if", "a", "an", "the"} | set(ATTRIBUTE_KINDS)
_PLACE_KINDS = r"continent|country|state|province|city|town|region|ocean|sea|island"


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": 0}
_SUPERLATIVE = (r"largest|biggest|smallest|longest|shortest|highest|tallest|deepest|oldest|"
                r"youngest|hottest|coldest|heaviest|lightest|fastest|slowest|greatest|brightest")
_DID = (r"invented|discovered|wrote|painted|composed|founded|built|designed|created|directed|"
        r"sculpted|led|explored|conquered|assassinated|killed|developed|formulated|proposed|"
        r"first climbed|ruled|unified")


def _parse_world(low, ent):
    """Questions about the world: dates and lives, origins, attributes,
    superlatives, counts, and who did what."""
    m = _rx(r"(?:when|what year|in what year|in which year) (?:was|were|is) " + _s("s") +
            r" born")(low)
    if m:
        return [Query("when", ent(m["s"]), "born")]
    m = _rx(r"(?:when|what year|in what year|in which year) (?:did|does|do) " + _s("s") +
            r" (?P<v>die|begin|start|end|finish|happen|take place|occur|live)")(low)
    if m:
        rel = {"die": "died", "begin": "began", "start": "began", "end": "ended",
               "finish": "ended"}.get(m["v"])
        return [Query("when", ent(m["s"]), rel)]
    m = _rx(r"(?:was|were) " + _s("s") + r" born in (?:the year )?(?P<y>\d{1,4}(?: bc)?)")(low) \
        or _rx(r"did " + _s("s") + r" (?P<d>die) in (?:the year )?(?P<y>\d{1,4}(?: bc)?)")(low)
    if m:
        return [Query("yesno", ent(m["s"]), "died" if "d" in m.groupdict() else "born",
                      m["y"].replace(" bc", " BC"))]
    m = _rx(r"(?:was|were) " + _s("s") + r" born in " + _s("p"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), "born_in", ent(m["p"]))]
    m = _rx(r"how old (?:was|is) " + _s("s") + r" when (?:[a-z' -]+ )?(?:died|passed away)")(low) or \
        _rx(r"(?:at what age|how old) did " + _s("s") + r" die")(low) or \
        _rx(r"how long did " + _s("s") + r" live")(low)
    if m:
        return [Query("lifespan", ent(m["s"]))]
    m = _rx(r"who (?:was|is) (?:born first|born earlier|older|the oldest) " + _s("a") +
            r" or " + _s("b"))(low) or \
        _rx(r"who (?:was born|came|lived) (?:first|earlier) " + _s("a") + r" or " + _s("b"))(low)
    if m:
        return [Query("born_first", ent(m["a"]), o=ent(m["b"]))]
    m = _rx(r"when (?:was|were|is) " + _s("s"))(low)
    if m and not m["s"].endswith(" born"):
        return [Query("when", ent(m["s"]), None)]
    m = _rx(r"what (?:nationality|country) (?:was|is|were|are) " + _s("s"))(low) or \
        _rx(r"what (?:was|is) " + _s("s") + r"'s nationality")(low)
    if m and not re.search(r" (?:in|on|the capital of)$", m["s"]):
        return [Query("objects", ent(m["s"]), "nationality")]
    m = _rx(r"where (?:was|is|were|are|did|do|does) " + _s("s") + r" (?P<v>from|come from|born)")(low)
    if m:
        return [Query("origin", ent(m["s"]), extra=m["v"] == "born")]
    m = _rx(r"what (?:is|are|was) (?:the )?(?P<a>area|size|chemical symbol|symbol|atomic number) "
            r"(?:of|for) " + _s("s"))(low)
    if m:
        a = {"chemical symbol": "symbol", "size": "area",
             "atomic number": "atomic_number"}.get(m["a"], m["a"])
        return [Query("objects", ent(m["s"]), a)]
    m = _rx(r"how (?:big|large) (?:is|are) " + _s("s"))(low)
    if m:
        return [Query("objects", ent(m["s"]), "area")]
    m = _rx(r"(?:what|which) (?:chemical )?element (?:has|is|with) (?:the )?(?P<a>symbol|atomic number) "
            r"(?:of )?(?P<v>[a-z0-9]+)")(low)
    if m:
        return [Query("by_attr", "element", "symbol" if m["a"] == "symbol" else "atomic_number",
                      m["v"])]
    m = _rx(r"what (?:is|are) (?:a |an |the )?" + _s("s") + r" used for")(low) or \
        _rx(r"what (?:do|does) (?:you|people|we|one) use (?:a |an |the )?" + _s("s") + r" for")(low) \
        or _rx(r"what is the (?:use|purpose) of (?:a |an |the )?" + _s("s"))(low)
    if m:
        return [Query("objects", ent(m["s"]), "used_for")]
    m = _rx(r"what (?:do you|do we|do people) call (?:people|someone|a person|the people|those) "
            r"from " + _s("o"))(low) or _rx(r"what are (?:people|the people) from " + _s("o") +
                                            r" called")(low)
    if m:
        return [Query("subjects", r="demonym_of", o=ent(m["o"]))]
    m = _rx(r"(?:what|which) (?:is|was|are) the (?P<adj>" + _SUPERLATIVE + r") (?P<c>[a-z][a-z -]*?)"
            r"(?: (?:in|of|on) (?:the )?(?P<y>[a-z][a-z -]*?))?")(low)
    if m:
        y = m["y"]
        y = None if y in (None, "world", "earth") else noun(y)
        return [Query("superlative", noun(m["c"]), m["adj"], y)]
    m = _rx(r"(?:what|which) (?P<c>[a-z]+) (?:is|was) the (?P<adj>" + _SUPERLATIVE + r")"
            r"(?: (?:one|of them|of all))?(?: (?:in|of|on) (?:the )?(?P<y>[a-z][a-z -]*?))?")(low)
    if m:
        y = m["y"]
        y = None if y in (None, "world", "earth") else noun(y)
        return [Query("superlative", noun(m["c"]), m["adj"], y)]
    m = _rx(r"how many (?P<c>[a-z]+) (?P<v>border|borders) " + _s("o"))(low)
    if m:
        return [Query("count", noun(m["c"]), "border", ent(m["o"]))]
    m = _rx(r"how many (?P<c>[a-z][a-z ]*?) (?:are there|are|exist|are there in the world)"
            r"(?: (?:in|on) (?:the )?(?P<o>[a-z][a-z -]*?))?")(low)
    if m:
        return [Query("count", noun(m["c"]), "in" if m["o"] else None,
                      ent(m["o"]) if m["o"] else None)]
    m = _rx(r"who (?P<v>" + _DID + r") (?:the )?(?P<o>[a-z0-9][a-z0-9' -]*)")(low)
    if m:
        return [Query("who_did", r=m["v"], o=m["o"])]
    m = _rx(r"who (?:was|is) the (?P<n>first|second|third|fourth|fifth|sixth|seventh|eighth|"
            r"ninth|tenth|last|\d+(?:st|nd|rd|th)) (?P<t>[a-z][a-z' -]*)")(low)
    if m:
        return [Query("who_title", extra=(m["n"], m["t"]))]
    return None


def _parse_natural(low, ent):
    """Question types people ask that the core patterns do not cover:
    why-questions about facts, kinds, lists and examples, comparisons,
    choices, differences, places of a given kind, inverse relations."""
    q = _parse_world(low, ent)
    if q:
        return q
    m = _rx(r"why (?P<q>(?:is|are|was|were|can|could|does|do|did|has|have) .+)")(low)
    if m:
        inner = m["q"]
        neg = re.search(r"\bnot\b", inner) is not None
        return [Query("why_q", extra=(re.sub(r"\s*\bnot\b", "", inner), neg))]
    m = _rx(r"(?:define|what does|what do) (?:the word )?'?(?P<s>[a-z][a-z' -]*?)'?(?: mean)?")(low)
    if m and (low.startswith("define") or low.endswith(" mean")):
        return [Query("describe", ent(m["s"].strip("' ")))]
    m = _rx(r"(?:what is |what's )?(?:the )?meaning of (?:the word )?'?(?P<s>[a-z][a-z -]*?)'?")(low)
    if m:
        return [Query("describe", ent(m["s"]))]
    m = _rx(r"what (?P<k>kind|type|sort) of (?P<c>[a-z]+(?: [a-z]+)?) (?:is|are) " + _s("s"))(low)
    if m:
        return [Query("kind", ent(m["s"]), extra=noun(m["c"]))]
    m = _rx(r"(?:which|what|who) (?:is|are) (?P<c>" + _COMP_RE + r") (?:of )?(?:the two )?"
            + _s("a") + r" or " + _s("b"))(low)
    if m:
        return [Query("compare", ent(m["a"]), COMPARATIVES[m["c"]], ent(m["b"]))]
    m = _rx(r"(?:is|are) " + _s("s") + r" (?P<c>" + _COMP_RE + r") or (?:" + _COMP_RE +
            r") than " + _s("o"))(low)
    if m:
        return [Query("compare", ent(m["s"]), COMPARATIVES[m["c"]], ent(m["o"]))]
    m = _rx(r"what is the difference between " + _s("a") + r" and " + _s("b"))(low) or \
        _rx(r"how (?:is|are|do|does) " + _s("a") + r" differ(?:ent)? from " + _s("b"))(low) or \
        _rx(r"how are " + _s("a") + r" and " + _s("b") + r" different")(low)
    if m:
        return [Query("difference", ent(m["a"]), o=ent(m["b"]))]
    m = _rx(r"(?:is|was) (?:a |an |the )?(?P<s>[a-z'-]+(?: [a-z'-]+)?) (?:a|an) "
            r"(?P<a>[a-z][a-z -]*?) or (?:a |an )?(?P<b>[a-z][a-z -]*?)")(low) or \
        _rx(r"(?:are|were) (?P<s>[a-z'-]+) (?P<a>[a-z'-]+) or (?P<b>[a-z'-]+)")(low)
    if m and " than" not in m["a"]:
        return [Query("choice", ent(m["s"]), extra=[noun(m["a"]), noun(m["b"])])]
    m = _rx(r"(?:which|what) (?P<k>" + _PLACE_KINDS + r") (?:is|are) " + _s("s") +
            r" (?:in|on|located in|part of)")(low) or \
        _rx(r"(?:in|on) (?:which|what) (?P<k>" + _PLACE_KINDS + r") (?:is|are) " + _s("s") +
            r"(?: located| found| situated)?")(low)
    if m:
        return [Query("place_kind", ent(m["s"]), extra=m["k"])]
    m = _rx(_s("s") + r" (?:is|are|was|were) the (?P<r>[a-z]+) of (?:which|what)(?: [a-z]+)?")(low) \
        or _rx(r"(?:which|what)(?: [a-z]+)? (?:is|are|was|were) " + _s("s") +
               r" the (?P<r>[a-z]+) of")(low)
    if m:
        return [Query("objects", ent(m["s"]), m["r"] + "_of")]
    m = _rx(r"(?:name|list|give (?:me |user )?|tell (?:me |user )?|show (?:me |user )?|"
            r"what are |which are |mention )(?:some|a few|several|all|any|the|two|three|"
            r"four|five|an|a|the names of)?(?: (?:examples?|kinds?|types?|sorts?) of)?"
            r"(?: (?:a|an|the))? (?P<c>[a-z][a-z -]*?)")(low)
    if m and (low.startswith(("name", "list", "give", "show", "mention")) or
              re.search(r"\b(some|a few|several|examples?|kinds? of|types? of|names of)\b", low)):
        c, _, place = m["c"].partition(" in ")
        if place:                                      # 'name some countries in Africa'
            return [Query("which", noun(c), "in", noun(place))]
        return [Query("examples", noun(c))]
    m = _rx(r"(?:which|what) (?P<c>[a-z]+(?: [a-z]+)?) (?P<v>can not|can|could|lives? in|"
            r"lives? on|are found in|is found in|are in|is in|eats?|has|have|makes?|borders?|"
            r"are|is) " + _s("o"))(low)
    if m and m["c"].split()[0] not in _NOT_A_KIND and m["c"].split()[-1] not in _NOT_A_KIND:
        v = m["v"]
        if v.startswith(("live", "are found", "is found", "are in", "is in")):
            return [Query("which", noun(m["c"]), "in", noun(m["o"]))]
        if v in ("can", "could", "can not"):
            return [Query("which", noun(m["c"]), "can", _vp(m["o"]), extra=v != "can not")]
        if v in ("are", "is"):
            return [Query("which", noun(m["c"]), "isa", noun(m["o"]))]
        if v in ("has", "have"):
            return [Query("which", noun(m["c"]), "has", quantity(m["o"])[1])]
        return [Query("which", noun(m["c"]), verb_lemma(v), noun(m["o"]))]
    return None


def _parse_question(low, wm):
    def ent(x):
        e = noun(x)
        wm.mention(e)
        return e

    m = _rx(r"(?:what do (?:you|mnemo) know about|what (?:does|do) mnemo know about|tell "
            r"(?:me|user) (?:everything )?about|describe) " + _s("s"))(low)
    if m:
        return [Query("describe", ent(m["s"]))]
    m = _rx(r"(?P<a>[a-z]+) is to (?P<b>[a-z]+) as (?P<c>[a-z]+) is to(?: what)?")(low)
    if m:
        return [Query("analogy", m["a"], m["b"], m["c"])]
    m = _rx(r"(?:what|which words?|who) (?:is|are) (?:similar|related|like) (?:to )?" + _s("s"))(low)
    if m:
        return [Query("similar", noun(m["s"]))]
    m = _rx(r"how (?:do|does|can) (?:you|one|i|we|mnemo|user) (?:go|get) from " + _s("a") +
            r" to " + _s("b"))(low)
    if m:
        return [Query("path", ent(m["a"]), o=ent(m["b"]))]
    m = _rx(r"why did " + _s("s") + r" (?:go|move|travel|journey|run|walk|come) (?:back )?to " +
            _s("p"))(low)
    if m:
        return [Query("why_go", ent(m["s"]), o=noun(m["p"]))]
    m = _rx(r"where will " + _s("s") + r" (?:go|move|travel|journey|run|walk)")(low)
    if m:
        return [Query("will_go", ent(m["s"]))]
    q = _parse_natural(low, ent)
    if q:
        return q
    m = _rx(r"where (?:is|are|was|were) " + _s("s") + r" before " + _s("p"))(low)
    if m:
        return [Query("where_before", ent(m["s"]), o=noun(m["p"]))]
    m = _rx(r"where (?:is|are|was|were|does|do) " + _s("s") + r"(?: live| located| now)?")(low)
    if m:
        return [Query("where", ent(m["s"]))]
    m = _rx(r"how many (?:objects|things|items) (?:is|are|was) " + _s("s") +
            r" (?:carrying|holding)")(low)
    if m:
        return [Query("count_carrying", ent(m["s"]))]
    m = _rx(r"what (?:is|are|was) " + _s("s") + r" (?:carrying|holding)")(low)
    if m:
        return [Query("carrying", ent(m["s"]))]
    m = _rx(r"who (?:has|had|is carrying|is holding|was carrying) " + _s("o"))(low)
    if m:
        return [Query("who_has", o=noun(m["o"]))]
    m = _rx(r"who (?:gave|handed|passed) " + _s("o") + r" to " + _s("r"))(low)
    if m:
        return [Query("who_gave", o=noun(m["o"]), extra=noun(m["r"]))]
    m = _rx(r"what did " + _s("s") + r" (?:give|hand|pass) to " + _s("r"))(low)
    if m:
        return [Query("what_gave", ent(m["s"]), extra=noun(m["r"]))]
    m = _rx(r"who (?:did )?(?:received|receive|got|get) " + _s("o"))(low)
    if m:
        return [Query("who_received", o=noun(m["o"]))]
    m = _rx(r"who did " + _s("s") + r" (?:give|hand|pass) " + _s("o") + r" to")(low)
    if m:
        return [Query("who_received", ent(m["s"]), o=noun(m["o"]))]
    m = _rx(r"how many " + _s("o") + r" (?:does|do|did) " + _s("s") + r" have")(low)
    if m:
        return [Query("how_many", ent(m["s"]), "has", noun(m["o"]))]
    m = _rx(r"what (?P<k>" + "|".join(ATTRIBUTE_KINDS) + r") (?:is|are|was|were) " + _s("s"))(low)
    if m:
        k = "color" if m["k"] == "colour" else m["k"]
        return [Query("attr", ent(m["s"]), extra=k)]
    m = _rx(r"what (?:is|are|was|were) the (?P<k>" + "|".join(ATTRIBUTE_KINDS) + r") of " +
            _s("s"))(low)
    if m:
        k = "color" if m["k"] == "colour" else m["k"]
        return [Query("attr", ent(m["s"]), extra=k)]
    m = _rx(r"what (?:is|are|was|were) " + _s("s") + r" (?:afraid|scared|frightened) of")(low)
    if m:
        return [Query("objects", ent(m["s"]), "afraid_of")]
    m = _rx(r"what (?:is|are|was|were) " + _s("s") + r" made of")(low)
    if m:
        return [Query("objects", ent(m["s"]), "made_of")]
    m = _rx(r"what can " + _s("s") + r" do")(low)
    if m:
        return [Query("objects", ent(m["s"]), "can")]
    m = _rx(r"what (?:does|do|did) " + _s("s") + r" have")(low)
    if m:
        return [Query("objects", ent(m["s"]), "has")]
    m = _rx(r"what (?:does|do|did) " + _s("s") + r" (?P<v>[a-z]+)")(low)
    if m:
        return [Query("objects", ent(m["s"]), verb_lemma(m["v"]))]
    m = _rx(r"what (?:is|are) the opposite of " + _s("s"))(low)
    if m:
        return [Query("objects", noun(m["s"]), "opposite")]
    m = _rx(r"(?:is|are) " + _s("s") + r" the opposite of " + _s("o"))(low)
    if m:
        return [Query("yesno", noun(m["s"]), "opposite", noun(m["o"]))]
    m = _rx(r"(?:who|what) (?:is|are|was|were) " + _s("o") + r"'s (?P<r>[a-z]+)")(low)
    if m:
        return [Query("subjects", r=m["r"] + "_of", o=ent(m["o"]), extra=m["r"])]
    m = _rx(r"(?:who|what) (?:is|are|was|were) (?:the )?(?P<r>[a-z]+) of " + _s("o"))(low)
    if m:
        return [Query("subjects", r=m["r"] + "_of", o=ent(m["o"]), extra=m["r"])]
    m = _rx(r"(?:who|what) (?:is|are) (?P<c>" + _COMP_RE + r") than " + _s("o"))(low)
    if m:
        return [Query("subjects", r=COMPARATIVES[m["c"]], o=ent(m["o"]))]
    m = _rx(r"(?:who|what) (?:is|are) (?P<d>north|south|east|west) of " + _s("o"))(low)
    if m:
        return [Query("subjects", r=DIRECTIONS[m["d"]], o=ent(m["o"]))]
    m = _rx(r"what (?:is|are) (?P<d>north|south|east|west) of " + _s("o"))(low)
    if m:
        return [Query("subjects", r=DIRECTIONS[m["d"]], o=ent(m["o"]))]
    m = _rx(r"(?:tell (?:me|user) about|describe|who is|who are|who was|who were|what is|"
            r"what are|what was|what were) " + _s("s"))(low)
    if m:
        return [Query("describe", ent(m["s"]))]
    m = _rx(r"who (?P<v>[a-z]+) " + _s("o"))(low)
    if m and m["v"] not in ("is", "are", "was", "were"):
        return [Query("subjects", r=verb_lemma(m["v"]), o=noun(m["o"]))]
    m = _rx(r"what (?P<v>[a-z]+) " + _s("o"))(low)
    if m and m["v"] not in ("is", "are", "was", "were", "do", "does", "did"):
        return [Query("subjects", r=verb_lemma(m["v"]), o=noun(m["o"]))]

    # yes / no
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?:a |an )?(?:kind|type|sort) of " +
            _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), "isa", noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?P<c>" + _COMP_RE + r") than " + _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), COMPARATIVES[m["c"]], noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?P<d>north|south|east|west) of " +
            _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), DIRECTIONS[m["d"]], noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?P<p>" + _SPATIAL_RE + r") " + _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), SPATIAL[m["p"]], noun(m["o"]))]
    m = _rx(r"(?:does|do) " + _s("s") + r" fit (?:inside|in|into) " + _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), "smaller_than", noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?:afraid|scared) of " + _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), "afraid_of", noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?P<o>(?:a|an|the) [a-z0-9' -]+)")(low)
    if m:
        return [Query("yesno", ent(m["s"]), "isa", noun(m["o"]))]
    m = _rx(r"(?:is|are|was|were) " + _s("s") + r" (?P<o>[a-z0-9'-]+)")(low)
    if m:
        s_np, o = m["s"], m["o"]
        if np_is_plural(s_np) and (is_plural(o) or o in ("fish", "sheep", "deer", "moose",
                                                          "people", "cattle")):
            return [Query("yesno", ent(s_np), "isa", singular(o))]
        return [Query("yesno", ent(s_np), "is", o)]
    m = re.match(r"^(?:can|could) (.+)$", low)
    if m:
        split = _split_subject_verb(m.group(1))
        if split:
            return [Query("yesno", ent(split[0]), "can", _vp(split[1]))]
    m = _rx(r"(?:does|do|did) " + _s("s") + r" (?:have|own) " + _s("o"))(low)
    if m:
        qty, o = quantity(m["o"])
        return [Query("yesno", ent(m["s"]), "has", o, extra=qty if qty and qty > 0 else None)]
    m = re.match(r"^(?:does|do|did) (.+)$", low)
    svo = _split_svo(m.group(1)) if m else None
    if svo and svo[1] not in ("have", "own"):          # 'does copper conduct electricity'
        s_np, v, o_np = svo
        if v in ("live", "stay") and o_np.startswith(("in ", "at ")):
            return [Query("yesno", ent(s_np), "in", noun(o_np[3:]))]
        return [Query("yesno", ent(s_np), v, noun(o_np))]
    m = _rx(r"(?:does|do|did) " + _s("s") + r" (?P<v>[a-z]+)")(low)
    if m and m["v"] not in ("have", "own"):
        return [Query("yesno", ent(m["s"]), "does", verb_lemma(m["v"]))]
    m = _rx(r"(?:does|do|did) " + _s("s") + r" (?P<v>[a-z]+) " + _s("o"))(low)
    if m:
        v = verb_lemma(m["v"])
        if v in ("live", "stay") and m["o"].startswith(("in ", "at ")):
            return [Query("yesno", ent(m["s"]), "in", noun(m["o"][3:]))]
        return [Query("yesno", ent(m["s"]), v, noun(m["o"]))]
    m = _rx(r"(?:has|have) " + _s("s") + r" (?:a |an |any )?" + _s("o"))(low)
    if m:
        return [Query("yesno", ent(m["s"]), "has", noun(m["o"]))]
    return []
