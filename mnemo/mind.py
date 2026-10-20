"""The integrated mind.

  perception   language.py  sentences -> facts, events, questions
  working mem  language.py  the last few entities, for pronouns
  episodic     world.py     what happened, in order; where things are
  declarative  knowledge.py facts with evidence-based truth, ACT-R activation
  semantic     semantic.py  meaning from co-occurrence; similarity, analogy
  intuition    cortex.py    the predictive cortex: fluent continuation,
                            surprise, plausibility of alternatives
  analogy      hdc.py       role-filler records in hyperdimensional space

Every answer comes with the evidence it rests on ("Why?").
"""
import re

from . import language as L
from .hdc import HyperSpace
from .rephrase import arithmetic, rephrase
from .knowledge import Answer, Fact, Knowledge, Truth
from .world import World

COLORS = "red orange yellow green blue purple violet pink brown black white gray grey gold silver".split()
SIZES = "big small large tiny huge little giant enormous tall short".split()
SHAPES = "round square triangular rectangular oval flat".split()
TASTES = "sweet sour bitter salty spicy".split()
TEMPERATURES = "hot cold warm cool freezing".split()
MOODS = sorted(L.FEELINGS)

_COMMON = set("""the a an what who where when why how which is are was were can could does do
did has have had then after afterwards following next later meanwhile finally also so well now
ok okay please yes no not i you he she it we they my your his her its our their this that these
those there here and or but if because tell describe complete continue write imagine generate
yesterday today this""".split())


class Mind:
    def __init__(self, cortex=None, semantic=None, innate=True, learn_language=True,
                 hd_dim=2048):
        self.kb = Knowledge(semantic)
        self.world = World()
        self.wm = L.WorkingMemory(capacity=4)
        self.cortex = cortex
        self.semantic = semantic
        self.hd = HyperSpace(hd_dim)
        self.learn_language = learn_language
        self.last = None            # (question text, Answer)
        self.surprises = []
        self._caps = {}             # capitalised names in the current sentence
        self._spoken = {}           # every name as the speaker wrote it ('John', not 'the john')
        self.llm = None             # an optional language model for what memory lacks (llm.py)
        self.llm_learn = True
        self.last_source = "memory"
        self._transcript = []       # recent (said, replied) pairs, for the model's context
        if innate:
            self._innate()

    OPPOSITES = [("big", "small"), ("large", "small"), ("tall", "short"), ("long", "short"),
                 ("hot", "cold"), ("warm", "cold"), ("fast", "slow"), ("old", "young"),
                 ("happy", "sad"), ("light", "dark"), ("heavy", "light"), ("wet", "dry"),
                 ("hard", "soft"), ("loud", "quiet"), ("rich", "poor"), ("strong", "weak"),
                 ("full", "empty"), ("open", "closed"), ("alive", "dead"), ("black", "white")]

    def _innate(self):
        for a, b in self.OPPOSITES:
            self.kb.tell(a, "opposite", b, source="innate")
            self.kb.tell(b, "opposite", a, source="innate")
        for group, kind in ((COLORS, "color"), (SIZES, "size"), (SHAPES, "shape"),
                            (TASTES, "taste"), (TEMPERATURES, "temperature"), (MOODS, "mood")):
            for w in group:
                self.kb.tell(w, "isa", kind, source="innate")
        self.kb.categories -= {"color", "size", "shape", "taste", "temperature", "mood"}

    # ------------------------------------------------------------------ input
    def hear(self, text):
        """Process an utterance (one or more sentences); returns the reply."""
        replies = []
        self.last_source = "memory"
        for sent in L.split_sentences(text):
            r = self._sentence(sent)
            if r:
                replies.append(r)
        reply = " ".join(replies)
        self._transcript = (self._transcript + [(text, reply)])[-4:]
        return reply

    # ------------------------------------------------------------------ a language model
    def attach_llm(self, llm, learn=True):
        """Pair the mind with a language model (llm.LLM): it is asked only what
        memory cannot answer, and what it says that Mnemo can read is kept
        as weak facts (learn=False keeps nothing)."""
        self.llm, self.llm_learn = llm, learn

    _UNKNOWN = ("I don't know", "I don't understand", "I haven't read", "I need a semantic",
                "I have no cortex", "I know the word", "I don't think I know")

    def _consult(self, sent, fallback):
        """Ask the model what memory cannot answer, showing it what memory
        holds about the question's words (retrieval-augmented, from a
        symbolic memory)."""
        from .llm import grounding, prompt, sentences
        facts = grounding(self, sent)
        try:
            history = [(q, re.sub(r" \[[^\]]+\].*$", "", a)) for q, a in self._transcript[-3:]]
            text = self.llm.ask(prompt(sent, facts), history=history)
        except (OSError, ValueError, KeyError) as e:   # not running, timed out, odd reply
            return f"{fallback} (The language model did not answer: {type(e).__name__}.)"
        text = " ".join(sentences(text)[:3])[:600].strip() or "I don't know."
        self.last = Answer(Truth(1.0, 0.5), [], "language model", stmt=(self.llm.name, len(facts)))
        self.last_source = self.llm.name
        note = self._learn_from(text) if self.llm_learn else ""
        return f"{text} [{self.llm.name}]{note}"

    def _learn_from(self, text):
        """Keep the model's simple statements as weak facts (source 'llm',
        confidence 0.6), except those that contradict what memory firmly
        holds; those are pointed out instead."""
        from .llm import readable, sentences
        doubts = []
        for line in readable(sentences(text))[:3]:
            names = self._remember_names(line)
            for it in L.parse_sentence(line, L.WorkingMemory(), names).items:
                if not isinstance(it, L.Fact):
                    continue
                self.canon_fact(it)
                prior = self.kb.ask(it.s, it.r, it.o)
                if prior.known and prior.truth.c >= 0.7 and (prior.truth.f >= 0.5) != it.positive:
                    doubts.append(self.kb.explain(prior, brief=True))
                elif self.kb.get(it.s, it.r, it.o) is None:
                    self.kb.tell(it.s, it.r, it.o, it.positive, 0.6, it.qty, "llm",
                                 generic=it.generic)
        return f" (I doubt part of that: {doubts[0][:1].lower() + doubts[0][1:]})" if doubts else ""

    @staticmethod
    def _from_llm(a, depth=0):
        """Does an answer rest on something a language model said?"""
        return depth < 6 and any(
            (isinstance(p, Fact) and p.source == "llm") or
            (isinstance(p, Answer) and Mind._from_llm(p, depth + 1)) for p in a.premises)

    def read(self, text, extract=True):
        """Read a document: the cortex learns its language and simple
        statements are stored as (less certain) facts."""
        if self.cortex is not None:
            self.cortex.learn(text)
        n = 0
        if extract:
            for sent in L.split_sentences(text):
                names = self._remember_names(sent)
                p = L.parse_sentence(sent, self.wm, names)
                for it in p.items:
                    if isinstance(it, L.Fact):
                        self._tell(it, conf=0.6, source="read")
                        n += 1
        return n

    def _sentence(self, sent):
        self.kb.tick()
        if self.cortex is not None and self.learn_language:
            bits = self.cortex.learn(sent + "\n")
            self.surprises.append(bits / max(1, len(sent) + 1))
        sum_ = arithmetic(sent)
        if sum_ is not None:
            return sum_
        sent, flip = rephrase(sent)
        social = self._social(sent)
        if social is not None:
            return social
        sent = self._persons(sent)
        names = self._remember_names(sent)
        p = L.parse_sentence(sent, self.wm, names)
        self._last_parse = p.items
        before = self.last
        if not p.items:
            p = self._fragment(sent, names)
        out = self._respond(p) if p is not None else self._fallback(sent)
        if flip and out:       # 'is it true that penguins can't fly' was asked the other way
            out = self._flip(out)
        if self.llm is not None and out.startswith(self._UNKNOWN) and \
                (sent.strip().endswith("?") or L._looks_like_question(sent.strip().lower())):
            return self._consult(sent, out)
        if self.last is not before and self.last is not None and self._from_llm(self.last):
            out += " (I learned that from a language model.)"
        return out

    _FLIP = {"Yes.": "No.", "No.": "Yes.", "Probably.": "Probably not.",
             "Probably not.": "Probably.", "Maybe.": "Maybe not.", "Maybe not.": "Maybe."}

    def _flip(self, reply):
        for a in sorted(self._FLIP, key=len, reverse=True):
            if reply.startswith(a):
                return self._FLIP[a] + reply[len(a):]
        return reply

    def _fragment(self, sent, names):
        """'Penguins?' or 'a giraffe' alone: read it as 'what is ...'."""
        body = sent.strip().rstrip("?.!").strip()
        if not body or len(body.split()) > 4:
            return None
        p = L.parse_sentence(f"What is {body}?", self.wm, names)
        return p if p.items and all(isinstance(i, L.Query) for i in p.items) else None

    def _respond(self, p):
        out = []
        for it in p.items:
            if isinstance(it, L.Fact):
                out.append(self._tell(it))
            elif isinstance(it, L.Event):
                self.world.observe(it)
                out.append("")
            elif isinstance(it, L.Query):
                out.append(self._answer(it, p.sentence))
            elif isinstance(it, L.Command):
                out.append(self._command(it))
        out = [o for o in out if o]
        if not out and any(isinstance(i, (L.Fact, L.Event)) for i in p.items):
            return "OK."
        return " ".join(dict.fromkeys(out))

    def _social(self, sent):
        low = re.sub(r"[^a-z' ]", "", sent.lower()).strip()
        if re.match(r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings)\b", low):
            name = self.kb.subjects("name_of", "user")
            return f"Hello{', ' + self.kb.name(name[0].value) if name else ''}! Teach me something."
        if re.match(r"^(thanks|thank you|cheers)\b", low):
            return "You're welcome."
        if low in ("who are you", "what are you", "what is your name", "whats your name"):
            return ("I am Mnemo, a small mind made of counting memories and logic rather than "
                    "a neural network. I learn from what you tell me and explain my reasoning.")
        if low in ("what can you do", "help", "how do i use you", "what do you do"):
            return ("Tell me facts ('Penguins are birds', 'Mary went to the kitchen') and ask me "
                    "about them ('Can Opus fly?', 'Where is the apple?', 'Why?'). I can generalise, "
                    "track people and objects, reason about sizes, places and directions, make "
                    "analogies, and, if I have read books, imagine text ('complete: Once upon a "
                    "time').")
        m = re.match(r"^(?:my name is|i am called|call me|i'm called) ([a-z' -]+)$", low)
        if m:
            name = m.group(1).strip()
            disp = " ".join(w.capitalize() for w in name.split())
            self.kb.display[name] = disp
            for f in self.kb.inn("user", "name_of"):
                self.kb.forget(f.s, f.r, f.o)
            self.kb.tell(name, "name_of", "user")
            return f"Nice to meet you, {disp}."
        if low in ("what is my name", "who am i", "whats my name", "do you know my name"):
            n = self.kb.subjects("name_of", "user")
            return f"You are {self.kb.name(n[0].value)}." if n else "You haven't told me your name."
        return None

    _PERSON = [(r"\bi am\b", "user is"), (r"\bi'm\b", "user is"), (r"\bam i\b", "is user"),
               (r"\bmy\b", "user's"), (r"\bme\b", "user"), (r"\bi\b", "user"),
               (r"\bdo user\b", "does user"), (r"\byour\b", "mnemo's"),
               (r"\bare you\b", "is mnemo"), (r"\byou are\b", "mnemo is"), (r"\byou\b", "mnemo")]

    _AUX = _COMMON | {"am", "should", "would", "will", "may", "shall", "must", "might"}

    def _persons(self, sent):
        """First and second person become the entities 'user' and 'mnemo'
        (but the I of 'World War I' and 'Elizabeth I' is a numeral)."""
        if not re.search(r"\b(i|i'm|me|my|you|your)\b", sent, re.I):
            return sent
        low = re.sub(r"\b([A-Z][a-z]+) I\b", lambda m: m.group(0) if m.group(1).lower() in
                     self._AUX else m.group(1) + " \u2160", sent)
        for pat, rep in self._PERSON:
            low = re.sub(pat, rep, low, flags=re.I)
        return low.replace("\u2160", "I")

    # small words inside names: the Bay of Biscay, Leonardo da Vinci, Vincent van Gogh
    _PARTICLES = {"of", "da", "de", "di", "du", "del", "van", "von", "der", "den", "la", "le", "bin",
                  "ibn", "al"}

    def _remember_names(self, sent):
        """Capitalised names in a sentence become display names; returns
        them (lower case) so that 'China' can mean the country."""
        toks = re.findall(r"[A-Za-z][A-Za-z'-]*", sent)
        names, i = {}, 0
        while i < len(toks):
            t = toks[i]
            if t[0].isupper() and t.lower() not in _COMMON and \
                    (i > 0 or self._proper_at_start(toks)):
                j = i
                while j + 1 < len(toks) and (toks[j + 1][0].isupper() or toks[j + 1] in self._PARTICLES
                                             and j + 2 < len(toks) and toks[j + 2][0].isupper()):
                    j += 1
                name = " ".join(toks[i:j + 1])
                low = name.lower()
                names[low] = name
                key = self.kb.canon(low, proper=True)
                if key not in self.kb.categories:
                    self.kb.display[key] = name       # the speaker's name for it wins
                self._spoken[low] = name              # how the speaker names it, for stories
                i = j + 1
            else:
                i += 1
        self._caps = names
        return names

    def _proper_at_start(self, toks):
        t = toks[0]
        low = t.lower()
        if low in self.kb._MASS or self.kb.canon(low) in self.kb.mass:
            return False      # 'Honey is sweet', 'Wood floats'
        if low in L.FEMALE or low in L.MALE:
            return True
        nxt = toks[1].lower() if len(toks) > 1 else ""
        if len(toks) > 1 and toks[1][:1].isupper():
            return True       # 'Buenos Aires', 'South Africa'
        if L.is_plural(low) and nxt in ("is", "was", "has"):
            return True       # 'Algiers is', 'Caracas has'
        if L.is_plural(low) or nxt in ("are", "were", "have", "do", "live", "eat", "can",
                                       "cannot", "can't", "don't", "never", "always",
                                       "often", "usually") or \
                nxt in L.VERBS or nxt in L.INTRANSITIVE:
            return False
        if low in self.kb.senses and low not in self.kb.proper and \
                self.kb.canon(low) not in self.kb.display:
            # a common noun ('Gold is a metal') -- unless it stands bare and singular
            # before 'is', which count nouns cannot do: 'Opus is a penguin' names someone
            stuff = any(k in self.kb.stuff or k in self.kb.mass
                        for k in self.kb.senses[low])
            return nxt in ("is", "was", "has") and not stuff
        if self.semantic is not None and low in self.semantic.index and \
                self.semantic.index[low] < 3000:
            return False
        return True

    # ------------------------------------------------------------------ facts
    _RAW_OBJECT = {"can", "does", "is", "opposite",   # object is a verb or an adjective,
                   "born", "died", "began", "ended",   # a year, or some other value
                   "nationality", "area", "symbol", "atomic_number", "used_for", "discovered_by",
                   "year"}
    # one value at a time: born in 1756 means not born in 1800
    _FUNCTIONAL = {"born", "died", "began", "ended", "born_in", "area", "symbol", "atomic_number",
                   "year"}

    def _c(self, x):
        """The concept a word in the current sentence names."""
        if not x:
            return x
        k = self.kb.canon(x, proper=x in self._caps)
        if k != x and self.kb.bare(k) != x and k not in self.kb.display:
            self.kb.surface[k] = x          # call it what the speaker calls it
        return k

    def canon_fact(self, f):
        s_word, o_word = f.s, f.o
        raw_o = f.r in self._RAW_OBJECT
        if not raw_o:
            f.o = self._c(f.o)
        f.s = self._c(f.s)
        if f.positive and self.kb.senses:
            f.s, o = self._senses_for(s_word, f.s, f.r, None if raw_o else o_word, f.o)
            if not raw_o:
                f.o = o
        elif f.r == "isa" and self.kb.senses and not self._own(f.s):
            # 'Pluto is not a planet' corrects the sense that is one (the planet, not the god)
            for k in [f.s] + self.kb.other_senses(s_word, f.s)[:4]:
                if self._fits(k, "isa", f.o):
                    if k != f.s:
                        self.kb.primed[s_word] = k
                    f.s = k
                    break
        for word, key in ((s_word, f.s), (o_word, f.o)):
            if word in self._caps and key not in self.kb.categories:
                self.kb.display[key] = self._caps[word]
        return f

    def _senses_for(self, s_word, s, r, o_word, o):
        """Word-sense choice from what is already known: 'Chad is a country'
        means the republic, not the lake; 'the currency of Japan is the yen'
        means the money, not a yearning.  Sense pairs are tried from the most
        to the least frequent (sum of sense ranks); the choice then holds for
        the rest of the discourse ('one sense per discourse', Gale, Church &
        Yarowsky 1992)."""
        def alts(word, key):
            # a sense chosen earlier sticks unless an is-a statement says otherwise
            if not word or self._own(key) or (r != "isa" and word in self.kb.primed):
                return []
            return self.kb.other_senses(word, key)[:4]
        s_c, o_c = [s] + alts(s_word, s), [o] + alts(o_word, o)
        if len(s_c) == 1 and len(o_c) == 1:
            return s, o
        pairs = sorted(((i + j, j, k, t) for i, k in enumerate(s_c) for j, t in enumerate(o_c)),
                       key=lambda p: p[:2])
        for _, _, k, t in pairs:
            if self._fits(k, r, t):
                if k != s:
                    self.kb.primed[s_word] = k
                if t != o:
                    self.kb.primed[o_word] = t
                return k, t
        return s, o

    _PLACE_RELS = {"in", "capital_of", "currency_of", "language_of"}

    def _fits(self, k, r, t):
        kb = self.kb
        if r == "isa":
            a = kb.ask(k, "isa", t)
            return a.known and a.truth.f >= 0.5 and a.how in ("direct", "transitivity")
        if kb.get(k, r, t) is not None:
            return True
        if r in self._PLACE_RELS and t is not None:
            if kb.get(k, "in", t) is not None:
                return True
            a = kb.ask(k, "in", t)
            return a.truth.f >= 0.5 and a.how in ("direct", "transitivity")
        return False

    def _tell(self, f, conf=0.9, source="told"):
        self.canon_fact(f)
        if f.r in ("north_of", "south_of", "east_of", "west_of"):
            self.world.add_direction(f.s, f.r, f.o)
        prior = self.kb.ask(f.s, f.r, f.o)
        self.kb.tell(f.s, f.r, f.o, f.positive, conf, f.qty, source, generic=f.generic)
        if f.generic:
            self.kb.display.pop(f.s, None)
        if prior.known and prior.truth.c >= 0.5 and (prior.truth.f >= 0.5) != f.positive \
                and prior.how != "direct":
            return ("Noted: " + self.kb.say(self.kb.get(f.s, f.r, f.o)) +
                    " (an exception to what I would have guessed).")
        return ""

    # ------------------------------------------------------------------ answers
    def _answer(self, q, text):
        h = getattr(self, "_q_" + q.kind, None)
        if h is None:
            return "I don't know how to answer that yet."
        return h(q)

    def _record(self, a):
        self.last = a
        return a

    @staticmethod
    def _cap(s):
        return s[:1].upper() + s[1:] if s else s

    def _verdict(self, a):
        t = a.truth
        if not a.known or t.c < 0.1:
            return None
        yes = t.f >= 0.5
        strength = abs(t.f - 0.5) * 2
        if t.c >= 0.7 and strength > 0.5:
            return "Yes" if yes else "No"
        if t.c >= 0.3:
            return "Probably" if yes else "Probably not"
        return "Maybe" if yes else "Maybe not"

    def _dynamic(self, x):
        """Things that move or are carried are tracked episodically; static
        facts (Paris is in France) live in declarative memory."""
        return any(e[2] == x or e[3] == x for e in self.world.events
                   if e[1] in ("move", "take", "drop", "give", "not_at", "either_at"))

    _AXIS = {"left_of": (-1, 0), "right_of": (1, 0), "above": (0, 1), "below": (0, -1)}

    def _positions(self):
        """Place objects on a grid from left/right/above/below statements
        (relative positions are composed, as in a mental map)."""
        rel = [f for f in self.kb.facts.values() if f.r in self._AXIS and f.truth.f >= 0.5]
        pos = {}
        changed = True
        while changed:
            changed = False
            for f in rel:
                dx, dy = self._AXIS[f.r]
                if f.o in pos and f.s not in pos:
                    pos[f.s] = (pos[f.o][0] + dx, pos[f.o][1] + dy)
                    changed = True
                elif f.s in pos and f.o not in pos:
                    pos[f.o] = (pos[f.s][0] - dx, pos[f.s][1] - dy)
                    changed = True
                elif f.s not in pos and f.o not in pos and not pos:
                    pos[f.o] = (0, 0)
                    pos[f.s] = (dx, dy)
                    changed = True
            if not changed:
                for f in rel:
                    if f.o not in pos and f.s not in pos:
                        pos[f.o] = (100 * len(pos), 0)
                        changed = True
                        break
        return pos

    def _spatial(self, q):
        pos = self._positions()
        if q.s not in pos or q.o not in pos:
            return None
        (x1, y1), (x2, y2) = pos[q.s], pos[q.o]
        if abs(x1 - x2) > 50 or abs(y1 - y2) > 50:
            return None
        holds = {"left_of": x1 < x2, "right_of": x1 > x2, "above": y1 > y2, "below": y1 < y2}
        return holds[q.r]

    def _q_yesno(self, q):
        if q.r in self._FUNCTIONAL:         # 'Was Mozart born in 1756?'
            k, f = self._first(q.s, q.r)
            if f is not None:
                objs = [q.o] if q.r in self._RAW_OBJECT else self._senses_of(q.o)
                same = any(f.o.lower() == x.lower() or x in self.kb.reach(f.o, "in")
                           for x in objs)
                self._record(Answer(f.truth, [f]))
                return "Yes." if same else f"No. {self._cap(self.kb.say(f))}."
        if q.r in self._AXIS:
            a = self.kb.ask(q.s, q.r, q.o)
            if not a.known:
                sp = self._spatial(q)
                if sp is not None:
                    self._record(Answer(Truth(1.0 if sp else 0.0, 0.85), [], "mental map"))
                    return "Yes." if sp else "No."
        if q.r == "in" and self._dynamic(q.s):
            r = self.world.is_at(q.s, q.o)
            if r is not None:
                self._record(Answer(Truth(1.0 if r == "yes" else 0.0, 0.9), [], "episodic"))
                return {"yes": "Yes.", "no": "No.", "maybe": "Maybe."}[r]
        s = self._c(q.s)
        o = q.o if q.r in self._RAW_OBJECT else self._c(q.o)
        if q.r == "has" and isinstance(q.extra, int) and q.extra > 0:
            known = self._qty(s, "has", o)
            if known is not None and known > 0:      # 'No. Birds have two legs.'
                self._record(Answer(Truth(1.0, 0.9), [Fact(s, "has", o, Truth(1.0, 0.9), known)]))
                verdict = "Yes" if known == q.extra else "No"
                return f"{verdict}. {self._cap(self.kb.say(Fact(s, 'has', o, Truth(1.0, 0.9), known)))}."
        a = self._record(self.kb.ask(s, q.r, o))
        if q.r == "is" and not a.known and not self._own(s):
            for k in self.kb.other_senses(q.s, s)[:4]:      # Darwin the naturalist, not the town
                f = self.kb.get(k, "is", o)
                if f is not None:
                    self._record(Answer(f.truth, [f]))
                    v = "Yes" if f.truth.f >= 0.5 else "No"
                    g = self._gloss(k)
                    if not g:
                        return f"{v}."
                    the = "" if re.match(r"(?:a|an|the)\b", g) else "the "
                    return f"{v}, if you mean {the}{g}" + ("" if g.endswith("\u2026") else ".")
        if q.r == "border" and not a.known:
            got = self._borders(q.s, q.o)
            if got:
                return got
        if q.r == "isa" and not (a.known and a.truth.f >= 0.5):
            # a clear 'no' from the usual sense of the subject stands ('a shark is not a
            # mammal', whatever a 'shark' of a person may be); the object may still be
            # meant in another sense ('liquid' as in water)
            sure_no = a.known and a.truth.f < 0.5 and a.truth.c >= 0.7
            other = self._other_sense_isa(q.s, s, q.o, o, subject_too=not sure_no)
            if other:
                return other
        elif not a.known and q.r not in ("is",):
            alt = self._other_pairs(q.s, s, q.r, None if q.r in self._RAW_OBJECT else q.o, o)
            if alt is not None:
                a = self._record(alt)
        v = self._verdict(a)
        if v is None:
            return "I don't know."
        self.kb.use(*[p for p in a.premises if isinstance(p, Fact)])
        if a.how == "direct":
            return f"{v}."
        return f"{v}. {self.kb.explain(a, brief=True)}"

    def _borders(self, s_word, o_word):
        """The border lists of the world data are complete, so a border that is
        not on them is not there ('Does Spain border Germany?' No)."""
        kb = self.kb
        complete = [k for k in self._senses_of(s_word) if any(
            f.source == "world" for f in kb.out(k, "border") + kb.out(k, "area"))]
        others = [k for k in self._senses_of(o_word) if any(
            f.source == "world" for f in kb.out(k, "border") + kb.out(k, "area"))]
        if not complete or not others:
            return None
        s, o = complete[0], others[0]
        if kb.get(s, "border", o) is not None:
            return "Yes."
        near = sorted(f.o for f in kb.out(s, "border") if f.truth.f >= 0.5)
        self._record(Answer(Truth(0.0, 0.85), kb.out(s, "border")[:3], "closed world"))
        if not near:
            return f"No. {self._cap(kb.ref(s))} borders no other country."
        return f"No. {self._cap(kb.ref(s))} borders {self._join([kb.ref(x) for x in near])}."

    def _own(self, x):
        """Has the user told me things about x (rather than only WordNet)?"""
        return any(f.source not in self.kb.authority for f in self.kb.out(x))

    def _gloss(self, x, n=90):
        d = self.kb.definitions.get(x, "")
        d = d.split(";")[0].strip()
        return d if len(d) <= n else d[:n].rsplit(" ", 1)[0] + "\u2026"

    def _other_pairs(self, s_word, s, r, o_word, o):
        """Another reading of the words that the memory has an answer for
        ('how many months has a year': the calendar month or the time unit)."""
        if self._own(s):
            return None
        s_c = [s] + self.kb.other_senses(s_word, s)[:3]
        o_c = [o] + (self.kb.other_senses(o_word, o)[:3] if o_word else [])
        for _, k, t in sorted((i + j, k, t) for i, k in enumerate(s_c)
                              for j, t in enumerate(o_c) if (k, t) != (s, o)):
            b = self.kb.ask(k, r, t)
            if b.known:
                return b
        return None

    def _other_sense_isa(self, word, s, o_word, o, subject_too=True):
        """'Is gold a metal?': a word is an X if any of its senses is (gold
        the element), as long as the word is not one of the user's own."""
        if self._own(s):
            return None
        targets = [o] + self.kb.other_senses(o_word, o)
        no = None
        for k in [s] + (self.kb.other_senses(word, s) if subject_too else []):
            for t in targets:
                if (k, t) == (s, o):
                    continue
                b = self.kb.ask(k, "isa", t)
                if b.known and b.truth.f >= 0.5 and b.how in ("direct", "transitivity") \
                        and b.truth.c >= 0.5:
                    self._record(b)
                    sense = f", in the sense of {self._gloss(k)}" if k != s and \
                        self._gloss(k) else ""
                    v = "Yes" if b.truth.c >= 0.7 else "Probably"
                    if b.how == "direct" and k == s:
                        return f"{v}."
                    return f"{v}{sense}. {self.kb.explain(b, brief=True)}"
                if no is None and k == s and b.how == "taxonomy" and \
                        not self.kb.ask(s, "isa", o).known:
                    no = b        # 'Is a dog a plant?': the botanical plant, not the factory
        if no is not None:
            self._record(no)
            return f"No. {self.kb.explain(no, brief=True)}"
        return None

    def _senses_of(self, word):
        """The concept a word names now, then its other senses."""
        k = self._c(word)
        return [k] + self.kb.other_senses(word, k)

    def _q_objects(self, q):
        ans = []
        for k in self._senses_of(q.s):
            ans = self.kb.objects(k, q.r)
            if ans:
                q.s = k
                break
        if not ans:
            return "I don't know."
        if q.r == "used_for":       # what a hammer is for, not what any tool is for
            near = min(a.dist for a in ans)
            ans = [a for a in ans if a.dist == near]
        self._record(ans[0])
        if q.r in ("can", "does"):
            verbs = [("probably " if a.truth.c < 0.7 else "") + a.value for a in ans[:6]]
            subj = self.kb.name(q.s, True) if self.kb.generic(q.s) else self.kb.ref(q.s)
            return self._cap(f"{subj} can {self._join(verbs)}.")
        if q.r == "opposite":
            return self._cap(self._join([a.value for a in ans[:3]])) + "."
        sure, unsure = [], []
        for a in ans[:5]:
            qty = self._qty(q.s, q.r, a.value)
            if qty is None and a.premises and isinstance(a.premises[-1], Fact):
                qty = a.premises[-1].qty
            phrase = self.kb.say(Fact(q.s, q.r, a.value, a.truth, qty))
            (sure if a.truth.c >= 0.7 else unsure).append(phrase)
        head, rest = self._merge(sure + unsure)
        if head is None:
            parts = sure + ["probably " + p for p in unsure]
            return self._cap(self._join(parts)) + "."
        sure_rest, unsure_rest = rest[:len(sure)], rest[len(sure):]
        text = head + " " + self._join(sure_rest) if sure_rest else ""
        if unsure_rest:
            maybe = "probably " + self._join(unsure_rest)
            text = f"{text}, and {maybe}" if text else f"{head} {maybe}"
        return self._cap(text) + "."

    @staticmethod
    def _merge(phrases):
        """'water is made of oxygen' + 'water is made of hydrogen' ->
        ('water is made of', ['oxygen', 'hydrogen']); (None, phrases) if they
        share no subject and verb."""
        split = [p.split() for p in phrases]
        n = 0
        while all(len(w) > n + 1 and w[n] == split[0][n] for w in split):
            n += 1
        if n < 2:
            return None, phrases
        return " ".join(split[0][:n]), [" ".join(w[n:]) for w in split]

    def _qty(self, s, r, o):
        f = self.kb.get(s, r, o)
        if f is not None:
            return f.qty
        for m, _ in self.kb.categories_of(s):
            g = self.kb.get(m, r, o)
            if g is not None:
                return g.qty
        return None

    @staticmethod
    def _join(parts):
        if len(parts) <= 1:
            return "".join(parts)
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    def _q_attr(self, q):
        q.s = self._c(q.s)
        vals = [a for a in self.kb.objects(q.s, "is")
                if self.kb.ask(a.value, "isa", q.extra).truth.f >= 0.5
                and self.kb.ask(a.value, "isa", q.extra).known]
        if not vals:
            return "I don't know."
        a = self._record(vals[0])
        phrase = self.kb.say(Fact(q.s, "is", a.value, a.truth))
        if a.how != "direct":
            return self._cap(phrase) + ". " + self.kb.explain(a)
        return self._cap(phrase) + "."

    def _q_how_many(self, q):
        s_word, o_word = q.s, q.o
        q.s, q.o = self._c(q.s), self._c(q.o)
        if self.kb.get(q.s, "has", q.o) is None and not self._own(q.s):
            for k in [q.s] + self.kb.other_senses(s_word, q.s)[:3]:
                hit = next((t for t in [q.o] + self.kb.other_senses(o_word, q.o)[:3]
                            if (g := self.kb.get(k, "has", t)) is not None and g.qty
                            and g.qty > 0), None)
                if hit is not None:
                    q.s, q.o = k, hit
                    break
        f = self.kb.get(q.s, "has", q.o)
        if f is None:
            for m, mem in self.kb.categories_of(q.s):
                g = self.kb.get(m, "has", q.o)
                if g is not None and g.qty is not None and g.qty >= 0:
                    self._record(Answer(Truth.deduction(mem.truth, g.truth), mem.premises + [g],
                                        "inheritance"))
                    return (f"{self._cap(str(L.NUMBER_WORDS.get(g.qty, g.qty)))}. "
                            f"{self.kb.explain(self.last)}")
            if self._members(self._senses_of(o_word))[1]:   # 'how many planets does the
                got = self._q_count(L.Query("count", o_word, "in", s_word))   # solar system have'
                if not got.startswith("I don't"):
                    return got
            return "I don't know."
        if f.qty is None or f.qty < 0:
            return "I don't know how many."
        self._record(Answer(f.truth, [f]))
        return f"{self._cap(str(L.NUMBER_WORDS.get(f.qty, f.qty)))}."

    def _q_subjects(self, q):
        ans = []
        for k in self._senses_of(q.o):
            ans = self.kb.subjects(q.r, k)
            if ans:
                break
        if ans:
            self._record(ans[0])
            names = [self.kb.name(a.value) for a in ans[:5]]
            return self._cap(self._join(names)) + "."
        if q.extra:
            cue, target = self._c(q.extra), self._c(q.o)
            hd = self._hd_analogy(cue, target) or self._map_analogy(cue, target)
            if hd:
                return hd
        return "I don't know."

    def _map_analogy(self, cue, target):
        """Structure mapping: the dollar is the currency of the USA, so 'the
        dollar of Mexico' is what stands in that relation to Mexico."""
        for f in self.kb.out(cue):
            if f.truth.f < 0.5 or f.r == "isa":
                continue
            ans = [a for a in self.kb.subjects(f.r, target) if a.value != cue]
            if ans:
                x = ans[0].value
                self._record(Answer(Truth(1.0, 0.8), [f] + ans[0].premises, "analogy"))
                rel = f.r[:-3].replace("_", " ") if f.r.endswith("_of") else f.r
                return (f"{self._cap(self.kb.name(x))} (by analogy: {self.kb.name(cue)} is the "
                        f"{rel} of {self.kb.ref(f.o)}, and {self.kb.name(x)} is the {rel} of "
                        f"{self.kb.ref(target)}).")
        return None

    def _hd_analogy(self, cue, target):
        """'What is the dollar of Mexico?': find a record that contains the
        cue, map it onto the target's record in hyperdimensional space."""
        recs = {}

        def record(e):
            if e not in recs:
                pairs = [(f.r, f.s) for f in self.kb.inn(e) if f.truth.f >= 0.5]
                pairs += [(f.r + "^", f.o) for f in self.kb.out(e) if f.truth.f >= 0.5]
                recs[e] = (self.hd.record(pairs) if pairs else None, [p[1] for p in pairs])
            return recs[e]

        sources = {f.o for f in self.kb.out(cue)} | {f.s for f in self.kb.inn(cue)}
        sources.discard(target)
        tv, fillers = record(target)
        if tv is None:
            return None
        for src in sources:
            sv, _ = record(src)
            if sv is None:
                continue
            best = self.hd.analogy(sv, tv, cue, names=fillers, k=1)
            if best and best[0][1] > 0.15:
                name = best[0][0]
                self._record(Answer(Truth(1.0, min(0.9, best[0][1] * 1.5)), [], "analogy"))
                return (f"{self._cap(self.kb.name(name))} (by analogy: {self.kb.name(cue)} is to "
                        f"{self.kb.name(src)} what {self.kb.name(name)} is to "
                        f"{self.kb.name(target)}; similarity {best[0][1]:.2f}).")
        return None

    def _knows(self, x):
        return bool(self.kb.out(x)) or x in self.kb.definitions or self.world.knows(x)

    def _q_describe(self, q):
        word = q.s
        senses = self._senses_of(word)
        s = next((k for k in senses if self._knows(k)), senses[0])
        if word in self._caps and not self._own(senses[0]):
            s = next((k for k in senses if k in self.kb.display and k in self.kb.definitions), s)
        kb = self.kb
        proper = s in kb.display
        subj = kb.ref(s) if proper else kb.name(s, True) if kb.generic(s) else kb.ref(s)
        be = "are" if not proper and kb.many(s) else "is"
        lines = []
        if s in kb.definitions:
            head = kb.display.get(s, kb.bare(s))
            lines.append(f"{self._cap(head)}: {kb.definitions[s]}")
        if self.world.knows(s):
            p, _ = self.world.where(s)
            if isinstance(p, str):
                lines.append(f"{kb.ref(s)} is in {kb.ref(p)}")
            carried = self.world.carrying(s)
            if carried:
                lines.append(f"{kb.ref(s)} is carrying " + self._join([kb.ref(o) for o in carried]))
        kinds = [m for m, a in kb.categories_of(s)
                 if kb.bare(m) not in kb._TOO_GENERAL | {"organism"}][:4]
        if kinds:
            terms = [kb.article(c) if be == "is" else kb.name(c, True) for c in kinds]
            lines.append(f"{subj} {be} {self._join(terms)}")
        facts = [f for f in kb.out(s) if f.source != "innate" and f.r != "isa"]
        nat = {f.o.lower() for f in facts if f.r == "nationality"}
        facts = [f for f in facts if not (f.r == "is" and f.o in nat)]    # 'English' once
        facts.sort(key=lambda f: (f.source in kb.authority, self._LIFE.get(f.r, 9),
                                  -kb.activation(f)))
        span = {f.r: f for f in facts if f.r in self._LIFE}
        for a, b, verb in (("born", "died", "lived"), ("began", "ended", "lasted")):
            if a in span and b in span:      # 'Newton lived from 1642 to 1727'
                lines.append(f"{subj} {verb} from {span[a].o} to {span[b].o}")
                facts = [f for f in facts if f is not span[a] and f is not span[b]]
        for f in facts[:5]:
            lines.append(kb.say(f))
        others = [w for w in kb.words.get(s, []) if w.lower() not in
                  (kb.display.get(s, "").lower(), word)]
        if others:
            lines.append("also called " + self._join(others[:3]))
        famous = proper and s in kb.definitions      # Einstein's abilities are not news
        inh = [] if famous else [a for a in kb.objects(s, "can") if a.how != "direct"][:4]
        if inh:
            verbs = [("probably " if a.truth.c < 0.7 else "") + a.value for a in inh]
            lines.append(f"{subj} can {self._join(verbs)}")
        cannot = [] if famous else \
            [a for a in kb.objects(s, "can", positive=False) if a.how != "direct"][:3]
        if cannot:
            lines.append(f"{subj} cannot {self._join([a.value for a in cannot])} "
                         f"({kb.say_answer(cannot[0]) if cannot[0].premises else ''})")
        if not lines:
            if self.semantic is not None and word in self.semantic:
                sim = ", ".join(w for w, _ in self.semantic.similar(word, 5))
                return f"I know the word '{word}' from reading; it goes with {sim}."
            return f"I don't know anything about {word}."
        return ". ".join(self._cap(x) for x in lines) + "."

    _LIFE = {"born": 0, "died": 1, "began": 0, "ended": 1, "year": 0, "nationality": 2,
             "from": 3, "born_in": 3}

    # ------------------------------------------------------------------ natural questions
    def _yes(self, reply):
        return reply.startswith(("Yes", "Probably")) and not reply.startswith("Probably not")

    def _no(self, reply):
        return reply.startswith(("No", "Probably not", "Maybe not"))

    def _q_why_q(self, q):
        """'Why is a whale not a fish?': answer the inner question and give
        the reasons -- or correct the question's assumption."""
        inner, neg = q.extra
        qs = [i for i in L.parse_sentence(inner + "?", self.wm, self._caps).items
              if isinstance(i, L.Query)]
        if not qs:
            return "I don't understand that yet."
        reply = self._answer(qs[0], inner)
        a = self.last
        if reply.startswith("I don't"):
            return reply
        if self._yes(reply) == neg or self._no(reply) != neg:
            if a is not None and a.how == "direct" and a.premises and reply.count(".") == 1:
                return f"Actually, {self.kb.say(a.premises[0])}."
            return "Actually, " + self._decap(reply)
        if a is None or not (a.premises or a.stmt):
            return reply
        if a.how == "direct":
            return f"I was told so: {self.kb.say(a.premises[0])}. I know no deeper reason."
        return "Because " + self._decap(self.kb.explain(a))

    def _decap(self, text):
        """Lower-case a sentence start, unless it is a name ('Tweety is ...')."""
        first = text.split(" ", 1)[0].rstrip(".,;")
        if first.lower() in self.kb.display or first in self.kb.display.values():
            return text
        return text[:1].lower() + text[1:]

    def _basic(self, chain):
        """The everyday level of a chain of kinds (Rosch): the one people talk
        about ('mammal', not 'placental'), else the nearest."""
        talked = [m for m in chain if any(f.source not in ("wordnet",) for f in self.kb.inn(m, "isa"))]
        return talked[0] if talked else chain[0]

    def _q_kind(self, q):
        """'What kind of animal is a whale?' -> 'A mammal (more precisely, a
        cetacean).'"""
        kb = self.kb
        for k in self._senses_of(q.extra):
            for s in self._senses_of(q.s):
                cats = [m for m, _ in kb.categories_of(s)]
                if k not in cats:
                    continue
                chain = [m for m in cats[:cats.index(k)]
                         if kb.bare(m) not in kb._TOO_GENERAL and kb.ask(m, "isa", k).known
                         and kb.ask(m, "isa", k).truth.f >= 0.5]
                if not chain:
                    return f"It is simply {kb.article(k)}."
                basic = self._basic(chain)
                self._record(kb.ask(s, "isa", basic))
                more = f" (more precisely, {kb.article(chain[0])})" if chain[0] != basic else ""
                return self._cap(kb.article(basic)) + more + "."
        return "I don't know."

    # comparisons the numbers decide: by area, or by year of birth (older = born earlier)
    _BY_NUMBER = {"bigger_than": ("area", 1), "smaller_than": ("area", -1),
                  "older_than": ("born", -1), "younger_than": ("born", 1)}

    def _q_compare(self, q):
        """'Which is bigger, an elephant or a mouse?', '... France or Spain?'"""
        kb = self.kb
        s, o = self._c(q.s), self._c(q.o)
        for x, y in ((s, o), (o, s)):
            a = kb.ask(x, q.r, y)
            if a.known:
                winner = x if a.truth.f >= 0.5 else y
                self._record(a)
                name = kb.display.get(winner) or (kb.article(winner) if kb.generic(winner)
                                                  else kb.ref(winner))
                return f"{self._cap(name)}. {kb.explain(a, brief=True)}"
        if q.r in self._BY_NUMBER:
            attr, sign = self._BY_NUMBER[q.r]
            got = []
            for word in (q.s, q.o):
                k, f = self._first(word, attr)
                v = None if f is None else self._year(f.o) if attr == "born" else float(f.o)
                if v is None:
                    break
                got.append((v, k, f))
            if len(got) == 2 and got[0][0] != got[1][0]:
                (va, a, fa), (vb, b, fb) = got
                self._record(Answer(Truth(1.0, min(fa.truth.c, fb.truth.c)), [fa, fb],
                                    "comparison"))
                winner = a if (va - vb) * sign > 0 else b
                return f"{self._cap(kb.ref(winner))}. {self._cap(kb.say(fa))}, and {kb.say(fb)}."
        return "I don't know."

    def _q_choice(self, q):
        """'Is a tomato a fruit or a vegetable?'"""
        kb = self.kb
        verdicts = [(t, self._q_yesno(L.Query("yesno", q.s, "isa", t)), self.last)
                    for t in q.extra]
        yes = [(t, r, a) for t, r, a in verdicts if self._yes(r)]
        no = [(t, r, a) for t, r, a in verdicts if self._no(r)]
        if len(yes) == 1:
            t, r, a = yes[0]
            self._record(a)
            reason = r.split(". ", 1)[1] if ". " in r else ""
            return f"{self._cap(kb.article(self._c(t)))}. {reason}".strip()
        s = self._c(q.s)
        many = kb.many(s)
        subj = kb.name(s, True) if many else kb.ref(s)
        kinds = self._join([kb.name(self._c(t), True) if many else kb.article(self._c(t))
                            for t in q.extra])
        if len(yes) == len(q.extra):
            return f"Both: {subj} {'are' if many else 'is'} {kinds}."
        if len(no) == len(q.extra):
            return f"Neither: {subj} {'are' if many else 'is'} not {kinds.replace(' and ', ' or ')}."
        return "I don't know."

    def _q_difference(self, q):
        """'What is the difference between a whale and a fish?': where they
        part in the taxonomy, and what each is."""
        kb = self.kb
        a, b = self._c(q.s), self._c(q.o)
        lines = []
        cats_b = dict(kb.categories_of(b))
        if a in cats_b:
            lines.append(f"{self._cap(kb.name(a, True))} are a kind of {kb.word(b)}")
        elif b in dict(kb.categories_of(a)):
            lines.append(f"{self._cap(kb.name(b, True))} are a kind of {kb.word(a)}")
        else:
            for p, ans in kb.categories_of(a):
                if p in cats_b and kb.bare(p) not in kb._TOO_GENERAL:
                    ca, cb = ans.premises, cats_b[p].premises
                    ba = ca[-2].o if len(ca) > 1 else a
                    bb = cb[-2].o if len(cb) > 1 else b
                    lines.append(f"{self._cap(kb.name(a, True))} are {kb.name(ba, True)}, while "
                                 f"{kb.name(b, True)} are {kb.name(bb, True)}; both are "
                                 f"{kb.name(p, True)}")
                    break
        for x in (a, b):
            if x in kb.definitions:
                lines.append(f"{self._cap(kb.word(x))}: {self._gloss(x, 120)}")
        return ". ".join(lines) + "." if lines else "I don't know."

    def _q_place_kind(self, q):
        """'Which continent is Kenya in?'"""
        kb = self.kb
        kinds = self._senses_of(q.extra)
        for s in self._senses_of(q.s):
            for x, a in kb.reach(s, "in").items():
                if any(kb.ask(x, "isa", k).known and kb.ask(x, "isa", k).truth.f >= 0.5
                       for k in kinds):
                    self._record(a)
                    reason = f" {kb.explain(a, brief=True)}" if len(a.premises) > 1 else ""
                    return f"{self._cap(kb.ref(x))}.{reason}"
        return "I don't know."

    def _familiar(self, facts):
        """What the user or the pack said first, then everyday words: a word
        with many senses is a frequent word (Zipf's law of meaning), so 'dog'
        comes before 'placental'."""
        def rank(f):
            w = self.kb.word(f.s)
            return (f.source == "wordnet", -self.kb.freq.get(f.s, 0), "#" in f.s,
                    -len(self.kb.senses.get(w, ())), " " in w, len(w), w)
        return sorted(facts, key=rank)

    def _names(self, xs, n=8):
        kb = self.kb
        out = []
        for x in xs:
            w = kb.display.get(x) or kb.name(x, kb.generic(x) or not kb.is_proper(x))
            if w not in out:
                out.append(w)
        return self._join(out[:n])

    def _q_examples(self, q):
        """'Name some birds.'"""
        kb = self.kb
        for c in self._senses_of(q.s):
            found, frontier, seen = [], [c], {c}
            for depth in range(4):              # kinds of kinds: 'dog' is a mammal too
                nxt = []
                for x in frontier:
                    for f in sorted(kb.inn(x, "isa"), key=lambda f: f.s):
                        if f.truth.f < 0.5 or f.s in seen or len(seen) >= 3000:
                            continue
                        seen.add(f.s)
                        nxt.append(f.s)
                        if depth and kb.is_proper(f.s):
                            continue            # 'Smith' is a mammal only via 'person'
                        if c in kb.taxa and f.s not in kb.taxa and f.source == "wordnet":
                            continue            # examples of an animal kind are animals
                        found.append(f)
                frontier = nxt
            if found:
                return self._cap(self._names([f.s for f in self._familiar(found)], 6)) + "."
        return "I don't know any."

    def _q_which(self, q):
        """'Which animals can fly?', 'What animals live in the ocean?'"""
        kb = self.kb
        c = self._c(q.s)
        positive = q.extra is not False
        if q.r == "isa":
            facts = [f for f in kb.inn(self._c(q.o), "isa") if f.truth.f >= 0.5]
        else:
            o = q.o if q.r in self._RAW_OBJECT else self._c(q.o)
            facts = [f for f in kb.inn(o, q.r) if (f.truth.f >= 0.5) == positive]
        keep = [f for f in self._familiar(facts)
                if f.s == c or (kb.ask(f.s, "isa", c).known and kb.ask(f.s, "isa", c).truth.f >= 0.5)]
        if not keep:
            return "I don't know any."
        self._record(Answer(keep[0].truth, [keep[0]]))
        return self._cap(self._names([f.s for f in keep])) + "."

    # ------------------------------------------------------------------ the wider world
    @staticmethod
    def _year(v):
        """'1564' -> 1564, '470 BC' -> -470; None if v is not a year."""
        m = re.fullmatch(r"(\d{1,4})( BC)?", str(v).strip())
        if not m:
            return None
        return -int(m.group(1)) if m.group(2) else int(m.group(1))

    def _first(self, word, *rels):
        """The first sense of a word with a fact of one of these relations, and
        the fact (Newton the physicist has a birth year; the unit has none)."""
        for k in self._senses_of(word):
            for r in rels:
                f = next((f for f in self.kb.out(k, r) if f.truth.f >= 0.5), None)
                if f is not None:
                    return k, f
        return None, None

    def _spelled(self, word):
        return self._caps.get(word, word)

    def _q_when(self, q):
        """'When was Shakespeare born?', 'When did World War II end?', 'When
        was the French Revolution?'"""
        kb = self.kb
        k, f = self._first(q.s, *([q.r, "year"] if q.r in ("began", "ended") else [q.r] if q.r
                                 else ["began", "born", "year"]))
        if f is None:
            return "I don't know."
        end = None if q.r or f.r == "year" else \
            next(iter(kb.out(k, "ended" if f.r == "began" else "died")), None)
        if end is not None:
            self._record(Answer(f.truth, [f, end]))
            verb = "lived" if f.r == "born" else "lasted"
            return f"{self._cap(kb.ref(k))} {verb} from {f.o} to {end.o}."
        self._record(Answer(f.truth, [f]))
        return self._cap(kb.say(f)) + "."

    def _q_lifespan(self, q):
        """'How old was Mozart when he died?' (from the years alone, so about)"""
        kb = self.kb
        for k in self._senses_of(q.s):
            b = next((f for f in kb.out(k, "born") if self._year(f.o) is not None), None)
            d = next((f for f in kb.out(k, "died") if self._year(f.o) is not None), None)
            if b is not None and d is not None:
                self._record(Answer(Truth(1.0, min(b.truth.c, d.truth.c)), [b, d], "arithmetic"))
                age = self._year(d.o) - self._year(b.o)
                return f"About {age}: {kb.ref(k)} was born in {b.o} and died in {d.o}."
        return "I don't know."

    def _q_born_first(self, q):
        """'Who was born first, Newton or Einstein?'"""
        kb = self.kb
        got = []
        for word in (q.s, q.o):
            k, f = self._first(word, "born", "began")
            if f is None or self._year(f.o) is None:
                return f"I don't know when {self._spelled(word)} was born."
            got.append((self._year(f.o), k, f))
        (ya, a, fa), (yb, b, fb) = got
        self._record(Answer(Truth(1.0, min(fa.truth.c, fb.truth.c)), [fa, fb], "comparison"))
        if ya == yb:
            return f"Neither: {kb.say(fa)}, and so was {kb.ref(b)}."
        return f"{self._cap(kb.ref(a if ya < yb else b))}. {self._cap(kb.say(fa))}, and {kb.say(fb)}."

    def _q_origin(self, q):
        """'Where was Gandhi from?', 'Where was Einstein born?', 'Where do
        kangaroos come from?'"""
        kb = self.kb
        k, f = self._first(q.s, *(("born_in",) if q.extra else ("from", "born_in")))
        if f is not None:
            self._record(Answer(f.truth, [f]))
            return self._cap(kb.say(f)) + "."
        k, f = self._first(q.s, *(("from", "nationality") if q.extra else ("nationality",)))
        if f is not None:                                # 'Mozart was Austrian'
            self._record(Answer(f.truth, [f]))
            return ("I don't know where, but " if q.extra else "") + \
                (kb.say(f) if q.extra else self._cap(kb.say(f))) + "."
        if q.extra:
            return "I don't know."
        for k in self._senses_of(q.s):
            ans = kb.objects(k, "in")
            if ans:
                a = self._record(ans[0])
                gen = kb.generic(k)
                subj = kb.name(k, True) if gen else kb.ref(k)
                return f"{self._cap(subj)} {'come' if gen else 'comes'} from {kb.ref(a.value)}."
        return "I don't know."

    def _q_by_attr(self, q):
        """'What element has the symbol Na?', 'Which element has atomic number 8?'"""
        kb = self.kb
        for v in dict.fromkeys((q.o, q.o.capitalize(), q.o.upper())):
            f = next((f for f in kb.inn(v, q.r) if f.truth.f >= 0.5), None)
            if f is not None:
                self._record(Answer(f.truth, [f]))
                return f"{self._cap(kb.word(f.s))}: {kb.say(f)}."
        return "I don't know."

    def _isa(self, x, kinds):
        cats = {x} | {m for m, _ in self.kb.categories_of(x)}
        return any(k in cats for k in kinds) and not any(
            self.kb.get(x, "isa", k) is not None and self.kb.get(x, "isa", k).truth.f < 0.5
            for k in kinds)

    def _facts_of(self, r):
        return [f for f in self.kb.facts.values() if f.r == r and f.truth.f >= 0.5]

    def _within(self, x, region):
        return not region or any(r in self.kb.reach(x, "in") for r in region)

    _EXTREME = {"largest": max, "biggest": max, "smallest": min}

    def _extreme(self, cats, region, pick, attr="area"):
        """The largest or smallest of a kind by the numbers ('the largest
        country in Africa' by area)."""
        kb = self.kb
        facts = []
        for f in self._facts_of(attr):
            try:
                facts.append((float(f.o), f))
            except ValueError:
                continue
        cands = []
        for c in cats:              # the usual sense of 'country' first
            cands = [(v, f.s, f) for v, f in facts if self._isa(f.s, [c])
                     and self._within(f.s, region)]
            if len(cands) >= 2:
                cats = [c]
                break
        if len(cands) < 2:
            return None
        _, x, f = pick(cands, key=lambda c: c[0])
        self._record(Answer(f.truth, [f], "comparison"))
        where = f" in {kb.ref(region[0])}" if region else ""
        most = "largest" if pick is max else "smallest"
        return (f"{self._cap(kb.ref(x))}. It {kb._parts(f)[1]}, the {most} of the {len(cands)} "
                f"{kb.name(cats[0], True)}{where} I know.")

    _SAME = {"largest": "largest|biggest", "biggest": "largest|biggest", "tallest": "tallest|highest",
             "highest": "highest|tallest", "greatest": "greatest|largest"}
    _HEADS = {"mountain": "mountain|peak|mountain peak", "peak": "peak|mountain"}
    _ORDINAL = r"\d+(?:st|nd|rd|th)|second|third|fourth|fifth|sixth"

    def _superlative_gloss(self, adj, word, cats, region):
        """The dictionary's own superlatives: 'the world's longest river'
        beats 'the longest river in England' (unless England was asked), and
        'the 2nd largest ocean' is not the largest."""
        kb = self.kb
        adjs = self._SAME.get(adj, adj)
        heads = self._HEADS.get(word, re.escape(word))
        heads = f"(?:{heads})(?:e?s)?"
        rx = re.compile(rf"\b(?:the )?(?P<w>world's |earth's )?(?P<n>(?:{self._ORDINAL}) )?"
                        rf"(?:{adjs}) (?P<mod>[a-z-]+ )?{heads}\b(?P<rest>[^;]*)", re.I)
        quick = adjs.split("|")
        places = {kb.word(r).lower() for r in region} | \
            {kb.display[r].lower() for r in region if r in kb.display}
        best = None
        for k, d in kb.definitions.items():
            if not any(a in d for a in quick):
                continue
            for clause in d.split(";"):
                m = rx.search(clause)
                if m is None or m["n"]:
                    continue
                pre = clause[:m.start()].strip().lower()
                if pre and not re.search(r"(?:^|\b)(?:is|was|and|also|now|it)$|,$", pre):
                    continue        # 'this range contains the world's highest mountain'
                rest = m["rest"].strip().lower()
                world = bool(m["w"]) or re.match(
                    r"(?:\w+ )?(?:in|on|of) (?:the )?(?:world|earth|globe)\b", rest) is not None
                place = None if world else re.match(r"(?:\w+ )?(?:in|of|on) (?:the )?"
                                                    r"([a-z][a-z .'-]*)", rest)
                if place is not None:
                    if not any(place.group(1).startswith(p) for p in places):
                        continue    # 'the longest river in England'
                    score = 3
                else:
                    score = 3 if world and not region else 2
                if m["mod"]:
                    score -= 1          # 'largest freshwater lake' is weaker than 'largest lake'
                key = (score, self._isa(k, cats), kb.freq.get(k, 0))
                if best is None or key > best[0]:
                    best = (key, k, clause.strip())
        if best is None:
            return None
        _, k, clause = best
        self._record(Answer(Truth(1.0, 0.8), [Fact(k, "defined_as", clause, Truth(1.0, 0.8))],
                            "definition"))
        return f"{self._cap(kb.ref(k))}: {clause}."

    def _q_superlative(self, q):
        """'What is the largest country (in Africa)?': by the numbers where the
        data has them; 'What is the longest river?': from the dictionary."""
        cats = self._senses_of(q.s)
        region = self._senses_of(q.o) if q.o else []
        if q.r in self._EXTREME:
            got = self._extreme(cats, region, self._EXTREME[q.r])
            if got:
                return got
        return self._superlative_gloss(q.r, q.s, cats, region) or "I don't know."

    def _listed(self, c):
        """The members of a kind that a list names -- the pack, the user, the
        world data: seven continents, not WordNet's Pangaea and Laurasia too;
        the 195 countries, not Prussia."""
        kb = self.kb
        return {f.s for f in kb.inn(c, "isa") if f.truth.f >= 0.5 and f.source != "wordnet"
                and kb.is_proper(f.s)}

    def _members(self, cats):
        for c in cats:
            listed = sorted(self._listed(c))
            if listed:
                return c, listed
        return None, []

    def _q_count(self, q):
        """'How many continents are there?', 'How many countries border Germany?'"""
        kb = self.kb
        cats = self._senses_of(q.s)
        place = kb.word(self._c(q.o)) if q.o is not None else None
        defined = [c for c in cats if place and re.search(rf"\b{re.escape(place)}\b",
                                                           kb.definitions.get(c, ""))]
        if defined and self._members(defined)[1]:
            q.o = None              # 'planets: bodies in the solar system that ...': all of them
        if q.o is not None:
            found, related = [], set()
            for o in self._senses_of(q.o):
                related = self._related(o, q.r)
                if related:
                    q.o = o
                    break
            for c in cats:              # the usual sense of 'country' first, not 'area'
                listed = self._listed(c)
                found = sorted(x for x in related if kb.is_proper(x) and (
                    x in listed if listed else self._isa(x, [c])))
                if found:
                    break
            if not found:
                return "I don't know."
            facts = [kb.get(x, q.r, q.o) or kb.get(q.o, q.r, x) for x in found]
            if any(facts):
                self._record(Answer(Truth(1.0, 0.9), [f for f in facts if f][:3]))
        else:
            c, found = self._members(defined or cats)
            if not found:
                return "I don't know how many there are."
            self._record(Answer(Truth(1.0, 0.9), [kb.get(found[0], "isa", c)]))
        n = len(found)
        num = self._cap(str(L.NUMBER_WORDS.get(n, n)))
        if n > 12:
            return f"{num}, among them {self._names(found, 8)}."
        return f"{num}: {self._names(found, 12)}."

    def _related(self, o, r):
        """Everything that stands in r to o: what borders it (either way
        round), or what is in it, at any depth (a country in Northern Africa
        is in Africa)."""
        kb = self.kb
        if r != "in":
            return {f.s for f in kb.inn(o, r) if f.truth.f >= 0.5} | \
                {f.o for f in kb.out(o, r) if f.truth.f >= 0.5 and r == "border"}
        seen, frontier = set(), [o]
        while frontier and len(seen) < 50000:
            nxt = []
            for x in frontier:
                for f in kb.inn(x, "in"):
                    if f.truth.f >= 0.5 and f.s not in seen:
                        seen.add(f.s)
                        nxt.append(f.s)
            frontier = nxt
        return seen

    _TITLES = {"president", "king", "queen", "sir", "saint", "st.", "lord", "general", "pope",
               "emperor", "czar", "tsar", "prince", "princess", "dr.", "mr.", "mrs.", "da", "de"}

    def _full_name(self, k):
        """'Alexander Graham Bell', not just 'Bell'."""
        name = self.kb.display.get(k) or self.kb.word(k)
        best = name
        for w in self.kb.words.get(k, []):
            parts = w.split()
            if name in parts and parts[0].lower() not in self._TITLES and \
                    len(parts) > len(best.split()):
                best = w
        return best

    def _person(self, k):
        return self._isa(k, ["person"])

    def _person_named(self, name):
        """The person a name in a definition refers to ('William Shakespeare')."""
        for n in (name, name.split()[-1]):
            k = self.kb.canon(n.lower(), proper=True)
            for x in [k] + self.kb.other_senses(n.lower(), k):
                if x in self.kb.display and self._person(x):
                    return x
        return None

    _IRREGULAR = {"wrote": r"wrote|writ|author", "built": r"built|build", "led": r"led|lead",
                  "first climbed": r"climb", "killed": r"kill|murder|slew",
                  "assassinated": r"assassin"}
    _WORKS = {"wrote": r"tragedy|comedy|play|novel|poem|book|story|novella|epic|essay",
              "painted": r"painting|portrait|fresco|mural",
              "composed": r"opera|symphony|concerto|oratorio|ballet|song|requiem",
              "sculpted": r"sculpture|statue"}

    def _cue(self, verb):
        if verb in self._IRREGULAR:
            return self._IRREGULAR[verb]
        stem = verb[:-3] if verb.endswith("ied") else verb[:-2] if verb.endswith("ed") else verb
        return re.escape(stem)

    def _q_who_did(self, q):
        """'Who discovered penicillin?', 'Who invented the telephone?', 'Who
        wrote Hamlet?': a fact if there is one (oxygen was discovered by
        Scheele), else the dictionary, which says who did what in its
        definitions of people ('bacteriologist who discovered penicillin')
        and of things ('the hero of William Shakespeare's tragedy')."""
        kb = self.kb
        obj = re.sub(r"^(?:the|a|an) ", "", q.o.strip())
        verb = L.verb_lemma(q.r.split()[-1])
        for k in self._senses_of(obj):       # what someone said: 'Leonardo painted the Mona Lisa'
            f = next((f for f in kb.inn(k, verb) if f.truth.f >= 0.5), None)
            if f is not None:
                self._record(Answer(f.truth, [f]))
                return f"{self._cap(kb.display.get(f.s) or kb.ref(f.s))}."
        if q.r == "discovered":
            for k in self._senses_of(obj):
                f = next(iter(kb.out(k, "discovered_by")), None)
                if f is not None:
                    self._record(Answer(f.truth, [f]))
                    return f"{f.o}: {kb.say(f)}."
        cue = re.compile(rf"\b(?:{self._cue(q.r)})", re.I)
        names = [obj] + [w.lower() for k in self._senses_of(obj)[:3]
                         for w in [kb.word(k)] + kb.words.get(k, [])]
        names = [n for n in dict.fromkeys(names) if len(n) > 2]
        orx = re.compile(r"\b(?:" + "|".join(r"[\s-]+".join(map(re.escape, n.split()))
                                             for n in names) + r")\b", re.I)
        firsts = {n.split()[0] for n in names}
        best = None
        for k, d in kb.definitions.items():
            if k not in kb.display:
                continue
            low = d.lower()
            if not any(w in low for w in firsts):
                continue
            # the thing itself, not part of a longer name ('America' in 'North America')
            m = next((x for x in orx.finditer(d) if not re.search(r"[A-Z][\w'-]* $", d[:x.start()])
                      and not re.match(r" [A-Z]", d[x.end():])), None)
            if m is None:
                continue
            cues = [c for c in cue.finditer(d) if c.start() < m.start()
                    and len(d[c.end():m.start()].split()) <= 6
                    and " by " not in d[c.end():m.start()]
                    and not re.search(r"\b(?:been|was|were|is|are)\s*$", d[:c.start()])]
            if not cues and not cue.search(d):
                continue
            score = (2 if cues else 1, kb.freq.get(k, 0))
            if (best is None or score > best[0]) and self._person(k):
                best = (score, k)
        if best is None or best[0][0] < 2:
            works = self._WORKS.get(q.r, "|".join(self._WORKS.values()))
            name = r"([A-Z][a-z]+(?: (?:[A-Z][a-z.]+|da|de|van|von))*)"
            for k in self._senses_of(obj):
                d = kb.definitions.get(k, "")
                m = re.search(name + rf"'s (?:\w+ )?(?:{works})\b", d) or \
                    re.search(rf"\b(?:{works})(?:e?s)? by " + name, d)
                p = self._person_named(m.group(1)) if m else None
                if p is not None:
                    self._record(Answer(Truth(1.0, 0.6), [Fact(k, "defined_as", d,
                                                               Truth(1.0, 0.6))], "definition"))
                    return (f"{self._full_name(p)}, I think: {kb.ref(k)} is "
                            f"{self._gloss(k, 120)}.")
        if best is None or best[0][0] < 2:     # 'gravitational attraction ... and discovered
            return "I don't know."             # the cellular structure of cork' is not it
        k = best[1]
        self._record(Answer(Truth(1.0, 0.8), [Fact(k, "defined_as", kb.definitions[k],
                                                   Truth(1.0, 0.8))], "definition"))
        return f"{self._full_name(k)}: {kb.definitions[k]}."

    _NTH = {1: "1st", 2: "2nd", 3: "3rd"}

    def _q_who_title(self, q):
        """'Who was the first president of the United States?'"""
        kb = self.kb
        n, title = q.extra
        num = L._ORDINALS.get(n) if n in L._ORDINALS else int(re.match(r"\d+", n).group())
        forms = {n}
        if num:
            forms |= {self._NTH.get(num, f"{num}th")} | {w for w, i in L._ORDINALS.items()
                                                         if i == num}
        # the head noun, then the rest of the title with a few words allowed between
        # ('the first man to set foot on the Moon', 'first president of independent Kenya')
        head, _, rest = title.partition(" ")
        rx = rf"\b(?:{'|'.join(map(re.escape, forms))}) {re.escape(head)}\b"
        if rest:
            words = [w for w in rest.split() if w != "the"]
            rx += r"(?:\s+\S+){0,4}?" + "".join(rf"\s+(?:the\s+)?(?:[a-z]+\s+)?{re.escape(w)}"
                                             if i == len(words) - 1 and len(words) > 1
                                             else rf"\s+(?:the\s+)?{re.escape(w)}"
                                             for i, w in enumerate(words)) + r"\b"
        rx = re.compile(rx, re.I)
        hits = [(m.start(), -kb.freq.get(k, 0), k) for k, d in kb.definitions.items()
                if k in kb.display and (m := rx.search(d)) is not None]
        for _, _, k in sorted(hits):
            if self._person(k):
                self._record(Answer(Truth(1.0, 0.8), [Fact(k, "defined_as", kb.definitions[k],
                                                           Truth(1.0, 0.8))], "definition"))
                return f"{self._full_name(k)}: {kb.definitions[k]}."
        return "I don't know."

    def _q_where(self, q):
        if self.world.knows(q.s) and (self._dynamic(q.s) or not self.kb.out(q.s, "in")):
            p, sup = self.world.where(q.s)
            if isinstance(p, str):
                self._record(Answer(Truth(1, 0.9), [], "episodic"))
                return f"{self._cap(self._ref(q.s))} is in {self._ref(p)}."
            if isinstance(p, tuple) and p and p[0] != "not":
                return f"Either in {self._ref(p[0])} or in {self._ref(p[1])}."
        for k in self._senses_of(q.s):
            if self.kb.out(k, "in") or self.kb.out(k, "part_of"):
                q.s = k
                break
        subj = self.kb.name(q.s, True) if self.kb.generic(q.s) else self.kb.ref(q.s)
        be = "are" if self.kb.generic(q.s) else "is"
        ans = self.kb.objects(q.s, "in") or self.kb.objects(q.s, "part_of")
        if ans:
            ans.sort(key=lambda a: a.how != "direct")
            a = self._record(ans[0])
            places = [self.kb.ref(x.value) for x in ans[:2]]
            return f"{self._cap(subj)} {be} in " + " and ".join(places) + "."
        for rel in ("on", "near", "above", "below"):
            ans = self.kb.objects(q.s, rel)
            if ans:
                self._record(ans[0])
                return f"{self._cap(subj)} {be} {rel} {self.kb.ref(ans[0].value)}."
        return "I don't know."

    def _ref(self, x):
        """A reference for something that happened in a story: the speaker's
        own name for it first ('John' is not WordNet's 'john')."""
        return self._spoken.get(x) or self.kb.ref(x)

    def _q_where_before(self, q):
        p = self.world.where_before(q.s, q.o)
        return f"{self._cap(self._ref(p))}." if p else "I don't know."

    def _q_carrying(self, q):
        objs = self.world.carrying(q.s)
        return (self._cap(self._join([self._ref(o) for o in objs])) + ".") if objs else "Nothing."

    def _q_count_carrying(self, q):
        n = len(self.world.carrying(q.s))
        return {0: "None."}.get(n, f"{L.NUMBER_WORDS.get(n, n)}.").capitalize()

    def _q_who_has(self, q):
        h = self.world.current_holder(q.o)
        return f"{self._cap(self._ref(h))}." if h else "Nobody, as far as I know."

    def _q_who_gave(self, q):
        g = self.world.who_gave(q.o, q.extra)
        return f"{self._cap(self._ref(g))}." if g else "I don't know."

    def _q_what_gave(self, q):
        o = self.world.what_gave(q.s, q.extra)
        return f"{self._cap(self._ref(o))}." if o else "I don't know."

    def _q_who_received(self, q):
        r = self.world.who_received(q.o, q.s)
        return f"{self._cap(self._ref(r))}." if r else "I don't know."

    def _q_why_go(self, q):
        st = self.world.why_went(q.s, q.o)
        return f"{self._cap(self._ref(q.s))} was {st}." if st else "I don't know."

    def _q_will_go(self, q):
        place, st = self.world.will_go(q.s)
        if place is None:
            return "I don't know."
        return (f"Probably to {self._ref(place)}: when someone is {st} they tend to go "
                f"there.")

    def _q_path(self, q):
        p = self.world.path(q.s, q.o)
        if p is None:
            return "I don't know the way."
        if not p:
            return "You are already there."
        return "Go " + ", then ".join(p) + "."

    def _q_why(self, q):
        if self.last is None:
            return "Why what?"
        a = self.last
        if a.how == "language model":
            name, n = a.stmt
            shown = f" I showed it {n} facts from my memory." if n else ""
            return (f"That answer came from the language model {name}, not from my memory, "
                    f"so I cannot give reasons for it.{shown}")
        if a.how == "episodic":
            return "That is what I last saw happen."
        if a.how == "analogy":
            return "I mapped one record onto the other in hyperdimensional space."
        if a.how == "mental map":
            return "I placed the objects on a mental map and compared their positions."
        txt = self.kb.explain(a)
        if "induction" in a.how:
            return (f"{txt} That is a generalisation from examples, so I am only "
                    f"{a.truth.c:.0%} confident.")
        return f"{txt} (confidence {a.truth.c:.2f})"

    def _q_similar(self, q):
        if self.semantic is None or q.s not in self.semantic:
            return "I haven't read enough to know that word."
        return self._cap(", ".join(w for w, _ in self.semantic.similar(q.s, 8))) + "."

    def _q_analogy(self, q):
        if self.semantic is None:
            return "I need a semantic memory for analogies."
        res = self.semantic.analogy(q.s, q.r, q.o, 3)
        if not res:
            return "I don't know those words."
        return f"{self._cap(res[0][0])} (also: {', '.join(w for w, _ in res[1:])})."

    # ------------------------------------------------------------------ commands
    def _command(self, c):
        if c.kind == "generate":
            if self.cortex is None:
                return "I have no cortex to imagine with."
            return c.text + self.cortex.generate(c.text, n=240, temperature=0.6)
        if c.kind == "stats":
            n = sum(1 for f in self.kb.facts.values() if f.source != "innate")
            mem = self.cortex.memory_bytes / 2 ** 20 if self.cortex is not None else 0
            return (f"I know {n} facts about {len(self.kb.concepts())} concepts, remember "
                    f"{len(self.world.events)} events, and my cortex uses {mem:.0f} MB.")
        return ""

    # ------------------------------------------------------------------ persistence
    _LEXICAL = ("definitions", "senses", "aliases", "proper", "words", "surface", "primed",
                "mass", "stuff", "taxa", "loaded", "freq")

    def save(self, directory):
        """Save the whole mind (cortex, semantic memory, facts, episodes)."""
        import os
        import pickle
        os.makedirs(directory, exist_ok=True)
        if self.cortex is not None:
            self.cortex.save(os.path.join(directory, "cortex.bin"))
        if self.semantic is not None:
            self.semantic.save(os.path.join(directory, "semantic.npz"))
        kb = self.kb
        state = {"kb": (kb.facts, kb.display, kb.categories, kb.clock), "spoken": self._spoken,
                 "lexical": {k: getattr(kb, k) for k in self._LEXICAL},
                 "keep_plural": L.KEEP_PLURAL, "world": self.world, "wm": self.wm}
        with open(os.path.join(directory, "mind.pkl"), "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, directory, learn_language=True):
        """Load a saved mind. Only load directories you created yourself
        (the episodic state is stored with pickle)."""
        import os
        import pickle
        from collections import defaultdict
        from .cortex import Cortex
        from .semantic import Semantic
        cp, sp, mp = (os.path.join(directory, n) for n in ("cortex.bin", "semantic.npz",
                                                            "mind.pkl"))
        cortex = Cortex.load(cp) if os.path.exists(cp) else None
        semantic = Semantic.load(sp) if os.path.exists(sp) else None
        m = cls(cortex=cortex, semantic=semantic, innate=not os.path.exists(mp),
                learn_language=learn_language)
        if os.path.exists(mp):
            with open(mp, "rb") as f:
                st = pickle.load(f)
            facts, display, cats, clock = st["kb"]
            m.kb.facts, m.kb.display, m.kb.categories, m.kb.clock = facts, display, cats, clock
            m.kb.by_s, m.kb.by_o = defaultdict(set), defaultdict(set)
            for k in facts:
                m.kb.by_s[k[0]].add(k)
                m.kb.by_o[k[2]].add(k)
            for k, v in st.get("lexical", {}).items():
                setattr(m.kb, k, v)
            L.KEEP_PLURAL |= st.get("keep_plural", set())
            m.world, m.wm = st["world"], st["wm"]
            m._spoken = st.get("spoken", {})
        return m

    def _fallback(self, sent):
        if self.cortex is not None and sent.strip().endswith("?"):
            guess = self.cortex.generate(sent.strip() + " ", n=120, temperature=0.5, stop="\n")
            return f"I don't understand that yet. (My intuition continues: \"{guess.strip()}\")"
        return "I don't understand that yet."
