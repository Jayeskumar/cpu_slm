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
        for sent in L.split_sentences(text):
            r = self._sentence(sent)
            if r:
                replies.append(r)
        return " ".join(replies)

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
        social = self._social(sent)
        if social is not None:
            return social
        sent = self._persons(sent)
        names = self._remember_names(sent)
        p = L.parse_sentence(sent, self.wm, names)
        if not p.items:
            return self._fallback(sent)
        out = []
        for it in p.items:
            if isinstance(it, L.Fact):
                out.append(self._tell(it))
            elif isinstance(it, L.Event):
                self.world.observe(it)
                out.append("")
            elif isinstance(it, L.Query):
                out.append(self._answer(it, sent))
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

    def _persons(self, sent):
        """First and second person become the entities 'user' and 'mnemo'."""
        if not re.search(r"\b(i|i'm|me|my|you|your)\b", sent, re.I):
            return sent
        low = sent
        for pat, rep in self._PERSON:
            low = re.sub(pat, rep, low, flags=re.I)
        return low

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
                while j + 1 < len(toks) and (toks[j + 1][0].isupper() or toks[j + 1] == "of"
                                             and j + 2 < len(toks) and toks[j + 2][0].isupper()):
                    j += 1
                name = " ".join(toks[i:j + 1])
                low = name.lower()
                names[low] = name
                key = self.kb.canon(low, proper=True)
                if key not in self.kb.categories:
                    self.kb.display[key] = name       # the speaker's name for it wins
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
                nxt in L.VERBS or nxt in L.INTRANSITIVE or low in self.kb._MASS:
            return False
        if low in self.kb.senses and low not in self.kb.proper and \
                self.kb.canon(low) not in self.kb.display:
            return False      # a common noun ('Gold is a metal')
        if self.semantic is not None and low in self.semantic.index and \
                self.semantic.index[low] < 3000:
            return False
        return True

    # ------------------------------------------------------------------ facts
    _RAW_OBJECT = {"can", "does", "is", "opposite"}   # object is a verb or an adjective

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
        elif f.r == "isa" and self.kb.senses:
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
        a = self._record(self.kb.ask(s, q.r, o))
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
            hd = self._hd_analogy(q.extra, q.o)
            if hd:
                return hd
        return "I don't know."

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
        facts.sort(key=lambda f: (f.source in kb.authority, -kb.activation(f)))
        for f in facts[:5]:
            lines.append(kb.say(f))
        if s in kb.words:
            lines.append("also called " + self._join(kb.words[s][:3]))
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

    def _q_where(self, q):
        if self.world.knows(q.s) and (self._dynamic(q.s) or not self.kb.out(q.s, "in")):
            p, sup = self.world.where(q.s)
            if isinstance(p, str):
                self._record(Answer(Truth(1, 0.9), [], "episodic"))
                return f"{self._cap(self.kb.ref(q.s))} is in {self.kb.ref(p)}."
            if isinstance(p, tuple) and p and p[0] != "not":
                return f"Either in {self.kb.ref(p[0])} or in {self.kb.ref(p[1])}."
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

    def _q_where_before(self, q):
        p = self.world.where_before(q.s, q.o)
        return f"{self._cap(self.kb.ref(p))}." if p else "I don't know."

    def _q_carrying(self, q):
        objs = self.world.carrying(q.s)
        return (self._cap(self._join([self.kb.ref(o) for o in objs])) + ".") if objs else "Nothing."

    def _q_count_carrying(self, q):
        n = len(self.world.carrying(q.s))
        return {0: "None."}.get(n, f"{L.NUMBER_WORDS.get(n, n)}.").capitalize()

    def _q_who_has(self, q):
        h = self.world.current_holder(q.o)
        return f"{self._cap(self.kb.ref(h))}." if h else "Nobody, as far as I know."

    def _q_who_gave(self, q):
        g = self.world.who_gave(q.o, q.extra)
        return f"{self._cap(self.kb.ref(g))}." if g else "I don't know."

    def _q_what_gave(self, q):
        o = self.world.what_gave(q.s, q.extra)
        return f"{self._cap(self.kb.ref(o))}." if o else "I don't know."

    def _q_who_received(self, q):
        r = self.world.who_received(q.o, q.s)
        return f"{self._cap(self.kb.ref(r))}." if r else "I don't know."

    def _q_why_go(self, q):
        st = self.world.why_went(q.s, q.o)
        return f"{self._cap(self.kb.ref(q.s))} was {st}." if st else "I don't know."

    def _q_will_go(self, q):
        place, st = self.world.will_go(q.s)
        if place is None:
            return "I don't know."
        return (f"Probably to {self.kb.ref(place)}: when someone is {st} they tend to go "
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
                "mass", "taxa")

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
        state = {"kb": (kb.facts, kb.display, kb.categories, kb.clock),
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
        return m

    def _fallback(self, sent):
        if self.cortex is not None and sent.strip().endswith("?"):
            guess = self.cortex.generate(sent.strip() + " ", n=120, temperature=0.5, stop="\n")
            return f"I don't understand that yet. (My intuition continues: \"{guess.strip()}\")"
        return "I don't understand that yet."
