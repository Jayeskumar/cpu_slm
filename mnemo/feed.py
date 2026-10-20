"""Feeding general knowledge into the mind.

Four doors in:

* ``load_statements``  plain English, one statement per line ("Paris is the
                       capital of France.") -- parsed by the same reader the
                       mind uses in conversation;
* ``load_triples``     structured facts: TSV ``subject  relation  object
                       [confidence]`` or JSON lines ``{"s":..,"r":..,"o":..,
                       "f":1.0,"c":0.9}``;
* ``load_wordnet``     Princeton WordNet 3.0 (downloaded by scripts/get_data.sh):
                       the noun taxonomy (is-a), part-whole and location links,
                       substances, adjective opposites, definitions, and
                       capitals / currencies read from definitions;
* ``load_names``       first-name lists for pronoun gender.

Imported facts carry their source; curated sources (WordNet, the bundled
packs) get high confidence and, for taxonomies, a closed-world reading of
sibling classes ("a whale is not a fish").
"""
import json
import os
import re

from . import language as L

PACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")


# ---------------------------------------------------------------- statements

FUNCTIONAL = {"capital_of", "currency_of"}   # newer statements replace imported ones


def load_statements(mind, lines, source="pack", conf=0.9):
    """Teach plain-English statements in bulk (no per-fact chatter).
    Returns (facts added, sentences not understood)."""
    kb = mind.kb
    added, failed = 0, []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sent in L.split_sentences(line):
            names = mind._remember_names(sent)
            items = L.parse_sentence(sent, mind.wm, names).items
            facts = [mind.canon_fact(it) for it in items if isinstance(it, L.Fact)]
            if not facts:
                failed.append(sent)
                continue
            for f in facts:
                if f.r in ("north_of", "south_of", "east_of", "west_of"):
                    mind.world.add_direction(f.s, f.r, f.o)
                if f.r in FUNCTIONAL:
                    for g in kb.inn(f.o, f.r):
                        if g.source == "wordnet" and g.s != f.s:
                            kb.forget(g.s, g.r, g.o)
                            kb.forget(g.s, "in", g.o)
                kb.tell(f.s, f.r, f.o, f.positive, conf, f.qty, source, generic=f.generic)
                if f.generic:
                    kb.display.pop(f.s, None)
                if f.r == "capital_of":
                    kb.tell(f.s, "in", f.o, True, conf, None, source)
                added += 1
    return added, failed


def load_pack(mind, name="general"):
    path = name if os.path.exists(name) else os.path.join(PACK_DIR, name + ".txt")
    with open(path, encoding="utf-8") as f:
        return load_statements(mind, f, source="pack")


# ---------------------------------------------------------------- triples

def load_triples(kb, path, source="file", conf=0.9):
    """TSV (subject, relation, object[, confidence]) or JSON lines."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                d = json.loads(line)
                s, r, o = d["s"], d["r"], d["o"]
                pos = float(d.get("f", 1.0)) >= 0.5
                c = float(d.get("c", conf))
            else:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                s, r, o = parts[:3]
                c = float(parts[3]) if len(parts) > 3 and parts[3] else conf
                pos = not r.startswith("not ")
                r = r[4:] if not pos else r
            s, o = s.strip(), o.strip()
            key_s, key_o = L.noun(s) if " " in s or s.islower() else s.lower(), o.lower()
            if s[:1].isupper():
                kb.display[s.lower()] = s
                key_s = s.lower()
            if o[:1].isupper():
                kb.display[o.lower()] = o
            kb.tell(key_s, r.strip().replace(" ", "_"), key_o, pos, c, None, source)
            n += 1
    return n


# ---------------------------------------------------------------- WordNet

def _wn_name(word):
    return word.replace("_", " ")


def _parse_data(path):
    """synset offset -> (lexfile, [words], [(symbol, offset, pos, src, dst)], gloss)"""
    syn = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("  "):
                continue  # license header
            head, _, gloss = line.partition(" | ")
            t = head.split()
            off, lexfile = t[0], int(t[1])
            wc = int(t[3], 16)
            words = [t[4 + 2 * i] for i in range(wc)]
            k = 4 + 2 * wc
            pc = int(t[k])
            ptrs = []
            for j in range(pc):
                sym, poff, pos, st = t[k + 1 + 4 * j: k + 5 + 4 * j]
                ptrs.append((sym, poff, pos, int(st[:2], 16), int(st[2:], 16)))
            syn[off] = (lexfile, words, ptrs, gloss.strip())
    return syn


def _parse_index(path):
    idx = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("  "):
                continue
            t = line.split()
            lemma, pc = t[0], int(t[3])
            sense_cnt = int(t[4 + pc])
            idx[lemma] = t[6 + pc: 6 + pc + sense_cnt]
    return idx


def _clean_gloss(g):
    g = re.sub(r';?\s*"[^"]*"', "", g)   # drop usage examples
    return g.strip().rstrip(";").strip()


_PLACE = r"(?:the )?([A-Z][\w'.-]*(?: (?:of |and |the )*[A-Z][\w'.-]*)*)"
# only at the start of a clause, so 'a former capital of China' and 'the crime
# capital of Italy' do not count
_CAPITAL = re.compile(r"(?:^|; )(?:the )?capital (?:and (?:the )?largest city )?of " + _PLACE)
_CURRENCY = re.compile(r"(?:^|; )(?:the )?(?:basic )?(?:monetary unit|unit of money|unit of "
                       r"currency) (?:in|of) " + _PLACE)


# WordNet lexicographer files (lexnames)
_ANIMAL, _FOOD, _LOCATION, _PERSON, _PLANT, _SUBSTANCE = 5, 13, 15, 18, 20, 27


def load_wordnet(kb, wn_dir, conf=0.99, definitions=True, max_senses=8):
    """Import WordNet 3.0 nouns.  Every synset becomes one concept, keyed by
    its first word when it is that word's dominant sense ('dog'), else
    'word#k' ('gold#3', the element).  Words know their senses; synonyms and
    capitalised uses ('China' the country, 'china' the porcelain) map to the
    right concept through kb.aliases / kb.proper.  Returns counts."""
    noun = _parse_data(os.path.join(wn_dir, "data.noun"))
    index = _parse_index(os.path.join(wn_dir, "index.noun"))
    first_names = L.FEMALE | L.MALE

    def surface(off, lemma):
        words = noun[off][1]
        if lemma in words:              # ['Earth', 'earth', 'world']: a common noun too
            return lemma
        return next((w for w in words if w.lower() == lemma), words[0])

    def former(o):      # 'Russia' today is the federation, not 'a former communist country'
        return re.match(r"(?:an? |the )?former\b|formerly\b", noun[o][3]) is not None

    common, proper = {}, {}
    for lemma, offs in index.items():
        offs = [o for o in offs if o in noun]
        offs = [o for o in offs if not former(o)] + [o for o in offs if former(o)]
        index[lemma] = offs
        low = [o for o in offs if not surface(o, lemma)[:1].isupper()]
        # 'China' (title case) can be a name; 'SALT' (an acronym) is not 'Salt'
        up = [o for o in offs if o not in low and not
              (surface(o, lemma).isupper() and len(lemma) > 1)]
        if low:
            d = low[0]
            if noun[d][0] in (_PERSON, _FOOD):     # 'whale', 'chicken': the animal
                d = next((o for o in low[1:3] if noun[o][0] == _ANIMAL), d)
            common[lemma] = d
        # a story's Mary is not the Virgin Mary: first names get no default person
        if up and not (lemma in first_names and noun[up[0]][0] == _PERSON):
            proper[lemma] = up[0]

    key = {}
    for lemma in index:
        d = common.get(lemma, proper.get(lemma))
        if d is not None and noun[d][1][0].lower() == lemma:
            key[d] = _wn_name(lemma)
    for off, (_, words, _, _) in noun.items():
        if off not in key:
            first = words[0].lower()
            offs = index.get(first, [])
            k = offs.index(off) + 1 if off in offs else len(offs) + 1
            key[off] = f"{_wn_name(first)}#{k}"

    def place_key(name):
        """'Japan' in a definition is the country, not the islands: prefer a
        sense that is a country or state, then any location."""
        low = name.lower().replace(" ", "_")
        offs = index.get(low, [])
        polities = {"country", "country#2", "state#4", "state", "nation", "political unit"}
        for o in offs:
            if any(m in polities for m, _ in kb.categories_of(key[o], 6)):
                return key[o]
        for o in offs:
            if noun[o][0] == _LOCATION:
                return key[o]
        d = proper.get(low, common.get(low))
        return key[d] if d is not None else _wn_name(low)

    # how often each sense was tagged in the SemCor corpus: everyday familiarity
    tags, lemma_tags = {}, {}
    sense_path = os.path.join(wn_dir, "index.sense")
    if os.path.exists(sense_path):
        with open(sense_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                t = line.split()
                if len(t) == 4 and "%1:" in t[0] and t[3] != "0":
                    tags[t[1]] = tags.get(t[1], 0) + int(t[3])
                    lemma_tags[(t[1], t[0].split("%")[0])] = int(t[3])
    for off, n in tags.items():
        if off in key:
            kb.freq[key[off]] = n
    # the everyday word for a concept: its most used synonym that means this
    # concept first ('metal', not 'metallic element'; 'human being', not 'homo')
    for off, (_, words, _, _) in noun.items():
        if len(words) < 2:
            continue
        own = [w.lower() for w in words
               if common.get(w.lower(), proper.get(w.lower())) == off]
        best = max(own, key=lambda w: lemma_tags.get((off, w), 0), default=None)
        first = words[0].lower()
        # only where the headword itself is hardly ever used: a speaker who says
        # 'hallway' must hear 'hallway', not WordNet's more frequent 'hall'
        if best and best != first and lemma_tags.get((off, first), 0) == 0 and \
                lemma_tags.get((off, best), 0) > 0:
            kb.surface.setdefault(key[off], _wn_name(best))
    gloss_facts = []   # resolved once the taxonomy is in
    stats = {"concepts": len(noun), "isa": 0, "in": 0, "part_of": 0, "has": 0, "made_of": 0,
             "definitions": 0, "capital_of": 0, "currency_of": 0, "opposite": 0, "aliases": 0}
    for off, (lexfile, words, ptrs, gloss) in noun.items():
        name = key[off]
        if words[0][:1].isupper():
            kb.display.setdefault(name, _wn_name(words[0]))
        if lexfile in (_ANIMAL, _PLANT):
            kb.taxa.add(name)     # Linnaean taxa: sibling classes really are disjoint
        if lexfile == _SUBSTANCE and " " not in kb.bare(name):
            kb.mass.add(name)     # 'gold', 'water' (but 'noble metals' are countable)
        if lexfile in (_SUBSTANCE, _FOOD):
            kb.stuff.add(name)
        others = [_wn_name(w) for w in words if w.lower() != kb.bare(name).replace(" ", "_")]
        if others:
            kb.words[name] = others[:4]
        for sym, poff, pos, _, _ in ptrs:
            if pos != "n" or poff not in noun:
                continue
            target = key[poff]
            if sym in ("@", "@i"):
                kb.tell(name, "isa", target, True, conf, None, "wordnet")
                stats["isa"] += 1
            elif sym == "#p":
                rel = "in" if lexfile == _LOCATION else "part_of"
                kb.tell(name, rel, target, True, conf, None, "wordnet")
                stats[rel] += 1
            elif sym == "%p":
                kb.tell(name, "has", target, True, conf * 0.85, -1, "wordnet")
                stats["has"] += 1
            elif sym == "%s":
                kb.tell(name, "made_of", target, True, conf * 0.85, None, "wordnet")
                stats["made_of"] += 1
        g = _clean_gloss(gloss)
        if definitions and g:
            kb.definitions.setdefault(name, g)
            stats["definitions"] += 1
        leaf = not any(p[0] in ("~", "~i") for p in ptrs)   # 'yen', not 'Japanese monetary unit'
        for rx, rel in ((_CAPITAL, "capital_of"), (_CURRENCY, "currency_of")):
            m = rx.search(g)
            if m and leaf:
                gloss_facts.append((name, rel, m.group(1)))
    for name, rel, place in gloss_facts:
        place = place_key(place)
        kb.tell(name, rel, place, True, conf * 0.9, None, "wordnet")
        stats[rel] += 1
        if rel == "capital_of":
            kb.tell(name, "in", place, True, conf * 0.9, None, "wordnet")
    for lemma, offs in index.items():
        word = _wn_name(lemma)
        d = common.get(lemma, proper.get(lemma))
        if d is None:
            if offs:
                kb.senses[word] = [key[o] for o in offs[:max_senses]]
            continue
        dk = key[d]
        kb.senses[word] = [dk] + [key[o] for o in offs[:max_senses] if key[o] != dk]
        if dk != word:
            kb.aliases[word] = dk
            stats["aliases"] += 1
        if lemma in proper and key[proper[lemma]] != dk:
            kb.proper[word] = key[proper[lemma]]
        # one-word names that look plural ('Honduras', 'the Netherlands') stay whole
        if word.endswith("s") and L.noun(word) != word and \
                L.noun(word).replace(" ", "_") not in index:
            L.KEEP_PLURAL.add(word)
    # adjective opposites (lexical antonym pointers)
    adj = _parse_data(os.path.join(wn_dir, "data.adj"))
    for off, (_, words, ptrs, _) in adj.items():
        for sym, poff, pos, src, dst in ptrs:
            if sym != "!" or poff not in adj or not src or not dst:
                continue
            a = re.sub(r"\(.*\)$", "", words[src - 1]).replace("_", " ").lower()
            b = re.sub(r"\(.*\)$", "", adj[poff][1][dst - 1]).replace("_", " ").lower()
            kb.tell(a, "opposite", b, True, conf, None, "wordnet")
            stats["opposite"] += 1
    kb.categories -= {"entity", "physical entity", "abstraction", "object", "whole"}
    return stats


def load_names(names_dir):
    """Extend the reader's first-name lists (used for 'he'/'she')."""
    n = 0
    for fn, target in (("female.txt", L.FEMALE), ("male.txt", L.MALE)):
        path = os.path.join(names_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    target.add(w)
                    n += 1
    ambiguous = L.FEMALE & L.MALE
    L.FEMALE.difference_update(ambiguous - {"mary", "sandra", "julie"})
    L.MALE.difference_update(ambiguous - {"john", "daniel", "bill", "fred"})
    return n
