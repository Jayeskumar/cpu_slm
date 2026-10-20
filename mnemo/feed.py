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
_ACT, _ANIMAL, _EVENT, _FOOD, _LOCATION, _PERSON, _PLANT, _SUBSTANCE = 4, 5, 11, 13, 15, 18, 20, 27


def load_wordnet(kb, wn_dir, conf=0.99, definitions=True, max_senses=8, read=True):
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
    if read:
        stats.update(_read_glosses(kb, noun, key))
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


# ---------------------------------------------------------------- world data

def _concept(kb, name, kinds=()):
    """The concept a proper name refers to, preferring a sense of the given
    kinds ('Georgia' the country, not the state, when kinds=['country'])."""
    low = name.lower()
    first = kb.canon(low, proper=True)
    if kinds:
        for k in [first] + kb.other_senses(low, first)[:6]:
            for kind in kinds:
                a = kb.ask(k, "isa", kind)
                if a.known and a.truth.f >= 0.5:
                    return k
    return first


def _named(kb, key, name):
    if key not in kb.display and key not in kb.categories:
        kb.display[key] = name


_CONTINENT = {"Africa": "africa", "Asia": "asia", "Europe": "europe", "Oceania": "oceania",
              "Antarctic": "antarctica", "South America": "south america",
              "North America": "north america", "Central America": "north america",
              "Caribbean": "north america"}


def load_countries(mind, path, conf=0.9, source="world"):
    """Countries from mledoze/countries: borders, area, landlocked, region,
    languages, currencies and demonyms.  Returns (count, name -> concept,
    demonym -> concept)."""
    kb = mind.kb
    data = json.load(open(path, encoding="utf-8"))
    kinds = [kb.canon("country"), "country#2", "state#4", "country"]
    country = kb.canon("country")
    colony = next((k for k in kb.senses.get("dependency", [])
                   if "politically controlled" in kb.definitions.get(k, "")), None)
    by_code, names, demonyms = {}, {}, {}
    for c in data:
        name = c["name"]["common"]
        key = _concept(kb, name, kinds)
        _named(kb, key, name)
        by_code[c["cca3"]] = key
        for alias in [name, c["name"]["official"]] + list(c.get("altSpellings", [])):
            names.setdefault(alias.lower(), key)
        # the list of countries: the 195 independent ones; Reunion is a dependency
        if c.get("independent") is not False:
            kb.tell(key, "isa", country, True, conf, None, source)
        elif colony and not kb.ask(key, "isa", country).known:
            kb.tell(key, "isa", colony, True, conf, None, source)
    n = 0
    for c in data:
        key = by_code[c["cca3"]]
        n += 1
        for b in c.get("borders", []):
            if b in by_code:
                kb.tell(key, "border", by_code[b], True, conf, None, source)
        if (c.get("area") or 0) > 0:          # -1 means unknown
            kb.tell(key, "area", str(int(c["area"])) if c["area"] >= 1 else str(c["area"]),
                    True, conf, None, source)
        kb.tell(key, "is", "landlocked", bool(c.get("landlocked")), conf, None, source)
        region = c.get("subregion") if c.get("region") == "Americas" else c.get("region")
        cont = _CONTINENT.get(region or "") or _CONTINENT.get(c.get("subregion") or "")
        if cont and not kb.out(key, "in"):
            kb.tell(key, "in", kb.canon(cont, proper=True), True, conf, None, source)
        sub = c.get("subregion")
        if sub and sub not in _CONTINENT:
            skey = sub.lower()
            _named(kb, skey, sub)
            kb.tell(key, "in", skey, True, conf, None, source)
            if cont and not kb.out(skey, "in"):         # Northern Africa is in Africa
                kb.tell(skey, "in", kb.canon(cont, proper=True), True, conf, None, source)
        if not kb.inn(key, "capital_of"):
            for cap in c.get("capital", [])[:1]:
                ck = _concept(kb, cap)
                _named(kb, ck, cap)
                kb.tell(ck, "capital_of", key, True, conf, None, source)
                kb.tell(ck, "in", key, True, conf, None, source)
        for lang in (c.get("languages") or {}).values():
            lk = kb.canon(lang.lower(), proper=True)
            _named(kb, lk, lang)
            kb.tell(lk, "language_of", key, True, conf, None, source)
        if not kb.inn(key, "currency_of"):
            for cur in (c.get("currencies") or {}).values():
                cur_key = kb.canon(cur["name"].lower())
                kb.tell(cur_key, "currency_of", key, True, conf, None, source)
        dem = ((c.get("demonyms") or {}).get("eng") or {}).get("m")
        if dem:
            dk = dem.lower()
            demonyms.setdefault(dem, key)
            kb.tell(dk, "demonym_of", key, True, conf, None, source)
            kb.display.setdefault(dk, dem)
    return n, names, demonyms


def load_cities(mind, path, countries, conf=0.8, source="world"):
    """Cities from datasets/world-cities (GeoNames): each city is in its
    region and country.  Cities the dictionary already places are left as
    they are."""
    import csv
    kb = mind.kb
    n = 0
    with open(path, encoding="utf-8") as f:
        rows = csv.reader(f)
        next(rows, None)
        for name, cname, sub, _ in rows:
            ckey = countries.get(cname.lower())
            if not ckey or not name:
                continue
            key = kb.canon(name.lower(), proper=True)
            if kb.senses.get(name.lower()):      # Darwin the town, not Darwin the naturalist
                key = _concept(kb, name, _TOWNS)
            if key == ckey or any(g.source != source for g in kb.out(key, "in")) or \
                    not _town(kb, key):
                continue
            gloss = kb.definitions.get(key)
            if gloss and kb.display.get(ckey, cname) not in gloss and cname not in gloss:
                continue        # the dictionary's Madrid is the one in Spain, not in Colombia
            _named(kb, key, name)
            kb.tell(key, "in", ckey, True, conf, None, source)
            if sub and sub.lower() != name.lower():
                skey = _region(kb, sub, ckey)
                if skey is not None:
                    _named(kb, skey, sub)
                    kb.tell(key, "in", skey, True, conf, None, source)
                    if not kb.out(skey, "in"):
                        kb.tell(skey, "in", ckey, True, conf, None, source)
            n += 1
    return n


_TOWNS = ("municipality", "port", "urban area")


def _town(kb, key):
    """May a city of this name be the concept key?  Only if key is a new name
    or already a town: the town of Asia in the Philippines is not the
    continent, and Lincoln, Nebraska is not the president."""
    if not kb.out(key, "isa"):
        return True
    return any(kb.ask(key, "isa", t).known and kb.ask(key, "isa", t).truth.f >= 0.5
               for t in _TOWNS)


def _region(kb, name, country):
    """The region of a country a name refers to (Georgia the US state, for a
    town in the United States), or None if the name means something else."""
    low = name.lower()
    first = kb.canon(low, proper=True)
    for k in [first] + kb.other_senses(low, first)[:6]:
        if k != country and country in kb.reach(k, "in"):
            return k
    if first == country or kb.out(first, "isa") or kb.out(first, "in"):
        return None
    return first


_ELEMENT_KINDS = {"noble gas": "noble gas", "alkali metal": "alkali metal",
                  "alkaline earth metal": "alkaline earth", "metalloid": "metalloid",
                  "lanthanide": "lanthanide", "actinide": "actinide",
                  "transition metal": "transition metal"}


def load_elements(mind, path, conf=0.95, source="world"):
    """The periodic table (Bowserinator/Periodic-Table-JSON): symbol, atomic
    number, kind, state at room temperature, discoverer."""
    kb = mind.kb
    data = json.load(open(path, encoding="utf-8"))["elements"]
    element = kb.canon("chemical element")
    states = {"solid": kb.canon("solid"), "liquid": kb.canon("liquid"), "gas": kb.canon("gas")}
    metal = kb.canon("metal")
    n = 0
    for e in data:
        key = _concept(kb, e["name"], [element])
        if kb.bare(key) != e["name"].lower():
            key = kb.canon(e["name"].lower())
        _named(kb, key, e["name"]) if key not in kb.definitions else None
        if not (kb.ask(key, "isa", element).known and kb.ask(key, "isa", element).truth.f >= 0.5):
            kb.tell(key, "isa", element, True, conf, None, source)
        kb.tell(key, "symbol", e["symbol"], True, conf, None, source)
        kb.tell(key, "atomic_number", str(e["number"]), True, conf, None, source)
        phase = (e.get("phase") or "").lower()
        if phase in states:
            kb.tell(key, "isa", states[phase], True, conf, None, source)
        cat = (e.get("category") or "").lower()
        for name, kind in _ELEMENT_KINDS.items():
            if name in cat:
                kb.tell(key, "isa", kb.canon(kind), True, conf, None, source)
        if "unknown" in cat:
            pass                    # superheavy elements: 'unknown, probably a transition metal'
        elif ("metal" in cat and "nonmetal" not in cat and "metalloid" not in cat) or \
                cat in ("lanthanide", "actinide"):
            kb.tell(key, "isa", metal, True, conf, None, source)
        elif "nonmetal" in cat or "noble gas" in cat:
            kb.tell(key, "isa", kb.canon("nonmetal"), True, conf, None, source)
            kb.tell(key, "isa", metal, False, conf, None, source)
        who = (e.get("discovered_by") or "").strip()
        if re.fullmatch(r"[A-Z][a-z]+(?:[ -](?:[A-Z][a-z.]*|de|von|van|da))+", who) and \
                not re.search(r"\b(?:Middle|Ancient|Unknown|East|Prehistoric)\b", who):
            kb.tell(key, "discovered_by", who, True, conf * 0.9, None, source)
        n += 1
    return n


# ---------------------------------------------------------------- reading the dictionary

DEMONYMS = {  # nationality words as WordNet writes them -> country (more come from data)
    "English": "united kingdom", "Scottish": "united kingdom", "Welsh": "united kingdom",
    "British": "united kingdom", "Irish": "ireland", "United States": "united states",
    "American": "united states", "French": "france", "German": "germany", "Italian": "italy",
    "Spanish": "spain", "Portuguese": "portugal", "Dutch": "netherlands", "Flemish": "belgium",
    "Belgian": "belgium", "Austrian": "austria", "Swiss": "switzerland", "Russian": "russia",
    "Polish": "poland", "Czech": "czech republic", "Hungarian": "hungary", "Greek": "greece",
    "Athenian": "greece", "Swedish": "sweden", "Norwegian": "norway", "Danish": "denmark",
    "Finnish": "finland", "Indian": "india", "Chinese": "china", "Japanese": "japan",
    "Canadian": "canada", "Mexican": "mexico", "Brazilian": "brazil", "Argentine": "argentina",
    "Australian": "australia", "Egyptian": "egypt", "Persian": "iran", "Israeli": "israel",
    "Turkish": "turkey", "Prussian": "germany", "Soviet": "russia", "Roman": "italy",
    "Venetian": "italy", "Florentine": "italy", "Viennese": "austria", "Scots": "united kingdom",
}

_YEARS = re.compile(r"\((?:[^()]*?[;,] )?(?:circa |c\. |about )?(\d{1,4})( BC)?\s*-\s*"
                    r"(?:circa |c\. )?(\d{1,4})( BC)?\)\s*$")
_BORN_YEAR = re.compile(r"\(born (?:in |circa )?(\d{3,4})\)\s*$")
_BORN_IN = re.compile(r"\bborn in ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)")
_SPAN = re.compile(r"\bfrom (\d{3,4}) (?:to|until) (\d{3,4})\b|[;,] (\d{3,4})-(\d{2,4})\s*$|"
                   r"\((\d{3,4})-(\d{2,4})\)")
_YEAR = re.compile(r"\((\d{3,4})\)|\b(?:in|on) (?:\d{1,2} )?(?:(?:January|February|March|April|"
                   r"May|June|July|August|September|October|November|December) )?(?:\d{1,2},? )?"
                   r"(\d{3,4})\b")
_PLACE_KIND = (r"(?:city|town|port|seaport|village|region|province|state|county|island|river|"
               r"lake|mountain|mountains|range|desert|valley|resort|district|capital)")
_IN_PLACE = re.compile(r"^(?:an? |the )?(?:[\w-]+ ){0,4}?" + _PLACE_KIND + r"\b[^;]*?\b(?:in|of) "
                       r"(?:the )?(?:(?:north|south|east|west|northern|southern|eastern|western|"
                       r"central|north-central|south-central|east-central|west-central|"
                       r"northeastern|northwestern|southeastern|southwestern) ){0,2}"
                       r"([A-Z][\w'.-]*(?: [A-Z][\w'.-]*)*)")
_USED = re.compile(r"\bused (to|for|as|in) ([^;(]+)")
_FOR_ING = re.compile(r"\b(?:instrument|device|tool|appliance|machine|implement|utensil|"
                      r"apparatus|container|vessel|equipment)\b[^;]*?\bfor (\w+ing\b[^;(]*)")
# the genus and its purpose: 'a regulator for automatically regulating temperature'
_GENUS_FOR = re.compile(r"(?:an? |the )?(?:[\w-]+ ){0,3}?[\w-]+ for ((?:\w+ly )?\w+ing\b[^;(]*)")
_MADE = re.compile(r"\bmade (?:of|from|with) (?:a |an |the )?([a-z][a-z-]*(?: [a-z-]+)?)")
_FEEDS = re.compile(r"\bfeeds? (?:chiefly |mainly |mostly |primarily |largely )?on "
                    r"([a-z][a-z-]*(?: [a-z-]+)?)")
_PROPS = ("flightless nocturnal diurnal carnivorous herbivorous omnivorous insectivorous venomous "
          "poisonous edible aquatic marine arboreal migratory extinct domesticated burrowing "
          "endangered freshwater").split()


def _short_np(text):
    words = []
    for w in text.split():
        if w in ("and", "or", "that", "which", "who", "with", "to", "of", "in", "by", "for",
                 "especially", "usually", "often", "etc"):
            break
        words.append(w.strip(".,"))
    return " ".join(words[:3])


def _read_glosses(kb, noun, concept, conf=0.8):
    """Turn WordNet definitions into facts: people's years and nationality,
    the years of events, where places and creatures are, what things are
    used for and made of, what animals feed on, and typical properties.
    concept: synset offset -> concept key."""
    dem = DEMONYMS

    def place(name):
        low = name.lower()
        if name in dem:
            return kb.canon(dem[name], proper=True)
        k = kb.canon(low, proper=True)
        return k if (k in kb.definitions or kb.out(k, "in") or kb.inn(k, "in")) else None

    stats = dict.fromkeys(("born", "died", "nationality", "event years", "in", "used_for",
                           "made_of", "eat", "is"), 0)
    for off, (lexfile, words, ptrs, gloss) in noun.items():
        k = concept.get(off)
        if not k:
            continue
        g = _clean_gloss(gloss)
        person = lexfile == _PERSON and any(p[0] == "@i" for p in ptrs)
        if person:
            m = _YEARS.search(g)
            if m:
                y1, bc1, y2, bc2 = m.groups()
                if int(y2) < 100 <= int(y1) and not (bc1 or bc2):
                    y2 = str(int(y1) // 100 * 100 + int(y2))
                bc = " BC" if (bc1 or bc2) else ""
                kb.tell(k, "born", y1 + (bc1 or bc), True, conf, None, "wordnet")
                kb.tell(k, "died", y2 + (bc2 or ""), True, conf, None, "wordnet")
                stats["born"] += 1
                stats["died"] += 1
            else:
                m = _BORN_YEAR.search(g)
                if m:
                    kb.tell(k, "born", m.group(1), True, conf, None, "wordnet")
                    stats["born"] += 1
            m = re.match(r"^(?:(?:an?|the|ancient|prolific|influential|celebrated|famous|great|"
                         r"leading|eminent|popular|noted|and) )*([A-Z][\w-]*(?: [A-Z][\w-]*)*)", g)
            if m and m.group(1) in dem:
                nat = m.group(1)
                kb.tell(k, "nationality", nat, True, conf, None, "wordnet")
                kb.tell(k, "is", nat.lower(), True, conf, None, "wordnet")
                ck = place(nat)
                if ck:
                    kb.tell(k, "from", ck, True, conf, None, "wordnet")
                stats["nationality"] += 1
            if not kb.out(k, "from"):        # '1st President of the United States'
                m = re.search(r"\b(?:[Pp]resident|[Kk]ing|[Qq]ueen|[Ee]mperor|[Pp]rime [Mm]inister|"
                              r"[Cc]hancellor|[Ss]ultan|[Tt]sar|[Pp]haraoh) of (?:the )?"
                              r"([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)", g)
                if m and place(m.group(1)):
                    kb.tell(k, "from", place(m.group(1)), True, conf, None, "wordnet")
            m = _BORN_IN.search(g)
            if m and place(m.group(1)):
                kb.tell(k, "born_in", place(m.group(1)), True, conf, None, "wordnet")
            continue
        m = _SPAN.search(g)
        if m and lexfile != _LOCATION:
            a, b = next((m.group(i), m.group(i + 1)) for i in (1, 3, 5) if m.group(i))
            if len(b) == 2:
                b = a[:2] + b
            if int(a) <= int(b):
                kb.tell(k, "began", a, True, conf, None, "wordnet")
                kb.tell(k, "ended", b, True, conf, None, "wordnet")
                stats["event years"] += 1
        elif lexfile in (_ACT, _EVENT) and any(p[0] == "@i" for p in ptrs):
            y = _YEAR.search(g)          # 'the decisive battle ... (1066)', '... in November 1917'
            if y:
                kb.tell(k, "year", y.group(1) or y.group(2), True, conf, None, "wordnet")
                stats["event years"] += 1
        if lexfile == _LOCATION and not kb.out(k, "in"):
            m = _IN_PLACE.match(g)
            if m:
                p = place(m.group(1))
                if p and p != k:
                    kb.tell(k, "in", p, True, conf, None, "wordnet")
                    stats["in"] += 1
        if lexfile == 6:  # artifacts
            m = _USED.search(g)
            if m:
                kb.tell(k, "used_for", f"{m.group(1)} {m.group(2).strip()}", True, conf, None,
                        "wordnet")
                stats["used_for"] += 1
            else:
                m = _FOR_ING.search(g) or _GENUS_FOR.match(g)
                if m:
                    kb.tell(k, "used_for", "for " + m.group(1).strip(), True, conf, None, "wordnet")
                    stats["used_for"] += 1
        if lexfile in (6, _FOOD, _SUBSTANCE):
            m = _MADE.search(g)
            if m:
                np_ = _short_np(m.group(1))
                if np_:
                    kb.tell(k, "made_of", kb.canon(L.noun(np_)), True, conf, None, "wordnet")
                    stats["made_of"] += 1
        if lexfile in (_ANIMAL, _PLANT):
            m = _FEEDS.search(g)
            if m:
                np_ = _short_np(m.group(1))
                if np_:
                    kb.tell(k, "eat", kb.canon(L.noun(np_)), True, conf, -1, "wordnet")
                    stats["eat"] += 1
            first = g.split(";")[0]
            for name in re.findall(r"\b(?:of|in|to|from) (?:the )?(?:[a-z-]+ ){0,3}?(?:of |in )?"
                                   r"(?:most of |parts of )?(?:the )?([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
                                   first) + re.findall(r"\b([A-Z][a-z]+)\b", first):
                p = place(name)
                if p and p != k and (name in dem or kb.ask(p, "isa", kb.canon("country")).known
                                     or kb.ask(p, "isa", kb.canon("continent")).known
                                     or kb.ask(p, "isa", "region").known):
                    kb.tell(k, "in", p, True, conf * 0.9, None, "wordnet")
                    stats["in"] += 1
            for adj in _PROPS:
                if re.search(r"\b" + adj + r"\b", g):
                    kb.tell(k, "is", adj, True, conf, None, "wordnet")
                    stats["is"] += 1
    return stats
