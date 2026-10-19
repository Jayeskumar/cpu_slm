"""Feeding general knowledge: plain-English packs, triples, and a WordNet
importer tested on a tiny fixture in WordNet 3.0's file format."""
import json
import os
import tempfile

import conftest  # noqa: F401  (puts the repo on sys.path)
from mnemo import feed
from mnemo.mind import Mind

# offset lexfile pos word-count words... pointer-count pointers... | gloss
DATA_NOUN = """\
00000001 03 n 01 entity 0 000 | that which is perceived or known
00000002 03 n 01 animal 0 001 @ 00000001 n 0000 | a living organism
00000003 05 n 01 vertebrate 0 001 @ 00000002 n 0000 | animals having a bony skeleton
00000004 05 n 01 mammal 0 001 @ 00000003 n 0000 | warm-blooded vertebrates that nourish their young with milk
00000005 05 n 01 bird 0 002 @ 00000003 n 0000 %p 00000010 n 0000 | warm-blooded egg-laying vertebrates
00000006 05 n 01 fish 0 001 @ 00000003 n 0000 | cold-blooded aquatic vertebrates
00000007 05 n 01 whale 0 001 @ 00000004 n 0000 | a large cetacean mammal
00000008 18 n 02 giant 0 whale 0 001 @ 00000009 n 0000 | a very large person
00000009 03 n 01 person 0 001 @ 00000001 n 0000 | a human being
00000010 08 n 01 wing 0 001 #p 00000005 n 0000 | a limb used for flying
00000011 27 n 02 metallic_element 0 metal 0 001 @ 00000001 n 0000 | an element that conducts heat
00000012 21 n 01 gold 0 001 @ 00000001 n 0000 | coins made of gold
00000013 27 n 02 gold 0 Au 0 001 @ 00000011 n 0000 | a soft yellow metallic element
00000014 15 n 01 France 0 001 @i 00000015 n 0000 | a republic in western Europe
00000015 15 n 01 country 0 001 @ 00000001 n 0000 | a politically organized body of people
00000016 15 n 01 Paris 0 002 @i 00000017 n 0000 #p 00000014 n 0000 | the capital and largest city of France
00000017 15 n 01 national_capital 0 001 @ 00000001 n 0000 | the capital city of a nation
00000018 18 n 02 Mary 0 Virgin_Mary 0 001 @i 00000009 n 0000 | the mother of Jesus
00000019 06 n 01 china 0 001 @ 00000001 n 0000 | high quality porcelain
00000020 15 n 01 China 0 001 @i 00000015 n 0000 | a communist nation in eastern Asia
00000021 15 n 01 Beijing 0 002 @i 00000017 n 0000 #p 00000020 n 0000 | capital of China
"""
INDEX_NOUN = """\
entity n 1 0 1 0 00000001
animal n 1 0 1 0 00000002
vertebrate n 1 0 1 0 00000003
mammal n 1 0 1 0 00000004
bird n 1 0 1 0 00000005
fish n 1 0 1 0 00000006
whale n 2 0 2 0 00000008 00000007
giant n 1 0 1 0 00000008
person n 1 0 1 0 00000009
wing n 1 0 1 0 00000010
metallic_element n 1 0 1 0 00000011
metal n 1 0 1 0 00000011
gold n 2 0 2 0 00000012 00000013
au n 1 0 1 0 00000013
france n 1 0 1 0 00000014
country n 1 0 1 0 00000015
paris n 1 0 1 0 00000016
national_capital n 1 0 1 0 00000017
mary n 1 0 1 0 00000018
virgin_mary n 1 0 1 0 00000018
china n 2 0 2 0 00000020 00000019
beijing n 1 0 1 0 00000021
"""
DATA_ADJ = """\
00000101 00 a 01 happy 0 001 ! 00000102 a 0101 | enjoying well-being
00000102 00 a 01 unhappy 0 001 ! 00000101 a 0101 | not happy
"""


def _wordnet_dir():
    d = tempfile.mkdtemp()
    for name, text in (("data.noun", DATA_NOUN), ("index.noun", INDEX_NOUN),
                       ("data.adj", DATA_ADJ)):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    return d


def _mind_with_wordnet():
    m = Mind()
    feed.load_wordnet(m.kb, _wordnet_dir())
    return m


def test_wordnet_taxonomy_and_closed_world():
    m = _mind_with_wordnet()
    assert m.hear("Is a whale a mammal?").startswith("Yes")
    reply = m.hear("Is a whale a fish?")
    assert reply.startswith("No") and "different kinds of vertebrate" in reply, reply
    assert m.hear("Do birds have wings?").startswith("Yes")


def test_wordnet_senses_aliases_and_names():
    m = _mind_with_wordnet()
    kb = m.kb
    assert kb.canon("whale") == "whale"                 # the animal, not 'a very large person'
    assert kb.canon("metal") == "metallic element"      # a synonym maps to its concept
    assert kb.canon("china") != kb.canon("china", proper=True)   # porcelain vs the country
    assert kb.canon("mary", proper=True) == "mary"      # a story's Mary is not the Virgin Mary
    assert m.hear("Is gold a metal?").startswith("Yes, in the sense of a soft yellow")
    assert m.hear("What is the capital of France?") == "Paris."
    assert m.hear("What is the capital of China?") == "Beijing."
    assert m.hear("What is a whale?").startswith("Whale: a large cetacean mammal")
    assert "unhappy" in m.hear("What is the opposite of happy?").lower()


def test_statements_and_exceptions():
    m = Mind()
    added, failed = feed.load_statements(m, ["Penguins are birds. Birds can fly.",
                                             "Penguins cannot fly.", "# a comment",
                                             "Quickly and quietly."])
    assert added == 3 and len(failed) == 1
    assert m.hear("Can a penguin fly?").startswith("No")
    assert m.hear("Opus is a penguin. Can Opus fly?").endswith(
        "Opus is a penguin, and penguins cannot fly.")


def test_triples_tsv_and_jsonl():
    d = tempfile.mkdtemp()
    tsv, jl = os.path.join(d, "facts.tsv"), os.path.join(d, "facts.jsonl")
    with open(tsv, "w") as f:
        f.write("Rome\tcapital of\tItaly\nsnakes\tnot has\tleg\n")
    with open(jl, "w") as f:
        f.write(json.dumps({"s": "Lima", "r": "capital_of", "o": "Peru", "c": 0.95}) + "\n")
    m = Mind()
    assert feed.load_triples(m.kb, tsv) == 2 and feed.load_triples(m.kb, jl) == 1
    assert m.hear("What is the capital of Italy?") == "Rome."
    assert m.hear("What is the capital of Peru?") == "Lima."
    assert m.hear("Do snakes have legs?").startswith("No")


def test_general_pack_is_fully_understood():
    m = Mind()
    added, failed = feed.load_pack(m, "general")
    assert added > 800 and failed == [], failed[:5]
    assert m.hear("What is the capital of Japan?") == "Tokyo."
    assert m.hear("How many legs does a spider have?") == "Eight."
    assert m.hear("Can people fly?").startswith("No")


def test_fed_mind_survives_save_and_load():
    m = _mind_with_wordnet()
    feed.load_statements(m, ["Gold is heavier than iron."])
    d = tempfile.mkdtemp()
    m.save(d)
    m2 = Mind.load(d)
    assert m2.kb.canon("metal") == "metallic element"
    assert m2.hear("Is gold a metal?").startswith("Yes")
    assert m2.hear("Is gold heavier than iron?").startswith("Yes")


def test_teacher_output_is_filtered_to_what_mnemo_can_read():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "tools"))
    import teacher_pack
    reply = """Here are some facts about owls:
1. Owls are birds.
2. **Owls can fly.**
- Owls hunt at night, using their excellent hearing to locate prey in complete darkness.
* Owls have big eyes
Owls are birds.
I hope this helps!"""
    kept = teacher_pack.keep(reply.splitlines())
    assert kept == ["Owls are birds.", "Owls can fly.", "Owls have big eyes."], kept
    m = Mind()
    added, failed = feed.load_statements(m, kept, source="teacher", conf=0.7)
    assert added == 3 and not failed
    assert m.kb.get("owl", "can", "fly").source == "teacher"
