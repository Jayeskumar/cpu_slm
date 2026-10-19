"""General-knowledge probe for a fed brain (WordNet + the general pack).

  python3 -m mnemo feed --brain models/kbrain        # build it once (~10 s)
  python3 tests/knowledge_probe.py models/kbrain [-v]

Three question sets:
  dev         questions used while building the importer and the pack;
  held-out    written before the system was run on them; its first score
              (57/64) was then used to fix bugs, so it is now development data;
  held-out 2  written after those fixes and run once.
A yes/no answer counts when it has the right polarity ('Yes'/'Probably' vs
'No'/'Probably not'); other answers count when they contain the expected word.
"I don't know" is always wrong.
"""
import re
import sys
import time

import conftest  # noqa: F401
from mnemo.mind import Mind

DEV = [
    ("Is a dog an animal?", "yes"), ("Is a whale a fish?", "no"), ("Is a bat a bird?", "no"),
    ("Is a spider an insect?", "no"), ("Can a penguin fly?", "no"), ("Can a sparrow fly?", "yes"),
    ("Can an ostrich fly?", "no"), ("Can a bat fly?", "yes"),
    ("Do penguins have feathers?", "yes"), ("What is the capital of France?", "paris"),
    ("What is the capital of Japan?", "tokyo"), ("What is the capital of australia?", "canberra"),
    ("What is the capital of Kuwait?", "kuwait city"), ("What is the capital of Russia?", "moscow"),
    ("What is the currency of Japan?", "yen"), ("What is the currency of Germany?", "euro"),
    ("What is the language of Brazil?", "portuguese"), ("Is Paris in Europe?", "yes"),
    ("Is Japan in Asia?", "yes"), ("Is Canada a country?", "yes"), ("Is gold a metal?", "yes"),
    ("Is gold heavier than wood?", "yes"), ("Does iron conduct electricity?", "yes"),
    ("What is water made of?", "hydrogen"), ("Is a tomato a fruit?", "yes"),
    ("How many legs does a spider have?", "eight"),
    ("How many legs does a beetle have?", "six"), ("How many bones do people have?", "206"),
    ("Is the sun a star?", "yes"), ("Is Pluto a planet?", "no"),
    ("Is a person an animal?", "yes"), ("Can people fly?", "no"), ("Do cats eat meat?", "yes"),
    ("Is a mouse bigger than an elephant?", "no"), ("What is the opposite of hot?", "cold"),
    ("Who is Shakespeare?", "dramatist"), ("What is a violin?", "stringed"),
]

HELD_OUT = [
    ("Is a cat a mammal?", "yes"), ("Is a shark a mammal?", "no"),
    ("Is a dolphin a fish?", "no"), ("Is an eagle a bird?", "yes"),
    ("Is a crocodile a reptile?", "yes"), ("Is a frog an amphibian?", "yes"),
    ("Is a butterfly an insect?", "yes"), ("Is a lobster an insect?", "no"),
    ("Is an oak a tree?", "yes"), ("Is a rose a plant?", "yes"),
    ("Is a potato a vegetable?", "yes"), ("Is a hammer a tool?", "yes"),
    ("Is a piano a musical instrument?", "yes"), ("Is a bus a vehicle?", "yes"),
    ("Is a chair a kind of furniture?", "yes"), ("Is copper a metal?", "yes"),
    ("Is oxygen a gas?", "yes"), ("Is Jupiter a planet?", "yes"),
    ("Is the moon a planet?", "no"), ("Is a whale an animal?", "yes"),
    ("What is the capital of Italy?", "rome"), ("What is the capital of Egypt?", "cairo"),
    ("What is the capital of Canada?", "ottawa"), ("What is the capital of Kenya?", "nairobi"),
    ("What is the capital of Peru?", "lima"), ("What is the capital of Norway?", "oslo"),
    ("What is the capital of Vietnam?", "hanoi"),
    ("What is the capital of Argentina?", "buenos aires"),
    ("What is the capital of Iceland?", "reykjavik"), ("What is the capital of Spain?", "madrid"),
    ("Is Berlin in Germany?", "yes"), ("Is Tokyo in Europe?", "no"),
    ("Where is Nairobi?", "kenya"), ("Is Brazil in South America?", "yes"),
    ("Is Egypt in Africa?", "yes"), ("Is London in Europe?", "yes"),
    ("What is the currency of India?", "rupee"), ("What is the currency of France?", "euro"),
    ("What is the language of Mexico?", "spanish"),
    ("What is the language of Germany?", "german"),
    ("Does a car have wheels?", "yes"), ("Does a tree have leaves?", "yes"),
    ("Do fish have gills?", "yes"), ("Does a bird have feathers?", "yes"),
    ("What is glass made of?", "sand"), ("What is cheese made of?", "milk"),
    ("How many legs does an ant have?", "six"), ("How many legs does a cat have?", "four"),
    ("How many months does a year have?", "twelve"),
    ("How many hours does a day have?", "24"),
    ("Can a duck swim?", "yes"), ("Can an emu fly?", "no"), ("Can a dog fly?", "no"),
    ("Can an eagle fly?", "yes"), ("Can a fish walk?", "no"), ("Can a snake swim?", "yes"),
    ("Is a horse bigger than a mouse?", "yes"), ("Is a cat bigger than a whale?", "no"),
    ("Is the earth bigger than the sun?", "no"), ("What is a giraffe?", "quadruped"),
    ("Who was Napoleon?", "emperor"), ("What is a volcano?", "lava"),
    ("What is the opposite of big?", "small"), ("What is the opposite of good?", "bad"),
]


# written after the fixes prompted by HELD_OUT, and run once
HELD_OUT_2 = [
    ("Is a wolf a mammal?", "yes"), ("Is a salmon a fish?", "yes"),
    ("Is a penguin a mammal?", "no"), ("Is a lizard a reptile?", "yes"),
    ("Is a mosquito an insect?", "yes"), ("Is a crab an insect?", "no"),
    ("Is a horse an animal?", "yes"), ("Is a parrot a bird?", "yes"),
    ("Is a tulip a flower?", "yes"), ("Is a pine a tree?", "yes"),
    ("Is an apple a fruit?", "yes"), ("Is a carrot a vegetable?", "yes"),
    ("Is a knife a tool?", "yes"), ("Is a guitar a musical instrument?", "yes"),
    ("Is a truck a vehicle?", "yes"), ("Is silver a metal?", "yes"),
    ("Is helium a gas?", "yes"), ("Is Saturn a planet?", "yes"),
    ("Is the sun a planet?", "no"), ("Is a mushroom an animal?", "no"),
    ("What is the capital of Germany?", "berlin"), ("What is the capital of China?", "beijing"),
    ("What is the capital of Brazil?", "brasilia"), ("What is the capital of Turkey?", "ankara"),
    ("What is the capital of Poland?", "warsaw"),
    ("What is the capital of Thailand?", "bangkok"),
    ("What is the capital of Nigeria?", "abuja"), ("What is the capital of Chile?", "santiago"),
    ("What is the capital of Sweden?", "stockholm"), ("What is the capital of Ukraine?", "kyiv"),
    ("Is Rome in Italy?", "yes"), ("Is Cairo in Europe?", "no"), ("Where is Lima?", "peru"),
    ("Is India in Asia?", "yes"), ("Is Madrid in Spain?", "yes"),
    ("What is the currency of Mexico?", "peso"), ("What is the currency of Russia?", "ruble"),
    ("What is the language of Italy?", "italian"),
    ("What is the language of Japan?", "japanese"),
    ("Does a bicycle have wheels?", "yes"), ("Do birds have beaks?", "yes"),
    ("What is butter made of?", "milk"), ("What is paper made of?", "wood"),
    ("How many legs does a horse have?", "four"), ("How many legs does a bee have?", "six"),
    ("How many days does a week have?", "seven"),
    ("How many wheels does a bicycle have?", "two"),
    ("Can a parrot fly?", "yes"), ("Can a penguin swim?", "yes"), ("Can a cow fly?", "no"),
    ("Can a frog jump?", "yes"), ("Is an elephant bigger than a dog?", "yes"),
    ("Is a dog bigger than a horse?", "no"), ("What is a zebra?", "striped"),
    ("Who was Mozart?", "composer"), ("What is a desert?", "arid"),
    ("What is the opposite of fast?", "slow"), ("What is the opposite of happy?", "sad"),
]


def correct(reply, gold):
    r = reply.strip().lower()
    if "don't know" in r or "don't understand" in r:
        return False
    if gold in ("yes", "no"):
        neg = r.startswith(("no", "probably not", "maybe not"))
        pos = r.startswith(("yes", "probably", "maybe")) and not neg
        return pos if gold == "yes" else neg
    return re.search(r"\b" + re.escape(gold) + r"\b", r) is not None


def run(mind, questions, verbose=False):
    ok, times = 0, []
    for q, gold in questions:
        t0 = time.perf_counter()
        reply = mind.hear(q)
        times.append(time.perf_counter() - t0)
        good = correct(reply, gold)
        ok += good
        if verbose and not good:
            print(f"  MISS {q} -> {reply!r} (want {gold})")
    return ok, sorted(times)[len(times) // 2]


if __name__ == "__main__":
    brain = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "models/kbrain"
    m = Mind.load(brain, learn_language=False)
    for name, qs in (("dev", DEV), ("held-out", HELD_OUT), ("held-out 2", HELD_OUT_2)):
        ok, med = run(m, qs, verbose="-v" in sys.argv)
        print(f"{name:9s} {ok}/{len(qs)} correct ({ok / len(qs):.0%}), "
              f"median {med * 1000:.1f} ms per question")
