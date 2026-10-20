"""How people actually ask: a probe of natural phrasings for a fed brain.

  python3 tests/natural_probe.py models/kbrain [-v]

The knowledge probe (tests/knowledge_probe.py) uses the phrasings the reader
was built for.  These questions are typed the way people type: contractions,
fragments, plurals, 'which ...', 'X or Y?', 'how many N in a M', politeness,
shouting, arithmetic, and statements to remember on the way.  Items run in
order in one conversation; an item with gold None is a statement to learn.

  NATURAL    written before the reader was widened for them; its first
             score is the baseline, after which it is development data;
  NATURAL_2  written after the widening, run once.
"""
import sys
import time

import conftest  # noqa: F401
from knowledge_probe import correct
from mnemo.mind import Mind

NATURAL = [
    # contractions, case, fragments, wrappers
    ("What's the capital of France?", "paris"),
    ("whats the capital of japan", "tokyo"),
    ("what is the capital of germany", "berlin"),
    ("Capital of Italy?", "rome"),
    ("capital of spain", "madrid"),
    ("Tell me the capital of Kenya.", "nairobi"),
    ("Do you know the capital of Egypt?", "cairo"),
    ("Which city is the capital of Peru?", "lima"),
    ("Paris is the capital of which country?", "france"),
    ("What country is Berlin the capital of?", "germany"),
    # plural yes/no
    ("Are dolphins fish?", "no"),
    ("Are whales mammals?", "yes"),
    ("Are bats birds?", "no"),
    ("Are spiders insects?", "no"),
    ("Can penguins fly?", "no"),
    ("Can eagles fly?", "yes"),
    ("Do spiders have eight legs?", "yes"),
    ("Do birds have four legs?", "no"),
    ("Is it true that bats are birds?", "no"),
    ("Are tomatoes fruits?", "yes"),
    # counting
    ("How many legs do spiders have?", "eight"),
    ("How many legs does an insect have?", "six"),
    ("How many days are in a week?", "seven"),
    ("How many months are there in a year?", "twelve"),
    ("How many hours are in a day?", "24"),
    # wh- questions with verbs
    ("What do cows eat?", "grass"),
    ("What do pandas eat?", "bamboo"),
    ("Where do kangaroos live?", "australia"),
    ("Where do fish live?", "water"),
    ("What do bees make?", "honey"),
    ("What does the earth orbit?", "sun"),
    # which, lists, examples
    ("Which animals can fly?", "bird|bat|bee|butterfly|duck|eagle"),
    ("Name some birds.", "penguin|eagle|owl|duck|ostrich|emu|kiwi|parrot|sparrow|robin"),
    ("Give me an example of a mammal.", "whale|dog|cat|bat|dolphin|seal|horse|cow|elephant"),
    ("What animals live in the ocean?", "whale|dolphin"),
    ("Name the planets.", "mars|jupiter|saturn|venus|mercury|neptune|uranus|earth"),
    ("List some metals.", "gold|iron|silver|copper|aluminium|lead"),
    # kinds and definitions
    ("What kind of animal is a whale?", "mammal|cetacean"),
    ("What type of animal is a frog?", "amphibian|anuran"),
    ("What is a violin?", "instrument"),
    ("Define volcano.", "lava|mountain|fissure"),
    ("What does 'penguin' mean?", "bird|birds"),
    ("What are penguins?", "bird|birds"),
    # comparison and choice
    ("Which is bigger, an elephant or a mouse?", "elephant"),
    ("What is bigger, the sun or the earth?", "sun"),
    ("Is a tomato a fruit or a vegetable?", "fruit"),
    ("Is a whale a fish or a mammal?", "mammal"),
    ("What is the difference between a whale and a fish?", "mammal|mammals"),
    # places, languages, money
    ("What language do they speak in Brazil?", "portuguese"),
    ("What language is spoken in Mexico?", "spanish"),
    ("What currency does Japan use?", "yen"),
    ("What is the currency in India?", "rupee"),
    ("Which continent is Kenya in?", "africa"),
    ("In which country is Tokyo?", "japan"),
    ("Where is Paris?", "france"),
    ("Is Rome in Italy?", "yes"),
    ("What continent is France on?", "europe"),
    # why
    ("Why is a whale not a fish?", "mammal|mammals"),
    ("Why can a sparrow fly?", "bird|birds"),
    # arithmetic
    ("What is 12 times 7?", "84"),
    ("What's 15 + 27?", "42"),
    ("How much is 100 divided by 4?", "25"),
    ("What is 3 plus 4 times 2?", "11"),
    ("What is 2 to the power of 10?", "1024"),
    # learning on the way
    ("Remember that Max is a cat.", None),
    ("Is Max an animal?", "yes"),
    ("Does Max have whiskers?", "yes"),
    ("Note that Bob likes pizza.", None),
    ("What does Bob like?", "pizza"),
    # robustness
    ("is a penguin a bird", "yes"),
    ("IS A WHALE A FISH?", "no"),
    ("Can a penguin fly???", "no"),
    ("Please tell me what a giraffe is.", "quadruped"),
    ("Could you tell me whether dolphins are mammals?", "yes"),
    ("I wonder if spiders are insects.", "no"),
]

NATURAL_2 = []      # written after the reader was widened; filled in then


def run(mind, items, verbose=False):
    ok = n = 0
    times = []
    for q, gold in items:
        t0 = time.perf_counter()
        reply = mind.hear(q)
        times.append(time.perf_counter() - t0)
        if gold is None:
            continue
        n += 1
        good = correct(reply, gold)
        ok += good
        if verbose and not good:
            print(f"  MISS {q} -> {reply!r} (want {gold})")
    return ok, n, sorted(times)[len(times) // 2]


if __name__ == "__main__":
    brain = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "models/kbrain"
    for name, items in (("natural", NATURAL), ("natural 2", NATURAL_2)):
        if not items:
            continue
        m = Mind.load(brain, learn_language=False)
        ok, n, med = run(m, items, verbose="-v" in sys.argv)
        print(f"{name:10s} {ok}/{n} correct ({ok / n:.0%}), median {med * 1000:.1f} ms")
