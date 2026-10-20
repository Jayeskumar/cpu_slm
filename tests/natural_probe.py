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

# written after the widening, before it was ever run, and run once
NATURAL_2 = [
    ("What's the capital of Canada?", "ottawa"),
    ("capital of norway?", "oslo"),
    ("Can you tell me the capital of Australia?", "canberra"),
    ("Which city is the capital of Thailand?", "bangkok"),
    ("Rome is the capital of which country?", "italy"),
    ("Are sharks mammals?", "no"),
    ("Are frogs reptiles?", "no"),
    ("Are owls birds?", "yes"),
    ("Can ostriches fly?", "no"),
    ("Can ducks swim?", "yes"),
    ("Do insects have six legs?", "yes"),
    ("Do spiders have six legs?", "no"),
    ("Is it true that whales are fish?", "no"),
    ("How many legs do cats have?", "four"),
    ("How many minutes are in an hour?", "60"),
    ("How many wheels does a bicycle have?", "two"),
    ("What do lions eat?", "meat"),
    ("What do frogs eat?", "insect"),
    ("Where do camels live?", "desert"),
    ("What do spiders make?", "web"),
    ("Which animals live in Australia?", "kangaroo|koala"),
    ("Name some reptiles.", "snake|lizard|crocodile|turtle"),
    ("Give me some examples of fruit.", "apple|banana|orange|grape|lemon|strawberry|tomato"),
    ("List the continents.", "africa|asia|europe|antarctica|australia|america"),
    ("What kind of animal is a shark?", "fish"),
    ("What type of animal is a snake?", "reptile"),
    ("Define giraffe.", "quadruped"),
    ("What does 'whale' mean?", "mammal|cetacean"),
    ("Which is heavier, gold or iron?", "gold"),
    ("What is bigger, a horse or a dog?", "horse"),
    ("Is a bat a bird or a mammal?", "mammal"),
    ("Is a penguin a fish or a bird?", "bird"),
    ("What is the difference between a frog and a lizard?", "amphibian|reptile"),
    ("What language do people speak in France?", "french"),
    ("Which language is spoken in Japan?", "japanese"),
    ("What currency is used in Mexico?", "peso"),
    ("What money do they use in the United Kingdom?", "pound"),
    ("Which continent is Brazil in?", "south america"),
    ("In which country is Berlin?", "germany"),
    ("What country is Nairobi in?", "kenya"),
    ("Why is a bat not a bird?", "mammal"),
    ("Why can an eagle fly?", "bird"),
    ("Why can't ostriches fly?", "cannot"),
    ("What is 9 times 8?", "72"),
    ("What's 100 minus 37?", "63"),
    ("How much is 144 divided by 12?", "12"),
    ("What is 5 squared?", "25"),
    ("Remember that Rex is a dog.", None),
    ("Can Rex fly?", "no"),
    ("Is Rex an animal?", "yes"),
    ("Note that Alice likes chess.", None),
    ("What does Alice like?", "chess"),
    ("is a shark a fish", "yes"),
    ("CAN A PENGUIN SWIM?", "yes"),
    ("Could you tell me whether owls are birds?", "yes"),
    ("I wonder if a whale is a mammal.", "yes"),
    ("Please tell me what a volcano is.", "lava|mountain|fissure"),
    ("Do you know what pandas eat?", "bamboo"),
]


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
