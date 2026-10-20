"""A generalist probe: does the fed brain know many kinds of thing?

  python3 tests/general_probe.py models/kbrain [-v]

People, places, science, objects, animals, history, superlatives and
counting.  GENERAL was written before the generalist work and its first score
is the baseline (it is development data afterwards); GENERAL_2 was written
after that work and run once.
"""
import sys

import conftest  # noqa: F401
from mnemo.mind import Mind
from natural_probe import run

GENERAL = [
    # people
    ("When was Shakespeare born?", "1564"),
    ("When did Einstein die?", "1955"),
    ("What nationality was Mozart?", "austrian"),
    ("Who was Isaac Newton?", "mathematician|physicist"),
    ("Was Beethoven German?", "yes"),
    ("What was Picasso?", "painter|artist|sculptor"),
    ("When was Abraham Lincoln born?", "1809"),
    ("Who was born first, Newton or Einstein?", "newton"),
    ("How old was Mozart when he died?", "35"),
    ("Where was Gandhi from?", "india|indian"),
    ("Who was Charles Darwin?", "naturalist"),
    ("When did Columbus die?", "1506"),
    ("Was Marie Curie a chemist?", "yes"),
    ("Who invented the telephone?", "bell"),
    ("Who discovered penicillin?", "fleming"),
    ("Who painted the Mona Lisa?", "leonardo"),
    ("Who wrote Hamlet?", "shakespeare"),
    ("Who was the first president of the United States?", "washington"),
    # geography
    ("What countries border France?",
     "spain|belgium|germany|italy|switzerland|luxembourg|andorra|monaco"),
    ("Does Germany border Poland?", "yes"),
    ("Is Switzerland landlocked?", "yes"),
    ("Is Japan landlocked?", "no"),
    ("What is the area of France?", "551"),
    ("Which is bigger, France or Spain?", "france"),
    ("Where is Lyon?", "france"),
    ("Which country is Barcelona in?", "spain"),
    ("Where is Toronto?", "canada"),
    ("Which countries border China?",
     "india|russia|mongolia|vietnam|nepal|pakistan|kazakhstan|laos|myanmar|bhutan|afghanistan"),
    ("Is Bolivia landlocked?", "yes"),
    ("What is the largest country?", "russia"),
    ("What is the largest country in Africa?", "algeria"),
    ("What is the smallest country?", "vatican"),
    ("How many countries border Germany?", "nine"),
    ("What do you call people from France?", "french"),
    # science
    ("What is the chemical symbol for gold?", "au"),
    ("What is the symbol of iron?", "fe"),
    ("What element has the symbol Na?", "sodium"),
    ("What is the atomic number of carbon?", "6"),
    ("Which element has atomic number 8?", "oxygen"),
    ("Is mercury a liquid?", "yes"),
    ("Who discovered oxygen?", "scheele|priestley|lavoisier"),
    ("Is helium a noble gas?", "yes"),
    ("Is neon a metal?", "no"),
    # objects
    ("What is a hammer used for?", "striking|strike|nails|nail|pounding|hitting"),
    ("What is a knife used for?", "cut|cutting"),
    ("What is a thermometer used for?", "temperature"),
    ("What is a saw used for?", "cut|cutting"),
    ("What is an oven used for?", "baking|roasting|cooking|heating"),
    ("What is wine made from?", "grape"),
    ("What is glass made of?", "sand"),
    # animals
    ("Where do kangaroos come from?", "australia"),
    ("Where do tigers live?", "asia"),
    ("Are penguins flightless?", "yes"),
    ("Is a koala a bear?", "no"),
    ("What do koalas eat?", "eucalyptus"),
    ("Are lions carnivores?", "yes"),
    # history
    ("When did World War II end?", "1945"),
    ("When was the French Revolution?", "1789"),
    ("When did the First World War begin?", "1914"),
    # counting
    ("How many continents are there?", "seven"),
    ("How many planets are there?", "eight"),
    ("How many moons does Mars have?", "two"),
    # superlatives and the rest
    ("What is the tallest mountain?", "everest"),
    ("What is the longest river?", "nile"),
    ("What is the largest ocean?", "pacific"),
    ("What is the biggest planet?", "jupiter"),
    ("What is the capital of Mongolia?", "ulaanbaatar"),
    ("What language do they speak in Argentina?", "spanish"),
]

# written after the generalist work, before it was ever run, and run once
GENERAL_2 = [
    # people
    ("When was Charles Darwin born?", "1809"),
    ("When did Beethoven die?", "1827"),
    ("What nationality was Napoleon?", "french|corsican"),
    ("Who was Galileo?", "astronomer|mathematician|physicist|scientist"),
    ("Was Darwin English?", "yes"),
    ("What was Beethoven?", "composer|pianist"),
    ("When was Galileo born?", "1564"),
    ("Who was born first, Mozart or Bach?", "bach"),
    ("How old was Shakespeare when he died?", "52"),
    ("Where was Darwin from?", "england|united kingdom|english|britain|british"),
    ("Who invented the telescope?", "lippershey|lipperhey|galileo"),
    ("Who discovered gravity?", "newton"),
    ("Who painted the Sistine Chapel?", "michelangelo"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("Who was the first man on the moon?", "armstrong"),
    ("Who was the third president of the United States?", "jefferson"),
    # geography
    ("What countries border Spain?", "portugal|france|andorra|morocco|gibraltar"),
    ("Does France border Germany?", "yes"),
    ("Does Spain border Germany?", "no"),
    ("Is Austria landlocked?", "yes"),
    ("Is Italy landlocked?", "no"),
    ("What is the area of Japan?", "377"),
    ("Which is bigger, Brazil or Australia?", "brazil"),
    ("Which is smaller, Belgium or the Netherlands?", "belgium"),
    ("Where is Marseille?", "france"),
    ("Which country is Munich in?", "germany"),
    ("What is the largest country in South America?", "brazil"),
    ("What is the smallest country in Europe?", "vatican"),
    ("How many countries border Italy?", "six|6"),
    ("What do you call people from Germany?", "german"),
    ("What is the capital of Vietnam?", "hanoi"),
    ("What language do they speak in Austria?", "german"),
    ("What currency is used in Switzerland?", "franc"),
    # science
    ("What is the chemical symbol for silver?", "ag"),
    ("What is the symbol of potassium?", "k"),
    ("What element has the symbol Fe?", "iron"),
    ("What is the atomic number of oxygen?", "8"),
    ("Which element has atomic number 1?", "hydrogen"),
    ("Is oxygen a gas?", "yes"),
    ("Is iron a metal?", "yes"),
    ("Is argon a noble gas?", "yes"),
    ("Who discovered hydrogen?", "cavendish"),
    # objects
    ("What is a thermostat used for?", "temperature"),
    ("What is a microscope used for?",
     "magnify|magnifying|magnified|enlarge|enlarged|small|tiny|view|viewing|examine|examining"),
    ("What is a spoon used for?", "stir|serve|eat|eating|food"),
    ("What is a compass used for?",
     "direction|navigation|navigating|north|circle|drawing|draw|bearing"),
    ("What is bread made from?", "flour|dough|grain|wheat|meal"),
    ("What is paper made of?", "pulp|wood|cellulose|rag"),
    # animals
    ("Where do pandas live?", "china"),
    ("Where do penguins live?", "antarctic|antarctica|southern hemisphere|south"),
    ("Where do lemurs live?", "madagascar"),
    ("Are bats nocturnal?", "yes"),
    ("Is a seal a fish?", "no"),
    ("Are tigers endangered?", "yes"),
    ("Are crocodiles reptiles?", "yes"),
    # history
    ("When did World War I end?", "1918"),
    ("When did the American Civil War begin?", "1861"),
    ("When was the Russian Revolution?", "1917"),
    ("When did the Korean War begin?", "1950"),
    ("When was the Battle of Hastings?", "1066"),
    # counting
    ("How many planets are there in the solar system?", "eight|8"),
    ("How many legs does an ant have?", "six|6"),
    # superlatives
    ("What is the highest mountain in Africa?", "kilimanjaro"),
    ("What is the largest island?", "greenland"),
    ("What is the longest river in Europe?", "volga"),
    ("What is the largest lake?", "caspian|superior"),
    ("What is the biggest animal?", "blue whale|whale"),
    ("What is the smallest continent?", "australia"),
]

# written after the general fixes that GENERAL_2's misses prompted, run once
GENERAL_3 = [
    # people
    ("When was Mozart born?", "1756"),
    ("When did Napoleon die?", "1821"),
    ("What nationality was Picasso?", "spanish"),
    ("Who was Aristotle?", "philosopher"),
    ("Was Einstein a physicist?", "yes"),
    ("How old was Napoleon when he died?", "51|52"),
    ("Who was born first, Newton or Galileo?", "galileo"),
    ("Where was Columbus from?", "italy|italian|genoa"),
    ("Who discovered America?", "columbus"),
    ("Who invented the printing press?", "gutenberg"),
    ("Who wrote Don Quixote?", "cervantes"),
    ("Who was the first emperor of Rome?", "augustus"),
    ("Who was Charles Dickens?", "novelist|writer"),
    ("When did Lincoln die?", "1865"),
    # geography
    ("What countries border Portugal?", "spain"),
    ("Does Canada border the United States?", "yes"),
    ("Does Brazil border Chile?", "no"),
    ("Is Nepal landlocked?", "yes"),
    ("What is the area of Italy?", "301"),
    ("Which is bigger, India or Argentina?", "india"),
    ("Where is Osaka?", "japan"),
    ("Which country is Milan in?", "italy"),
    ("What is the largest country in Europe?", "russia|ukraine"),
    ("How many countries border Austria?", "eight|8"),
    ("What do you call people from Italy?", "italian"),
    ("What is the capital of Turkey?", "ankara"),
    # science
    ("What is the chemical symbol for lead?", "pb"),
    ("Which element has the symbol He?", "helium"),
    ("What is the atomic number of iron?", "26"),
    ("Is mercury a metal?", "yes"),
    ("Is chlorine a gas?", "yes"),
    ("Who discovered helium?", "janssen|lockyer|ramsay"),
    # objects
    ("What is a scalpel used for?", "cut|cutting|surgery|surgical|incision"),
    ("What is a kettle used for?", "boil|boiling|heat|heating|water"),
    ("What is cheese made from?", "milk|curd"),
    # animals
    ("Where do polar bears live?", "arctic"),
    ("Are owls nocturnal?", "yes"),
    ("Is a frog an amphibian?", "yes"),
    # history
    ("When did the Vietnam War end?", "1975"),
    ("When was the Battle of Waterloo?", "1815"),
    # counting
    ("How many legs does a dog have?", "four|4"),
    # superlatives
    ("What is the longest river in Africa?", "nile"),
    ("What is the largest planet in the solar system?", "jupiter"),
    ("What is the highest mountain in Japan?", "fuji"),
    ("What is the smallest country in the world?", "vatican"),
    ("What is the largest bird?", "ostrich"),
]


if __name__ == "__main__":
    brain = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "models/kbrain"
    for name, items in (("general", GENERAL), ("general 2", GENERAL_2),
                        ("general 3", GENERAL_3)):
        if not items:
            continue
        m = Mind.load(brain, learn_language=False)
        ok, n, med = run(m, items, verbose="-v" in sys.argv)
        print(f"{name:10s} {ok}/{n} correct ({ok / n:.0%}), median {med * 1000:.1f} ms")
