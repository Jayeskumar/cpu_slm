"""The wider world: facts read from definitions, open country / city /
element data, and the questions a generalist is asked (dates, lives,
origins, superlatives, counts, who did what).  Everything runs on small
fixtures in the real files' formats, so no download is needed."""
import json
import os
import tempfile

import conftest  # noqa: F401
from mnemo import feed
from mnemo import language as L
from mnemo.knowledge import Knowledge
from mnemo.mind import Mind

# offset lexfile pos word-count words... pointer-count pointers... | gloss
DATA_NOUN = """\
00000001 03 n 01 entity 0 000 | that which is perceived or known
00000002 03 n 01 person 0 001 @ 00000001 n 0000 | a human being
00000003 18 n 01 dramatist 0 001 @ 00000002 n 0000 | someone who writes plays
00000004 18 n 02 Shakespeare 0 William_Shakespeare 0 001 @i 00000003 n 0000 | English poet and dramatist (1564-1616)
00000005 18 n 01 inventor 0 001 @ 00000002 n 0000 | someone who is the first to think of or make something
00000006 18 n 02 Bell 0 Alexander_Graham_Bell 0 001 @i 00000005 n 0000 | United States inventor (born in Scotland) of the telephone (1847-1922)
00000007 18 n 01 statesman 0 001 @ 00000002 n 0000 | a man who is a respected leader in national affairs
00000008 18 n 02 Washington 0 George_Washington 0 001 @i 00000007 n 0000 | 1st President of the United States; commander-in-chief of the Continental Army (1732-1799)
00000009 18 n 01 Hamlet 0 001 @i 00000002 n 0000 | the hero of William Shakespeare's tragedy who hoped to avenge the murder of his father
00000010 06 n 01 telephone 0 001 @ 00000001 n 0000 | electronic equipment that converts sound into electrical signals
00000011 06 n 01 hammer 0 001 @ 00000001 n 0000 | a hand tool used to deliver an impulsive force by striking
00000012 15 n 01 country 0 001 @ 00000001 n 0000 | a politically organized body of people under a single government
00000013 15 n 01 France 0 001 @i 00000012 n 0000 | a republic in western Europe
00000014 15 n 01 Spain 0 001 @i 00000012 n 0000 | a parliamentary monarchy in southwestern Europe
00000015 15 n 02 United_States 0 America 0 001 @i 00000012 n 0000 | North American republic containing 50 states
00000016 15 n 01 Scotland 0 001 @i 00000012 n 0000 | one of the four countries that make up the United Kingdom
00000017 15 n 01 continent 0 001 @ 00000001 n 0000 | one of the large landmasses of the earth
00000018 15 n 01 Europe 0 001 @i 00000017 n 0000 | the 2nd smallest continent
00000019 15 n 01 Asia 0 001 @i 00000017 n 0000 | the largest continent with 60% of the earth's population
00000020 15 n 01 river 0 001 @ 00000001 n 0000 | a large natural stream of water
00000021 15 n 01 Nile 0 001 @i 00000020 n 0000 | the world's longest river (4150 miles); flows northward through eastern Africa
00000022 15 n 01 Thames 0 001 @i 00000020 n 0000 | the longest river in England; flows eastward through London
00000023 15 n 01 Amazon 0 001 @i 00000020 n 0000 | a major South American river; the world's 2nd longest river (4000 miles)
00000024 04 n 01 war 0 001 @ 00000001 n 0000 | the waging of armed conflict against an enemy
00000025 04 n 02 World_War_II 0 Second_World_War 0 001 @i 00000024 n 0000 | a war between the Allies and the Axis from 1939 to 1945
00000026 05 n 01 animal 0 001 @ 00000001 n 0000 | a living organism
00000027 05 n 01 marsupial 0 001 @ 00000026 n 0000 | mammals of Australia and nearby islands
00000028 05 n 01 koala 0 001 @ 00000027 n 0000 | sluggish tailless Australian arboreal marsupial; feeds on eucalyptus leaves
00000029 15 n 01 Australia 0 001 @i 00000012 n 0000 | a nation occupying the whole of the Australian continent
00000030 15 n 01 municipality 0 001 @ 00000001 n 0000 | an urban district having corporate status
00000031 15 n 01 city 0 001 @ 00000030 n 0000 | a large and densely populated urban area
00000032 15 n 01 Valencia 0 001 @i 00000031 n 0000 | a city on the Mediterranean coast; third city of Spain
00000033 27 n 02 chemical_element 0 element 0 001 @ 00000001 n 0000 | any of the more than 100 known substances
00000034 27 n 01 neon 0 001 @ 00000033 n 0000 | a colorless odorless gaseous element
00000035 27 n 02 gold 0 Au 0 001 @ 00000033 n 0000 | a soft yellow malleable metallic element
00000036 27 n 01 metal 0 001 @ 00000033 n 0000 | any of several chemical elements that are usually shiny solids
00000037 18 n 02 Fleming 0 Alexander_Fleming 0 001 @i 00000002 n 0000 | Scottish bacteriologist who discovered penicillin (1881-1955)
00000038 18 n 01 Florey 0 001 @i 00000002 n 0000 | British pathologist who isolated penicillin, which had been discovered in 1928 by Sir Alexander Fleming (1898-1968)
00000039 06 n 01 penicillin 0 001 @ 00000001 n 0000 | any of various antibiotics obtained from Penicillium molds
00000040 04 n 01 battle 0 001 @ 00000001 n 0000 | an open clash between two opposing groups
00000041 04 n 02 Hastings 0 battle_of_Hastings 0 001 @i 00000040 n 0000 | the decisive battle in which William the Conqueror defeated the Saxons under Harold II (1066) and thus left England open
00000042 04 n 01 Vietnam_War 0 001 @i 00000024 n 0000 | a prolonged war (1954-1975) between the communist armies of North Vietnam and the government of South Vietnam
00000043 04 n 02 World_War_I 0 First_World_War 0 001 @i 00000024 n 0000 | a war between the allies and the central powers from 1914 to 1918
00000044 27 n 01 helium 0 001 @ 00000033 n 0000 | a very light colorless element that is one of the six inert gasses
00000045 18 n 01 Picasso 0 001 @i 00000002 n 0000 | prolific and influential Spanish artist who lived in France (1881-1973)
00000046 15 n 01 Darwin 0 001 @i 00000031 n 0000 | provincial capital of the Northern Territory of Australia
00000047 18 n 02 Darwin 0 Charles_Darwin 0 001 @i 00000002 n 0000 | English natural scientist who formulated a theory of evolution by natural selection (1809-1882)
00000048 18 n 02 Armstrong 0 Neil_Armstrong 0 001 @i 00000002 n 0000 | United States astronaut; the first man to set foot on the Moon (1930-2012)
00000049 15 n 01 lake 0 001 @ 00000001 n 0000 | a body of water surrounded by land
00000050 15 n 02 Lake_Superior 0 Superior 0 001 @i 00000049 n 0000 | the largest freshwater lake in the world; the deepest of the Great Lakes
00000051 06 n 01 thermostat 0 001 @ 00000001 n 0000 | a regulator for automatically regulating temperature by starting or stopping the supply of heat
00000052 18 n 01 Cabot 0 001 @i 00000002 n 0000 | Italian explorer who discovered the mainland of North America (1450-1498)
00000053 18 n 01 Columbus 0 001 @i 00000002 n 0000 | Italian navigator who discovered America in the service of Spain (1451-1506)
"""
LEMMAS = {
    "entity": [1], "person": [2], "dramatist": [3], "shakespeare": [4],
    "william_shakespeare": [4], "inventor": [5], "bell": [6], "alexander_graham_bell": [6],
    "statesman": [7], "washington": [8], "george_washington": [8], "hamlet": [9],
    "telephone": [10], "hammer": [11], "country": [12], "france": [13], "spain": [14],
    "united_states": [15], "america": [15], "scotland": [16], "continent": [17], "europe": [18],
    "asia": [19], "river": [20], "nile": [21], "thames": [22], "amazon": [23], "war": [24],
    "world_war_ii": [25], "second_world_war": [25], "animal": [26], "marsupial": [27],
    "koala": [28], "australia": [29], "municipality": [30], "city": [31], "valencia": [32],
    "chemical_element": [33], "element": [33], "neon": [34], "gold": [35], "au": [35],
    "metal": [36], "fleming": [37], "alexander_fleming": [37], "florey": [38],
    "penicillin": [39], "battle": [40], "hastings": [41], "battle_of_hastings": [41],
    "vietnam_war": [42], "world_war_i": [43], "first_world_war": [43], "helium": [44],
    "picasso": [45], "darwin": [46, 47], "charles_darwin": [47], "armstrong": [48],
    "neil_armstrong": [48], "lake": [49], "lake_superior": [50], "superior": [50],
    "thermostat": [51], "cabot": [52], "columbus": [53],
}
COUNTRIES = [
    {"name": {"common": "France", "official": "French Republic"}, "cca3": "FRA",
     "borders": ["ESP", "AND"], "area": 551695, "landlocked": False, "region": "Europe",
     "subregion": "Western Europe", "capital": ["Paris"], "languages": {"fra": "French"},
     "demonyms": {"eng": {"m": "French"}}, "independent": True},
    {"name": {"common": "Spain", "official": "Kingdom of Spain"}, "cca3": "ESP",
     "borders": ["FRA", "AND"], "area": 505992, "landlocked": False, "region": "Europe",
     "subregion": "Southern Europe", "capital": ["Madrid"], "languages": {"spa": "Spanish"},
     "demonyms": {"eng": {"m": "Spanish"}}, "independent": True},
    {"name": {"common": "Andorra", "official": "Principality of Andorra"}, "cca3": "AND",
     "borders": ["FRA", "ESP"], "area": 468, "landlocked": True, "region": "Europe",
     "subregion": "Southern Europe", "capital": ["Andorra la Vella"], "independent": True},
    {"name": {"common": "Philippines", "official": "Republic of the Philippines"},
     "cca3": "PHL", "area": 342353, "landlocked": False, "region": "Asia",
     "subregion": "South-Eastern Asia", "capital": ["Manila"], "independent": True},
    {"name": {"common": "Venezuela", "official": "Bolivarian Republic of Venezuela"},
     "cca3": "VEN", "area": 916445, "region": "Americas", "subregion": "South America",
     "independent": True},
    {"name": {"common": "Réunion", "official": "Réunion Island"}, "cca3": "REU",
     "area": 2511, "region": "Africa", "subregion": "Eastern Africa", "independent": False},
    {"name": {"common": "Svalbard", "official": "Svalbard og Jan Mayen"}, "cca3": "SJM",
     "area": -1, "region": "Europe", "subregion": "Northern Europe", "independent": False},
]
CITIES = """name,country,subcountry,geonameid
Lyon,France,Auvergne-Rhone-Alpes,2996944
Valencia,Spain,Valencia,2509954
Valencia,Venezuela,Carabobo,3625549
Asia,Philippines,Western Visayas,1730000
"""
ELEMENTS = {"elements": [
    {"name": "Gold", "symbol": "Au", "number": 79, "phase": "Solid",
     "category": "transition metal", "discovered_by": "Middle East"},
    {"name": "Neon", "symbol": "Ne", "number": 10, "phase": "Gas", "category": "noble gas",
     "discovered_by": "Morris Travers"},
    {"name": "Helium", "symbol": "He", "number": 2, "phase": "Gas", "category": "noble gas",
     "discovered_by": "Pierre Janssen"},
]}


def _world_mind():
    d = tempfile.mkdtemp()
    index = "".join(f"{lemma} n {len(offs)} 0 {len(offs)} 0 " +
                    " ".join(f"{o:08d}" for o in offs) + "\n" for lemma, offs in LEMMAS.items())
    for name, text in (("data.noun", DATA_NOUN), ("index.noun", index), ("data.adj", "")):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    with open(os.path.join(d, "countries.json"), "w", encoding="utf-8") as f:
        json.dump(COUNTRIES, f)
    with open(os.path.join(d, "cities.csv"), "w", encoding="utf-8") as f:
        f.write(CITIES)
    with open(os.path.join(d, "elements.json"), "w", encoding="utf-8") as f:
        json.dump(ELEMENTS, f)
    m = Mind(learn_language=False)
    feed.load_wordnet(m.kb, d)
    _, names, _ = feed.load_countries(m, os.path.join(d, "countries.json"))
    feed.load_cities(m, os.path.join(d, "cities.csv"), names)
    feed.load_elements(m, os.path.join(d, "elements.json"))
    return m


def test_definitions_are_read_into_facts():
    kb = _world_mind().kb
    assert kb.get("shakespeare", "born", "1564") and kb.get("shakespeare", "died", "1616")
    assert kb.get("shakespeare", "nationality", "English")
    assert kb.get("bell", "born_in", "scotland")
    assert kb.get("washington", "from", "united states")      # '1st President of the US'
    assert kb.get("world war ii", "began", "1939") and kb.get("world war ii", "ended", "1945")
    assert kb.get("hammer", "used_for", "to deliver an impulsive force by striking")
    assert kb.get("koala", "eat", "eucalyptus leaf") and kb.get("koala", "in", "australia")


def test_world_data_and_its_pitfalls():
    m = _world_mind()
    kb = m.kb
    assert kb.get("france", "border", "spain") and kb.get("spain", "border", "france")
    assert kb.get("lyon", "in", "france")
    assert kb.get("valencia", "in", "spain") and not kb.get("valencia", "in", "venezuela")
    assert not kb.get("asia", "in", "philippines")   # the town of Asia is not the continent
    assert not kb.out("svalbard", "area")            # -1 means unknown
    assert m.hear("Is Andorra landlocked?").startswith("Yes")
    assert m.hear("What do you call people from France?") == "French."
    assert m.hear("What is the chemical symbol for gold?") == "The chemical symbol of gold is Au."
    assert m.hear("Which element has atomic number 10?").startswith("Neon")
    assert m.hear("Is neon a metal?").startswith("No")
    assert m.hear("Who discovered neon?").startswith("Morris Travers")


def test_lives_and_dates():
    m = _world_mind()
    assert m.hear("When was Shakespeare born?") == "Shakespeare was born in 1564."
    assert m.hear("When did Shakespeare live?") == "Shakespeare lived from 1564 to 1616."
    assert m.hear("How old was Shakespeare when he died?").startswith("About 52")
    assert m.hear("Who was born first, Washington or Shakespeare?").startswith("Shakespeare")
    assert m.hear("Was Shakespeare born in 1600?") == "No. Shakespeare was born in 1564."
    assert m.hear("Where was Bell born?") == "Bell was born in Scotland."
    assert m.hear("When did the Second World War begin?") == "The Second World War began in 1939."
    assert m.hear("When did World War II end?") == "World War II ended in 1945."


def test_who_did_what():
    m = _world_mind()
    assert m.hear("Who invented the telephone?").startswith("Alexander Graham Bell:")
    assert m.hear("Who discovered penicillin?").startswith("Alexander Fleming:")  # not Florey
    assert m.hear("Who wrote Hamlet?").startswith("William Shakespeare, I think")
    assert m.hear("Who was the first president of the United States?").startswith(
        "George Washington:")
    # ('the hammer' here: in the tiny fixture nothing is a kind of hammer)
    assert m.hear("What is a hammer used for?").endswith(
        "used to deliver an impulsive force by striking.")
    assert m.hear("What do koalas eat?").endswith("eucalyptus leaves.")
    assert m.hear("Where do koalas come from?").endswith("from Australia.")


def test_superlatives_counts_and_comparisons():
    m = _world_mind()
    assert m.hear("What is the longest river?").startswith("The Nile: the world's longest")
    assert m.hear("What is the largest country in Europe?").startswith("France.")
    assert m.hear("What is the smallest country?").startswith("Andorra.")   # Reunion is not one
    assert m.hear("Which is bigger, France or Spain?").startswith("France.")
    assert m.hear("How many countries border Spain?") == "Two: Andorra and France."
    assert m.hear("How many countries are there?").startswith("Five:")


def test_what_the_fresh_sets_taught():
    """General bugs that the fresh question sets exposed (each a rule, not an
    answer): years written several ways, the numeral I, the symbol He, the
    other sense of a name, complete border lists, looser titles and
    superlatives, purposes after the genus, a name inside a longer name."""
    m = _world_mind()
    assert m.hear("When was the Battle of Hastings?") == \
        "The Battle of Hastings took place in 1066."
    assert m.hear("When did the Vietnam War end?") == "The Vietnam War ended in 1975."
    assert m.hear("When did World War I end?") == "World War I ended in 1918."
    assert m.hear("Which element has the symbol He?").startswith("Helium")
    assert m.hear("What nationality was Picasso?") == "Picasso was Spanish."
    assert m.hear("Was Darwin English?").startswith("Yes, if you mean the English natural")
    assert m.hear("Does Spain border Venezuela?") == "No. Spain borders Andorra and France."
    assert m.hear("Who was the first man on the moon?").startswith("Neil Armstrong:")
    assert m.hear("What is the largest lake?").startswith("Lake Superior:")
    assert m.hear("What is a thermostat used for?").endswith(
        "used for automatically regulating temperature by starting or stopping the supply of "
        "heat.")
    assert m.hear("Who discovered America?").startswith("Columbus")    # not North America


def test_lives_can_be_taught():
    m = Mind(learn_language=False)
    for s in ("Tom was born in 1930.", "Tom died in 2001.", "Ann is from Paris."):
        m.hear(s)
    assert m.hear("When was Tom born?") == "Tom was born in 1930."
    assert m.hear("How old was Tom when he died?").startswith("About 71")
    assert m.hear("Was Tom born in 1950?") == "No. Tom was born in 1930."
    assert m.hear("Where is Ann from?") == "Ann is from Paris."


def test_names_that_take_the_and_plurals():
    kb = Knowledge()
    for k, name in (("nile", "Nile"), ("united states", "United States"),
                    ("world war ii", "World War II"), ("paris", "Paris"),
                    ("french revolution", "French Revolution")):
        kb.display[k] = name
    kb.tell("nile", "isa", "river")
    assert [kb.ref(k) for k in ("nile", "united states", "world war ii", "paris",
                                "french revolution")] == \
        ["the Nile", "the United States", "World War II", "Paris", "the French Revolution"]
    assert L.plural("eucalyptus leaf") == "eucalyptus leaves"
