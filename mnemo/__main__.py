"""Command line: python3 -m mnemo {chat,train,feed,demo,generate}

  chat      talk to a mind (teach it facts, ask questions, ask it to imagine)
  train     build a brain directory from text files (cortex + semantic memory)
  feed      add general knowledge: WordNet, knowledge packs, your own fact files
  demo      a scripted tour of what the mind can do
  generate  continue a prompt with the predictive cortex
"""
import argparse
import os
import sys
import time

from .mind import Mind

PRESETS = {  # table_bits, hist_bits, match_bits, lexicon order
    "tiny": (18, 21, 18, 3),
    "small": (21, 24, 21, 4),
    "large": (23, 25, 22, 5),
}


def cmd_train(a):
    from .cortex import Cortex
    from .semantic import Semantic
    tb, hb, mb, lex = PRESETS[a.preset]
    cortex = Cortex(table_bits=tb, hist_bits=hb, match_bits=mb, lex_order=lex, spaced=a.spaced)
    total, t0 = 0, time.time()
    texts = []
    for path in a.files:
        bits, n = cortex.learn_file(path)
        total += n
        print(f"read {path}: {n / 1e6:.2f} MB, {bits / max(1, n):.3f} bits/char "
              f"({time.time() - t0:.0f} s)")
        if a.semantic and not a.semantic_files:
            texts.append(open(path, encoding="utf-8", errors="replace").read())
    for path in a.semantic_files or []:
        texts.append(open(path, encoding="utf-8", errors="replace").read())
    semantic = None
    if (a.semantic or a.semantic_files) and texts:
        t1 = time.time()
        semantic = Semantic(dim=a.dim).fit("\n".join(texts))
        print(f"semantic memory: {len(semantic.words)} words x {semantic.dim} dims "
              f"({time.time() - t1:.0f} s)")
    m = Mind(cortex=cortex, semantic=semantic)
    m.save(a.out)
    print(f"saved brain to {a.out}: {total / 1e6:.1f} MB read in {time.time() - t0:.0f} s, "
          f"cortex {cortex.memory_bytes / 2**20:.0f} MB")


def cmd_feed(a):
    """Feed general knowledge into a brain: WordNet, knowledge packs, and
    your own fact files (plain English .txt, or .tsv / .jsonl triples)."""
    from . import feed
    m = load_mind(a.brain)
    t0 = time.time()
    loaded = m.kb.loaded
    if a.names and os.path.isdir(a.names):
        print(f"names: {feed.load_names(a.names)} first names")
    if a.wordnet and "wordnet" in loaded:
        print("WordNet: already in this brain")
    elif a.wordnet and os.path.isdir(a.wordnet):
        st = feed.load_wordnet(m.kb, a.wordnet)
        loaded.add("wordnet")
        print(f"WordNet: {st['concepts']} concepts, {st['isa']} is-a links, "
              f"{st['part_of'] + st['has'] + st['made_of'] + st['in']} part/substance/place links, "
              f"{st['definitions']} definitions ({time.time() - t0:.0f} s)")
    elif a.wordnet:
        print(f"(no WordNet at {a.wordnet}; run scripts/get_data.sh to fetch it)")
    for name in a.pack or []:
        if "pack:" + name in loaded:
            print(f"pack {name}: already in this brain")
            continue
        added, failed = feed.load_pack(m, name)
        loaded.add("pack:" + name)
        print(f"pack {name}: {added} facts" + (f", {len(failed)} lines not understood: "
                                              f"{failed[:3]}" if failed else ""))
    for path in a.files:
        if path.endswith((".tsv", ".jsonl", ".json")):
            print(f"{path}: {feed.load_triples(m.kb, path, a.source, a.conf)} triples")
        else:
            with open(path, encoding="utf-8") as f:
                added, failed = feed.load_statements(m, f, source=a.source, conf=a.conf)
            print(f"{path}: {added} facts" + (f", {len(failed)} sentences not understood"
                                              if failed else ""))
    m.save(a.brain)
    n = sum(1 for f in m.kb.facts.values() if f.source != "innate")
    print(f"saved {a.brain}: {n} facts about {len(m.kb.concepts())} concepts "
          f"({time.time() - t0:.0f} s)")


def load_mind(brain, learn=True):
    if brain and os.path.isdir(brain):
        return Mind.load(brain, learn_language=learn)
    return Mind(learn_language=learn)


def cmd_chat(a):
    m = load_mind(a.brain, learn=not a.no_learn)
    print("Mnemo is listening. Teach me facts, tell me stories, ask questions, "
          "'complete: <text>' to imagine, 'why?' for my reasoning, 'quit' to stop.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("quit", "exit", "bye"):
            break
        if not line:
            continue
        t0 = time.perf_counter()
        reply = m.hear(line)
        dt = (time.perf_counter() - t0) * 1000
        print(f"mnemo> {reply}" + (f"   [{dt:.1f} ms]" if a.timing else ""))
    if a.brain and a.save:
        m.save(a.brain)
        print(f"saved to {a.brain}")


DEMO = [
    ("Default reasoning with exceptions",
     ["Tweety is a bird.", "Birds can fly.", "Penguins are birds.", "Penguins can't fly.",
      "Opus is a penguin.", "Can Tweety fly?", "Can Opus fly?", "Why?", "Is Opus an animal?",
      "Birds are animals.", "Is Opus an animal?"]),
    ("Learning by induction",
     ["Robins are birds.", "Robins can sing.", "Sparrows are birds.", "Sparrows can sing.",
      "Can birds sing?", "Why?", "What can Tweety do?"]),
    ("Episodic memory (who is where, who has what)",
     ["Mary went to the kitchen.", "Mary picked up the apple.", "Daniel went to the garden.",
      "Mary went to the hallway.", "Where is the apple?", "She gave the apple to Daniel.",
      "Who has the apple?", "Where was the apple before the hallway?"]),
    ("Spatial and size reasoning",
     ["The kitchen is north of the hallway.", "The garden is west of the hallway.",
      "How do you go from the kitchen to the garden?", "The box is bigger than the chocolate.",
      "The chest is bigger than the box.", "Does the chocolate fit in the chest?",
      "Is the chocolate bigger than the box?"]),
    ("Analogy in hyperdimensional space (Kanerva's 'dollar of Mexico')",
     ["The currency of the USA is the dollar.", "The capital of the USA is Washington.",
      "The currency of Mexico is the peso.", "The capital of Mexico is Mexico City.",
      "What is the dollar of Mexico?"]),
    ("Motivation learned from observation",
     ["John is hungry.", "John went to the kitchen.", "Sumit is tired.",
      "Sumit went to the bedroom.", "Fred is hungry.", "Where will Fred go?"]),
]


def cmd_demo(a):
    m = load_mind(a.brain, learn=False)
    for title, lines in DEMO:
        print(f"\n== {title}")
        for line in lines:
            t0 = time.perf_counter()
            r = m.hear(line)
            dt = (time.perf_counter() - t0) * 1000
            if line.endswith("?") or r not in ("OK.", ""):
                print(f"  you> {line}\n  mnemo> {r}   [{dt:.2f} ms]")
            else:
                print(f"  you> {line}")
    if m.semantic is not None:
        print("\n== Semantic memory (learned from reading)")
        for q in ["What is similar to king?", "france is to paris as germany is to what?"]:
            print(f"  you> {q}\n  mnemo> {m.hear(q)}")
    if m.cortex is not None:
        print("\n== Imagination (predictive cortex)")
        for p in ["The history of the", "Once upon a time"]:
            print(f"  you> complete: {p}\n  mnemo> {m.hear('complete: ' + p)}")


def cmd_generate(a):
    m = load_mind(a.brain, learn=False)
    if m.cortex is None:
        sys.exit("this brain has no cortex; train one with `python3 -m mnemo train`")
    print(a.prompt + m.cortex.generate(a.prompt, n=a.n, temperature=a.temp, seed=a.seed))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mnemo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="build a brain from text files")
    t.add_argument("files", nargs="+")
    t.add_argument("--out", default="models/brain")
    t.add_argument("--preset", choices=sorted(PRESETS), default="small")
    t.add_argument("--spaced", action="store_true", help="input is pre-tokenized (PTB/WikiText)")
    t.add_argument("--semantic", action="store_true", help="also build the semantic memory")
    t.add_argument("--dim", type=int, default=200)
    t.add_argument("--semantic-files", nargs="+", metavar="FILE",
                   help="build the semantic memory from these files instead")
    t.set_defaults(fn=cmd_train)
    c = sub.add_parser("chat", help="talk to a mind")
    c.add_argument("--brain", default="models/brain")
    c.add_argument("--no-learn", action="store_true")
    c.add_argument("--save", action="store_true", help="save what was learned on exit")
    c.add_argument("--timing", action="store_true")
    c.set_defaults(fn=cmd_chat)
    f = sub.add_parser("feed", help="feed general knowledge (WordNet, packs, fact files)")
    f.add_argument("files", nargs="*", help="statements (.txt) or triples (.tsv, .jsonl)")
    f.add_argument("--brain", default="models/brain")
    f.add_argument("--wordnet", nargs="?", const="data/nltk/wordnet", default="data/nltk/wordnet",
                   help="WordNet 3.0 directory (default data/nltk/wordnet)")
    f.add_argument("--no-wordnet", dest="wordnet", action="store_const", const=None)
    f.add_argument("--names", default="data/nltk/names", help="first-name lists")
    f.add_argument("--pack", nargs="*", default=["general"], help="knowledge packs (knowledge/*.txt)")
    f.add_argument("--source", default="file", help="source label for the files' facts")
    f.add_argument("--conf", type=float, default=0.9, help="confidence of the files' facts")
    f.set_defaults(fn=cmd_feed)
    d = sub.add_parser("demo", help="scripted tour")
    d.add_argument("--brain", default="models/brain")
    d.set_defaults(fn=cmd_demo)
    g = sub.add_parser("generate", help="continue a prompt")
    g.add_argument("prompt")
    g.add_argument("--brain", default="models/brain")
    g.add_argument("-n", type=int, default=400)
    g.add_argument("--temp", type=float, default=0.6)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(fn=cmd_generate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
