"""Distil a large language model's knowledge into a Mnemo knowledge pack.

Mnemo cannot load Gemma's weights (2.5 billion of them, 5 GB), but it can
learn from Gemma the way a pupil learns from a teacher: the model, running on
your own machine, is asked for short, simple statements about each topic, and
Mnemo's own reader keeps the ones it understands.

  ollama pull gemma:2b                     # or any model your server offers
  python3 tools/teacher_pack.py --model gemma:2b --topics topics.txt \\
      --out knowledge/teacher.txt
  python3 -m mnemo feed --brain models/kbrain --source teacher --conf 0.7 \\
      knowledge/teacher.txt

Facts keep their source ('teacher') and a lower confidence, so what you tell
Mnemo yourself, and WordNet, win when they disagree; Mnemo also notes every
statement that contradicts what it would have concluded.

  --api ollama   POST {endpoint}/api/generate           (default endpoint
                 http://localhost:11434)
  --api openai   POST {endpoint}/v1/chat/completions    (llama.cpp server,
                 vLLM, LM Studio, ...)
  --from FILE    filter a file of model output instead of calling a server
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnemo.llm import LLM, readable as keep  # noqa: E402,F401  (keep: the filter)

PROMPT = """List {n} short, true, general facts about {topic}.
Write one fact per line as a simple sentence in one of these forms:
X is a Y. / Xs are Ys. / Xs can VERB. / Xs cannot VERB. / Xs have Ys. /
Xs have NUMBER Ys. / X is in Y. / X is bigger than Y. / X is made of Y. /
The capital of X is Y. / The currency of X is the Y. / The language of X is Y.
No numbering, no explanations, no other text."""


def ask(api, endpoint, model, prompt, timeout=300):
    """One request to the teacher (see mnemo/llm.py for the backends)."""
    url = endpoint.rstrip("/") + ("/v1" if api == "openai" else "")
    return LLM(f"{api}:{model}@{url}", timeout=timeout, max_tokens=800).ask(prompt, system=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--topics", help="file with one topic per line")
    ap.add_argument("--topic", action="append", default=[], help="a topic (repeatable)")
    ap.add_argument("--model", default="gemma:2b")
    ap.add_argument("--api", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--endpoint", default="http://localhost:11434")
    ap.add_argument("-n", type=int, default=12, help="facts asked per topic")
    ap.add_argument("--from", dest="src", help="filter this file of model output instead")
    ap.add_argument("--out", default="knowledge/teacher.txt")
    a = ap.parse_args()

    topics = list(a.topic)
    if a.topics:
        topics += [t.strip() for t in open(a.topics, encoding="utf-8") if t.strip()]
    kept, total = [], 0
    if a.src:
        lines = open(a.src, encoding="utf-8").read().splitlines()
        total = len(lines)
        kept = keep(lines)
    else:
        if not topics:
            sys.exit("give --topic/--topics, or --from FILE")
        for t in topics:
            text = ask(a.api, a.endpoint, a.model, PROMPT.format(n=a.n, topic=t))
            lines = text.splitlines()
            total += len(lines)
            new = keep(lines)
            kept += [x for x in new if x not in kept]
            print(f"{t}: {len(new)}/{len(lines)} lines kept", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# Taught by {a.model if not a.src else a.src} on "
                f"{datetime.date.today()}; {len(kept)} of {total} lines kept.\n")
        f.write("# Check before trusting: a teacher model can be wrong.\n")
        for line in kept:
            f.write(line + "\n")
    print(f"wrote {a.out}: {len(kept)} statements Mnemo can read (from {total} lines)")


if __name__ == "__main__":
    main()
