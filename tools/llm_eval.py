"""How a language model does on Mnemo's probes, alone and paired with Mnemo.

  python3 tools/llm_eval.py --llm ollama:qwen2.5:0.5b            # all sets, all modes
  python3 tools/llm_eval.py --llm openai:smollm2@http://127.0.0.1:8080/v1 \\
      --sets general2,general3 --limit 20 -v
  python3 tools/llm_eval.py --modes mnemo                        # no model needed

Three ways to answer each question:
  mnemo   Mnemo alone: the fed brain, no model
  llm     the model alone, with the conversation so far but none of Mnemo's facts
  paired  Mnemo first; the model only for what memory cannot answer, shown the
          facts Mnemo holds about the question (mnemo/llm.py)

Each probe set is one conversation, in order, as in tests/*_probe.py, and is
scored with the probes' own checker.  The model runs on your machine or your
server; nothing is downloaded.  Write the table with --out.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import general_probe  # noqa: E402
import knowledge_probe  # noqa: E402
import natural_probe  # noqa: E402
from knowledge_probe import correct  # noqa: E402
from mnemo.llm import LLM  # noqa: E402
from mnemo.mind import Mind  # noqa: E402

SETS = {
    "knowledge": [(q, g) for q, g in knowledge_probe.DEV],
    "knowledge-held-out": knowledge_probe.HELD_OUT,
    "knowledge-held-out-2": knowledge_probe.HELD_OUT_2,
    "natural": natural_probe.NATURAL,
    "natural2": natural_probe.NATURAL_2,
    "general": general_probe.GENERAL,
    "general2": general_probe.GENERAL_2,
    "general3": general_probe.GENERAL_3,
}


def run(items, answer):
    """(correct, scored, latencies, misses) over one conversation."""
    ok = n = 0
    times, misses = [], []
    for q, gold in items:
        t0 = time.perf_counter()
        reply = answer(q)
        times.append(time.perf_counter() - t0)
        if gold is None:                        # a statement to remember on the way
            continue
        n += 1
        if correct(reply, gold):
            ok += 1
        else:
            misses.append((q, reply, gold))
    return ok, n, times, misses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llm", help="ollama:MODEL[@URL] or openai:MODEL[@URL]")
    ap.add_argument("--brain", default="models/kbrain")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated: " + ", ".join(SETS))
    ap.add_argument("--modes", default="mnemo,llm,paired")
    ap.add_argument("--limit", type=int, default=0, help="first N questions of each set")
    ap.add_argument("--no-learn", action="store_true",
                    help="paired: do not keep the model's statements as facts")
    ap.add_argument("--out", help="also write the table to this file")
    ap.add_argument("-v", action="store_true", help="list the misses")
    a = ap.parse_args()
    modes = a.modes.split(",")
    if any(m != "mnemo" for m in modes) and not a.llm:
        sys.exit("give --llm for the 'llm' and 'paired' modes (or --modes mnemo)")
    llm = LLM(a.llm) if a.llm else None

    rows = [f"# tools/llm_eval.py --llm {a.llm} --brain {a.brain}"
            f"{f' --limit {a.limit}' if a.limit else ''}",
            f"{'set':22s} {'n':>4s} " + " ".join(f"{m:>16s}" for m in modes) + "  model answered"]
    totals = {m: [0, 0] for m in modes}
    for name in a.sets.split(","):
        items = SETS[name][:a.limit] if a.limit else SETS[name]
        cells, asked = [], ""
        for mode in modes:
            if mode == "llm":
                history = []

                def answer(q, history=history):
                    try:
                        r = llm.ask(q, history=history[-6:])
                    except (OSError, ValueError, KeyError) as e:
                        r = f"(no answer: {type(e).__name__})"
                    history.append((q, r))
                    return r
            else:
                mind = Mind.load(a.brain, learn_language=False)
                if mode == "paired":
                    mind.attach_llm(llm, learn=not a.no_learn)
                used = []

                def answer(q, mind=mind, used=used):
                    r = mind.hear(q)
                    used.append(mind.last_source != "memory")
                    return r
            ok, n, times, misses = run(items, answer)
            totals[mode][0] += ok
            totals[mode][1] += n
            med = sorted(times)[len(times) // 2] * 1000
            cells.append(f"{ok:>3d}/{n:<3d} {med:7.1f} ms")
            if mode == "paired":
                asked = f"  {sum(used)}/{len(used)}"
            if a.v:
                for q, r, gold in misses:
                    print(f"  [{name} / {mode}] MISS {q} -> {r[:160]!r} (want {gold})")
        rows.append(f"{name:22s} {len(items):4d} " + " ".join(f"{c:>16s}" for c in cells) + asked)
        print(rows[-1], flush=True)
    rows.append(f"{'total':22s} {'':4s} " + " ".join(
        f"{totals[m][0]:>4d}/{totals[m][1]:<4d} {totals[m][0] / max(1, totals[m][1]):5.0%}  "
        for m in modes))
    print(rows[-1])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
