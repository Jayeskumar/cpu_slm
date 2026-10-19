# cpu_slm: Mnemo, a tiny mind without neural networks

Mnemo is a small, CPU-only system that learns language and reasons in a few
tens of megabytes of memory. It contains no neural network weights: no
gradient descent and no GPU. Every component is a classic idea from cognitive
science or data compression, built from counting, hashing, hypervectors and
logic. Each learns online from a single pass over its input.

* It **predicts text.** On Penn Treebank, with long-term memory frozen, it
  matches the classic Kneser-Ney + cache baseline (125.4 vs 125.7 perplexity).
  Learning online, it beats a small LSTM (113.0 vs 114.5), and it still
  reaches 117.7 in only 35 MB.
* On Tiny Shakespeare it scores 2.04 bits per character with learning frozen
  (1.85 online) after 7 s of CPU training. nanoGPT's 10.65M-parameter GPU
  transformer scores 2.12.
* It **compresses losslessly**, so its predictions are honest probabilities.
  Tiny Shakespeare goes to 1.94 bits per character, against 2.36 for bzip2 and
  2.61 for xz.
* It **learns meaning by reading.**
  * happy → glad, pleased, delighted;
  * emma → harriet, knightley, weston (the characters of Austen's *Emma*);
  * france:paris :: germany:berlin;
  * concept clusters of colours, months and royal titles.
* It **remembers facts one-shot and reasons** with uncertainty (defaults,
  exceptions, induction, analogy) and explains every answer.
* It **tracks a changing world** (who is where, who holds what) and passes
  19 bAbI-style story tasks. The stories come from our own generator, not
  the official set.
* It runs on one CPU core: about 40 s to learn the 5 MB Penn Treebank, and
  answers in well under a millisecond.

The full survey, design rationale, measurements and limitations are in
**[RESEARCH.md](RESEARCH.md)**.

## Quick start

```bash
make                       # builds build/mnemo (CLI) and build/libmnemo.so (C99, no deps)
python3 -m mnemo demo      # reasoning tour; needs only numpy
python3 -m mnemo chat      # talk to it: teach facts, tell stories, ask questions
python3 tests/run_tests.py # 24 unit tests (also runnable with pytest)
python3 tests/story_tasks.py 200   # bAbI-style story suite
```

To give it language and word meanings, let it read:

```bash
scripts/get_data.sh                          # PTB, WikiText-2, Tiny Shakespeare, Gutenberg
python3 -m mnemo train data/gutenberg/gutenberg/*.txt --semantic --out models/brain
python3 -m mnemo chat --brain models/brain   # now 'complete: Once upon a time' works too
```

To reproduce the language-modelling tables:

```bash
scripts/benchmark.sh                         # results/*.txt and a summary table
python3 tools/kn_reference.py data/ptb/ptb.train.txt data/ptb/ptb.test.txt 5   # KN5 = 141.19
```

## A conversation

```text
you> Tweety is a bird. Birds can fly. Penguins are birds. Penguins can't fly.
mnemo> Noted: penguins cannot fly (an exception to what I would have guessed).
you> Opus is a penguin. Can Tweety fly?
mnemo> Yes. Tweety is a bird, and birds can fly.
you> Can Opus fly?
mnemo> No. Opus is a penguin, and penguins cannot fly.
you> Robins are birds. Robins can sing. Sparrows are birds. Sparrows can sing.
you> Can birds sing?
mnemo> Probably. Judging from robins and sparrows, birds can sing.
you> Why?
mnemo> Judging from robins and sparrows, birds can sing. That is a generalisation from examples, so I am only 62% confident.
you> Mary went to the kitchen. Mary picked up the apple. Mary went to the hallway.
you> Where is the apple?
mnemo> The apple is in the hallway.
you> The currency of the USA is the dollar. The currency of Mexico is the peso.
you> What is the dollar of Mexico?
mnemo> Peso (by analogy: dollar is to USA what peso is to Mexico; similarity 0.26).
```

## How it works

```text
                      text in
                         |
        +----------------v-----------------+
        |  PREDICTIVE CORTEX (C, src/)     |   one pass, online, fixed memory
        |  12-16 context columns (bytes,   |   - bit histories in hashed tables
        |  words, concepts) + MENTAL       |   - Kneser-Ney word expectations
        |  LEXICON + PRIMING + EPISODIC    |   - recency (ACT-R-like decay)
        |  RECALL, pooled by learned gates |   - induction-head-like match model
        +----------------+-----------------+
                         | surprise, fluency, plausibility, imagination
        +----------------v-----------------+
        |  LANGUAGE (mnemo/language.py)    |   sentences -> facts / events / questions
        |  WORKING MEMORY (4 entities)     |   pronouns ("she", "they")
        +---+--------------+-----------+---+
            |              |           |
   +--------v------+ +-----v------+ +--v-----------------+
   | DECLARATIVE   | | EPISODIC   | | SEMANTIC           |
   | facts, NARS   | | events in  | | PPMI + SVD (LSA),  |
   | truth <f,c>,  | | time order,| | concepts, analogy, |
   | ACT-R recall  | | world state| | hypervector codes  |
   +-------+-------+ +-----+------+ +--+-----------------+
           +---------------+-----------+
                           v
                REASONING + ANSWERS (mnemo/mind.py)
   deduction, induction, abduction, revision, transitivity,
   mental map, HDC analogy, path search, learned motives; "Why?"
```

| Module | Idea | Brain analogue |
|---|---|---|
| `src/cortex.c` | context mixing: many hashed count memories pooled by context-gated weights (PAQ family) | cortical columns voting on the next input |
| `src/lexicon.c` | online interpolated modified Kneser-Ney over words, read out letter by letter through a trie | the mental lexicon (word-superiority effect) |
| priming caches | two exponentially decaying token caches, which approximate power-law forgetting | repetition priming |
| match model | find the last occurrence of the recent context and propose what followed | hippocampal pattern completion; induction heads |
| `mnemo/semantic.py` | PPMI + randomized SVD; spherical k-means concepts | semantic memory (LSA) |
| `mnemo/hdc.py` | bind, bundle, permute, cleanup in 2048-D bipolar space | distributed, compositional codes |
| `mnemo/knowledge.py` | NARS truth values and inference; ACT-R base-level activation | declarative memory |
| `mnemo/world.py` | ordered event memory; learned motive associations | episodic memory |
| `mnemo/language.py` | pattern parser; 4-slot working memory | language comprehension |

## Results

Full tables, with sources for every baseline, are in [RESEARCH.md](RESEARCH.md#results).

### Penn Treebank, word level, test perplexity

All Mnemo rows run on one CPU core. Online learning is the analogue of dynamic
evaluation.

| Model | Memory | PPL |
|---|---|---|
| Kneser-Ney 5-gram (published / our reference) | | 141.2 / 141.19 |
| KN5 + cache | | 125.7 |
| RNNLM | | 124.7 |
| LSTM small (Zaremba 2014) | | 114.5 |
| **Mnemo, long-term memory frozen** | 138 MB | **125.4** |
| **Mnemo, online, tiny** | **35 MB** | **117.7** |
| **Mnemo, online, default** | 138 MB | **113.0** |
| **Mnemo, online, large** | 225 MB | **112.0** |

Training takes about 40 s on one core.

The rest of the results:
* **WikiText-2.** Kneser-Ney 5-gram, our reference: 220.1. Mnemo with
  long-term memory frozen: 165.5. Mnemo online: **107.5** at 203 MB, and
  **111.0** at 84 MB. A basic LSTM scores about 100.
* **Tiny Shakespeare (character level).** Mnemo frozen: **2.04 bits per
  character**. Mnemo online: **1.85**, and 1.87 at 25 MB. nanoGPT's
  10.65M-parameter GPU transformer: 2.12. bzip2: 2.36.
* **Ablation.** Removing the mental lexicon raises PTB perplexity from 113 to
  162 (online). The word-level expectations matter most.

A sample of imagination, after 46 s of reading 4 MB of 19th-century novels and
children's stories:

> **Once upon a time** there was a great deal to leave the box--he wished to her a bad heart,
> for with a most decided, man for the little the gentleman, and there he saw the sister of
> the bed ...

It is locally fluent and globally incoherent; see Limitations below.

## Repository layout

```text
src/            C engine: cortex.c (context mixing), lexicon.c (word model), main.c (CLI)
mnemo/          Python mind: cortex wrapper, language, knowledge, world, semantic, hdc, CLI
tests/          unit tests, bAbI-style story generator/evaluator
tools/          kn_reference.py (exact KN5 baseline)
scripts/        get_data.sh, benchmark.sh
RESEARCH.md     survey + design + results + limitations + roadmap
```

## Limitations

These are stated plainly here, and in more detail in RESEARCH.md:
* **Generated text** is locally fluent but globally incoherent, like any
  n-gram-family model.
* **Hand-written rules.** The parser understands a controlled subset of
  English, and its inference rules are designed by hand, not learned.
* **Knowledge.** It knows only what it has been told or has read.
* **Story-task scores.** These come from a self-generated suite in bAbI style.
  The official bAbI files were not reachable from this environment.
