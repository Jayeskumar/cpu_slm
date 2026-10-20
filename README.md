# cpu_slm: Mnemo, a tiny mind that runs on one CPU core

Mnemo is a small, CPU-only system that learns language and reasons in a few
tens of megabytes of memory. Its core contains no neural network: no gradient
descent and no GPU. Every core component is a classic idea from cognitive
science or data compression, built from counting, hashing, hypervectors and
logic, and each learns online from a single pass over its input. An optional
9.3M-parameter LSTM, trained on a CPU in plain numpy, can be mixed in to show
what weight-based learning adds.

* It **predicts text.** On Penn Treebank, with long-term memory frozen, it
  matches the classic Kneser-Ney + cache baseline (125.4 vs 125.7 perplexity).
  Learning online, it beats a small LSTM (113.0 vs 114.5), and it still
  reaches 117.7 in only 35 MB.
* **Counts plus a few weights** do better still. A 9.3M-parameter LSTM,
  trained on the CPU in numpy, scores 79.4 on Penn Treebank. Mixed with
  Mnemo's counts it reaches **69.5**. That beats a 66M-parameter GPU LSTM
  (78.4).
* It **can be fed general knowledge.** WordNet (82k concepts) and a curated
  pack of about 870 facts load in 6 s into a 23 MB brain (about 200 MB of RAM
  once loaded). It scored 89% on held-out general-knowledge questions the
  first time they were asked. After fixing what they exposed, it scored 100%
  on a second held-out set, asked once. Each answer takes about 0.2 ms and
  comes with an explanation:
  > *Is a whale a fish?* No. Whales are mammals; mammals and fish are
  > different kinds of vertebrate.
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
  19 bAbI-style story tasks. The stories come from our own generator, not the
  official set.
* It runs on one CPU core: about 40 s to learn the 5 MB Penn Treebank, and
  answers in well under a millisecond.

The full survey, design rationale, measurements and limitations are in
**[RESEARCH.md](RESEARCH.md)**.

## Quick start

```bash
make                       # builds build/mnemo (CLI) and build/libmnemo.so (C99, no deps)
python3 -m mnemo demo      # reasoning tour; needs only numpy
python3 -m mnemo chat      # talk to it: teach facts, tell stories, ask questions
python3 tests/run_tests.py # 34 unit tests (also runnable with pytest)
python3 tests/story_tasks.py 200   # bAbI-style story suite
```

To give it general knowledge (WordNet and the bundled pack):

```bash
scripts/get_data.sh                               # corpora, plus WordNet 3.0 and name lists
python3 -m mnemo feed --brain models/kbrain       # ~6 s, 108k facts, 23 MB
python3 -m mnemo chat --brain models/kbrain
python3 tests/knowledge_probe.py models/kbrain    # 159 general-knowledge questions
```

To give it language and word meanings, let it read:

```bash
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

## Feeding it general knowledge

`python3 -m mnemo feed` fills the declarative memory from three kinds of
source, and keeps the source and confidence of every fact.

* **WordNet 3.0**, from the NLTK data mirror:
  * 82,115 concepts with their definitions;
  * 84,427 is-a links;
  * 18,991 part, substance and place links;
  * opposites, plus capitals and currencies read from the definitions.
* **`knowledge/general.txt`**, about 870 facts in plain English:
  * animals with their defaults and exceptions;
  * plants, the body, matter and astronomy;
  * 154 countries with capitals, continents, currencies and languages.
* **Your own files**, in any of three formats:
  * plain English, one statement per line: `Penguins cannot fly.`;
  * TSV: `Rome<TAB>capital of<TAB>Italy[<TAB>0.95]`;
  * JSON lines: `{"s": "Lima", "r": "capital_of", "o": "Peru"}`.

```bash
python3 -m mnemo feed --brain models/kbrain my_facts.txt more.tsv   # adds to the brain
```

A dictionary is only useful if the reasoner gets its senses right. Each
WordNet meaning becomes its own concept: "gold" the coins is a different
concept from gold the element, and "China" the country from "china" the
porcelain. The sense of a word is chosen in three ways:
* **From what is already known.** "Chad is a country" means the republic, not
  the lake.
* **One sense per discourse.** Once chosen, a sense is kept for the rest of the
  conversation.
* **The usual sense first.** A clear "no" from it is not overridden by a rare
  sense.

A closed world over curated sibling categories is what lets it say "no"
instead of "I don't know".

```text
you> Is a bat a bird?
mnemo> No. Bats are mammals; mammals and birds are different kinds of vertebrate.
you> Is Tokyo in Europe?
mnemo> No. Tokyo is in Japan, which is in Asia, and Asia is not part of Europe.
you> How many legs does a beetle have?
mnemo> Six. Beetles are insects, and insects have six legs.
you> Who was Mozart?
mnemo> Mozart: prolific Austrian composer and child prodigy; master of the classical style in all its forms of his time (1756-1791). Mozart is a composer, a musician, an artist and a creator. Also called Wolfgang Amadeus Mozart.
you> Kiri is a kiwi.
mnemo> OK.
you> Can Kiri fly?
mnemo> No. Kiri is a kiwi, and kiwis cannot fly.
```

| General-knowledge probe (`tests/knowledge_probe.py`) | Correct |
|---|---|
| dev questions | 37/37 |
| held-out, first run (failures were then fixed) | 57/64 (89%) |
| second held-out set, written after the fixes, run once | 58/58 |

One caveat: these questions are in the phrasing the reader understands, about
what WordNet and the pack cover. The story tasks still score 100% with all
108k facts loaded underneath.

## Weights: a small LSTM next to the counts

`mnemo/neural.py` is a 2×512 word-level LSTM in plain numpy, with
hand-written backpropagation that is gradient-checked in the tests. It has
9.3M parameters, an 18.8 MB file in fp16. It trains on 3 CPU threads at about
7.5 minutes per PTB epoch.

```bash
python3 tools/train_lstm.py --train data/ptb/ptb.train.txt --valid data/ptb/ptb.valid.txt \
    --test data/ptb/ptb.test.txt --out models/lstm_ptb --hidden 512 --dropout 0.5 --epochs 25
./build/mnemo bench --train data/ptb/ptb.train.txt --test data/ptb/ptb.test.txt --spaced \
    --tb 22 --hb 24 --mb 22 --lex 4 --dump models/mnemo_ptb.test.online.f32   # same for valid
python3 tools/hybrid_eval.py --text data/ptb/ptb.test.txt --mnemo models/mnemo_ptb.test.online.f32 \
    --lstm models/lstm_ptb.test_logp.npy --valid-text data/ptb/ptb.valid.txt \
    --valid-mnemo models/mnemo_ptb.valid.online.f32 --valid-lstm models/lstm_ptb.valid_logp.npy
```

On Penn Treebank it reaches 79.4 test perplexity after 25 epochs (about 3
hours). Counting memories remember exactly and adapt within a document. The
LSTM generalises across similar contexts. Because their errors differ, a
simple word-level mixture beats both: **69.5** with Mnemo learning online,
and 72.8 with its long-term memory frozen (see the table below and
RESEARCH.md §12).

## Can it be better than Gemma 1?

**Not in general, and we will not pretend otherwise.** Gemma 1 2B has 2.51
billion weights (about 5 GB), trained on 3 trillion tokens on 512 TPUv5e
chips. On the
[Gemma report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)'s
benchmarks it scores:
* MMLU 42.3;
* HellaSwag 71.4;
* ARC-e 73.2;
* GSM8K 17.7;
* HumanEval 22.0.

Mnemo cannot attempt these: it reads a controlled subset of English, does no
arithmetic word problems and writes no code. Its training compute is 10⁷ to
10¹¹ times smaller.

Gemma's weights are also not used here. They are licence-gated, and Hugging
Face and Kaggle are unreachable from this environment. They would also make
the system 200× larger, and a transformer's knowledge cannot be lifted out in
pieces.

**What it is better at, by design and measurably:**
* **Size.** A 23 MB knowledge brain and a 138 MB language cortex.
* **Cost per answer.** About 0.2 ms on one core, against about 5 GFLOPs per
  generated token for a 2.5B-parameter model.
* **One-shot learning.** Tell it a fact once and it is used and saved at once.
* **Exact, explained reasoning** with a confidence, and "I don't know" when
  the evidence is missing.

**How to use a large model with it anyway:** as a teacher. Run Gemma, or any
LLM, on your own machine, and let Mnemo keep only what it can read:

```bash
ollama pull gemma:2b
python3 tools/teacher_pack.py --model gemma:2b --topic volcanoes --topic owls \
    --out knowledge/teacher.txt        # asks for simple statements, filters them
python3 -m mnemo feed --brain models/kbrain --source teacher --conf 0.7 knowledge/teacher.txt
```

The teacher's facts carry a lower confidence, so your own statements and
WordNet win when they disagree. The filter drops chatter, long sentences and
anything the reader cannot parse. The tool also works with any
OpenAI-compatible server (`--api openai`), or on saved output (`--from FILE`).
You can also generate text for the cortex to read, or for the LSTM to train on
(distillation).

RESEARCH.md §13 has the full comparison table.

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
        +----------------+-----------------+   (+ optional LSTM, mixed per word)
                         | surprise, fluency, plausibility, imagination
        +----------------v-----------------+
        |  LANGUAGE (mnemo/language.py)    |   sentences -> facts / events / questions
        |  WORKING MEMORY (4 entities)     |   pronouns ("she", "they"), word senses
        +---+--------------+-----------+---+
            |              |           |
   +--------v------+ +-----v------+ +--v-----------------+
   | DECLARATIVE   | | EPISODIC   | | SEMANTIC           |
   | facts, NARS   | | events in  | | PPMI + SVD (LSA),  |
   | truth <f,c>,  | | time order,| | concepts, analogy, |
   | ACT-R recall, | | world state| | hypervector codes  |
   | WordNet + pack| |            | |                    |
   +-------+-------+ +-----+------+ +--+-----------------+
           +---------------+-----------+
                           v
                REASONING + ANSWERS (mnemo/mind.py)
   deduction, induction, abduction, revision, transitivity, closed world,
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
| `mnemo/knowledge.py` | NARS truth values and inference; ACT-R base-level activation; closed world over curated taxonomies | declarative memory |
| `mnemo/feed.py` | WordNet, knowledge packs, fact files; one concept per word sense | schooling: learning from a dictionary and a textbook |
| `mnemo/world.py` | ordered event memory; learned motive associations | episodic memory |
| `mnemo/language.py` | pattern parser; 4-slot working memory | language comprehension |
| `mnemo/neural.py` | 2-layer LSTM in numpy, mixed with the counts (optional) | slow, generalising cortical learning |

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
| our LSTM, 2×512, numpy on CPU | 9.3M weights | 79.4 |
| **Mnemo frozen + our LSTM** | 138 MB + 9.3M weights | **72.8** |
| **Mnemo online + our LSTM** | 138 MB + 9.3M weights | **69.5** |

Training takes about 40 s on one core for Mnemo, and about 3 h on 3 cores for
the LSTM. Zaremba et al.'s GPU LSTMs score 82.7 (medium, 20M parameters) and
78.4 (large, 66M).

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
mnemo/          Python mind: cortex wrapper, language, knowledge, world, semantic, hdc, CLI,
                feed (WordNet / packs / fact files), neural (numpy LSTM)
knowledge/      general.txt, the curated knowledge pack
tests/          unit tests, bAbI-style story suite, general-knowledge probe
tools/          kn_reference.py (exact KN5), train_lstm.py, hybrid_eval.py, teacher_pack.py
scripts/        get_data.sh, benchmark.sh
RESEARCH.md     survey + design + results + limitations + roadmap
```

## Limitations

These are stated plainly here, and in more detail in RESEARCH.md:
* **Generated text** is locally fluent but globally incoherent, like any
  n-gram-family model.
* **Hand-written rules.** The parser understands a controlled subset of
  English, and its inference rules are designed by hand, not learned.
* **Knowledge.** It knows only what it has been told, has read, or was fed
  (WordNet and the pack): no events, dates or numbers beyond those.
* **Story-task scores.** These come from a self-generated suite in bAbI style.
  The official bAbI files were not reachable from this environment.
