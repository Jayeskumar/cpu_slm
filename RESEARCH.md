# A tiny mind without neural networks

Research notes for **cpu_slm / Mnemo**: how far can a CPU-only system, built from
counting, hashing, hypervectors and logic instead of gradient-trained neural
networks, go towards language and human-like intelligence in a small memory and
compute budget?

This document has three parts:

* **Part I: survey.** What is known about non-neural (or minimally neural)
  approaches to prediction, memory, meaning and reasoning, with numbers.
* **Part II: the system.** The design that came out of the survey, how it
  maps onto brain systems, what it measurably does, and what it cannot do.
* **Part III: knowledge and weights.** Feeding it general knowledge (WordNet,
  a curated pack, your own files), adding a small CPU-trained LSTM, an
  honest comparison with Gemma 1, understanding how people actually ask,
  becoming a generalist (definitions read into facts, open world data), and
  pairing with a language model.

Figures from the literature carry a source link. Three research agents
gathered them. Their web fetches were mostly blocked, so many figures come from
search-engine snippets, GitHub copies of primary sources, or the agents' own
background knowledge. Claims that could not be checked against a primary source
are marked *(unverified)*. Every result of our own was measured in this
repository, and scripts are given to reproduce it.

---

## Part I: survey

### 1. The guiding idea: intelligence as prediction, memory and inference

Three independent lines of work converge on the same picture:

* **Compression = prediction = intelligence.** A model that predicts the next
  symbol well can compress, and vice versa. The Hutter Prize uses compression
  of Wikipedia under tight compute limits as a proxy for intelligence. Its
  rules allow roughly 50–70 hours, one CPU core, under 10 GB of RAM and no GPU
  ([prize.hutter1.net](http://prize.hutter1.net/), as transcribed in
  [ai-discovery-data](https://github.com/tecunningham/ai-discovery-data/blob/main/problems/algorithms-enwik9/README.md)).
  DeepMind's *Language Modeling Is Compression*
  ([arXiv:2309.10668](https://arxiv.org/abs/2309.10668)) gives the converse:
  Chinchilla 70B compresses enwik9 to 8.3% of its size. Counting its
  parameters, though, the rate becomes 14,008%. Model size only pays off when
  there is a great deal of data.
* **The memory-prediction view of the cortex.** The neocortex constantly
  predicts its next input from stored sequences (Hawkins). In HTM sequence
  memory this works online with local learning
  ([Cui, Ahmad & Hawkins 2016](https://arxiv.org/abs/1512.05463)). In
  predictive-coding accounts, feedback carries predictions and feedforward
  carries errors ([Rao & Ballard 1999](https://www.cs.utexas.edu/~dana/nn.pdf)).
* **Intelligence under insufficient knowledge and resources (AIKR).** NARS
  defines intelligence as *adaptation with insufficient knowledge and
  resources*. A system that is finite, open and real-time must reason with
  evidence-weighted uncertain beliefs
  ([Wang](https://cis.temple.edu/~wangp/AGI-course/03-Validity.html)).

Humans do this cheaply. The brain runs on about 20 W
([Balasubramanian 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8364152)),
with only about 1% of cortical neurons active at a time (Lennie 2003). A child
hears at most about 10⁶ words a month, roughly 2×10⁸ words by age 20. LLMs
train on 10¹²–10¹³ tokens
([Frank 2023](https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/Frank2023_ChildrenLLMs.pdf)).
Working memory holds about 4 chunks
([Cowan 2001](https://memory.psych.missouri.edu/assets/doc/articles/2001/cowan-bbs-2001.pdf)).
Memory traces decay as a power law and strengthen with use. This is ACT-R's
base-level learning, B = ln Σ tⱼ^−0.5.

### 2. Non-neural sequence prediction: compressors and count-based LMs

#### 2.1 Context mixing (PAQ, lpaq, paq8, cmix)

These are the strongest non-neural sequence predictors. Our cortex is built on
the same mechanisms
([Mahoney, *Data Compression Explained*](https://github.com/mco-gh/notes/blob/main/articles/Data%20Compression%20Explained.md);
[paq8px](https://github.com/hxim/paq8px); [cmix](https://github.com/byronknoll/cmix)):

* **Bit-level prediction.** Each of the 8 bits of a byte is predicted in turn.
* **Hashed contexts.** Context tables use 64-byte buckets. Each nibble lookup
  returns 15 bit-history slots, guarded by an 8-bit checksum.
* **Nonstationary bit histories.** One byte holds the counts n0 and n1. When a
  bit is seen, a large count of the opposite bit is halved. A *StateMap* then
  learns, online, what probability each history actually means.
* **A match model.** It finds the last occurrence of the recent context and
  predicts what followed it.
* **Logistic mixing.** The output is p = σ(Σ wᵢ·logit pᵢ), with the local
  update wᵢ += λ(y−p)·logit pᵢ, and the weight set chosen by context.
* **APM/SSE.** Calibration stages refine the final probability.

For binary symbols, logistic mixing is exactly normalised *geometric* mixing,
which minimises KL divergence (Mattern, DCC 2012). Gated Linear Networks
generalise this to deep, backprop-free networks
([Veness et al.](https://arxiv.org/abs/1712.01897)).

Large Text Compression Benchmark figures, as bits per character (bpc) on enwik8
or enwik9, collected by the compression research agent:

| Program | enwik8 bpc | enwik9 bpc | Notes |
|---|---|---|---|
| gzip -9 | 2.916 | 2.581 | |
| bzip2 -9 | 2.321 | 2.032 | |
| xz -9e | 1.987 | 1.579 | |
| ppmd J | 1.711 | 1.472 | PPM |
| lpaq1 -9 | 1.580 | 1.316 | 7 models, 1 mixer, ~0.9 MB/s, 1.5 GB |
| ppmonstr J | 1.524 | 1.256 | PPM + SSE |
| paq8px v206 | — | 0.998 | general purpose |
| cmix v21 | 1.170 | 0.864 | hundreds of models + LSTM mixer, ~31 GB RAM |
| nncp v3.2 | 1.193 | 0.853 | transformer trained online (Bellard) |

**The Hutter Prize trajectory.** In 2019 phda9 compressed enwik9 to
116.7 MB. fx2-cmix (2024-09) reached 110.8 MB. In 2026 the record was broken by
**fx2-cmix-transformer at 100.4 MB (0.803 bpc)**
([kaitz/fx2-cmix](https://github.com/kaitz/fx2-cmix);
[cmix-lex-transformer write-up](https://github.com/Neel49/cmix-lex-transformer)).
It added a *pretrained* 6M-parameter int4 byte transformer, which still runs on
one CPU core.

Under strict compute limits, then:
* pure count-based context mixing (paq8px, 0.998 bpc on enwik9) is about 25%
  behind the record;
* cmix-family mixers with a small online LSTM are about 10% behind;
* a tiny *pretrained* neural column closed the last gap.

The winners' preprocessing orders the dictionary *semantically*: related words
get codes with common prefixes, so a model can merge them by ignoring the low
bits (DCE §5.3.2). This is a hand-made concept hierarchy, and it is the reason
we tried learned concept classes as contexts (§ 7.5, § 9).

#### 2.2 Count-based word language models

Penn Treebank word-level test perplexity (lower is better):

| Model | PPL | Source |
|---|---|---|
| Kneser-Ney 5-gram | **141.2** | [Mikolov & Zweig 2012](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rnn_ctxt.pdf) |
| KN5 + cache | **125.7** | same |
| Maximum-entropy 5-gram | 142.1 | [Mikolov et al. 2011](https://www.fit.vut.cz/person/imikolov/public/rnnlm/is2011_emp.pdf) |
| Random-forest LM | 131.9 | same |
| Feed-forward NNLM / log-bilinear | 140.2 / 144.5 | same |
| RNNLM | 124.7 | same |
| RNNLM + KN5 + cache | 97.5 | same |
| LSTM small / medium / large (Zaremba 2014) | 114.5 / 82.7 / 78.4 | arXiv:1409.2329 *(unverified)* |
| AWD-LSTM (+cache) | 57.3 (52.8) | *(unverified)* |
| Transformer-XL | 54.5 | [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl) |

Goodman (2001) combined caching, clustering, skipping, higher orders, KN
smoothing and sentence mixtures for a 38–50% perplexity reduction over a Katz
trigram ([cs/0108005](https://arxiv.org/abs/cs/0108005)).
**Infini-gram** backs off to the longest matching suffix in a suffix array over
5T tokens, at about 7 bytes per token, using CPU only. It predicts the next
token correctly 47% of the time against 29% for a 5-gram. Interpolated with
neural LMs, it cuts their perplexity by up to 73%
([arXiv:2401.17377](https://arxiv.org/abs/2401.17377)).

#### 2.3 Lessons for a tiny system

1. Count-based predictors learn in **one pass, online**. With little data they
   are hard to beat: Tiny Shakespeare in § 8.3 is an example.
2. Their weakness is **generalisation across similar contexts**: an n-gram
   model does not know that "dog" is like "cat". Class-based models, caches
   and mixing patch this partially.
3. A **match model is an induction head.** Transformers learn heads that
   attend to the token after the previous occurrence of the current token and
   copy it ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)). A
   longest-suffix-match pointer does exactly this, with no training.

### 3. Brain-inspired representations and memories

| Method | Learning rule | Memory | Compute | Strengths | Weaknesses | Language maturity |
|---|---|---|---|---|---|---|
| Hyperdimensional computing / VSA ([Kanerva 2009](https://redwood.berkeley.edu/wp-content/uploads/2018/01/kanerva2009hyperdimensional.pdf)) | bundle (add) encoded examples | 1.25 KB per 10k-bit vector | XOR/popcount or multiply, O(D) | one-shot, compositional, robust | capacity linear in D; hand-made encoders | language ID 97.4% over 21 languages ([Joshi et al. 2016](https://redwood.berkeley.edu/wp-content/uploads/2018/12/joshi2016language.pdf)); no competitive LM |
| Random Indexing / PPMI+SVD | counting + projection | V·d floats | one pass + randomized SVD | equals word2vec when tuned ([Levy, Goldberg & Dagan 2015](https://aclanthology.org/Q15-1016/)) | type-level, no generation | TOEFL: RI 72.5%, LSA 64.4%, humans 64.5% ([ACL wiki](https://aclweb.org/aclwiki/TOEFL_Synonym_Questions_(State_of_the_art))) |
| Sparse Distributed Memory (Kanerva 1988) | counters within a Hamming radius | M·N counters | O(MN) per read, done naively | content-addressable; approximates attention ([Bricken & Pehlevan 2021](https://arxiv.org/abs/2111.05498)) | assumes random addresses | none directly |
| HTM / Thousand Brains / Monty ([tbp.monty](https://github.com/thousandbrainsproject/tbp.monty)) | Hebbian permanence, voting | sparse | sparse | online, noise-robust; Monty 98.6% on 77 YCB objects with ~34,000× fewer FLOPs than a ViT ([arXiv:2507.04494](https://arxiv.org/abs/2507.04494)) | not competitive on standard benchmarks | none |
| Tsetlin machines ([Granmo 2018](https://arxiv.org/abs/1804.01508)) | automaton reward/penalty | bits | AND + popcount | interpretable; 16.6 µW keyword spotting ([TsetlinKWS](https://arxiv.org/abs/2510.24282)) | slow for large vocabularies | IMDb ~90.6%; no LM |
| Weightless NNs (WiSARD, [ULEEN](https://arxiv.org/abs/2304.10618), [DWN](https://arxiv.org/abs/2410.11112)) | write-once lookup tables | classes·k·2ⁿ bits | table lookups | fastest at the edge | poor generalisation at scale | hobby LM: WikiText-2 ppl ~9,400 |
| Modern Hopfield / dense associative memory ([Ramsauer 2020](https://arxiv.org/abs/2008.02217)) | store patterns | K·d | O(Kd) | exponential capacity; its update *is* attention | cost grows with K | underlies attention |
| Complementary Learning Systems ([McClelland et al. 1995](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)) | fast episodic store + slow statistics | grows with experience | retrieval | solves catastrophic forgetting | retrieval cost | kNN-LM, RETRO, HippoRAG |
| Predictive coding / active inference | local error-driven updates | model size | iterative | principled, sample-efficient | scaling is the open problem ([PCX](https://arxiv.org/abs/2407.01163)) | none |
| Forward-Forward, Hebbian, spiking | layer-local rules | like an ANN | like an ANN | biologically plausible | large accuracy gap; spiking LMs are trained with backprop ([SpikeGPT](https://arxiv.org/abs/2302.13939)) | none without backprop |

Two results matter most here:

* **Count-based word meaning is enough.** Skip-gram with negative sampling
  implicitly factorises a shifted PMI matrix
  ([Levy & Goldberg 2014](https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)).
  With the same hyperparameters (context-distribution smoothing 0.75,
  eigenvalue weighting), PPMI+SVD matches it on most similarity tasks. Latent
  Semantic Analysis was proposed as a model of how children acquire vocabulary
  (Landauer & Dumais 1997).
* **HDC gives structure for free.** With binding (elementwise product),
  bundling (sum + sign) and permutation, records and sequences live in one
  fixed-width vector. Kanerva's *"What is the dollar of Mexico?"* is answered
  by mapping the USA record onto the Mexico record and decoding: binding
  `dollar` with that mapping gives a vector closest to `peso`.

**No backprop-free method has produced a language model near GPT-2 small.** The
practical non-neural route to language is count-based prediction plus
symbolic or vector structure on top.

### 4. Cognitive architectures and reasoning

| System | What it contributes | Status |
|---|---|---|
| ACT-R ([pyactr](https://github.com/jakdot/pyactr)) | Activation A = B + Σ W·S + noise; base-level B = ln Σ t⁻ᵈ; retrieval probability 1/(1+e^−(A−τ)/s); latency F·e^−A | fits human data; models are hand-built |
| Soar 9.6.5 (2026) ([SoarGroup](https://github.com/SoarGroup/Soar)) | working memory, chunking, semantic and episodic memory | knowledge engineered by hand |
| LIDA / Global Workspace, CLARION, Sigma | conscious broadcast, implicit/explicit levels, factor graphs | research prototypes |
| **NARS / ONA** ([OpenNARS for Applications](https://github.com/opennars/OpenNARS-for-Applications)) | truth ⟨f, c⟩ with c = w/(w+1); deduction, induction, abduction and revision truth functions; written in C with statically bounded memory | 51/51 QA tests; Pong, Cartpole |
| OpenCog Hyperon / MeTTa | probabilistic logic networks | pre-1.0 in 2026 |
| Bayesian Program Learning ([Lake et al. 2015](https://cims.nyu.edu/~brenden/papers/LakeEtAl2015Science.pdf)) | one-shot concept learning | Omniglot: 3.3% error vs 4.5% for humans |

**ARC-AGI** results show where symbolic search stands:

* Pure DSL search (icecuber, C++) reaches about 20% on ARC-AGI-1 but only about
  1.6% on ARC-AGI-2 ([ARC-AGI-2 paper](https://arxiv.org/pdf/2505.11831)).
* Tiny recursive networks score higher: TRM (~7M parameters) gets 45% / 8%.
  They are small at inference but need GPU-days of training
  ([arcprize.org](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
* ARC-AGI-3 (March 2026) is interactive, and frontier AI scored under 1% at
  launch ([arXiv:2603.24621](https://arxiv.org/abs/2603.24621)).

**bAbI** ([Weston et al. 2015](https://arxiv.org/abs/1502.05698)):

* Memory networks needed supervision of the supporting facts.
* Symbolic answer-set programming plus inductive logic programming solves
  nearly every task ([Mitra & Baral 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10354)).
* Given a parser, most tasks reduce to bookkeeping, transitive closure, breadth-first search or
  induction.

### 5. Calibration: how small is "small" for neural models?

* TinyStories models under 10M parameters write fluent children's stories
  ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)).
* llama2.c's 15M model runs at about 110 tokens/s on an M1
  ([karpathy/llama2.c](https://github.com/karpathy/llama2.c)).
* BitNet b1.58 2B4T needs 0.4 GB and 29 ms per token on CPU
  ([report](https://arxiv.org/pdf/2504.12285)).
* nanoGPT's character-level Shakespeare model (10.65M parameters, about 3
  minutes on an A100) reaches 1.4697 nats = **2.12 bits per character**. The
  CPU configuration of the same recipe reaches 2.71 bpc, which is worse than
  bzip2 ([nanoGPT](https://github.com/karpathy/nanoGPT)).

### 6. Design conclusions drawn from the survey

1. **Prediction:** context mixing of many count-based memories, learned online
   in one pass. This is the strongest, best-understood non-neural predictor.
2. **Word-level expectation:** an interpolated modified Kneser-Ney model over
   whole words. This addresses the classic weakness of byte models on
   word-level structure.
3. **Recency:** a priming memory that approximates ACT-R base-level decay. The
   "cache" effect alone is worth 11% perplexity on PTB (KN5 → KN5+cache).
4. **Episodic recall:** a match model, i.e. an induction head without
   training.
5. **Meaning:** PPMI+SVD distributional semantics, plus hypervector codes for
   structure and analogy.
6. **Reasoning:** NARS-style evidence-weighted truth, so that defaults,
   exceptions, induction and revision come from one calculus.
7. **Events:** a time-ordered episodic store (the hippocampus of
   Complementary Learning Systems), separate from slow statistics.
8. **Working memory** of about 4 entities, used for reference resolution.

---

## Part II: the system (Mnemo)

### 7. Architecture

```text
 text --> [ PREDICTIVE CORTEX ] --surprise/plausibility/imagination--> [ MIND ]
           |  byte columns (orders 1-12)      gates pick whose vote counts      |
           |  word columns (current word,     calibration (APM/SSE)             |
           |   previous 1-3 words, skips)                                       |
           |  concept columns (optional)                                        |
           |  MENTAL LEXICON: online modified Kneser-Ney over whole tokens,     |
           |   read out letter by letter through a vocabulary trie              |
           |  PRIMING: fast and slow recency caches (power-law-like decay)      |
           |  EPISODIC RECALL: longest previous match -> its continuation       |
           v                                                                    v
   language.py: parse -> facts | events | questions ; working memory (4) for pronouns
   knowledge.py: facts with NARS truth <f,c>; deduction / induction / abduction /
                 revision / choice; transitivity, inverse & asymmetric relations;
                 ACT-R activation for recall
   world.py:     time-ordered episodes -> locations, holders, transfers, motives, routes
   semantic.py:  PPMI + randomized SVD meaning vectors, concepts, analogies, hypervectors
   hdc.py:       bind / bundle / permute / cleanup; record-to-record analogy
```

**One pass, local updates, fixed memory.** Every structure is updated
incrementally by the input stream. The cortex uses hashed counts and
mixer-weight nudges, the lexicon counts n-grams, the caches decay, and facts
are revised. Nothing is iterated to convergence and nothing is backpropagated.
The only batch step is the semantic memory, an SVD over co-occurrence counts.
We run it as a "sleep" consolidation pass.

#### 7.1 The predictive cortex (`src/cortex.c`)

Bytes are predicted bit by bit, MSB first. For each byte, 12 context hashes
are computed (16 if concept classes are on):

* the last 2, 3, 4, 5, 6, 8 and 12 bytes;
* the current word prefix (or the separator run before it);
* that prefix combined with 1, 2 or 3 previous words, or with the word two back;
* optionally, the concept classes of the previous words.

**Memory.** Each context owns 16-byte buckets in one shared table, one bucket per
nibble (an 8-bit checksum and 15 bit-history slots, organised as 4-way
associative cache lines). Lookups for all columns are issued together with
prefetch, and the tables use transparent huge pages. A bit history is one byte
holding two saturating counts (n0, n1). The count of the opposite bit is
roughly halved when it exceeds 2. A per-column adaptive map turns each history
into a probability. These choices are our own, simpler than PAQ's 253-state
machine.

**Inputs to the vote.** Each column contributes logit(p) and a "deterministic"
copy, used when only one bit value has ever been seen. Further inputs are:

* direct order-0 and order-1 statistics;
* the episodic recall model (a hash of the last 5 and 12 bytes points to the
  previous occurrence; the expected bit is weighted by match length);
* two inputs from the lexicon;
* a bias.

**Pooling.** Five gates, each a weight vector chosen by a different situation,
combine the votes in the log-odds domain:

* the partial byte;
* the match state;
* the previous byte;
* familiarity, i.e. the highest order with a known context;
* the lexicon state.

A final gate takes the five outputs plus a direct lexicon vote. Its weight set
is chosen by the lexicon state and the bit position. Its result is averaged
50/50 with three adaptive probability maps, which calibrate it.

All weights follow the local delta rule w += λ(y−p)x. Each gate learns from
its own error. This is online gradient descent on code length for a
single-layer geometric mixture, a learning rule for trust that is local to each
gate. Nothing is backpropagated through layers.

**Correctness check.** `mnemo compress` drives an arithmetic coder with exactly
these probabilities, and `decompress` reproduces the input byte for byte
(tested). Every bits-per-character figure is therefore a real code length,
with no future information leaking in.

#### 7.2 The mental lexicon (`src/lexicon.c`)

**Problem.** The first cortex scored 219 perplexity on PTB with its long-term
memory frozen. KN5 scores 141. Word statistics spread over per-letter hash
slots are both memory-hungry and badly smoothed.

**Fix.** The lexicon is the classic answer, done online and
read out per letter:

* **Tokens.** The stream is cut into tokens: runs of letters, of punctuation
  or of line breaks. Pre-tokenized corpora use whitespace-separated runs
  instead (`--spaced`). Each token is stored once in a trie.
* **Counts.** Interpolated *modified* Kneser-Ney counts are kept online. The
  top order holds raw counts and lower orders hold continuation counts. A
  cascade updates a lower order only when the higher-order n-gram is new. The
  three discounts per order are re-estimated from count-of-counts every 512
  tokens.
* **Readout.** The next-token distribution is never materialised. It is a
  sparse part (discounted counts of candidates seen in the current contexts)
  plus a dense part (the back-off weight times the unigram continuation
  distribution). The dense mass under any spelling prefix is a subtree sum
  stored in the trie. As letters arrive, the candidate list narrows, and the
  masses of the children of the current trie node give the next-byte
  distribution. A spacing model handles token ends.
* **Storage.** Contexts live in an open-addressing table keyed by 32-bit
  fingerprints. A context with one follower stores it inline; otherwise the
  followers are a sorted array searched by binary search. Compacting this way
  cut memory roughly threefold with identical results (PTB: 256 → 161 MB;
  WikiText-2: 584 → 203 MB).

**Validation.** `tools/kn_reference.py` is an independent batch implementation
of interpolated modified KN. It reproduces the published PTB KN5 perplexity
exactly: **141.19** against 141.2. In whitespace mode, the lexicon alone
scores 1.3108 bpc on the PTB test set, which is **142.5 perplexity**. The 1%
gap is the ε-uniform floor it keeps for robustness. Per-token comparison
against the reference located the one remaining discrepancy, tokens with
internal punctuation such as `corp.` and `500-stock`, which motivated the
whitespace mode.

#### 7.3 Priming

Two caches with decay rates 1/64 and 1/1024 per token approximate a
power-law forgetting curve as a mixture of exponentials, as ACT-R base-level
learning does. Their weights are learned by online EM. They are stored as
scaled subtree sums in the same trie, so a prefix's cache mass costs O(1).

The effect was large. These are development measurements, taken before the
final-gate changes; the final numbers are in § 8:

| PTB | Before priming | With priming |
|---|---|---|
| Long-term memory frozen | 148 | 128 |
| Online | 119.5 | 113 |

This mirrors the published KN5 → KN5+cache improvement (141.2 → 125.7).

#### 7.4 Imagination: snapshot and restore

`mn_snapshot` captures the short-term state:
* the stream registers;
* the lexicon's history, candidates and caches;
* an undo log of the recall-index and history-buffer writes.

It also freezes learning. `mn_restore` rewinds all of this. The mind can
therefore evaluate hypothetical continuations (`Cortex.surprise`,
`Cortex.choose`) and generate text "in its head" without leaving a trace.
Tests check that the restore is bit-exact.

#### 7.5 Semantic memory (`mnemo/semantic.py`)

* **Counts.** Co-occurrence counts use a ±4-word window with harmonic weights,
  over the 12,000 most frequent words and 3,000 context words.
* **Weighting.** Counts are converted to PPMI with the context distribution
  smoothed by exponent 0.75.
* **Factorisation.** A randomized truncated SVD (two power iterations,
  Halko–Martinsson–Tropp) keeps 200 dimensions with √S eigenvalue weighting.
* **Uses.** Cosine similarity, 3CosMul analogies, spherical k-means concepts,
  and sign-random-projection hypervectors.

WikiText-2 (2M words) takes 14 s.

#### 7.6 Declarative memory and reasoning (`mnemo/knowledge.py`)

Facts are (subject, relation, object) triples with NARS truth ⟨f, c⟩. Told
facts get c = 0.9, read facts c = 0.6. Repeating a fact revises it (w adds up).

**Exceptions without exception machinery.** A query collects candidate
conclusions from every applicable rule and picks the most confident (the
NARS choice rule). This is why "Opus cannot fly" (direct and specific,
c = 0.81) beats "Opus can fly" (inherited through one more link,
c = 0.73).

The rules:
* direct facts;
* inverse relations;
* transitive chains (isa, in, comparatives, directions);
* property inheritance down isa chains (deduction, carrying polarity);
* induction from instances, revised together;
* abduction;
* asymmetry: "the box is bigger" ⇒ "the chocolate is not bigger";
* innate opposites: "elephants are big" ⇒ "elephants are not small";
* semantic analogy: similar words share properties, at low confidence.

Retrieval order uses ACT-R base-level activation.

#### 7.7 Episodic memory and the mind (`mnemo/world.py`, `mnemo/mind.py`)

**Episodes.** Events are stored in time order. Positions are derived by replay:
objects follow their holder, and drops leave things where the holder was.
Coverage includes:
* negation ("no longer in") and disjunction ("either … or" → "maybe");
* routes, found by breadth-first search over learned directions;
* a 2-D mental map for left/right/above/below;
* motives, counted as state → destination associations.

**Analogy.** For the hyperdimensional query, a record is built per entity
(relation ⊗ filler), and a source record is mapped onto a target record, as in
Kanerva 2010.

**Language and explanation.** The parser covers a controlled subset of
English. Every answer keeps its premises, so "Why?" replays the chain with its
confidence.

<a id="results"></a>
### 8. Results

All runs were measured in this repository on a 4-core Intel Xeon at 2.8 GHz
(cloud container). Each run uses **one core**, so four ran in parallel.
`scripts/benchmark.sh` reproduces every Mnemo number; the raw logs are in
`results/`.

**Two evaluation modes are reported:**

* **online** is the model's natural mode. It keeps learning while it reads the
  test text, exactly as when it compresses. This corresponds to "dynamic
  evaluation" for neural LMs, which also helps them, so it is not
  like-for-like with static neural results.
* **frozen** keeps all long-term memory fixed: context tables, lexicon counts,
  gate weights, calibration. Short-term memory still works: the recall index,
  the priming caches and the current context. That is the analogue of a
  transformer using its context window. With caches, the comparable classic
  baseline is KN5 + cache.

Word perplexity for byte models is 2^(total bits / number of words), with one
end-of-sentence token per line (82,430 words in the PTB test set and 245,569 in
WikiText-2). Since a byte model also spends probability on non-words, this
conversion favours word-level models.

#### 8.1 Penn Treebank (word level, test)

| Model | Memory | Train time | PPL | Source |
|---|---|---|---|---|
| Kneser-Ney 5-gram | — | — | 141.2 | published |
| KN5, our reference (`tools/kn_reference.py`) | — | — | 141.19 | measured |
| KN5 + cache | — | — | 125.7 | published |
| RNNLM | — | — | 124.7 | published |
| LSTM small / medium (Zaremba et al. 2014) | — | GPU | 114.5 / 82.7 | published *(unverified)* |
| **Mnemo, frozen** (default) | 138 MB | 39 s | **125.4** | measured |
| **Mnemo, online, tiny** | **35 MB** | 37 s | **117.7** | measured |
| **Mnemo, online, small** | 90 MB | 39 s | 113.3 | measured |
| **Mnemo, online, default** | 138 MB | 39 s | **113.0** | measured |
| **Mnemo, online, large** | 225 MB | 40 s | **112.0** | measured |

Ablations (default memory):

| Variant | Online | Frozen |
|---|---|---|
| Full model | 113.0 | 125.4 |
| Cortex without the lexicon | 161.7 | 204.9 |
| Lexicon alone (its own predictions, measured inside the run) | 121.6 | 128.4 |

**Reading the tables:**
* **Frozen.** The full system matches KN5 + cache (125.4 vs 125.7).
* **Online.** It is better than every classic count-based result here and
  ahead of a small LSTM. It is still behind medium LSTMs and every modern
  neural model.
* **The lexicon** is the single most important component: it removes 30% of
  the perplexity.
* **The byte-level cortex** adds about 7% on top of the lexicon online. It
  helps only slightly when frozen, because its nonstationary bit histories
  are built for adaptation.

#### 8.2 WikiText-2 (word level, test)

| Model | Memory | Train time | PPL | Source |
|---|---|---|---|---|
| KN5, our reference (sentence-reset) | — | — | 220.1 | measured |
| LSTM baselines | — | GPU | ≈ 99–101 | Merity et al. 2016; Grave et al. 2016 *(unverified)* |
| AWD-LSTM | — | GPU | 65.8 | *(unverified)* |
| **Mnemo, frozen** (default) | 182 MB | 112 s | **165.5** | measured |
| **Mnemo, online, tiny** | 84 MB | 105 s | **111.0** | measured |
| **Mnemo, online, small** | 163 MB | 111 s | 107.7 | measured |
| **Mnemo, online, default** | 203 MB | 112 s | **107.5** | measured |
| Lexicon alone (online / frozen) | | | 140.5 / 166.5 | measured |

WikiText-2 articles repeat their entities, so priming and episodic recall are
worth a great deal on it: frozen 165 against online 107.5.

#### 8.3 Tiny Shakespeare (character level, the 90/10 split used by nanoGPT)

| Model | Validation bits/char | Source |
|---|---|---|
| gzip -9 (validation given the training text) | 3.10 | measured by research agent |
| bzip2 -9 | ≈ 2.40 | same |
| xz -9e | 2.52 | same |
| nanoGPT, CPU configuration (4 layers, d = 128, ~3 min) | 2.71 | published |
| nanoGPT, GPU (6 layers, d = 384, 10.65M params, ~3 min on an A100) | 2.12 | published |
| **Mnemo cortex without lexicon, frozen** | **2.09** | measured, 7 s training |
| **Mnemo, frozen** | **2.04** | measured |
| **Mnemo, online, tiny** (25 MB) | **1.87** | measured |
| **Mnemo, online, default** | **1.85** | measured |

On a 1 MB corpus a one-pass count-based learner on one CPU core
**beats a 10.65M-parameter transformer trained on a GPU**, and does so even
with learning frozen. With little data, sample efficiency wins.

#### 8.4 Lossless compression (whole file, from scratch)

The probabilities are honest: arithmetic coding with them round-trips.

| File | gzip -9 | bzip2 -9 | xz -9e | **Mnemo** |
|---|---|---|---|---|
| Tiny Shakespeare (1,115,394 B) | 3.110 | 2.356 | 2.614 | **1.942** (270,808 B) |
| PTB train (5,101,618 B) | 2.647 | 1.879 | 1.868 | **1.341** (855,016 B) |

#### 8.5 Reasoning and memory

* **Story suite.** `tests/story_tasks.py` covers 19 bAbI-style task types with
  self-generated stories. It ran 200 stories per type, **3,800 stories in
  2.9 s, about 0.8 ms each**, and scored 100% on every type:
  * single, two and three supporting facts;
  * two- and three-argument relations;
  * yes/no, counting, lists, negation and indefinite knowledge;
  * coreference, conjunction and compound coreference;
  * deduction and induction;
  * positional and size reasoning;
  * path finding and motivations.

  The caveats from § 9 apply.
* **Latency.** A typical question takes 0.05–0.6 ms in Python.
* **Footprint.** Without the cortex, the whole reasoning mind is a few hundred
  KB of Python objects.
* **Semantic memory.** It is built from 3.4M words (WikiText-2 plus four
  Gutenberg books) in **15 s**, and takes 9 MB as float32 (12,000 × 200).
  Examples:

  | Query | Result |
  |---|---|
  | king | henry, ii, son, iii, successor, george, edward |
  | happy | glad, pleased, fortunate, agreeable, delighted |
  | river | missouri, mississippi, valley, lake, platte |
  | emma | harriet, knightley, weston, elton, woodhouse (characters of *Emma*, learned by reading) |
  | alice | gryphon, hatter, dormouse, turtle |
  | france : paris :: germany : ? | berlin |
  | he : his :: she : ? | her |
  | england : english :: france : ? | french |
  | january : february :: march : ? | april |
  | man : woman :: king : ? | henry (fails) |
  | big : bigger :: small : ? | fails |

#### 8.6 Imagination: generation samples

The cortex was trained for 46 s on 4.2 MB of Austen, Chesterton, Carroll,
Bryant, Burgess and Edgeworth, with a 101 MB cortex. Sampling ran at
temperature 0.5. These samples are not cherry-picked: the first sample for
each prompt.

> **Once upon a time** there was a great deal to leave the box--he wished to
> her a bad heart, for with a most decided, man for the little the gentleman,
> and there he saw the sister of the bed, with more than to has not made a
> little more harm her brother's scotcher and of nothing else to do anything
> dangerous to the land

> **The little rabbit** for a moment with the station, by our long expected
> all that was any news for you in the former, and the maid, who was the more
> than half and the guinea is honestly blanket, and he had a little boy and
> girl went with their basket ...

The texture is right: period vocabulary, dialogue punctuation, phrase-level
syntax. Global coherence is absent, as § 9 explains.

### 9. What this is not: honest limitations

* **Not a general conversational AI.** It knows what it has been told, has
  read, or was fed (WordNet and the knowledge pack, Part III), and nothing
  more. It has no open-domain knowledge of events, people or numbers beyond
  those sources.
* **Generation is locally fluent and globally incoherent.** The cortex
  reproduces the style, vocabulary and short-range syntax of what it read. It
  keeps no plan or topic beyond a few sentences, like every model in the
  n-gram family. Small neural LMs (TinyStories, 1–30M parameters) remain far
  better at coherent long text.
* **The reader and the inference rules are hand-written.** The parser covers a
  controlled subset of English, and paraphrases outside its patterns are not
  understood. The rule set (inheritance, transitivity, induction, routes,
  mental map) is designed, not learned. The facts, categories,
  generalisations, motives, word meanings and language statistics are what
  the system learns.
* **The story tasks are self-generated.** We could not obtain the official
  bAbI files here. Our 19 task types copy bAbI's structure, but our templates
  are easier, with fixed phrasing and no noise. The 100% figure therefore
  shows that the episodic memory and reasoner cover these skills cheaply. It
  is not comparable to learned bAbI results.
* **Speed.** The cortex learns at about 0.1–0.25 MB/s on one core: 40 s for
  PTB, 2 min for WikiText-2. The 5-gate mixer and the lexicon readout
  dominate, and lpaq-class engines are about 5× faster. Inference is cheap,
  but generation runs byte by byte.
* **Frozen mode is its weaker mode.** The cortex is designed as a lifelong
  learner, and with long-term memory frozen it relies mostly on its lexicon.
* **Memory grows with experience.** The lexicon grows with the number of
  distinct n-grams, about 15 bytes each. It has no forgetting or pruning yet.
  The cortex tables have a fixed size and evict under pressure.
* **Concept classes did not help prediction much.** Using the semantic
  clusters as contexts gained about 1% on WikiText-2 and nothing on PTB. A
  class-based Kneser-Ney component would be the principled next step.

### 10. Roadmap

1. **A class-based lexicon:** Kneser-Ney over concept sequences combined with
   P(word | concept). This brings semantic generalisation into prediction.
2. **Learning to read:** induce parse patterns from (sentence, meaning)
   examples, so that pattern generalisation replaces the hand-written regexes.
3. **Learning inference rules:** treat rules as hypotheses with NARS truth
   values, revised by experience, as ONA does for sensorimotor rules.
4. **Sleep:** periodic consolidation. This means rebuilding the semantic
   memory, inducing generalisations from episodes, and forgetting by ACT-R
   activation threshold. It bounds memory and mirrors Complementary Learning
   Systems.
5. **Speed:** a fast preset with 2 gates and 8 columns (lpaq-class speed), and
   SIMD readout of the lexicon.
6. **Scale the reading:** BabyLM-sized corpora (10M/100M words). Evaluate on
   BLiMP-style minimal pairs with `Cortex.choose`, which needs no training
   change.
7. **Official benchmarks:** bAbI and enwik8, when the files are reachable.
8. **A tighter hybrid** (a first version is in Part III): feed the LSTM's
   word distribution into the cortex as one more mixer input, so that the
   gates learn when to trust it, as the 2026 Hutter Prize entries do with a
   tiny pretrained network.
9. **More knowledge:** ConceptNet-style common-sense relations (used-for,
   capable-of, located-at), and Wikidata facts with dates and numbers.

---

## Part III: feeding it knowledge, and adding weights

Two follow-up questions shaped this part: can the system be fed general facts
in bulk, and can it use weight-based learning (for example Gemma 1's weights)
while staying small, and be better than Gemma 1?

### 11. General knowledge without a neural network

#### 11.1 Sources

`python3 -m mnemo feed` fills the declarative memory from three kinds of source
(`mnemo/feed.py`):

* **WordNet 3.0** (Miller 1995). Every noun synset becomes a concept:
  * 82,115 concepts with 82,115 definitions;
  * 84,427 is-a links (hypernyms and instances);
  * 18,991 part, substance and place links: "a bird has wings", "water is made
    of hydrogen", "Paris is in France";
  * 4,024 adjective opposites;
  * 168 capitals and 156 currencies, read from definitions with clause-initial
    patterns, so "a *former* capital of China" and "the *crime* capital of
    Italy" do not count.
* **A curated knowledge pack.** `knowledge/general.txt` holds 567 lines of the
  same controlled English the mind reads in conversation, which yield about 870
  facts. It covers:
  * animals with their defaults and exceptions;
  * plants, food, the body, matter and astronomy;
  * 154 countries with capitals and continents;
  * 46 currency and 57 language statements.
  Pack facts replace WordNet's older capitals and currencies (WordNet dates from
  2006): Naypyidaw, Astana, Gitega, Kyiv.
* **Your own files**, in any of three formats:
  * plain English, one statement per line;
  * TSV triples: `subject<TAB>relation<TAB>object[<TAB>confidence]`;
  * JSON lines: `{"s": .., "r": .., "o": .., "c": ..}`.

The fed brain has 108,009 facts about 85,150 concepts. It takes 23 MB on disk
and about 200 MB of RAM once loaded, and takes 5–8 s to build. (§15 later adds facts
read from the definitions and open world data: 200k facts, 34 MB.)

#### 11.2 Word senses: the hard part

A dictionary is useless to a reasoner that confuses its senses. A plain lemma
import makes several such mistakes:
* "whale" is first a *very large person*;
* "metal" has no entry of its own: it is a synonym under *metallic element*;
* gold is first *coins*;
* "yen" is first a *yearning*;
* "China" is the country, but "china" is porcelain;
* "Chad" is first Lake Chad;
* one capitalised sense of "Guinea" is an ethnic slur.

The importer and the mind handle this in six ways.

* **One concept per synset.** A concept is keyed by its first word when it is
  that word's dominant sense (`dog`), and by `word#k` otherwise (`gold#3`, the
  element). Every word keeps its list of senses. Synonyms map to the dominant
  concept (`metal` → `metallic element`).
* **Case matters.** A capitalised word prefers a proper-noun sense ('China'),
  a lower-case one a common-noun sense ('china'). Acronym senses never capture
  ordinary words ('SALT' is not 'Salt').
* **Exceptions to "first sense dominant".** An animal sense beats a person
  or food sense ('whale', 'chicken'). A common first name gets no default
  person sense, so a story's Mary is not the Virgin Mary. A sense whose
  definition starts "a former ..." is demoted: 'Russia' is the federation, not
  the Soviet Union.
* **Sense choice from what is known.** For a new statement, the system tries
  sense pairs of subject and object, most frequent first, and picks the first
  pair that fits what it already knows. "Chad is a country" picks the
  republic. "The currency of Japan is the yen" picks the money, which WordNet
  already ties to Japan. "Elephants have trunks" picks the proboscis. This is a
  small, knowledge-based relative of Lesk's algorithm.
* **One sense per discourse** (Gale, Church & Yarowsky 1992). A sense chosen
  once is *primed* and stays chosen. The pack therefore states categories first
  ("Dolphins are mammals.", "Kiwis are birds.", "Trees are plants."). Later
  lines then get the right sense: "kiwis cannot fly" must not land on the
  fruit. A negated is-a corrects the sense that would otherwise hold: "Pluto
  is not a planet" is about the planet, not the god.
* **The speaker's word is kept for display.** WordNet files countries under
  the synset 'state, nation, country, …'. Once the pack has said "country",
  the answers say "country" too.

Questions read the same way. If the usual sense gives no answer, other sense
pairs are tried ("Is gold a metal?" → "Yes, in the sense of a soft yellow
malleable ductile … metallic element"). A *confident* "no" from the usual sense
is never overridden by a rare one. A shark is not a mammal, whatever a "shark"
of a person may be.

#### 11.3 A closed world where the taxonomy is curated

In an open world, "Is a whale a fish?" can only be answered "I don't know".
People answer "No, it's a mammal", because they treat sibling categories as
disjoint. Mnemo does the same, but only where the taxonomy comes from a curated
source.

* **Different branches.** If s and o sit on different branches under a common
  parent that o is close to, the answer is "No. Whales are mammals; mammals and
  fish are different kinds of vertebrate."
* **Confidence.** Biological taxa (WordNet's animal and plant files) are
  disjoint by construction, so the confidence is 0.7 ("No"). Other WordNet
  siblings often overlap. "Chemical" and "mineral" are an example, so there the
  confidence is 0.45 ("Probably not").
* **Places.** Continents and countries do not overlap: "Is Tokyo in Europe?" →
  "No. Tokyo is in Japan, which is in Asia, and Asia is not part of Europe."
* **Exceptions.** A told exception beats the inherited chain: "Pluto is not a
  planet" beats "superior planets are planets".

#### 11.4 Choosing between conclusions

The NARS choice rule picks the more confident conclusion. On top of it, three
preferences apply in order, following Touretzky's inferential distance.
1. What is said about the thing itself wins.
2. Otherwise, what is stated about a kind beats a guess from examples: "mammals
   cannot fly" beats "judging from bats, placentals can fly".
3. Otherwise, the most specific category with an opinion wins.

Induction from a single example is allowed when the user gave the example: "Lily
is a swan; Lily is white" is a deliberate hint. Reference data needs two
examples: "salt is made of chlorine" says nothing about other compounds.

#### 11.5 Results

`tests/knowledge_probe.py` asks a fed brain 159 questions:
* is-a, with yes and no answers;
* capitals, places, currencies and languages;
* parts, materials and counts;
* abilities with exceptions, sizes, definitions and opposites.

| Question set | Correct |
|---|---|
| dev (used while building the importer and pack) | 37/37 |
| held-out, first run, before any fix | **57/64 (89%)** |
| the same set after fixing what it exposed (now development data) | 64/64 |
| held-out 2, written after those fixes, run once | **58/58** |

The median answer time is 0.1–0.2 ms. The seven first-run misses showed real
mechanism bugs, and the fixes were general mechanisms, not per-question
patches:
* rare senses overriding a clear "no" (shark the person);
* pack facts landing on the wrong sense (factory plants);
* a guess from bats beating "mammals cannot fly";
* 'Napoleon' resolving to the pastry;
* no closed world for places ("Is Tokyo in Europe?");
* two sense-ordering problems (dolphinfish; month).

Held-out 2 checks that the fixes generalise. The caveat: all three sets were
written by the same author, in the phrasing the reader understands, about
topics that WordNet and the pack cover. They measure the reasoning and the
import, not open-domain question answering.

The 19 story tasks still score 100% with the whole 108k-fact brain loaded
underneath (30 stories per task). They take 68 ms per story instead of 0.7 ms.

```text
you> Is a whale a fish?
mnemo> No. Whales are mammals; mammals and fish are different kinds of vertebrate.
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
you> Is a mouse bigger than an elephant?
mnemo> Probably not. It is the other way round: elephants are bigger than horses, which are bigger than dogs, which are bigger than cats, which are bigger than mice.
```

#### 11.6 Limitations of the fed brain

* **Coverage is WordNet's plus the pack's.** WordNet is a dictionary: it has
  no dates, populations or events, and many common-sense facts are missing
  ("a bus is a vehicle" is not in its taxonomy, so the pack adds it).
* **The reader is still a controlled-English parser.** "Is a chair furniture?"
  must be asked as "Is a chair a kind of furniture?".
* **Sense errors remain possible** wherever neither the data nor the discourse
  disambiguates a word.

### 12. Weight-based learning: a small LSTM, and the hybrid

#### 12.1 The model

`mnemo/neural.py` is a word-level LSTM language model in plain numpy, with
forward and backward passes written out by hand. `tests/test_neural.py` checks
the gradients against finite differences (tolerance 1e-5).

* **Architecture.** Two layers of 512 units, with tied input/output embeddings
  (Press & Wolf 2017).
* **Regularisation.** Variational dropout of 0.5 on non-recurrent connections
  (Gal & Ghahramani 2016).
* **Training.** Truncated BPTT over 35 steps, SGD with gradient clipping at 5
  and plateau learning-rate decay, as in Zaremba et al. 2014.
* **Size.** 9.33M parameters, an 18.8 MB file in fp16.
* **Cost.** On PTB, one epoch takes about 7.5 minutes on 3 CPU threads.

The training script, `tools/train_lstm.py`, resumes from checkpoints.

#### 12.2 Results on Penn Treebank (test perplexity)

| Model | Parameters / memory | PPL |
|---|---|---|
| Kneser-Ney 5-gram (our exact reference) | counts | 141.2 |
| Mnemo, long-term memory frozen | 138 MB of counts | 125.4 |
| Mnemo, online | 138 MB of counts | 113.0 |
| LSTM, 2×512, numpy on CPU (ours; validation 82.2) | 9.3M weights | 79.4 |
| **Mnemo frozen + LSTM** (weight tuned on validation) | both | **72.8** |
| **Mnemo online + LSTM** (weight tuned on validation) | both | **69.5** |
| Mnemo online + LSTM, weight learned online (no validation data) | both | 69.4 |
| *for reference:* RNNLM + KN5 + cache (Mikolov et al. 2011) | | 97.5 |
| *for reference:* LSTM medium, 650 units (Zaremba et al. 2014) | 20M, GPU | 82.7 |
| *for reference:* LSTM large, 1500 units (Zaremba et al. 2014) | 66M, GPU | 78.4 |

The mixture is linear at the word level. Mnemo's surprise for a word is the
sum over its letters plus the space that ends it (the line break for `<eos>`).
Its probabilities over in-vocabulary words sum to at most one, so the mixed
perplexity is an honest upper bound (`tools/hybrid_eval.py`). A context-gated
weight (bins of the previous word's surprise) gained only 0.1 perplexity, so the
tool uses one weight.

* **The weight.** Tuned on validation, the LSTM's weight comes out at
  λ = 0.74, next to the test-set optimum (0.745).
* **Mnemo's contribution.** It is largest where it adapts. Online, it cuts
  the LSTM's perplexity by 12%; frozen, by 8%.
* **An earlier checkpoint.** At epoch 6 the LSTM scored 112.8 alone, and
  86.6 mixed with online Mnemo.
* **Reproducing the run.** The logged run (`results/ptb_lstm_hybrid.txt`)
  restarted from its epoch-3 checkpoint after a learning-rate-schedule
  bug was fixed. Running the current script gives a clean run.

#### 12.3 Why the mixture works

The two learners make different errors.
* **Counting memories** remember exactly, in one pass. They adapt within the
  document, which is the cache effect, and they are sharp on frequent and
  repeated n-grams.
* **The LSTM** generalises across similar contexts, through shared embeddings
  and a continuous state. It is better on rare and unseen combinations.

This is the Complementary Learning Systems picture (McClelland et al. 1995): a
fast, exact, episodic learner next to a slow, generalising one. It is also why
KN5+cache mixed with an RNNLM was the state of the art on PTB in 2011, and why
the best compressors add a small network to context mixing.

### 13. Gemma 1: weights, distillation, and an honest comparison

#### 13.1 Why Gemma's weights are not inside Mnemo

* **They could not be obtained here.** The weights sit behind a licence gate on
  Hugging Face and Kaggle, and both are unreachable from this sandbox.
* **They are not small.** Gemma 2B has 2.51 billion parameters: 524M embedding
  and 1.98B non-embedding. That is about 5 GB in 16-bit, and about 1.5 GB even
  at 4 bits. It was trained on 3 trillion tokens on 512 TPUv5e chips
  ([Gemma report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf),
  [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)). Loading it would make
  this project a 200× larger wrapper around Gemma.
* **Its parts do not transfer one by one.** A transformer's knowledge is spread
  across all its weights. The 256k-entry embedding table on its own is a
  decent word-similarity space (much like our PPMI-SVD memory), but it carries
  no facts or reasoning.

#### 13.2 What a large model can still contribute

With Gemma, or any LLM, running on your own machine, three routes keep Mnemo
small.

1. **Facts.** `tools/teacher_pack.py` asks the teacher, through Ollama or any
   OpenAI-compatible server, for simple statements about each topic ("X is a
   Y", "Xs can Z", "the capital of X is Y"). It keeps the lines Mnemo's reader
   understands and drops chatter, headings and long sentences. Then `python3
   -m mnemo feed --source teacher --conf 0.7 teacher.txt` adds them. Every fact
   keeps its source and its lower confidence, so the user's statements and
   WordNet win a conflict.
2. **Text.** Generate a corpus and let the cortex read it (`python3 -m mnemo
   train`). This is distillation into counts.
3. **Soft targets.** Train the LSTM on teacher-generated text, or on teacher
   probabilities (Hinton et al. 2015). This is distillation into a
   few-million-weight network.

#### 13.3 Comparison

| | Gemma 1 2B | Mnemo (this repository) |
|---|---|---|
| Parameters / memory | 2.51B weights (~5 GB) | counts + facts: 34 MB knowledge brain, 138 MB language cortex; optional 9.3M-weight LSTM (19 MB) |
| Training | 3T tokens, 512 TPUv5e | 5 MB of text; 40 s on one CPU core (cortex), ~3 h on 3 cores (LSTM); 5–8 s to import WordNet |
| MMLU (5-shot) | 42.3 | not attempted: it cannot read exam questions |
| HellaSwag / PIQA / ARC-e (0-shot) | 71.4 / 77.3 / 73.2 | not attempted |
| GSM8K (maj@1) / HumanEval | 17.7 / 22.0 | not attempted: no arithmetic word problems, no code |
| PTB perplexity | not reported (different tokenizer) | 113.0 counts only; 69.5 mixed with the LSTM |
| General-knowledge probe (§11.5) | not measured here | 89% held-out (first run), 100% on a second held-out set |
| Story reasoning (bAbI-style, ours) | not measured here | 100% on 19 tasks |
| Teach a new fact | fine-tuning, or keep it in the prompt | one sentence; used immediately and saved |
| Explains its answers | no faithful trace | every answer has its derivation and a confidence |
| Says "I don't know" | not reliably (small LMs hallucinate) | whenever evidence is missing |
| Compute per answer | ~5 GFLOPs per generated token (2 × parameters) | ~0.1–1 ms of one CPU core |

Gemma's benchmark numbers are from Table 6 of the Gemma report.

#### 13.4 What "better than Gemma 1" can and cannot mean

* **It cannot mean broad capability.** Gemma 2B was trained with roughly
  4.5·10²² FLOPs (6 × parameters × tokens). Our LSTM used about 10¹⁵, and the
  counting cortex about 10¹¹ operations. That is a factor of 10⁷ to 10¹¹ less.
  Nothing in the literature, and nothing here, closes that gap on open-ended
  language understanding. On MMLU, HellaSwag, GSM8K or HumanEval, Mnemo would
  score at chance or not at all. Saying otherwise would be dishonest.
* **It can mean specific, measurable axes,** and on these Mnemo is better by
  design:
  * it is 30–200× smaller;
  * it is orders of magnitude cheaper per answer;
  * it learns new facts in one shot and updates them without retraining;
  * every conclusion is exact within its domain and explained;
  * its uncertainty is explicit, and it does not invent facts.
* **The hybrid result shows the right use of weights.** A few million weights
  next to the counting memories buy most of what weights buy in language
  modelling, at a tiny fraction of an LLM's size.

### 14. Understanding how people ask

The reader (§7.7) parses a controlled subset of English. That suits teaching
and testing, but it is not how people type. A probe of 73 natural phrasings
(`tests/natural_probe.py`), written before this work, scored **24/73 (33%)**.
Almost every miss was "I don't understand that yet":
* contractions written without an apostrophe ("whats");
* fragments ("capital of Spain?");
* wrappers ("could you tell me whether …");
* lists ("Which animals can fly?");
* choices, comparisons and sums.

#### 14.1 Rephrasing

`mnemo/rephrase.py` maps common ways of asking onto the reader's forms, one
rule at a time, and keeps the speaker's capitals for the names:
* **Wrappers.** It strips politeness and embedded questions: "please tell me
  what a giraffe is" → "what is a giraffe"; "do you know what cows eat" →
  "what do cows eat".
* **Checks.** "Is it true that bats are birds" → "are bats birds". A
  negated statement is asked the other way round and its verdict flipped:
  "is it true that penguins can't fly" → "can penguins fly", then flip.
* **Teaching wrappers.** "Remember that …" and "note that …" become plain
  statements.
* **Fragments and paraphrases.** It handles fragments, shouting, "how many N
  are in a M", and language and currency paraphrases.
* **Sums.** A small recursive-descent parser does exact arithmetic with
  number words, precedence, powers and roots. It never calls `eval`.

#### 14.2 The question types people expect

New handlers in `mnemo/mind.py` answer the following:
* **Why plus a question.** It gives the reasons, or corrects the question's
  assumption: "Why can a penguin fly?" → "Actually, penguins cannot fly."
* **"What kind of animal is a whale?"** The answer is at the everyday level,
  the category that people (the user or the pack) talk about. That follows
  Rosch's basic level: "A mammal (more precisely, a cetacean)."
* **Examples and lists.** "Name some birds", "Which animals can fly?" and
  "Which countries are in South America?"
* **Comparisons.** "Which is bigger, an elephant or a mouse?"
* **Choices.** "Is a tomato a fruit or a vegetable?" → "Both".
* **Differences.** For "the difference between X and Y", it gives where X and
  Y part in the taxonomy, plus both definitions.
* **Places of a kind.** "Which continent is Kenya in?"
* **Inverse relations.** "Paris is the capital of which country?"
* **Yes/no with a number.** "Do birds have four legs?" → "No. Birds have two
  legs."

#### 14.3 Familiarity

A list of examples is only useful if it names familiar things. Candidates are
ranked in this order:
1. What the user or the pack said.
2. How often the sense was tagged in SemCor, from WordNet's `index.sense`.
3. Whether the concept is its word's dominant sense.
4. The word's polysemy. Frequent words have many senses, by Zipf's law of
   meaning.

Deep proper names are left out: Adam Smith is a mammal only through
"person". Examples of an animal kind must be animals. "Give me an example of a
mammal" now answers "People, seals, bats, dolphins, mice and monkeys", not
"tuskers and placentals".

The same counts choose the everyday word for about 2,000 concepts whose
WordNet headword is almost never used: "human being" for *homo*, "metal" for
*metallic element*. The speaker's own word always wins. A story that says
"hallway" hears "hallway", not WordNet's more frequent "hall".

#### 14.4 Results

| Natural-phrasing probe | Correct |
|---|---|
| 73 questions, baseline before this work | 24/73 (33%) |
| the same 73 after it (development data) | 73/73 |
| 56 fresh questions, written after the work, run once | **55/56 (98%)** |

The one fresh miss was "What money do they use in the United Kingdom?", a
paraphrase the rules did not cover. It was fixed afterwards. Answers still take
about 0.2 ms. All earlier suites are unchanged:
* 159/159 on the knowledge probe;
* 100% on the story tasks, with and without the fed brain;
* 41 unit tests.

#### 14.5 Limits

These are rules, not learned reading. They cover the common shapes of a
question, but any new shape needs a new rule. "Is Mary in the kitchen or the
garden?" is still read as one place called "kitchen or garden". The
principled next step remains roadmap item 2: induce parse patterns from
(sentence, meaning) pairs. The rephrasing rules and the probes now provide a
training and test set for exactly that.

### 15. A generalist: knowing many kinds of thing

The next request was to make Mnemo a generalist that knows many things. The fed
brain knew WordNet's taxonomy and parts, the pack, and the capitals and
currencies that definitions name. A generalist probe of 68 questions
(`tests/general_probe.py`) was written before this work. It covers people,
geography, science, objects, animals, history, counting and superlatives. It
scored **13/68 (19%)**.

#### 15.1 Reading the dictionary

WordNet's definitions are short, carefully edited and regular in shape. A
pattern reader turns those shapes into facts. It is related to Hearst
patterns and open information extraction, but it works over a single
well-edited source.

**People** (instances in `noun.person`):
* life years "(1564-1616)", BC years, "(born in 1934)";
* a leading nationality adjective ("English poet", "prolific and influential
  Spanish artist"), mapped to a country;
* "President of the United States" → from;
* "born in Scotland" → born in.

This gives 3,059 birth years, 2,785 death years and 2,652 nationalities.

**Events:**
* spans "from 1939 to 1945", "; 1789-1799", "(1954-1975)";
* single years for named acts and events: "(1066)", "in November 1917",
  "on 18 June 1815".

This gives 123 spans and 134 single years.

**Places, artifacts and creatures:**
* places that have no WordNet location link: "a city in east-central France";
* artifacts: "used to/for/as …", "an instrument … for cutting", "a regulator
  for automatically regulating temperature". That gives 2,339 purposes, plus
  what things are made of;
* animals and plants: what they feed on, where they live ("Australian
  marsupial"), and typical properties from a short whitelist (flightless,
  nocturnal, endangered, …).

The facts carry confidence 0.8. That is below WordNet's own links (0.99) and
below what the user says. Facts about a single thing (years, areas, symbols,
borders, nationality) are not inherited: France's area is not every
country's.

#### 15.2 Open world data

`scripts/get_data.sh` fetches three small open datasets. They are not stored
in the repository.
* **mledoze/countries** (ODbL): 250 entries with borders, area, landlocked,
  region and subregion, capital, languages, currencies and demonyms. Its
  "independent" flag decides whether an entry is a country (195) or a
  dependency (Réunion, Greenland).
* **datasets/world-cities** (GeoNames, CC BY 3.0): 32k cities, placed in their
  region and country.
* **Periodic-Table-JSON** (CC BY-SA 3.0): 119 elements with symbol, atomic
  number, phase, kind, whether each is a metal, and a discoverer where one is
  named.

Most of the effort went into matching names to the dictionary's concepts:
* **Name collisions.** A town called Asia in the Philippines had put the
  continent inside the Philippines, so a city name now attaches only to a new
  concept or to an existing town. That also keeps Lincoln, Nebraska, away
  from the president.
* **Namesakes.** When the dictionary defines a town, the data row's country
  must appear in the definition. This is a Lesk-style check: the dictionary's
  Madrid is in Spain, not Colombia.
* **The dictionary's own sense first.** The town of Darwin is WordNet's
  provincial capital, not a fresh concept.
* **Sentinel values.** An area of −1 means unknown.
* **"the".** Names that take "the" get it: the Nile, the United States, the
  Battle of Hastings, but not World War II.

The brain grows from 23 MB to 34 MB and about 200k facts. Feeding takes 10 s,
and loading takes 1.4 s.

#### 15.3 Answering

New question types (`mnemo/language.py`, `mnemo/mind.py`):

**Dates and lives:**
* when someone was born or died, and when something began, ended or took
  place;
* age at death, from the years alone, so "about 35";
* who was born first;
* where someone was from or born.

Some facts hold one value at a time: "Was Mozart born in 1800?" → "No. Mozart
was born in 1756." These can also be taught: "Tom was born in 1930."

**Attributes both ways**, and comparisons by numbers:
* area, symbol and atomic number: "What element has the symbol Na?";
* "Which is bigger, France or Spain?" by area;
* "older" by year of birth.

**Superlatives:**
* By the numbers where the data has them: the largest country in Africa by
  area.
* Otherwise, the dictionary's own superlatives, ranked:
  * "the world's longest river" beats "the longest river in England", unless
    England was asked;
  * "the 2nd longest" is not the longest;
  * "this range contains the world's highest mountain" is about something
    else;
  * a modifier is allowed but ranks lower ("the largest freshwater lake").

**Counting** uses members that a list names: the pack, the user or the world
data.
* Seven continents, not WordNet's Pangaea and Laurasia as well.
* 195 countries, and 54 in Africa.
* Bordering countries.
* A region that the kind's own definition names: planets are "bodies in the
  solar system".

**Who did what:**
1. A fact, if there is one: oxygen was discovered by Scheele.
2. Otherwise, people's definitions in which the verb governs the thing:
   "bacteriologist who discovered penicillin", but not "penicillin, which had
   been discovered by …". The thing's other names count ("incandescent
   electric light" for the light bulb). Matches inside a longer name do not
   ("North America" for America).
3. Otherwise, the thing's own definition: "the hero of William Shakespeare's
   tragedy" → probably Shakespeare. This answer is hedged.

A weak match is refused rather than guessed. Before that rule, "Who
discovered gravity?" returned Robert Hooke from a definition that mentions
both.

**Titles:** "the first president of the United States" →
"1st President of the United States". "First man on the moon" → "the first
man to set foot on the Moon".

**Closed world over complete data:** the border lists are complete, so "Does
Spain border Germany?" → "No. Spain borders Andorra, France, Gibraltar,
Morocco and Portugal." This is Reiter's closed-world assumption, applied only
where the data is complete.

**Small language points:**
* "World War I" is a name, not "World War me";
* "the symbol He" is helium, not "he";
* the dead *were* from somewhere, while Ann *is* from Paris.

#### 15.4 Results

| Generalist probe | Correct |
|---|---|
| GENERAL, 68 questions, baseline before this work | 13/68 (19%) |
| the same 68 after it (development data) | 67/68 |
| GENERAL_2, 68 fresh questions, written afterwards, run once | **52/68 (76%)** |
| GENERAL_2 after the general fixes it prompted (development data) | 61/68 |
| GENERAL_3, 46 fresh questions, written after those fixes, run once | **39/46 (85%)** |
| GENERAL_3 after the fixes it prompted (development data) | 43/46 |

**GENERAL_2's first-run misses (16):**
* 7 were knowledge the sources lack: the telescope's inventor, *Romeo and
  Juliet*, the Sistine Chapel, where lemurs live, what a microscope is for,
  the largest animal, and gravity. The dictionary has Newton's law of
  gravitation but never says he "discovered" gravity, and the one answer
  given (Hooke) was wrong.
* 9 were general bugs:
  * the "I" of "World War I";
  * Darwin the naturalist against the town;
  * borders as a closed world;
  * two single-year events;
  * looser titles;
  * modifiers in superlatives;
  * a purpose after the genus;
  * the solar system count.

**GENERAL_3's first-run misses (7):**
* 3 were knowledge gaps:
  * Gutenberg's definition never says "invented";
  * Augustus' never says "first emperor";
  * Columbus "discovered the New World", which the dictionary does not link
    to America. The first run named John Cabot instead, from "discovered the
    mainland of North America". Matches inside a longer name are now refused.
* 4 were general bugs:
  * dates written "(1954-1975)" and "on 18 June 1815";
  * "the symbol He";
  * "prolific and influential Spanish artist".

Each fix is a rule, tested on fixtures in `tests/test_world.py`, not an
answer. All earlier suites are unchanged:
* 159/159 on the knowledge probe;
* 73/73 and 56/56 on the natural-phrasing probes, since the country data now
  answers the one earlier miss, the UK's money;
* 100% on the story tasks, with and without the 200k facts underneath;
* 49 unit tests.

The median answer takes 0.3 ms. Questions that search all 82k definitions
(who did what, dictionary superlatives) take 20 to 80 ms.

#### 15.5 Limits

* **Coverage** is WordNet 3.0's, which dates from 2006, plus the three
  datasets. There are no populations, no current affairs, no works of art (no
  Mona Lisa), no sport and no companies. The remaining misses are all of this
  kind.
* **The reader is a set of patterns.** It has errors: same-named places can
  still merge, and a year in a definition may not be the event's year.
* **Counts depend on which list is taken as complete.** "195 countries" is
  the dataset's count of independent states, and "four oceans" is the pack's.
* **This is not MMLU.** The probe asks single facts. Nothing here changes the
  comparison with Gemma 1 in §13.
* **The next step for breadth** is a large open knowledge base (Wikidata
  dumps) through the same importers, with the same care over names. The
  extraction patterns should be learned rather than written.

### 16. Pairing with a language model

The next request was for "the performance of an LLM" in Mnemo. On its own it
cannot have that, for three reasons:
* **Capacity.** An LLM's breadth comes from billions of weights fitted to
  trillions of tokens. That holds orders of magnitude more facts and phrasings
  than a 34 MB brain built from a dictionary and three datasets.
* **Reading.** An LLM reads any English. Mnemo's reader is a set of rules
  (§14.5).
* **Generation.** An LLM writes fluent text. Mnemo's cortex is locally fluent
  and globally incoherent (§9).

The two kinds of system fail in opposite ways, though:
* a language model is broad but slow, opaque and sometimes confidently wrong;
* Mnemo is narrow but instant, exact, explained, and able to learn a fact in
  one shot.

That is the case for pairing them. The idea parallels complementary
learning systems in the brain: a fast, specific memory beside a slow
statistical learner. It is also retrieval-augmented generation (Lewis et al.
2020), with a symbolic memory as the retriever.

#### 16.1 Design (`mnemo/llm.py`, `Mind.attach_llm`)

* **Backends.** Ollama's `/api/chat`, or any OpenAI-compatible
  `/chat/completions` server: llama.cpp's `llama-server`, vLLM, LM Studio.
  The client uses the standard library, at temperature 0. A model on the
  same machine is reached directly, never through a proxy.
* **Routing.** Memory answers first. The model is called only when the reply
  would be "I don't know" or "I don't understand", and only for a question or
  request, never for a statement.
* **Grounding.** The prompt carries up to 12 lines of what Mnemo holds about
  the things the question mentions: a definition and the most confident facts
  for each. The things are found from:
  * the names the speaker capitalised;
  * the slots of Mnemo's own parse of the question;
  * nouns, recognised by a determiner or a plural.

  Verbs ("how does it *work*") and a word's rarer senses (Anatole France for
  France) are left out unless nothing else is known. The last three exchanges
  of the conversation go along as context.
* **Provenance.** The model's answers carry its name. "Why?" answers that they
  came from the model, not from memory, and gives no reasons. Later answers
  that rest on facts the model supplied add "(I learned that from a language
  model.)".
* **Learning and checking.** The model's answer is filtered by the same
  reader-based filter as the teacher tool (§13.2), which keeps short
  statements Mnemo can parse. Each kept statement is compared with memory
  first:
  * if Mnemo firmly believes the opposite (confidence at least 0.7), the
    statement is pointed out and not stored;
  * otherwise it is stored with source `llm` and confidence 0.6. That is below
    the user, WordNet and the world data, so they win any conflict.

  A question the model once answered can then be answered from memory in
  well under a millisecond.

#### 16.2 What is measured, and what is not yet

`tests/test_llm.py` tests the plumbing against a stand-in server on localhost
that replays canned replies:
* both API formats;
* memory first, so the model is not called for what Mnemo knows;
* grounding in the prompt;
* provenance, and the answer to "Why?";
* learning, and refusing a contradiction;
* no server running.

It deliberately says nothing about how well any model answers.

`tools/llm_eval.py` runs every probe set in three modes:
* the model alone, with the conversation but none of Mnemo's facts;
* Mnemo alone;
* the pair.

It reports accuracy, latency and the share of questions the model answered.
It could not be run with a real model here: the environment's network policy
blocks `huggingface.co` and `registry.ollama.ai`, so no weights could be
fetched. No model numbers are reported until they are measured. Two cautions
for whoever runs it:
* **Biased sets.** The probe sets are partly Mnemo's development data, which
  biases the comparison in Mnemo's favour (459/470 for Mnemo alone). The
  sets' first-run scores are the fair figures for Mnemo.
* **The checker.** Yes/no answers count only if they begin with yes or no.
  The system prompt asks for that, but small models do not always comply.

#### 16.3 Limits and next steps

* **Speed.** An answer that goes to the model takes as long as the model:
  seconds on a CPU, rather than Mnemo's milliseconds.
* **Checking is shallow.** Only what Mnemo can parse is checked. A wrong
  answer in a form Mnemo cannot read passes unchecked, though it is still
  labelled.
* **Only gaps go to the model.** The model is never asked about what Mnemo
  answers itself, so a wrong answer from memory is not caught by the model. A
  mode that asks both and reports disagreement would be easy to add.
* **The next step** is to run `tools/llm_eval.py` with a small CPU model
  (0.5–2B parameters) where one can be downloaded, and report the three
  columns. A pure-numpy runtime for a small open model, like the numpy LSTM of
  §12, would put the model inside the repository. It should be written only
  where real weights are available to check it against.

### 17. References

**Compression and count-based language models**
* M. Mahoney, *Data Compression Explained* (mirror):
  https://github.com/mco-gh/notes/blob/main/articles/Data%20Compression%20Explained.md
* Large Text Compression Benchmark: https://www.mattmahoney.net/dc/text.html
  (transcription: https://github.com/mitiko/weath3rb0i/blob/main/web/js/ltcb.js)
* Hutter Prize: http://prize.hutter1.net/
  * fx2-cmix: https://github.com/kaitz/fx2-cmix
  * cmix: https://github.com/byronknoll/cmix
  * paq8px: https://github.com/hxim/paq8px
  * cmix-lex-transformer: https://github.com/Neel49/cmix-lex-transformer
* C. Mattern, "Mixing strategies in data compression", DCC 2012.
* J. Veness et al., "Gated Linear Networks": https://arxiv.org/abs/1712.01897
* G. Delétang et al., "Language Modeling Is Compression", ICLR 2024:
  https://arxiv.org/abs/2309.10668
* J. Liu et al., "Infini-gram": https://arxiv.org/abs/2401.17377
* S. Chen & J. Goodman, "An empirical study of smoothing techniques for
  language modeling", 1998.
* J. Goodman, "A bit of progress in language modeling":
  https://arxiv.org/abs/cs/0108005
* T. Mikolov et al., Interspeech 2011:
  https://www.fit.vut.cz/person/imikolov/public/rnnlm/is2011_emp.pdf
* T. Mikolov & G. Zweig 2012:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rnn_ctxt.pdf
* C. Olsson et al., "In-context learning and induction heads":
  https://arxiv.org/abs/2209.11895

**Brain-inspired representations and memory**
* P. Kanerva, "Hyperdimensional computing" 2009:
  https://redwood.berkeley.edu/wp-content/uploads/2018/01/kanerva2009hyperdimensional.pdf
* D. Kleyko et al., HDC/VSA survey: https://arxiv.org/abs/2111.06077 and
  https://arxiv.org/abs/2112.15424
* A. Joshi, J. Halseth & P. Kanerva, "Language geometry using random
  indexing" 2016:
  https://redwood.berkeley.edu/wp-content/uploads/2018/12/joshi2016language.pdf
* O. Levy & Y. Goldberg 2014:
  https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html
* O. Levy, Y. Goldberg & I. Dagan 2015: https://aclanthology.org/Q15-1016/
* T. Landauer & S. Dumais, "A solution to Plato's problem", Psychological
  Review 1997.
* N. Halko, P.-G. Martinsson & J. Tropp, "Finding structure with randomness",
  SIAM Review 2011.
* T. Bricken & C. Pehlevan, "Attention approximates sparse distributed
  memory": https://arxiv.org/abs/2111.05498
* H. Ramsauer et al., "Hopfield networks is all you need":
  https://arxiv.org/abs/2008.02217
* Y. Cui, S. Ahmad & J. Hawkins 2016: https://arxiv.org/abs/1512.05463
* Thousand Brains Project / Monty: https://github.com/thousandbrainsproject/tbp.monty
  and https://arxiv.org/abs/2507.04494
* O.-C. Granmo, Tsetlin machine: https://arxiv.org/abs/1804.01508
* J. McClelland, B. McNaughton & R. O'Reilly 1995:
  https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf
* R. Rao & D. Ballard 1999: https://www.cs.utexas.edu/~dana/nn.pdf
* G. Hinton, Forward-Forward: https://arxiv.org/abs/2212.13345

**Cognitive architectures and reasoning**
* J. Anderson & L. Schooler, "Reflections of the environment in memory",
  Psychological Science 1991.
* pyactr: https://github.com/jakdot/pyactr
* Soar: https://github.com/SoarGroup/Soar
* P. Wang, Non-Axiomatic Logic; course notes:
  https://cis.temple.edu/~wangp/AGI-course/03-Validity.html
* OpenNARS for Applications: https://github.com/opennars/OpenNARS-for-Applications
  (truth functions in src/Truth.c)
* J. Weston et al., bAbI: https://arxiv.org/abs/1502.05698
* A. Mitra & C. Baral, AAAI 2016:
  https://ojs.aaai.org/index.php/AAAI/article/view/10354
* B. Lake, R. Salakhutdinov & J. Tenenbaum, Science 2015:
  https://cims.nyu.edu/~brenden/papers/LakeEtAl2015Science.pdf
* ARC Prize analyses: https://arcprize.org/blog/arc-prize-2025-results-analysis
* N. Cowan 2001:
  https://memory.psych.missouri.edu/assets/doc/articles/2001/cowan-bbs-2001.pdf
* M. Frank 2023, children vs LLM data:
  https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/Frank2023_ChildrenLLMs.pdf

**Knowledge, word senses and inheritance**
* E. Rosch et al., "Basic objects in natural categories", Cognitive Psychology
  1976.
* G. Zipf, "The meaning-frequency relationship of words", J. General
  Psychology 1945.
* G. Miller et al., "A semantic concordance" (SemCor), HLT 1993.
* G. Miller, "WordNet: a lexical database for English", CACM 1995; WordNet 3.0
  via the NLTK data mirror: https://github.com/nltk/nltk_data
* W. Gale, K. Church & D. Yarowsky, "One sense per discourse", HLT 1992.
* M. Lesk, "Automatic sense disambiguation using machine readable
  dictionaries", SIGDOC 1986.
* D. Touretzky, *The Mathematics of Inheritance Systems*, 1986 (inferential
  distance).
* M. Hearst, "Automatic acquisition of hyponyms from large text corpora",
  COLING 1992.
* M. Banko et al., "Open information extraction from the web", IJCAI 2007.
* S. Harabagiu, G. Miller & D. Moldovan, "WordNet 2: a morphologically and
  semantically enhanced resource" (Extended WordNet: parsed glosses),
  SIGLEX 1999.
* R. Reiter, "On closed world data bases", in *Logic and Data Bases*, 1978.
* P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP
  tasks", NeurIPS 2020: https://arxiv.org/abs/2005.11401
* J. McClelland, B. McNaughton & R. O'Reilly, "Why there are complementary
  learning systems in the hippocampus and neocortex", Psychological Review
  1995.
* World data: mledoze/countries (ODbL), https://github.com/mledoze/countries;
  datasets/world-cities (GeoNames, CC BY 3.0),
  https://github.com/datasets/world-cities; Periodic-Table-JSON
  (CC BY-SA 3.0), https://github.com/Bowserinator/Periodic-Table-JSON

**Weights and mixtures**
* W. Zaremba, I. Sutskever & O. Vinyals, "Recurrent neural network
  regularization": https://arxiv.org/abs/1409.2329
* O. Press & L. Wolf, "Using the output embedding to improve language
  models": https://arxiv.org/abs/1608.05859
* Y. Gal & Z. Ghahramani, "A theoretically grounded application of dropout in
  recurrent neural networks": https://arxiv.org/abs/1512.05287
* G. Hinton, O. Vinyals & J. Dean, "Distilling the knowledge in a neural
  network": https://arxiv.org/abs/1503.02531
* Gemma Team, "Gemma: open models based on Gemini research and technology",
  2024: https://arxiv.org/abs/2403.08295 (PDF:
  https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)

**Small neural models, for calibration**
* TinyStories: https://arxiv.org/abs/2305.07759
* nanoGPT: https://github.com/karpathy/nanoGPT
* llama2.c: https://github.com/karpathy/llama2.c
* BitNet b1.58 2B4T: https://arxiv.org/pdf/2504.12285
