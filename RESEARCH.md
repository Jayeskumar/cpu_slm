# A tiny mind without neural networks

Research notes for **cpu_slm / Mnemo**: how far can a CPU-only system, built from
counting, hashing, hypervectors and logic instead of gradient-trained neural
networks, go towards language and human-like intelligence in a small memory and
compute budget?

This document has two parts:

* **Part I: survey.** What is known about non-neural (or minimally neural)
  approaches to prediction, memory, meaning and reasoning, with numbers.
* **Part II: the system.** The design that came out of the survey, how it
  maps onto brain systems, what it measurably does, and what it cannot do.

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

* **Not a general conversational AI.** It knows what it has been told or has
  read, and nothing more. Without a reading phase it has no world knowledge,
  and with one it knows only what the parser could extract.
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
8. **Hybrid option:** the 2026 Hutter Prize result shows that a tiny
   pretrained network (about 6M parameters, int4) added to context mixing
   closes most of the gap to large models. It could be an optional "intuition"
   column, at the cost of the no-neural-network constraint.

### 11. References

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

**Small neural models, for calibration**
* TinyStories: https://arxiv.org/abs/2305.07759
* nanoGPT: https://github.com/karpathy/nanoGPT
* llama2.c: https://github.com/karpathy/llama2.c
* BitNet b1.58 2B4T: https://arxiv.org/pdf/2504.12285
