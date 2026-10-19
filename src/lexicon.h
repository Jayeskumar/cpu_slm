/*
 * lexicon.h - the mental lexicon: whole-word expectations for the cortex.
 *
 * Words are stored once in a vocabulary trie.  Word n-gram statistics
 * (interpolated modified Kneser-Ney, learned online) give a distribution over
 * the next word; as the word is spelled out, the distribution is narrowed to
 * the words consistent with the letters seen so far and turned into a
 * prediction of the next byte.  Humans read the same way: expectations about
 * whole words shape the perception of their letters.
 */
#ifndef MNEMO_LEXICON_H
#define MNEMO_LEXICON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct mn_lex mn_lex;

mn_lex *lex_new(int order, int spaced);
void lex_free(mn_lex *L);
size_t lex_memory(const mn_lex *L);

/* Observe the next byte of the stream (learn = update statistics). After the
 * call, lex_tree() describes the predicted distribution of the following
 * byte as a binary tree of masses: node 1 is the root, the children of node i
 * are 2i and 2i+1, and the leaf of byte b is 256 + b. Returns NULL when the
 * lexicon has no opinion. */
void lex_observe(mn_lex *L, int c, int learn);
const float *lex_tree(const mn_lex *L);

/* 0 = silent, 1 = between words, 2 = inside a word (diffuse), 3 = inside a
 * word (focused on few candidates) */
int lex_state(const mn_lex *L);

/* Vocabulary access (for the semantic memory and for diagnostics). */
uint32_t lex_num_words(const mn_lex *L);
const char *lex_word(const mn_lex *L, uint32_t id, int *len);
uint32_t lex_count(const mn_lex *L, uint32_t id); /* continuation count */
int64_t lex_find(const mn_lex *L, const char *w, int len);

/* The next-word distribution in the current context: writes up to max (word
 * id, probability) pairs with the highest probabilities. */
int lex_top_next(mn_lex *L, uint32_t *ids, float *probs, int max);

/* Imagination: capture and restore the short-term (stream) state so that a
 * continuation can be evaluated without leaving a trace. */
typedef struct lex_snap lex_snap;
lex_snap *lex_snapshot(const mn_lex *L);
void lex_restore(mn_lex *L, const lex_snap *z);
void lex_snap_free(lex_snap *z);

int lex_save(const mn_lex *L, FILE *f);
mn_lex *lex_load(FILE *f);

#endif
