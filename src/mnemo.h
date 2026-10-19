/*
 * Mnemo - a tiny, CPU-only, non-neural predictive memory.
 *
 * The "cortex" is an online byte-level predictor built from counting memories
 * (context columns), an episodic match memory and a gated vote of experts.
 * There is no backpropagation, no pretraining and no GPU: every structure
 * learns from a single pass over the data with local update rules.
 */
#ifndef MNEMO_H
#define MNEMO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mn_model mn_model;

typedef struct {
    int table_bits;   /* log2 #16-byte buckets in the shared context table   */
    int hist_bits;    /* log2 size of the episodic history buffer (bytes)    */
    int match_bits;   /* log2 #entries of the episodic recall index          */
    int class_bits;   /* log2 #entries of the word->concept table (0 = off)  */
    float lr;         /* final mixer learning-rate floor                      */
    float lr0;        /* extra learning rate at the start (decays)            */
    uint32_t ctx_mask; /* bit i enables context column i (0 = all)            */
    int lex_order;    /* word n-gram order of the mental lexicon (0 = off)    */
    int lex_spaced;   /* 1: tokens are runs of non-space bytes (pre-tokenized) */
} mn_config;

void mn_default_config(mn_config *c);
mn_model *mn_new(const mn_config *c);
void mn_free(mn_model *m);
size_t mn_memory(const mn_model *m);          /* bytes allocated            */

/* Learning on/off. When off ("frozen") the model still tracks context but no
 * statistic, weight or index is modified. */
void mn_set_learning(mn_model *m, int on);
int mn_get_learning(const mn_model *m);

/* Bit-level interface (used by the arithmetic coder). */
int mn_p16(mn_model *m);                       /* P(next bit = 1) * 65536   */
void mn_update(mn_model *m, int bit);          /* observe the bit           */

/* Byte-level helpers. Return the information content in bits. */
double mn_observe_byte(mn_model *m, int byte);
double mn_observe(mn_model *m, const uint8_t *buf, size_t n);

/* Full distribution of the next byte (out[256], sums to 1). Requires frozen
 * mode or temporarily freezes the model. */
void mn_next_byte_dist(mn_model *m, double *out);

/* Sample the next byte bit by bit with temperature, then observe it. */
int mn_sample_byte(mn_model *m, double temperature, uint64_t *rng);

/* Concept (semantic class) hook: words are hashed with mn_word_hash; a class
 * id (1..65535) can be attached to each word and is used as context. */
uint32_t mn_word_hash(const char *word, size_t len);
void mn_set_class(mn_model *m, uint32_t word_hash, uint32_t cls);
uint32_t mn_get_class(const mn_model *m, uint32_t word_hash);

/* Imagination: mn_snapshot captures the short-term state and freezes
 * learning; everything observed afterwards (e.g. a candidate continuation) is
 * undone by mn_restore, which also frees the snapshot. Snapshots nest. */
typedef struct mn_snap mn_snap;
mn_snap *mn_snapshot(mn_model *m);
void mn_restore(mn_model *m, mn_snap *z);

/* Sample n bytes (bit-level temperature); stops after stop_byte (-1 none). */
size_t mn_generate(mn_model *m, uint8_t *out, size_t n, double temperature, uint64_t seed,
                   int stop_byte);

/* The mental lexicon (see lexicon.h), NULL when disabled. */
struct mn_lex *mn_lexicon(mn_model *m);

/* Persistence. */
int mn_save(const mn_model *m, const char *path);
mn_model *mn_load(const char *path);

/* Diagnostics: number of mixer inputs and their names. */
int mn_num_inputs(const mn_model *m);
void mn_lex_stats(mn_model *m, double *bits, uint64_t *bytes, int reset);

#ifdef __cplusplus
}
#endif
#endif
