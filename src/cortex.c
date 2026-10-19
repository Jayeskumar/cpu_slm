/*
 * cortex.c - the predictive "cortex" of Mnemo.
 *
 * A byte stream is predicted one bit at a time.  Many small memories
 * ("columns") each look at the past through a different window - the last
 * 2..12 bytes, the current word, the previous words, the concept classes of
 * the previous words - and remember, for every context they have ever seen, a
 * compact bit history (two saturating, recency-weighted counts in one byte).
 * An episodic recall memory finds the last time the recent past occurred and
 * proposes whatever followed (the analogue of an induction head).
 *
 * The columns' votes are combined by gated geometric pooling: several gates,
 * each selected by a different situation (partial byte, previous byte, match
 * state, familiarity), weight the votes in the log-odds domain and learn their
 * trust online with a local delta rule.  A last calibration stage (adaptive
 * probability maps) corrects systematic over/under-confidence.
 *
 * Everything is updated online, in one pass, in a fixed memory budget.
 */
#include "mnemo.h"
#include "lexicon.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#define NCTX_MAX 16
#define NIN_MAX 40              /* padded input vector (multiple of 8) */
#define NM 5                    /* first-layer gates */
#define M0SETS 256              /* partial byte */
#define M1SETS (16 * 2 * 8)     /* match length bucket x expected bit x bitpos */
#define M2SETS 256              /* previous byte */
#define M3SETS (8 * 8)          /* familiarity (longest known order) x bitpos */
#define M4SETS (4 * 8)          /* lexicon state x bitpos */
#define NFIN (NM + 2)            /* gates + direct lexicon vote + bias */
#define FSETS (4 * 8)            /* final gate: lexicon state x bitpos */
#define MINLEN_S 5              /* short episodic cue (bytes) */
#define MINLEN_L 12             /* long episodic cue (bytes) */
#define SM_LIMIT 1023

typedef struct {
    uint8_t chk;
    uint8_t st[15];
} Bucket;

struct mn_model {
    mn_config cfg;
    int learn;
    int nctx, nin;
    /* stream state */
    uint64_t pos;
    uint64_t nbits;
    uint32_t c0, bitpos, nib;
    uint32_t c4, c8, c12;
    uint32_t w0, w1, w2, w3, wlen, sep;
    uint32_t k1, k2;
    uint32_t h[NCTX_MAX];
    uint32_t hraw[NCTX_MAX];
    uint8_t active[NCTX_MAX];
    /* episodic recall */
    uint32_t mptr, mlen;
    int mexp;
    uint32_t mlenb;
    /* per-bit bookkeeping */
    Bucket *bhi[NCTX_MAX], *blo[NCTX_MAX];
    uint8_t *sp[NCTX_MAX];
    uint8_t sv[NCTX_MAX];
    uint8_t o1s;
    float x[NIN_MAX];
    float ms[NM], mp[NM];
    int msel[NM];
    float fx[NFIN];
    float fs, fp;
    int fsel;
    uint32_t aidx[3];
    float aw[3], ap[3];
    float pfin;
    int p16;
    int predicted;
    float lr_now;
    /* adaptive maps */
    uint32_t sm[NCTX_MAX + 1][256];
    uint32_t smm[16];
    uint32_t o0[256];
    uint8_t o1[65536];
    float wm0[M0SETS][NIN_MAX];
    float wm1[M1SETS][NIN_MAX];
    float wm2[M2SETS][NIN_MAX];
    float wm3[M3SETS][NIN_MAX];
    float wm4[M4SETS][NIN_MAX];
    float wf[FSETS][NFIN];
    uint16_t apm1[65536][33];
    uint16_t apm2[65536][33];
    uint16_t apm3[8192][33];
    /* large memories */
    Bucket *tab;
    uint32_t tmask;
    uint8_t *hist;
    uint32_t hmask;
    uint32_t *midx;
    uint32_t mmask;
    uint32_t *ckey;
    uint16_t *cval;
    uint32_t cmask;
    mn_lex *lex;
    int lstate;
    /* undo log of short-term writes while imagining */
    int undo_on;
    uint32_t *undo_i, *undo_v;
    uint8_t *undo_k; /* 0 = recall index entry, 1 = history byte */
    size_t undo_n, undo_cap;
    double lex_bits;     /* diagnostics: cost of the lexicon alone */
    uint64_t lex_bytes;
};

/* ------------------------------------------------------------------ tables */

static float stretch_tab[4096];
static float squash_tab[4097]; /* squash over [-16, 16] in steps of 1/128 */
static uint8_t nex_tab[256][2];
static int32_t recip_tab[1024];
static Bucket zero_bucket;
static int tables_ready = 0;

static void init_tables(void) {
    if (tables_ready) return;
    for (int i = 0; i < 4096; i++) {
        double p = (i + 0.5) / 4096.0;
        stretch_tab[i] = (float)log(p / (1.0 - p));
    }
    /* Bit history: high nibble = n0, low nibble = n1 (saturating at 15).
     * Observing a bit increments its count; if the opposite count exceeds 2
     * it is roughly halved (recent evidence outweighs old evidence). */
#ifndef NS_KEEP
#define NS_KEEP 2
#endif
    for (int s = 0; s < 256; s++) {
        for (int y = 0; y < 2; y++) {
            int n0 = s >> 4, n1 = s & 15;
            if (y) {
                if (n1 < 15) n1++;
                if (n0 > NS_KEEP) n0 = (n0 + NS_KEEP) / 2;
            } else {
                if (n0 < 15) n0++;
                if (n1 > NS_KEEP) n1 = (n1 + NS_KEEP) / 2;
            }
            nex_tab[s][y] = (uint8_t)((n0 << 4) | n1);
        }
    }
    for (int n = 0; n < 1024; n++) recip_tab[n] = (int32_t)(65536.0 / (n + 1.5));
    for (int i = 0; i <= 4096; i++) squash_tab[i] = (float)(1.0 / (1.0 + exp(-(i - 2048) / 128.0)));
    tables_ready = 1;
}

static inline float stretchf(float p) {
    if (p < 1e-6f) p = 1e-6f;
    if (p > 1 - 1e-6f) p = 1 - 1e-6f;
    return logf(p / (1 - p));
}
static inline float squashf(float x) {
    float t = (x + 16.0f) * 128.0f;
    if (t <= 0) return squash_tab[0];
    if (t >= 4095.999f) return squash_tab[4096];
    int i = (int)t;
    float f = t - (float)i;
    return squash_tab[i] + f * (squash_tab[i + 1] - squash_tab[i]);
}
static inline float sm_stretch(uint32_t e) { return stretch_tab[e >> 20]; }

static inline void sm_update(uint32_t *e, int y, uint32_t limit) {
    uint32_t v = *e, n = v & 1023;
    int32_t p = (int32_t)(v >> 10);
    int32_t target = y ? (1 << 22) - 1 : 0;
    p += (int32_t)(((int64_t)(target - p) * recip_tab[n]) >> 16);
    if (p < 0) p = 0;
    if (p > (1 << 22) - 1) p = (1 << 22) - 1;
    if (n < limit) n++;
    *e = ((uint32_t)p << 10) | n;
}

static inline uint32_t sm_init_value(int s) {
    int n0 = s >> 4, n1 = s & 15;
    double p = (n1 + 0.4) / (n0 + n1 + 0.8);
    return ((uint32_t)(p * ((1 << 22) - 1)) << 10);
}

static inline uint32_t fmix(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}
static inline uint32_t H2(uint32_t a, uint32_t b) {
    return fmix(a * 0x9E3779B1u + fmix(b + 0x7F4A7C15u));
}
static inline uint32_t H3(uint32_t a, uint32_t b, uint32_t c) { return H2(H2(a, b), c); }

static inline int is_word_char(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '\'' || c >= 128;
}
static inline int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
static inline uint32_t word_step(uint32_t w, int c) {
    return (w + (uint32_t)lower(c) + 1u) * 0x2F0B3A49u;
}

uint32_t mn_word_hash(const char *word, size_t len) {
    uint32_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int c = (unsigned char)word[i];
        if (is_word_char(c)) w = word_step(w, c);
    }
    return w;
}

/* ------------------------------------------------------- vector helpers */

/* Fixed-length loops over the padded input vector so the compiler can use
 * SIMD; unused inputs are kept at zero. */
static inline float dotp(const float *restrict w, const float *restrict x) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < NIN_MAX; i += 8)
        for (int j = 0; j < 8; j++) acc[j] += w[i + j] * x[i + j];
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}
static inline void train(float *restrict w, const float *restrict x, float e) {
    for (int i = 0; i < NIN_MAX; i++) w[i] += e * x[i];
}
static inline float dot_small(const float *w, const float *x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += w[i] * x[i];
    return s;
}
static inline void train_small(float *w, const float *x, int n, float e) {
    for (int i = 0; i < n; i++) w[i] += e * x[i];
}

/* ------------------------------------------------- APM (calibration) */

static inline float apm_p(uint16_t (*t)[33], uint32_t cx, float s, uint32_t *idx, float *wt) {
    float pos = (s + 8.0f) * 2.0f;
    if (pos < 0) pos = 0;
    if (pos > 31.999f) pos = 31.999f;
    int lo = (int)pos;
    float w = pos - (float)lo;
    *idx = cx * 33 + (uint32_t)lo;
    *wt = w;
    const uint16_t *e = &t[0][0] + *idx;
    return ((1.0f - w) * e[0] + w * e[1]) * (1.0f / 65536.0f);
}
static inline void apm_update(uint16_t (*t)[33], uint32_t idx, float w, int y, int rate) {
    uint16_t *e = &t[0][0] + idx;
    int g = y ? 65535 : 0;
    e[0] = (uint16_t)(e[0] + (int)((g - (int)e[0]) * (1.0f - w)) / (1 << rate));
    e[1] = (uint16_t)(e[1] + (int)((g - (int)e[1]) * w) / (1 << rate));
}
static void apm_init(uint16_t (*t)[33], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 33; j++) {
            double p = 1.0 / (1.0 + exp(-((j - 16) * 0.5)));
            int v = (int)(p * 65536.0);
            if (v > 65535) v = 65535;
            t[i][j] = (uint16_t)v;
        }
}

/* ----------------------------------------------------- hash table */

static inline int bpri(uint8_t s) { return (s >> 4) + (s & 15); }

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(p) __builtin_prefetch(p)
#else
#define PREFETCH(p) ((void)(p))
#endif

static inline uint32_t tab_hash(uint32_t h) { return fmix(h ^ 0xA5A5A5A5u); }

static Bucket *tab_find_hh(mn_model *m, uint32_t hh) {
    uint8_t chk = (uint8_t)(hh & 255u);
    uint32_t idx = (hh >> 8) & m->tmask;
    Bucket *line = m->tab + (idx & ~3u);
    int best = 0, bp = 1 << 30;
    for (int j = 0; j < 4; j++) {
        if (line[j].chk == chk) return &line[j];
        int p = bpri(line[j].st[0]);
        if (p < bp) {
            bp = p;
            best = j;
        }
    }
    if (!m->learn) return &zero_bucket;
    memset(&line[best], 0, sizeof(Bucket));
    line[best].chk = chk;
    return &line[best];
}

/* Fetch the buckets of all columns: issue every memory request first so the
 * cache misses overlap, then search the lines. */
static void fetch_buckets(mn_model *m, Bucket **out, uint32_t add) {
    uint32_t hh[NCTX_MAX];
    const int n = m->nctx;
    for (int i = 0; i < n; i++) {
        hh[i] = tab_hash(m->h[i] + add);
        PREFETCH(m->tab + (((hh[i] >> 8) & m->tmask) & ~3u));
    }
    for (int i = 0; i < n; i++) out[i] = tab_find_hh(m, hh[i]);
}

/* ------------------------------------------------------- concepts */

uint32_t mn_get_class(const mn_model *m, uint32_t wh) {
    if (!m->cmask || !wh) return 0;
    uint32_t i = fmix(wh) & m->cmask;
    while (m->ckey[i]) {
        if (m->ckey[i] == wh) return m->cval[i];
        i = (i + 1) & m->cmask;
    }
    return 0;
}

void mn_set_class(mn_model *m, uint32_t wh, uint32_t cls) {
    if (!m->cmask || !wh) return;
    uint32_t i = fmix(wh) & m->cmask;
    while (m->ckey[i] && m->ckey[i] != wh) i = (i + 1) & m->cmask;
    m->ckey[i] = wh;
    m->cval[i] = (uint16_t)cls;
}

/* ------------------------------------------------------- contexts */

static inline uint32_t len_bucket(uint32_t len) {
    if (len == 0) return 0;
    if (len < 16) return 1 + (len >> 2);          /* 1..4  */
    if (len < 32) return 5 + ((len - 16) >> 3);   /* 5..6  */
    if (len < 64) return 7 + ((len - 32) >> 4);   /* 7..8  */
    if (len < 128) return 9 + ((len - 64) >> 5);  /* 9..10 */
    if (len < 256) return 11;
    if (len < 512) return 12;
    if (len < 1024) return 13;
    if (len < 4096) return 14;
    return 15;
}

static void compute_contexts(mn_model *m) {
    uint32_t c4 = m->c4, c8 = m->c8, c12 = m->c12;
    uint32_t cur = m->wlen ? m->w0 : (m->sep + 0x9E3779B9u);
    uint32_t *h = m->hraw;
    h[0] = H2(1, c4 & 0xFFFFu);
    h[1] = H2(2, c4 & 0xFFFFFFu);
    h[2] = H2(3, c4);
    h[3] = H3(4, c4, c8 & 0xFFu);
    h[4] = H3(5, c4, c8 & 0xFFFFu);
    h[5] = H3(6, c4, c8);
    h[6] = H3(7, H2(c4, c8), c12);
    h[7] = H2(8, cur);
    h[8] = H3(9, cur, m->w1);
    h[9] = H3(10, H2(cur, m->w1), m->w2);
    h[10] = H3(11, cur, m->w2);
    h[11] = H3(12, H2(cur, m->w1), H2(m->w2, m->w3));
    h[12] = H3(13, cur, m->k1);
    h[13] = H3(14, cur, m->k1 | (m->k2 << 16));
    h[14] = H3(15, H2(cur, m->w1), m->k2);
    h[15] = H3(16, H2(cur, m->k1), m->w2);
    for (int i = 0; i < m->nctx; i++) m->h[i] = h[m->active[i]];
}

static void undo_push(mn_model *m, int kind, uint32_t i, uint32_t v) {
    if (m->undo_n == m->undo_cap) {
        size_t nc = m->undo_cap ? m->undo_cap * 2 : 4096;
        uint32_t *a = (uint32_t *)realloc(m->undo_i, nc * 4);
        if (!a) return;
        m->undo_i = a;
        uint32_t *b = (uint32_t *)realloc(m->undo_v, nc * 4);
        if (!b) return;
        m->undo_v = b;
        uint8_t *k = (uint8_t *)realloc(m->undo_k, nc);
        if (!k) return;
        m->undo_k = k;
        m->undo_cap = nc;
    }
    m->undo_i[m->undo_n] = i;
    m->undo_v[m->undo_n] = v;
    m->undo_k[m->undo_n] = (uint8_t)kind;
    m->undo_n++;
}

static void byte_update(mn_model *m, int c) {
    /* episodic recall: did the recalled continuation come true? */
    if (m->mlen) {
        if (m->hist[m->mptr & m->hmask] == (uint8_t)c) {
            m->mlen++;
            m->mptr++;
        } else {
            m->mlen = 0;
        }
    }
    if (m->undo_on) undo_push(m, 1, (uint32_t)(m->pos & m->hmask), m->hist[m->pos & m->hmask]);
    m->hist[m->pos & m->hmask] = (uint8_t)c;
    m->pos++;
    m->c12 = (m->c12 << 8) | (m->c8 >> 24);
    m->c8 = (m->c8 << 8) | (m->c4 >> 24);
    m->c4 = (m->c4 << 8) | (uint32_t)c;

    if (m->pos >= MINLEN_L) {
        uint32_t hs = H2(m->c4, m->c8 & 0xFFu) & m->mmask;
        uint32_t hl = H3(m->c4, m->c8, m->c12 & 0xFFFFFFFFu) & m->mmask;
        if (m->mlen == 0) {
            uint32_t cands[2] = {m->midx[hl], m->midx[hs]};
            for (int k = 0; k < 2 && m->mlen == 0; k++) {
                uint32_t cand = cands[k];
                uint64_t p = m->pos;
                if (cand == 0 || cand >= p || p - cand >= (uint64_t)m->hmask) continue;
                uint32_t L = 0;
                while (L < 64 && L < cand &&
                       m->hist[(cand - 1 - L) & m->hmask] == m->hist[(p - 1 - L) & m->hmask])
                    L++;
                if (L >= MINLEN_S) {
                    m->mlen = L;
                    m->mptr = cand;
                }
            }
        }
        /* the recall index is short-term memory: it keeps recording even
         * when long-term learning is frozen (like a context window) */
        if (m->undo_on) {
            undo_push(m, 0, hs, m->midx[hs]);
            undo_push(m, 0, hl, m->midx[hl]);
        }
        m->midx[hs] = (uint32_t)m->pos;
        m->midx[hl] = (uint32_t)m->pos;
    }

    if (m->lex) {
        const float *t = lex_tree(m->lex);
        if (t) {
            m->lex_bits -= log2((double)t[256 + c] / (double)t[1]);
            m->lex_bytes++;
        }
        lex_observe(m->lex, c, m->learn);
    }

    /* words and concepts */
    if (is_word_char(c)) {
        if (m->wlen == 0) m->w0 = 0;
        m->w0 = word_step(m->w0, c);
        m->wlen++;
    } else {
        if (m->wlen) {
            m->w3 = m->w2;
            m->w2 = m->w1;
            m->w1 = m->w0;
            m->k2 = m->k1;
            m->k1 = mn_get_class(m, m->w1);
            m->wlen = 0;
            m->sep = 0;
        }
        m->sep = (m->sep + (uint32_t)c + 1u) * 0x3C6EF372u;
    }
    compute_contexts(m);
}

/* ------------------------------------------------------- prediction */

static void predict(mn_model *m) {
    const uint32_t bp = m->bitpos, c0 = m->c0;
    const int nctx = m->nctx;
    if (bp == 0)
        fetch_buckets(m, m->bhi, 0);
    else if (bp == 4)
        fetch_buckets(m, m->blo, (c0 + 1u) * 0x6F4F2A35u);
    Bucket **b = (bp < 4) ? m->bhi : m->blo;
    const int slot = (int)m->nib - 1;
    float *x = m->x;
    int k = 0, fam = 0;
    for (int i = 0; i < nctx; i++) {
        uint8_t *sp = &b[i]->st[slot];
        uint8_t s = *sp;
        m->sp[i] = sp;
        m->sv[i] = s;
        if (s == 0) {
            x[k++] = 0;
            x[k++] = 0;
        } else {
            float st = sm_stretch(m->sm[i][s]);
            int det = ((s >> 4) == 0) != ((s & 15) == 0);
            x[k++] = st;
            x[k++] = det ? st : 0;
            if (m->active[i] < 7) fam = m->active[i] + 1;
        }
    }
    /* order 0 and order 1 */
    x[k++] = sm_stretch(m->o0[c0]);
    {
        uint8_t s = m->o1[((m->c4 & 255u) << 8) | c0];
        m->o1s = s;
        x[k++] = s ? sm_stretch(m->sm[nctx][s]) : 0;
    }
    /* episodic recall */
    m->mexp = -1;
    m->mlenb = 0;
    if (m->mlen) {
        uint32_t pb = m->hist[m->mptr & m->hmask];
        if (((pb + 256u) >> (8 - bp)) == c0) {
            m->mexp = (int)((pb >> (7 - bp)) & 1u);
            m->mlenb = len_bucket(m->mlen);
        }
    }
    if (m->mexp >= 0) {
        float st = sm_stretch(m->smm[m->mlenb]);
        float sg = m->mexp ? 1.0f : -1.0f;
        float ml = (float)(m->mlen > 32 ? 32 : m->mlen);
        x[k++] = sg * st;
        x[k++] = sg * ml * (1.0f / 8.0f);
    } else {
        x[k++] = 0;
        x[k++] = 0;
    }
    /* mental lexicon: whole-word expectation of the next letter */
    m->lstate = 0;
    float lexst = 0;
    if (m->lex) {
        const float *t = lex_tree(m->lex);
        if (t && t[c0] > 0) {
            float p = t[2 * c0 + 1] / t[c0];
            float st = stretchf(p < 1.0f / 4096 ? 1.0f / 4096 : p > 4095.0f / 4096 ? 4095.0f / 4096 : p);
            m->lstate = lex_state(m->lex);
            lexst = st;
            x[k++] = st;
            x[k++] = m->lstate == 3 ? st : 0;
        } else {
            x[k++] = 0;
            x[k++] = 0;
        }
    }
    x[k++] = 0.5f; /* bias */
    m->nin = k;
    while (k < NIN_MAX) x[k++] = 0;

    /* gated pooling: four gates selected by different situations */
    m->msel[0] = (int)c0;
    m->msel[1] = (int)(((m->mlenb * 2u + (m->mexp > 0)) << 3) | bp);
    m->msel[2] = (int)(m->c4 & 255u);
    m->msel[3] = (fam << 3) | (int)bp;
    m->msel[4] = (m->lstate << 3) | (int)bp;
    float *ws[NM] = {m->wm0[m->msel[0]], m->wm1[m->msel[1]], m->wm2[m->msel[2]],
                     m->wm3[m->msel[3]], m->wm4[m->msel[4]]};
    for (int j = 0; j < NM; j++) {
        float s = dotp(ws[j], x);
        if (s > 16) s = 16;
        if (s < -16) s = -16;
        m->ms[j] = s;
        m->mp[j] = squashf(s);
        m->fx[j] = s * 0.25f;
    }
    m->fx[NM] = lexst * 0.25f;
    m->fx[NM + 1] = 0.5f;
    m->fsel = (m->lstate << 3) | (int)bp;
    float fs = dot_small(m->wf[m->fsel], m->fx, NFIN);
    if (fs > 16) fs = 16;
    if (fs < -16) fs = -16;
    m->fs = fs;
    m->fp = squashf(fs);

    /* calibration */
    uint32_t c1 = m->c4 & 255u;
    m->ap[0] = apm_p(m->apm1, c0 | (c1 << 8), fs, &m->aidx[0], &m->aw[0]);
    m->ap[1] = apm_p(m->apm2, H2(m->c4 & 0xFFFFu, c0) & 0xFFFFu, fs, &m->aidx[1], &m->aw[1]);
    m->ap[2] = apm_p(m->apm3, ((m->mlenb * 2u + (m->mexp > 0)) << 8) | c0, fs, &m->aidx[2],
                     &m->aw[2]);
#ifndef FINAL_W
#define FINAL_W 0.5f
#endif
    const float wa = (1.0f - FINAL_W) / 3.0f;
    float p = FINAL_W * m->fp + wa * (m->ap[0] + m->ap[1] + m->ap[2]);
    m->pfin = p;
    int p16 = (int)(p * 65536.0f);
    if (p16 < 16) p16 = 16;
    if (p16 > 65520) p16 = 65520;
    m->p16 = p16;
    m->predicted = 1;
}

static void update(mn_model *m, int y) {
    if (!m->predicted) predict(m);
    if (m->learn) {
        const int nctx = m->nctx;
        for (int i = 0; i < nctx; i++) {
            uint8_t s = m->sv[i];
            if (s) sm_update(&m->sm[i][s], y, SM_LIMIT);
            *m->sp[i] = nex_tab[s][y];
        }
        sm_update(&m->o0[m->c0], y, SM_LIMIT);
        {
            uint32_t oi = ((m->c4 & 255u) << 8) | m->c0;
            if (m->o1s) sm_update(&m->sm[nctx][m->o1s], y, SM_LIMIT);
            m->o1[oi] = nex_tab[m->o1s][y];
        }
        if (m->mexp >= 0) sm_update(&m->smm[m->mlenb], y == m->mexp, SM_LIMIT);

        float lr = m->cfg.lr + m->cfg.lr0 / (1.0f + (float)m->nbits * (1.0f / 4194304.0f));
        m->lr_now = lr;
        float *ws[NM] = {m->wm0[m->msel[0]], m->wm1[m->msel[1]], m->wm2[m->msel[2]],
                         m->wm3[m->msel[3]], m->wm4[m->msel[4]]};
        for (int j = 0; j < NM; j++) train(ws[j], m->x, ((float)y - m->mp[j]) * lr);
        train_small(m->wf[m->fsel], m->fx, NFIN, ((float)y - m->fp) * lr * 0.5f);
        apm_update(m->apm1, m->aidx[0], m->aw[0], y, 6);
        apm_update(m->apm2, m->aidx[1], m->aw[1], y, 6);
        apm_update(m->apm3, m->aidx[2], m->aw[2], y, 6);
        m->nbits++;
    }
    m->predicted = 0;
    m->c0 = (m->c0 << 1) | (uint32_t)y;
    m->nib = (m->nib << 1) | (uint32_t)y;
    m->bitpos++;
    if (m->bitpos == 4) m->nib = 1;
    if (m->bitpos == 8) {
        byte_update(m, (int)(m->c0 & 255u));
        m->c0 = 1;
        m->bitpos = 0;
        m->nib = 1;
    }
}

/* ------------------------------------------------------------ public */

void mn_default_config(mn_config *c) {
    c->table_bits = 22;
    c->hist_bits = 24;
    c->match_bits = 22;
    c->class_bits = 0;
    c->lr = 0.002f;
    c->lr0 = 0.02f;
    c->ctx_mask = 0;
    c->lex_order = 4;
    c->lex_spaced = 0;
}

/* Large zeroed allocation; on Linux ask for transparent huge pages so that
 * random probes into the context table do not thrash the TLB. */
static void *big_alloc(size_t n) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    void *p = NULL;
    if (n >= ((size_t)2 << 20) && posix_memalign(&p, (size_t)2 << 20, n) == 0) {
        madvise(p, n, MADV_HUGEPAGE);
        memset(p, 0, n);
        return p;
    }
#endif
    return calloc(1, n);
}

static int alloc_big(mn_model *m) {
    const mn_config *c = &m->cfg;
    m->tmask = (1u << c->table_bits) - 1u;
    m->hmask = (1u << c->hist_bits) - 1u;
    m->mmask = (1u << c->match_bits) - 1u;
    m->tab = (Bucket *)big_alloc(((size_t)1 << c->table_bits) * sizeof(Bucket));
    m->hist = (uint8_t *)big_alloc((size_t)1 << c->hist_bits);
    m->midx = (uint32_t *)big_alloc(((size_t)1 << c->match_bits) * sizeof(uint32_t));
    if (c->class_bits > 0) {
        m->cmask = (1u << c->class_bits) - 1u;
        m->ckey = (uint32_t *)calloc((size_t)1 << c->class_bits, sizeof(uint32_t));
        m->cval = (uint16_t *)calloc((size_t)1 << c->class_bits, sizeof(uint16_t));
    } else {
        m->cmask = 0;
        m->ckey = NULL;
        m->cval = NULL;
    }
    return m->tab && m->hist && m->midx && (c->class_bits == 0 || (m->ckey && m->cval));
}

mn_model *mn_new(const mn_config *cfg) {
    init_tables();
    mn_model *m = (mn_model *)calloc(1, sizeof(mn_model));
    if (!m) return NULL;
    if (cfg)
        m->cfg = *cfg;
    else
        mn_default_config(&m->cfg);
    if (m->cfg.table_bits < 10) m->cfg.table_bits = 10;
    if (m->cfg.table_bits > 24) m->cfg.table_bits = 24;
    if (m->cfg.hist_bits < 12) m->cfg.hist_bits = 12;
    if (m->cfg.hist_bits > 30) m->cfg.hist_bits = 30;
    if (m->cfg.match_bits < 10) m->cfg.match_bits = 10;
    if (m->cfg.match_bits > 28) m->cfg.match_bits = 28;
    if (m->cfg.class_bits > 24) m->cfg.class_bits = 24;
    if (!alloc_big(m)) {
        mn_free(m);
        return NULL;
    }
    m->learn = 1;
    {
        uint32_t mask = m->cfg.ctx_mask ? m->cfg.ctx_mask : 0xFFFFu;
        if (m->cfg.class_bits == 0) mask &= 0x0FFFu;
        m->nctx = 0;
        for (int i = 0; i < NCTX_MAX; i++)
            if (mask >> i & 1u) m->active[m->nctx++] = (uint8_t)i;
    }
    m->c0 = 1;
    m->nib = 1;
    for (int i = 0; i <= NCTX_MAX; i++)
        for (int s = 0; s < 256; s++) m->sm[i][s] = sm_init_value(s);
    for (int i = 0; i < 16; i++) m->smm[i] = (uint32_t)(0.75 * ((1 << 22) - 1)) << 10;
    for (int i = 0; i < 256; i++) m->o0[i] = 1u << 31;
    for (int s = 0; s < M0SETS; s++)
        for (int i = 0; i < NIN_MAX; i++) m->wm0[s][i] = 0.15f;
    for (int s = 0; s < M1SETS; s++)
        for (int i = 0; i < NIN_MAX; i++) m->wm1[s][i] = 0.15f;
    for (int s = 0; s < M2SETS; s++)
        for (int i = 0; i < NIN_MAX; i++) m->wm2[s][i] = 0.15f;
    for (int s = 0; s < M3SETS; s++)
        for (int i = 0; i < NIN_MAX; i++) m->wm3[s][i] = 0.15f;
    for (int s = 0; s < M4SETS; s++)
        for (int i = 0; i < NIN_MAX; i++) m->wm4[s][i] = 0.15f;
    if (m->cfg.lex_order > 0) {
        m->lex = lex_new(m->cfg.lex_order, m->cfg.lex_spaced);
        if (!m->lex) {
            mn_free(m);
            return NULL;
        }
    }
    for (int s = 0; s < FSETS; s++)
        for (int i = 0; i < NFIN; i++) m->wf[s][i] = (i < NM) ? 1.0f : 0.0f;
    apm_init(m->apm1, 65536);
    apm_init(m->apm2, 65536);
    apm_init(m->apm3, 8192);
    compute_contexts(m);
    return m;
}

void mn_free(mn_model *m) {
    if (!m) return;
    free(m->tab);
    free(m->hist);
    free(m->midx);
    free(m->ckey);
    free(m->cval);
    lex_free(m->lex);
    free(m->undo_i);
    free(m->undo_v);
    free(m->undo_k);
    free(m);
}

size_t mn_memory(const mn_model *m) {
    size_t s = sizeof(mn_model);
    s += ((size_t)m->tmask + 1) * sizeof(Bucket);
    s += (size_t)m->hmask + 1;
    s += ((size_t)m->mmask + 1) * sizeof(uint32_t);
    if (m->cmask) s += ((size_t)m->cmask + 1) * (sizeof(uint32_t) + sizeof(uint16_t));
    if (m->lex) s += lex_memory(m->lex);
    return s;
}

void mn_set_learning(mn_model *m, int on) {
    on = on ? 1 : 0;
    if (on == m->learn) return;
    m->learn = on;
    m->predicted = 0;
    if (on) {
        /* buckets fetched while frozen may be the shared read-only zero bucket */
        if (m->bitpos > 0 && m->bitpos < 4) fetch_buckets(m, m->bhi, 0);
        if (m->bitpos > 4)
            fetch_buckets(m, m->blo, ((m->c0 >> (m->bitpos - 4)) + 1u) * 0x6F4F2A35u);
    }
}
int mn_get_learning(const mn_model *m) { return m->learn; }
void mn_lex_stats(mn_model *m, double *bits, uint64_t *bytes, int reset) {
    if (bits) *bits = m->lex_bits;
    if (bytes) *bytes = m->lex_bytes;
    if (reset) {
        m->lex_bits = 0;
        m->lex_bytes = 0;
    }
}
int mn_num_inputs(const mn_model *m) { return m->nctx * 2 + 5 + (m->lex ? 2 : 0); }

int mn_p16(mn_model *m) {
    predict(m);
    return m->p16;
}

void mn_update(mn_model *m, int bit) { update(m, bit ? 1 : 0); }

double mn_observe_byte(mn_model *m, int byte) {
    double bits = 0;
    for (int i = 7; i >= 0; i--) {
        int y = (byte >> i) & 1;
        predict(m);
        double p = m->p16 / 65536.0;
        bits -= log2(y ? p : 1.0 - p);
        update(m, y);
    }
    return bits;
}

double mn_observe(mn_model *m, const uint8_t *buf, size_t n) {
    double bits = 0;
    for (size_t i = 0; i < n; i++) bits += mn_observe_byte(m, buf[i]);
    return bits;
}

static void dist_rec(mn_model *m, uint32_t c0, uint32_t bp, uint32_t nib, double q, double *out) {
    if (bp == 8) {
        out[c0 & 255u] = q;
        return;
    }
    if (q < 1e-12) {
        /* negligible branch: spread nothing, leave zeros */
        return;
    }
    m->c0 = c0;
    m->bitpos = bp;
    m->nib = nib;
    predict(m);
    double p1 = m->pfin;
    uint32_t nn1 = (bp == 3) ? 1 : nib * 2 + 1, nn0 = (bp == 3) ? 1 : nib * 2;
    dist_rec(m, c0 * 2 + 1, bp + 1, nn1, q * p1, out);
    dist_rec(m, c0 * 2, bp + 1, nn0, q * (1.0 - p1), out);
}

void mn_next_byte_dist(mn_model *m, double *out) {
    int was = m->learn;
    m->learn = 0;
    for (int i = 0; i < 256; i++) out[i] = 0;
    /* must start at a byte boundary */
    uint32_t c0 = m->c0, bp = m->bitpos, nib = m->nib;
    dist_rec(m, 1, 0, 1, 1.0, out);
    double s = 0;
    for (int i = 0; i < 256; i++) s += out[i];
    if (s > 0)
        for (int i = 0; i < 256; i++) out[i] /= s;
    m->c0 = c0;
    m->bitpos = bp;
    m->nib = nib;
    m->predicted = 0;
    m->learn = was;
}

static inline double rng01(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return (double)((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

int mn_sample_byte(mn_model *m, double temperature, uint64_t *rng) {
    int c = 0;
    for (int i = 0; i < 8; i++) {
        predict(m);
        double p = m->pfin;
        if (temperature > 0 && temperature != 1.0) {
            double s = log(p / (1 - p)) / temperature;
            p = 1.0 / (1.0 + exp(-s));
        } else if (temperature <= 0) {
            p = p >= 0.5 ? 1.0 : 0.0;
        }
        int y = rng01(rng) < p;
        update(m, y);
        c = (c << 1) | y;
    }
    return c;
}

/* ------------------------------------------------------------ persistence */

#define MN_MAGIC 0x4F4D4E4Du /* "MNMO" */

int mn_save(const mn_model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = MN_MAGIC, size = (uint32_t)sizeof(mn_model);
    int ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&size, 4, 1, f) == 1 &&
             fwrite(m, sizeof(mn_model), 1, f) == 1 &&
             fwrite(m->tab, sizeof(Bucket), (size_t)m->tmask + 1, f) == (size_t)m->tmask + 1 &&
             fwrite(m->hist, 1, (size_t)m->hmask + 1, f) == (size_t)m->hmask + 1 &&
             fwrite(m->midx, 4, (size_t)m->mmask + 1, f) == (size_t)m->mmask + 1;
    if (ok && m->cmask)
        ok = fwrite(m->ckey, 4, (size_t)m->cmask + 1, f) == (size_t)m->cmask + 1 &&
             fwrite(m->cval, 2, (size_t)m->cmask + 1, f) == (size_t)m->cmask + 1;
    if (ok && m->lex) ok = lex_save(m->lex, f) == 0;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

mn_model *mn_load(const char *path) {
    init_tables();
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = 0, size = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&size, 4, 1, f) != 1 || magic != MN_MAGIC ||
        size != sizeof(mn_model)) {
        fclose(f);
        return NULL;
    }
    mn_model *m = (mn_model *)calloc(1, sizeof(mn_model));
    if (!m || fread(m, sizeof(mn_model), 1, f) != 1) {
        free(m);
        fclose(f);
        return NULL;
    }
    m->undo_i = m->undo_v = NULL;
    m->undo_k = NULL;
    m->undo_n = m->undo_cap = 0;
    m->undo_on = 0;
    m->lex = NULL;
    m->ckey = NULL;
    m->cval = NULL;
    if (!alloc_big(m)) {
        mn_free(m);
        fclose(f);
        return NULL;
    }
    int ok = fread(m->tab, sizeof(Bucket), (size_t)m->tmask + 1, f) == (size_t)m->tmask + 1 &&
             fread(m->hist, 1, (size_t)m->hmask + 1, f) == (size_t)m->hmask + 1 &&
             fread(m->midx, 4, (size_t)m->mmask + 1, f) == (size_t)m->mmask + 1;
    if (ok && m->cmask)
        ok = fread(m->ckey, 4, (size_t)m->cmask + 1, f) == (size_t)m->cmask + 1 &&
             fread(m->cval, 2, (size_t)m->cmask + 1, f) == (size_t)m->cmask + 1;
    m->lex = NULL;
    if (ok && m->cfg.lex_order > 0) {
        m->lex = lex_load(f);
        ok = m->lex != NULL;
    }
    fclose(f);
    if (!ok) {
        mn_free(m);
        return NULL;
    }
    m->predicted = 0;
    /* pointers in the saved image belong to the saving process */
    m->undo_on = 0;
    m->undo_i = m->undo_v = NULL;
    m->undo_k = NULL;
    m->undo_n = m->undo_cap = 0;
    if (m->bitpos != 0) {
        /* saved mid-byte: re-fetch both nibble buckets on next predict */
        m->c0 = 1;
        m->bitpos = 0;
        m->nib = 1;
    }
    return m;
}

/* ------------------------------------------------------------ imagination */

struct mn_snap {
    int learn;
    uint64_t pos;
    uint32_t c0, bitpos, nib, c4, c8, c12, w0, w1, w2, w3, wlen, sep, k1, k2;
    uint32_t h[NCTX_MAX], hraw[NCTX_MAX];
    uint32_t mptr, mlen;
    size_t undo_mark;
    int outer_undo;
    lex_snap *lex;
};

mn_snap *mn_snapshot(mn_model *m) {
    mn_snap *z = (mn_snap *)calloc(1, sizeof(mn_snap));
    if (!z) return NULL;
    z->learn = m->learn;
    z->pos = m->pos;
    z->c0 = m->c0;
    z->bitpos = m->bitpos;
    z->nib = m->nib;
    z->c4 = m->c4;
    z->c8 = m->c8;
    z->c12 = m->c12;
    z->w0 = m->w0;
    z->w1 = m->w1;
    z->w2 = m->w2;
    z->w3 = m->w3;
    z->wlen = m->wlen;
    z->sep = m->sep;
    z->k1 = m->k1;
    z->k2 = m->k2;
    memcpy(z->h, m->h, sizeof z->h);
    memcpy(z->hraw, m->hraw, sizeof z->hraw);
    z->mptr = m->mptr;
    z->mlen = m->mlen;
    z->undo_mark = m->undo_n;
    z->outer_undo = m->undo_on;
    if (m->lex) {
        z->lex = lex_snapshot(m->lex);
        if (!z->lex) {
            free(z);
            return NULL;
        }
    }
    mn_set_learning(m, 0);
    m->undo_on = 1;
    return z;
}

void mn_restore(mn_model *m, mn_snap *z) {
    if (!z) return;
    while (m->undo_n > z->undo_mark) {
        size_t i = --m->undo_n;
        if (m->undo_k[i])
            m->hist[m->undo_i[i]] = (uint8_t)m->undo_v[i];
        else
            m->midx[m->undo_i[i]] = m->undo_v[i];
    }
    m->undo_on = z->outer_undo;
    m->pos = z->pos;
    m->c0 = z->c0;
    m->bitpos = z->bitpos;
    m->nib = z->nib;
    m->c4 = z->c4;
    m->c8 = z->c8;
    m->c12 = z->c12;
    m->w0 = z->w0;
    m->w1 = z->w1;
    m->w2 = z->w2;
    m->w3 = z->w3;
    m->wlen = z->wlen;
    m->sep = z->sep;
    m->k1 = z->k1;
    m->k2 = z->k2;
    memcpy(m->h, z->h, sizeof z->h);
    memcpy(m->hraw, z->hraw, sizeof z->hraw);
    m->mptr = z->mptr;
    m->mlen = z->mlen;
    if (m->lex && z->lex) lex_restore(m->lex, z->lex);
    lex_snap_free(z->lex);
    m->predicted = 0;
    m->learn = z->learn;
    free(z);
}

size_t mn_generate(mn_model *m, uint8_t *out, size_t n, double temperature, uint64_t seed,
                   int stop_byte) {
    uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
    if (!rng) rng = 1;
    size_t i = 0;
    while (i < n) {
        int c = mn_sample_byte(m, temperature, &rng);
        out[i++] = (uint8_t)c;
        if (c == stop_byte) break;
    }
    return i;
}

mn_lex *mn_lexicon(mn_model *m) { return m->lex; }
