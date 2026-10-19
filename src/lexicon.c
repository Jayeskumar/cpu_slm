/*
 * lexicon.c - the mental lexicon (see lexicon.h).
 *
 * The text is cut into tokens by character class: runs of letters/digits,
 * runs of punctuation and runs of line breaks are tokens; spaces and tabs only
 * separate them.  Every token is stored once in a vocabulary trie.
 *
 * Token statistics are interpolated modified Kneser-Ney (Chen & Goodman),
 * maintained online: the top order keeps raw counts, lower orders keep
 * continuation counts (how many distinct contexts a token followed) and the
 * discounts are re-estimated from count-of-counts as data arrives.  The next
 * token distribution is never materialised: it is a sparse part (discounted
 * counts of tokens seen in the current contexts) plus a dense part (back-off
 * weight times the unigram continuation distribution) whose mass under every
 * spelling prefix is read from subtree sums kept in the trie.
 */
#include "lexicon.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LEX_MAXLEN 40
#define LEX_MAXORD 6
#define LEX_MAXWORDS (1u << 22)
#define UNK_TOKEN 0xFFFFFFFEu
#define NONE 0xFFFFFFFFu

typedef struct {
    uint64_t *k;
    uint32_t *v;
    uint32_t cap, n;
} HIdx;

typedef struct {
    uint32_t w;
    uint32_t cc; /* count << 8 | first byte of the token */
} Foll;
#define FCOUNT(f) ((f).cc >> 8)
#define FMAXC 0xFFFFFFu

/* A context slot: 32-bit fingerprint of the context, total count, number of
 * distinct followers and either the follower itself (nf == 1, its count is
 * total) or the arena offset of a follower list sorted by token id. */
typedef struct {
    uint32_t fp, total, nf, x;
} CSlot;

typedef struct {
    CSlot *s;
    uint32_t cap, n;
} CTab;

typedef struct {
    uint32_t first, next, parent, word; /* word = id + 1 when terminal */
    uint64_t sub;                       /* continuation counts in subtree */
    uint8_t ch, depth;
} Node;

typedef struct {
    uint32_t word, off; /* token id and spelling offset in the pool */
    float wt;
    uint8_t len, ch;    /* spelling length, first byte */
} Cand;

struct mn_lex {
    int order;
    int spaced;
    const uint8_t *cls; /* byte -> class table for the tokenization mode */
    /* vocabulary */
    HIdx widx;
    uint32_t nw, wcap;
    uint32_t *woff, *wnode, *wcc1;
    uint8_t *wlen;
    char *pool;
    size_t pool_n, pool_cap;
    uint64_t total_cc1;
    /* trie */
    Node *nodes;
    uint32_t nn, ncap;
    /* n-gram statistics */
    CTab ct;
    Foll *ar;
    uint32_t ar_n, ar_cap;
    uint32_t freel[32];
    uint32_t coc[LEX_MAXORD + 1][5];
    float disc[LEX_MAXORD + 1][3];
    uint32_t disc_age;
    /* stream state */
    uint32_t tok[LEX_MAXORD];
    int ntok;
    uint8_t cur[LEX_MAXLEN + 1];
    int curlen, curcls, lost, root_valid, prev;
    uint32_t node;
    /* candidates for the next token */
    Cand *cand;
    uint32_t ncand, candcap;
    double g_dense;
    /* spacing */
    uint32_t wsb[256][2];   /* at a boundary, by previous byte: token starts / space */
    uint32_t wsa[4][2];     /* after a token of class k: attached token / space */
    uint32_t wsc[256][3];   /* which space character, by previous byte */
    uint16_t att[256][256]; /* first byte of an attached token, by previous byte */
    uint32_t attt[256];
    uint32_t attsum[256][4]; /* att[pv][*] summed per byte class */
    uint16_t nclass[4];      /* number of byte values in each class */
    float root_mass[256];    /* sparse first-byte masses of the root candidates */
    /* priming: two recency-weighted token caches (fast and slow decay); a
     * mixture of exponentials approximating power-law forgetting */
    double *pc[2];          /* per trie node: subtree cache weight (scaled) */
    double *wc[2];          /* per token: own cache weight (scaled) */
    double cscale[2], ctot[2], cdecay[2];
    float lam[2];
    double last_kn, last_c[2]; /* term masses of the token just completed */
    uint32_t pccap;
    /* output */
    float tree[512];
    int state;
};

/* ------------------------------------------------------------ helpers */

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* 0 = space, 1 = word, 2 = punctuation, 3 = line break.  In "spaced" mode
 * (pre-tokenized text) every run of non-space bytes is one token. */
static int lclass_raw(int c, int spaced) {
    if (c == ' ' || c == '\t' || c == '\r') return 0;
    if (c == '\n') return 3;
    if (spaced) return 1;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '\'' ||
        c >= 128)
        return 1;
    return 2;
}
static uint8_t lclass_tab[2][256];
static int lclass_ready = 0;
static void lclass_init(void) {
    if (lclass_ready) return;
    for (int m = 0; m < 2; m++)
        for (int c = 0; c < 256; c++) lclass_tab[m][c] = (uint8_t)lclass_raw(c, m);
    lclass_ready = 1;
}
#define lclass(c) (L->cls[(c) & 255])
static inline int space_index(int c) { return c == ' ' ? 0 : c == '\t' ? 1 : 2; }

static int grow(void **p, uint32_t *cap, uint32_t need, size_t elem) {
    if (need <= *cap) return 1;
    uint64_t nc = *cap ? *cap : 16;
    while (nc < need) nc = nc * 3 / 2 + 1;
    if (nc > 0xFFFFFFF0u) nc = 0xFFFFFFF0u;
    void *q = realloc(*p, (size_t)nc * elem);
    if (!q) return 0;
    *p = q;
    *cap = (uint32_t)nc;
    return 1;
}

static int hidx_init(HIdx *h, uint32_t cap) {
    h->cap = cap;
    h->n = 0;
    h->k = (uint64_t *)calloc(cap, sizeof(uint64_t));
    h->v = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    return h->k && h->v;
}
static void hidx_free(HIdx *h) {
    free(h->k);
    free(h->v);
    h->k = NULL;
    h->v = NULL;
}
static inline uint64_t fixkey(uint64_t k) { return k ? k : 1; }
static inline uint32_t hslot(uint64_t key, uint32_t cap) {
    return (uint32_t)(key ^ (key >> 32)) & (cap - 1);
}
static uint32_t *hidx_find(const HIdx *h, uint64_t key) {
    key = fixkey(key);
    uint32_t m = h->cap - 1, i = hslot(key, h->cap);
    while (h->k[i]) {
        if (h->k[i] == key) return &h->v[i];
        i = (i + 1) & m;
    }
    return NULL;
}
static int hidx_grow(HIdx *h) {
    HIdx n;
    if (!hidx_init(&n, h->cap * 2)) return 0;
    uint32_t m = n.cap - 1;
    for (uint32_t i = 0; i < h->cap; i++)
        if (h->k[i]) {
            uint32_t j = hslot(h->k[i], n.cap);
            while (n.k[j]) j = (j + 1) & m;
            n.k[j] = h->k[i];
            n.v[j] = h->v[i];
        }
    n.n = h->n;
    hidx_free(h);
    *h = n;
    return 1;
}
/* returns NULL on allocation failure; the pointer is valid until the next insert */
static uint32_t *hidx_insert(HIdx *h, uint64_t key, int *isnew) {
    if ((uint64_t)(h->n + 1) * 4 > (uint64_t)h->cap * 3 && !hidx_grow(h)) return NULL;
    key = fixkey(key);
    uint32_t m = h->cap - 1, i = hslot(key, h->cap);
    while (h->k[i]) {
        if (h->k[i] == key) {
            *isnew = 0;
            return &h->v[i];
        }
        i = (i + 1) & m;
    }
    h->k[i] = key;
    h->n++;
    *isnew = 1;
    return &h->v[i];
}

/* context table: open addressing on 32-bit fingerprints, home slot by
 * multiplicative range reduction so the table can grow by 1.5x */
static inline uint32_t key_fp(uint64_t key) {
    uint32_t fp = (uint32_t)(key >> 32) ^ (uint32_t)key;
    return fp ? fp : 1u;
}
static inline uint32_t ct_home(uint32_t fp, uint32_t cap) {
    return (uint32_t)(((uint64_t)fp * cap) >> 32);
}
static int ct_init(CTab *t, uint32_t cap) {
    t->s = (CSlot *)calloc(cap, sizeof(CSlot));
    t->cap = cap;
    t->n = 0;
    return t->s != NULL;
}
static CSlot *ct_find(const CTab *t, uint32_t fp) {
    uint32_t i = ct_home(fp, t->cap);
    while (t->s[i].fp) {
        if (t->s[i].fp == fp) return &t->s[i];
        if (++i == t->cap) i = 0;
    }
    return NULL;
}
static int ct_grow(CTab *t) {
    CTab n;
    uint64_t nc = (uint64_t)t->cap + t->cap / 2 + 1024;
    if (nc > 0xFFFFFFF0u || !ct_init(&n, (uint32_t)nc)) return 0;
    for (uint32_t i = 0; i < t->cap; i++)
        if (t->s[i].fp) {
            uint32_t j = ct_home(t->s[i].fp, n.cap);
            while (n.s[j].fp)
                if (++j == n.cap) j = 0;
            n.s[j] = t->s[i];
        }
    n.n = t->n;
    free(t->s);
    *t = n;
    return 1;
}
/* the returned pointer is valid until the next insertion */
static CSlot *ct_insert(CTab *t, uint32_t fp, int *isnew) {
    if ((uint64_t)(t->n + 1) * 5 > (uint64_t)t->cap * 4 && !ct_grow(t)) return NULL;
    uint32_t i = ct_home(fp, t->cap);
    while (t->s[i].fp) {
        if (t->s[i].fp == fp) {
            *isnew = 0;
            return &t->s[i];
        }
        if (++i == t->cap) i = 0;
    }
    memset(&t->s[i], 0, sizeof(CSlot));
    t->s[i].fp = fp;
    t->n++;
    *isnew = 1;
    return &t->s[i];
}

/* follower arena: blocks of 2^k entries with per-size free lists */
static uint32_t arena_alloc(mn_lex *L, int lg) {
    uint32_t sz = 1u << lg;
    if (L->freel[lg] != NONE) {
        uint32_t off = L->freel[lg];
        L->freel[lg] = L->ar[off].w;
        return off;
    }
    if (!grow((void **)&L->ar, &L->ar_cap, L->ar_n + sz, sizeof(Foll))) return NONE;
    uint32_t off = L->ar_n;
    L->ar_n += sz;
    return off;
}
static void arena_free(mn_lex *L, uint32_t off, int lg) {
    L->ar[off].w = L->freel[lg];
    L->freel[lg] = off;
}
static inline int cap_lg(uint32_t nf) { /* log2 of block size holding nf entries */
    int lg = 0;
    while ((1u << lg) < nf) lg++;
    return lg;
}

/* keep the per-node and per-token cache arrays as large as the trie/vocab */
static int cache_reserve(mn_lex *L) {
    uint32_t need = L->ncap > L->wcap ? L->ncap : L->wcap;
    if (need <= L->pccap) return 1;
    for (int i = 0; i < 2; i++) {
        double *a = (double *)realloc(L->pc[i], (size_t)need * sizeof(double));
        if (!a) return 0;
        L->pc[i] = a;
        double *b = (double *)realloc(L->wc[i], (size_t)need * sizeof(double));
        if (!b) return 0;
        L->wc[i] = b;
        for (uint32_t j = L->pccap; j < need; j++) a[j] = b[j] = 0;
    }
    L->pccap = need;
    return 1;
}

static void cache_rescale(mn_lex *L, int i) {
    double s = L->cscale[i];
    for (uint32_t j = 0; j < L->nn; j++) L->pc[i][j] /= s;
    for (uint32_t j = 0; j < L->nw; j++) L->wc[i][j] /= s;
    L->ctot[i] /= s;
    L->cscale[i] = 1.0;
}

/* a token was seen: decay every cache entry and add the token */
static void cache_add(mn_lex *L, uint32_t w) {
    if (!cache_reserve(L)) return;
    for (int i = 0; i < 2; i++) {
        L->cscale[i] /= (1.0 - L->cdecay[i]);
        if (L->cscale[i] > 1e200) cache_rescale(L, i);
        double a = L->cscale[i];
        L->wc[i][w] += a;
        L->ctot[i] += a;
        for (uint32_t n = L->wnode[w];; n = L->nodes[n].parent) {
            L->pc[i][n] += a;
            if (n == 0) break;
        }
    }
}

/* ------------------------------------------------------------ setup */

mn_lex *lex_new(int order, int spaced) {
    lclass_init();
    mn_lex *L = (mn_lex *)calloc(1, sizeof(mn_lex));
    if (!L) return NULL;
    L->spaced = spaced ? 1 : 0;
    L->cls = lclass_tab[L->spaced];
    for (int c = 0; c < 256; c++) L->nclass[L->cls[c]]++;
    if (order < 2) order = 2;
    if (order > LEX_MAXORD) order = LEX_MAXORD;
    L->order = order;
    for (int i = 0; i < 32; i++) L->freel[i] = NONE;
    if (!hidx_init(&L->widx, 1u << 12) || !ct_init(&L->ct, 1u << 12) ||
        !grow((void **)&L->nodes, &L->ncap, 1024, sizeof(Node))) {
        lex_free(L);
        return NULL;
    }
    memset(&L->nodes[0], 0, sizeof(Node));
    L->nn = 1;
    L->prev = ' ';
    for (int k = 0; k <= LEX_MAXORD; k++) {
        L->disc[k][0] = 0.5f;
        L->disc[k][1] = 1.0f;
        L->disc[k][2] = 1.4f;
    }
    L->cdecay[0] = 1.0 / 64;
    L->cdecay[1] = 1.0 / 1024;
    L->cscale[0] = L->cscale[1] = 1.0;
    L->lam[0] = L->lam[1] = 0.02f;
    return L;
}

void lex_free(mn_lex *L) {
    if (!L) return;
    hidx_free(&L->widx);
    free(L->ct.s);
    free(L->ar);
    free(L->woff);
    free(L->wnode);
    free(L->wcc1);
    free(L->wlen);
    free(L->pool);
    free(L->nodes);
    free(L->cand);
    for (int i = 0; i < 2; i++) {
        free(L->pc[i]);
        free(L->wc[i]);
    }
    free(L);
}

size_t lex_memory(const mn_lex *L) {
    size_t s = sizeof(mn_lex);
    s += (size_t)L->widx.cap * 12 + (size_t)L->ct.cap * sizeof(CSlot);
    s += (size_t)L->wcap * 13 + L->pool_cap + (size_t)L->ncap * sizeof(Node);
    s += (size_t)L->ar_cap * sizeof(Foll);
    s += (size_t)L->candcap * sizeof(Cand);
    s += (size_t)L->pccap * 32;
    return s;
}

/* ------------------------------------------------------------ vocabulary */

static uint64_t word_key(const uint8_t *w, int n) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int i = 0; i < n; i++) h = (h ^ w[i]) * 0x100000001B3ULL;
    return mix64(h + (uint64_t)n);
}

static uint32_t find_child(const mn_lex *L, uint32_t node, int c) {
    for (uint32_t ch = L->nodes[node].first; ch; ch = L->nodes[ch].next)
        if (L->nodes[ch].ch == (uint8_t)c) return ch;
    return 0;
}

static int64_t add_word(mn_lex *L, const uint8_t *w, int n, uint64_t key) {
    if (L->nw >= LEX_MAXWORDS) return -1;
    uint32_t id = L->nw;
    if (id + 1 > L->wcap) {
        uint32_t c1 = L->wcap, c2 = L->wcap, c3 = L->wcap, c4 = L->wcap;
        if (!grow((void **)&L->woff, &c1, id + 1, 4) || !grow((void **)&L->wnode, &c2, id + 1, 4) ||
            !grow((void **)&L->wcc1, &c3, id + 1, 4) || !grow((void **)&L->wlen, &c4, id + 1, 1))
            return -1;
        L->wcap = c1;
    }
    if (L->pool_n + (size_t)n > L->pool_cap) {
        size_t nc = L->pool_cap ? L->pool_cap : 4096;
        while (nc < L->pool_n + (size_t)n) nc *= 2;
        char *q = (char *)realloc(L->pool, nc);
        if (!q) return -1;
        L->pool = q;
        L->pool_cap = nc;
    }
    uint32_t node = 0;
    for (int i = 0; i < n; i++) {
        uint32_t ch = find_child(L, node, w[i]);
        if (!ch) {
            if (!grow((void **)&L->nodes, &L->ncap, L->nn + 1, sizeof(Node))) return -1;
            ch = L->nn++;
            Node *x = &L->nodes[ch];
            memset(x, 0, sizeof(Node));
            x->ch = w[i];
            x->depth = (uint8_t)(i + 1);
            x->parent = node;
            x->next = L->nodes[node].first;
            L->nodes[node].first = ch;
        }
        node = ch;
    }
    int isnew;
    uint32_t *slot = hidx_insert(&L->widx, key, &isnew);
    if (!slot) return -1;
    *slot = id;
    L->nodes[node].word = id + 1;
    memcpy(L->pool + L->pool_n, w, (size_t)n);
    L->woff[id] = (uint32_t)L->pool_n;
    L->pool_n += (size_t)n;
    L->wlen[id] = (uint8_t)n;
    L->wnode[id] = node;
    L->wcc1[id] = 0;
    L->nw++;
    return id;
}

uint32_t lex_num_words(const mn_lex *L) { return L->nw; }
const char *lex_word(const mn_lex *L, uint32_t id, int *len) {
    if (id >= L->nw) return NULL;
    if (len) *len = L->wlen[id];
    return L->pool + L->woff[id];
}
uint32_t lex_count(const mn_lex *L, uint32_t id) { return id < L->nw ? L->wcc1[id] : 0; }
int64_t lex_find(const mn_lex *L, const char *w, int len) {
    if (len <= 0 || len > LEX_MAXLEN) return -1;
    uint32_t *p = hidx_find(&L->widx, word_key((const uint8_t *)w, len));
    return p ? (int64_t)*p : -1;
}

/* ------------------------------------------------------------ n-grams */

static void push_token(mn_lex *L, uint32_t t) {
    for (int i = LEX_MAXORD - 1; i > 0; i--) L->tok[i] = L->tok[i - 1];
    L->tok[0] = t;
    if (L->ntok < LEX_MAXORD - 1) L->ntok++;
}

static int ctx_keys(const mn_lex *L, uint64_t *keys) {
    int mk = L->order;
    if (mk > L->ntok + 1) mk = L->ntok + 1;
    uint64_t kk = 0x9E3779B97F4A7C15ULL;
    for (int k = 2; k <= mk; k++) {
        kk = mix64(kk + (uint64_t)L->tok[k - 2] * 0x100000001B3ULL + (uint64_t)k);
        keys[k] = kk;
    }
    return mk;
}

static void update_discounts(mn_lex *L) {
    for (int k = 2; k <= L->order; k++) {
        double n1 = L->coc[k][1], n2 = L->coc[k][2], n3 = L->coc[k][3], n4 = L->coc[k][4];
        float *D = L->disc[k];
        if (n1 < 1 || n2 < 1 || n3 < 1 || n4 < 1) continue;
        double Y = n1 / (n1 + 2 * n2);
        double d1 = 1 - 2 * Y * n2 / n1, d2 = 2 - 3 * Y * n3 / n2, d3 = 3 - 4 * Y * n4 / n3;
        D[0] = (float)(d1 < 0.05 ? 0.05 : d1 > 0.95 ? 0.95 : d1);
        D[1] = (float)(d2 < 0.1 ? 0.1 : d2 > 1.9 ? 1.9 : d2);
        D[2] = (float)(d3 < 0.15 ? 0.15 : d3 > 2.9 ? 2.9 : d3);
    }
}

static inline void bump_coc(mn_lex *L, int k, uint32_t old) {
    if (old >= 1 && old <= 4) L->coc[k][old]--;
    if (old + 1 <= 4) L->coc[k][old + 1]++;
}

static inline uint32_t first_byte(const mn_lex *L, uint32_t w) {
    return (uint8_t)L->pool[L->woff[w]];
}

/* position of w in a sorted follower list, or -(insertion point) - 1 */
static int64_t foll_find(const Foll *f, uint32_t n, uint32_t w) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (f[mid].w < w)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < n && f[lo].w == w) return lo;
    return -(int64_t)lo - 1;
}

/* insert follower w (count 1) at position pos of context c's list */
static int foll_insert(mn_lex *L, CSlot *c, uint32_t pos, uint32_t w) {
    uint32_t nf = c->nf;
    if ((nf & (nf - 1)) == 0) { /* the block is full: sizes are powers of two */
        int lg = cap_lg(nf);
        uint32_t off = arena_alloc(L, lg + 1);
        if (off == NONE) return 0;
        memcpy(&L->ar[off], &L->ar[c->x], (size_t)nf * sizeof(Foll));
        arena_free(L, c->x, lg);
        c->x = off;
    }
    Foll *f = &L->ar[c->x];
    memmove(&f[pos + 1], &f[pos], (size_t)(nf - pos) * sizeof(Foll));
    f[pos].w = w;
    f[pos].cc = (1u << 8) | first_byte(L, w);
    c->nf = nf + 1;
    return 1;
}

/* count one occurrence of token w in context slot c; returns its old count */
static int64_t ctx_count(mn_lex *L, CSlot *c, uint32_t w) {
    if (c->nf == 0) {
        c->nf = 1;
        c->x = w;
        c->total = 1;
        return 0;
    }
    if (c->nf == 1) {
        if (c->x == w) return c->total < FMAXC ? (int64_t)c->total++ : (int64_t)c->total;
        uint32_t off = arena_alloc(L, 1);
        if (off == NONE) return -1;
        Foll a = {c->x, (c->total > FMAXC ? FMAXC : c->total) << 8 | first_byte(L, c->x)};
        Foll b = {w, (1u << 8) | first_byte(L, w)};
        L->ar[off] = a.w < w ? a : b;
        L->ar[off + 1] = a.w < w ? b : a;
        c->x = off;
        c->nf = 2;
        c->total++;
        return 0;
    }
    Foll *f = &L->ar[c->x];
    int64_t pos = foll_find(f, c->nf, w);
    if (pos >= 0) {
        uint32_t old = FCOUNT(f[pos]);
        if (old < FMAXC) {
            f[pos].cc += 256;
            c->total++;
        }
        return old;
    }
    if (!foll_insert(L, c, (uint32_t)(-pos - 1), w)) return -1;
    c->total++;
    return 0;
}

static void learn_token(mn_lex *L, uint32_t wi) {
    uint64_t keys[LEX_MAXORD + 1];
    int mk = ctx_keys(L, keys);
    int new_above = 1;
    for (int k = mk; k >= 2; k--) {
        if (k < mk && !new_above) break;
        int isnew;
        CSlot *c = ct_insert(&L->ct, key_fp(keys[k]), &isnew);
        if (!c) return;
        int64_t old = ctx_count(L, c, wi);
        if (old < 0) return;
        bump_coc(L, k, (uint32_t)old);
        new_above = (old == 0);
    }
    if (new_above) {
        L->wcc1[wi]++;
        L->total_cc1++;
        for (uint32_t n = L->wnode[wi];; n = L->nodes[n].parent) {
            L->nodes[n].sub++;
            if (n == 0) break;
        }
    }
    if (++L->disc_age >= 512) {
        L->disc_age = 0;
        update_discounts(L);
    }
}

static void build_cands(mn_lex *L) {
    L->ncand = 0;
    memset(L->root_mass, 0, sizeof L->root_mass);
    double g = 1.0;
    uint64_t keys[LEX_MAXORD + 1];
    int mk = ctx_keys(L, keys);
    for (int k = mk; k >= 2; k--) {
        const CSlot *c = ct_find(&L->ct, key_fp(keys[k]));
        if (!c || c->total == 0) continue;
        const float *D = L->disc[k];
        const double inv = 1.0 / c->total;
        if (!grow((void **)&L->cand, &L->candcap, L->ncand + c->nf, sizeof(Cand))) return;
        Foll one;
        const Foll *f;
        if (c->nf == 1) {
            one.w = c->x;
            one.cc = (c->total > FMAXC ? FMAXC : c->total) << 8 | first_byte(L, c->x);
            f = &one;
        } else {
            f = &L->ar[c->x];
        }
        uint32_t n1 = 0, n2 = 0, n3 = 0;
        Cand *cd = L->cand + L->ncand;
        const double gi = g * inv;
        for (uint32_t i = 0; i < c->nf; i++) {
            uint32_t cnt = FCOUNT(f[i]);
            float d;
            if (cnt == 1) {
                d = D[0];
                n1++;
            } else if (cnt == 2) {
                d = D[1];
                n2++;
            } else {
                d = D[2];
                n3++;
            }
            uint32_t w = f[i].w;
            float wt = (float)(gi * ((double)cnt - d));
            uint8_t ch = (uint8_t)(f[i].cc & 255u);
            cd[i].word = w;
            cd[i].off = L->woff[w];
            cd[i].len = L->wlen[w];
            cd[i].ch = ch;
            cd[i].wt = wt;
            L->root_mass[ch] += wt;
        }
        L->ncand += c->nf;
        g *= (D[0] * n1 + D[1] * n2 + D[2] * n3) * inv;
    }
    L->g_dense = g;
}

/* ------------------------------------------------------------ prediction */

static void prepare(mn_lex *L) {
    L->state = 0;
    if (L->lost || L->total_cc1 == 0) return;
    float mass[256];
    double term = 0;
    const int d = L->curlen;
    if (d == 0) {
        memcpy(mass, L->root_mass, sizeof mass);
    } else {
        memset(mass, 0, sizeof mass);
        const char *pool = L->pool;
        for (uint32_t i = 0; i < L->ncand; i++) {
            const Cand *cd = &L->cand[i];
            if (cd->len == d)
                term += cd->wt;
            else
                mass[(uint8_t)pool[cd->off + d]] += cd->wt;
        }
    }
    const double gd = L->g_dense / (double)L->total_cc1;
    const Node *n = &L->nodes[L->node];
    for (uint32_t ch = n->first; ch; ch = L->nodes[ch].next)
        mass[L->nodes[ch].ch] += (float)(gd * (double)L->nodes[ch].sub);
    if (n->word) term += gd * L->wcc1[n->word - 1];
    /* priming: interpolate with the recency caches at the token level */
    L->last_kn = term;
    L->last_c[0] = L->last_c[1] = 0;
    if (L->pccap >= L->ncap && L->ctot[0] > 0) {
        const double l0 = L->lam[0], l1 = L->lam[1], lk = 1.0 - l0 - l1;
        const double i0 = 1.0 / L->ctot[0], i1 = 1.0 / L->ctot[1];
        for (int c = 0; c < 256; c++) mass[c] *= (float)lk;
        for (uint32_t ch = n->first; ch; ch = L->nodes[ch].next)
            mass[L->nodes[ch].ch] += (float)(l0 * L->pc[0][ch] * i0 + l1 * L->pc[1][ch] * i1);
        if (n->word) {
            L->last_c[0] = L->wc[0][n->word - 1] * i0;
            L->last_c[1] = L->wc[1][n->word - 1] * i1;
        }
        term = lk * term + l0 * L->last_c[0] + l1 * L->last_c[1];
    }
    double W = 0, mx = 0;
    for (int c = 0; c < 256; c++) {
        W += mass[c];
        if (mass[c] > mx) mx = mass[c];
    }
    if (W + term <= 0) return;

    const int pv = L->prev & 255;
    double p_space, p_cont, p_att;
    if (d == 0) {
        p_space = (L->wsb[pv][1] + 0.5) / (L->wsb[pv][0] + L->wsb[pv][1] + 1.0);
        p_cont = 1 - p_space; /* the next token starts here */
        p_att = 0;
    } else {
        double pend = term / (W + term);
        const uint32_t *a = L->wsa[L->curcls];
        double ps = (a[1] + 0.5) / (a[0] + a[1] + 1.0);
        p_cont = 1 - pend;
        p_space = pend * ps;
        p_att = pend * (1 - ps);
    }
    /* spaces */
    const double sden = L->wsc[pv][0] + L->wsc[pv][1] + L->wsc[pv][2] + 0.3;
    /* attached tokens: bytes of another class, smoothed */
    double aden = 0;
    if (p_att > 0)
        for (int k = 1; k < 4; k++)
            if (k != L->curcls) aden += L->attsum[pv][k] + 0.02 * L->nclass[k];
    const double eps = 0.002;
    float *t = L->tree;
    const uint8_t *cls = L->cls;
    const double kc = W > 0 ? (1 - eps) * p_cont / W : 0;
    const double ka = p_att > 0 ? (1 - eps) * p_att / aden : 0;
    const double ks = (1 - eps) * p_space / sden, e0 = eps / 256.0;
    const int cc = L->curcls;
    for (int c = 0; c < 256; c++) {
        const int k = cls[c];
        double p = e0;
        if (k == 0)
            p += ks * (L->wsc[pv][space_index(c)] + 0.1);
        else {
            p += kc * mass[c];
            if (k != cc) p += ka * (L->att[pv][c] + 0.02);
        }
        t[256 + c] = (float)p;
    }
    for (int i = 255; i >= 1; i--) t[i] = t[2 * i] + t[2 * i + 1];
    if (d == 0)
        L->state = 1;
    else
        L->state = (W > 0 && mx / (W + term) >= 0.5) ? 3 : 2;
}

static void end_token(mn_lex *L, int learn) {
    uint32_t tokid = UNK_TOKEN;
    if (L->curlen <= LEX_MAXLEN) {
        uint64_t key = word_key(L->cur, L->curlen);
        uint32_t *wi = hidx_find(&L->widx, key);
        int64_t w = wi ? (int64_t)*wi : -1;
        if (w < 0 && learn) w = add_word(L, L->cur, L->curlen, key);
        if (w >= 0) {
            /* the term masses computed when the token's last byte was predicted
             * are its probabilities under each memory */
            if (learn && !L->lost && L->node == L->wnode[w] && L->ctot[0] > 0) {
                double pk = L->last_kn, p0 = L->last_c[0], p1 = L->last_c[1];
                double l0 = L->lam[0], l1 = L->lam[1];
                double pm = (1 - l0 - l1) * pk + l0 * p0 + l1 * p1;
                if (pm > 0) {
                    const double eta = 0.002;
                    l0 += eta * (l0 * p0 / pm - l0);
                    l1 += eta * (l1 * p1 / pm - l1);
                    L->lam[0] = (float)(l0 < 0.001 ? 0.001 : l0 > 0.45 ? 0.45 : l0);
                    L->lam[1] = (float)(l1 < 0.001 ? 0.001 : l1 > 0.45 ? 0.45 : l1);
                }
            }
            if (learn) learn_token(L, (uint32_t)w);
            cache_add(L, (uint32_t)w);
            tokid = (uint32_t)w;
        }
    }
    push_token(L, tokid);
    L->curlen = 0;
}

/* descend the trie with byte c, narrowing the candidates */
static void descend(mn_lex *L, int c) {
    const int d = L->curlen;
    if (d < LEX_MAXLEN) L->cur[d] = (uint8_t)c;
    L->curlen++;
    L->root_valid = 0;
    if (L->lost) return;
    uint32_t ch = d < LEX_MAXLEN ? find_child(L, L->node, c) : 0;
    if (!ch) {
        L->lost = 1;
        L->ncand = 0;
        return;
    }
    L->node = ch;
    uint32_t k = 0;
    if (d == 0) {
        for (uint32_t i = 0; i < L->ncand; i++)
            if (L->cand[i].ch == (uint8_t)c) L->cand[k++] = L->cand[i];
    } else {
        const char *pool = L->pool;
        for (uint32_t i = 0; i < L->ncand; i++) {
            const Cand *cd = &L->cand[i];
            if (cd->len > d && (uint8_t)pool[cd->off + d] == (uint8_t)c) L->cand[k++] = *cd;
        }
    }
    L->ncand = k;
}

void lex_observe(mn_lex *L, int c, int learn) {
    c &= 255;
    const int cls = lclass(c);
    const int pv = L->prev & 255;
    if (learn) {
        if (L->curlen == 0) {
            L->wsb[pv][cls == 0]++;
        } else if (cls != L->curcls) {
            L->wsa[L->curcls][cls == 0]++;
            if (cls != 0) {
                L->attt[pv]++;
                L->attsum[pv][cls]++;
                if (++L->att[pv][c] >= 65535) {
                    uint32_t t = 0;
                    memset(L->attsum[pv], 0, sizeof L->attsum[pv]);
                    for (int i = 0; i < 256; i++) {
                        t += (L->att[pv][i] >>= 1);
                        L->attsum[pv][lclass(i)] += L->att[pv][i];
                    }
                    L->attt[pv] = t;
                }
            }
        }
        if (cls == 0) L->wsc[pv][space_index(c)]++;
    }
    if (L->curlen > 0 && cls == L->curcls) {
        descend(L, c);
    } else {
        int changed = 0;
        if (L->curlen > 0) {
            end_token(L, learn);
            changed = 1;
        }
        L->lost = 0;
        L->node = 0;
        if (changed || !L->root_valid) {
            build_cands(L);
            L->root_valid = 1;
        }
        if (cls != 0) {
            L->curcls = cls;
            descend(L, c);
        }
    }
    L->prev = c;
    prepare(L);
}

const float *lex_tree(const mn_lex *L) { return L->state ? L->tree : NULL; }
int lex_state(const mn_lex *L) { return L->state; }

int lex_top_next(mn_lex *L, uint32_t *ids, float *probs, int max) {
    if (L->total_cc1 == 0 || max <= 0) return 0;
    /* probability of each candidate token at the current trie position */
    uint32_t n = 0;
    Cand *tmp = (Cand *)malloc(((size_t)L->ncand + (size_t)max + 1) * sizeof(Cand));
    HIdx seen;
    if (!tmp || !hidx_init(&seen, 1u << 12)) {
        free(tmp);
        return 0;
    }
    for (uint32_t i = 0; i < L->ncand; i++) {
        int isnew;
        uint32_t *s = hidx_insert(&seen, (uint64_t)L->cand[i].word + 1, &isnew);
        if (!s) break;
        if (isnew) {
            *s = n;
            tmp[n].word = L->cand[i].word;
            tmp[n].wt = 0;
            n++;
        }
        tmp[*s].wt += L->cand[i].wt;
    }
    const double gd = L->g_dense / (double)L->total_cc1;
    for (uint32_t i = 0; i < n; i++) tmp[i].wt += (float)(gd * L->wcc1[tmp[i].word]);
    /* the most frequent tokens can win through the dense part alone */
    for (int r = 0; r < max; r++) {
        uint32_t bu = NONE;
        uint32_t bc = 0;
        for (uint32_t w = 0; w < L->nw; w++)
            if (L->wcc1[w] > bc && !hidx_find(&seen, (uint64_t)w + 1)) {
                bc = L->wcc1[w];
                bu = w;
            }
        if (bu == NONE) break;
        int isnew;
        uint32_t *s = hidx_insert(&seen, (uint64_t)bu + 1, &isnew);
        if (!s) break;
        *s = n;
        tmp[n].word = bu;
        tmp[n].wt = (float)(gd * bc);
        n++;
    }
    int got = 0;
    for (; got < max; got++) {
        int best = -1;
        float bw = 0;
        for (uint32_t i = 0; i < n; i++)
            if (tmp[i].wt > bw) {
                bw = tmp[i].wt;
                best = (int)i;
            }
        if (best < 0) break;
        ids[got] = tmp[best].word;
        probs[got] = bw;
        tmp[best].wt = -1;
    }
    hidx_free(&seen);
    free(tmp);
    return got;
}

/* ------------------------------------------------------------ persistence */

#define W(ptr, sz, cnt) ok = ok && ((cnt) == 0 || fwrite((ptr), (sz), (cnt), f) == (size_t)(cnt))
#define R(ptr, sz, cnt) ok = ok && ((cnt) == 0 || fread((ptr), (sz), (cnt), f) == (size_t)(cnt))

static int hidx_save(const HIdx *h, FILE *f) {
    int ok = 1;
    W(&h->cap, 4, 1);
    W(&h->n, 4, 1);
    W(h->k, 8, h->cap);
    W(h->v, 4, h->cap);
    return ok;
}
static int hidx_load(HIdx *h, FILE *f) {
    int ok = 1;
    uint32_t cap = 0, n = 0;
    R(&cap, 4, 1);
    R(&n, 4, 1);
    if (!ok || cap == 0 || (cap & (cap - 1))) return 0;
    if (!hidx_init(h, cap)) return 0;
    h->n = n;
    R(h->k, 8, cap);
    R(h->v, 4, cap);
    return ok;
}

int lex_save(const mn_lex *L, FILE *f) {
    int ok = 1;
    uint32_t magic = 0x3258454Cu;
    W(&magic, 4, 1);
    W(L, sizeof(mn_lex), 1);
    ok = ok && hidx_save(&L->widx, f);
    W(&L->ct.cap, 4, 1);
    W(&L->ct.n, 4, 1);
    W(L->ct.s, sizeof(CSlot), L->ct.cap);
    W(L->woff, 4, L->nw);
    W(L->wnode, 4, L->nw);
    W(L->wcc1, 4, L->nw);
    W(L->wlen, 1, L->nw);
    W(L->pool, 1, L->pool_n);
    W(L->nodes, sizeof(Node), L->nn);
    W(L->ar, sizeof(Foll), L->ar_n);
    W(L->cand, sizeof(Cand), L->ncand);
    for (int i = 0; i < 2; i++) {
        W(L->pc[i], sizeof(double), L->pccap);
        W(L->wc[i], sizeof(double), L->pccap);
    }
    return ok ? 0 : -1;
}

mn_lex *lex_load(FILE *f) {
    lclass_init();
    int ok = 1;
    uint32_t magic = 0;
    R(&magic, 4, 1);
    if (!ok || magic != 0x3258454Cu) return NULL;
    mn_lex *L = (mn_lex *)calloc(1, sizeof(mn_lex));
    if (!L) return NULL;
    R(L, sizeof(mn_lex), 1);
    if (!ok) {
        free(L);
        return NULL;
    }
    memset(&L->widx, 0, sizeof(HIdx));
    memset(&L->ct, 0, sizeof(CTab));
    L->cls = lclass_tab[L->spaced ? 1 : 0];
    L->woff = L->wnode = L->wcc1 = NULL;
    L->wlen = NULL;
    L->pool = NULL;
    L->nodes = NULL;
    L->ar = NULL;
    L->cand = NULL;
    const uint32_t pccap = L->pccap;
    for (int i = 0; i < 2; i++) L->pc[i] = L->wc[i] = NULL;
    L->pccap = 0;
    const uint32_t nw = L->nw, nn = L->nn, ar_n = L->ar_n, ncand = L->ncand;
    const size_t pool_n = L->pool_n;
    L->wcap = nw ? nw : 1;
    L->ncap = nn ? nn : 1;
    L->ar_cap = ar_n ? ar_n : 1;
    L->candcap = ncand ? ncand : 1;
    L->pool_cap = pool_n ? pool_n : 1;
    ok = hidx_load(&L->widx, f);
    uint32_t ccap = 0, cn = 0;
    R(&ccap, 4, 1);
    R(&cn, 4, 1);
    if (ok && (ccap == 0 || !ct_init(&L->ct, ccap))) ok = 0;
    if (ok) L->ct.n = cn;
    R(L->ct.s, sizeof(CSlot), ccap);
    if (ok) {
        L->woff = (uint32_t *)malloc((size_t)L->wcap * 4);
        L->wnode = (uint32_t *)malloc((size_t)L->wcap * 4);
        L->wcc1 = (uint32_t *)malloc((size_t)L->wcap * 4);
        L->wlen = (uint8_t *)malloc(L->wcap);
        L->pool = (char *)malloc(L->pool_cap);
        L->nodes = (Node *)malloc((size_t)L->ncap * sizeof(Node));
        L->ar = (Foll *)malloc((size_t)L->ar_cap * sizeof(Foll));
        L->cand = (Cand *)malloc((size_t)L->candcap * sizeof(Cand));
        ok = L->woff && L->wnode && L->wcc1 && L->wlen && L->pool && L->nodes && L->ar && L->cand;
    }
    R(L->woff, 4, nw);
    R(L->wnode, 4, nw);
    R(L->wcc1, 4, nw);
    R(L->wlen, 1, nw);
    R(L->pool, 1, pool_n);
    R(L->nodes, sizeof(Node), nn);
    R(L->ar, sizeof(Foll), ar_n);
    R(L->cand, sizeof(Cand), ncand);
    for (int i = 0; i < 2 && ok; i++) {
        L->pc[i] = (double *)calloc(pccap ? pccap : 1, sizeof(double));
        L->wc[i] = (double *)calloc(pccap ? pccap : 1, sizeof(double));
        ok = L->pc[i] && L->wc[i];
        R(L->pc[i], sizeof(double), pccap);
        R(L->wc[i], sizeof(double), pccap);
    }
    if (ok) L->pccap = pccap;
    if (!ok) {
        lex_free(L);
        return NULL;
    }
    return L;
}

/* ------------------------------------------------------------ imagination */

struct lex_snap {
    uint32_t tok[LEX_MAXORD];
    int ntok;
    uint8_t cur[LEX_MAXLEN + 1];
    int curlen, curcls, lost, root_valid, prev;
    uint32_t node;
    uint32_t ncand;
    Cand *cand;
    double g_dense;
    float root_mass[256];
    double cscale[2], ctot[2];
    float lam[2];
    double last_kn, last_c[2];
    float tree[512];
    int state;
    uint32_t pccap;
    double *pc[2], *wc[2];
};

lex_snap *lex_snapshot(const mn_lex *L) {
    lex_snap *z = (lex_snap *)calloc(1, sizeof(lex_snap));
    if (!z) return NULL;
    memcpy(z->tok, L->tok, sizeof z->tok);
    z->ntok = L->ntok;
    memcpy(z->cur, L->cur, sizeof z->cur);
    z->curlen = L->curlen;
    z->curcls = L->curcls;
    z->lost = L->lost;
    z->root_valid = L->root_valid;
    z->prev = L->prev;
    z->node = L->node;
    z->ncand = L->ncand;
    z->g_dense = L->g_dense;
    memcpy(z->root_mass, L->root_mass, sizeof z->root_mass);
    memcpy(z->cscale, L->cscale, sizeof z->cscale);
    memcpy(z->ctot, L->ctot, sizeof z->ctot);
    memcpy(z->lam, L->lam, sizeof z->lam);
    z->last_kn = L->last_kn;
    memcpy(z->last_c, L->last_c, sizeof z->last_c);
    memcpy(z->tree, L->tree, sizeof z->tree);
    z->state = L->state;
    z->pccap = L->pccap;
    int ok = 1;
    z->cand = (Cand *)malloc(((size_t)L->ncand + 1) * sizeof(Cand));
    ok = z->cand != NULL;
    if (ok) memcpy(z->cand, L->cand, (size_t)L->ncand * sizeof(Cand));
    for (int i = 0; i < 2 && ok; i++) {
        size_t bytes = (size_t)L->pccap * sizeof(double);
        z->pc[i] = (double *)malloc(bytes ? bytes : 8);
        z->wc[i] = (double *)malloc(bytes ? bytes : 8);
        ok = z->pc[i] && z->wc[i];
        if (ok && bytes) {
            memcpy(z->pc[i], L->pc[i], bytes);
            memcpy(z->wc[i], L->wc[i], bytes);
        }
    }
    if (!ok) {
        lex_snap_free(z);
        return NULL;
    }
    return z;
}

void lex_snap_free(lex_snap *z) {
    if (!z) return;
    free(z->cand);
    for (int i = 0; i < 2; i++) {
        free(z->pc[i]);
        free(z->wc[i]);
    }
    free(z);
}

void lex_restore(mn_lex *L, const lex_snap *z) {
    memcpy(L->tok, z->tok, sizeof z->tok);
    L->ntok = z->ntok;
    memcpy(L->cur, z->cur, sizeof z->cur);
    L->curlen = z->curlen;
    L->curcls = z->curcls;
    L->lost = z->lost;
    L->root_valid = z->root_valid;
    L->prev = z->prev;
    L->node = z->node;
    if (grow((void **)&L->cand, &L->candcap, z->ncand + 1, sizeof(Cand))) {
        memcpy(L->cand, z->cand, (size_t)z->ncand * sizeof(Cand));
        L->ncand = z->ncand;
    } else {
        L->ncand = 0;
    }
    L->g_dense = z->g_dense;
    memcpy(L->root_mass, z->root_mass, sizeof z->root_mass);
    memcpy(L->cscale, z->cscale, sizeof z->cscale);
    memcpy(L->ctot, z->ctot, sizeof z->ctot);
    memcpy(L->lam, z->lam, sizeof z->lam);
    L->last_kn = z->last_kn;
    memcpy(L->last_c, z->last_c, sizeof z->last_c);
    memcpy(L->tree, z->tree, sizeof z->tree);
    L->state = z->state;
    if (z->pccap == L->pccap && z->pccap)
        for (int i = 0; i < 2; i++) {
            memcpy(L->pc[i], z->pc[i], (size_t)z->pccap * sizeof(double));
            memcpy(L->wc[i], z->wc[i], (size_t)z->pccap * sizeof(double));
        }
}
