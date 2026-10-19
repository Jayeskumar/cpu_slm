/*
 * mnemo - command line front end for the predictive cortex.
 *
 *   mnemo bench    --train FILE... --test FILE [--static] [model options]
 *   mnemo train    -o MODEL [model options] FILE...
 *   mnemo eval     -m MODEL [--static] FILE
 *   mnemo gen      -m MODEL [--prompt TEXT] [-n N] [--temp T] [--seed S]
 *   mnemo compress   [model options] IN OUT
 *   mnemo decompress IN OUT
 *
 * model options: --tb BITS (context table), --hb BITS (history), --mb BITS
 *                (recall index), --cb BITS (concept table), --lr X, --lr0 X,
 *                --classes FILE (word<TAB>class lines)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mnemo.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    buf[sz] = 0;
    *n = (size_t)sz;
    return buf;
}

/* tokens = whitespace-separated words plus one end-of-line token per '\n' */
static size_t count_tokens(const uint8_t *b, size_t n) {
    size_t t = 0;
    int in = 0;
    for (size_t i = 0; i < n; i++) {
        int sp = b[i] == ' ' || b[i] == '\n' || b[i] == '\t' || b[i] == '\r';
        if (!sp && !in) t++;
        in = !sp;
        if (b[i] == '\n') t++;
    }
    return t;
}

static int load_classes(mn_model *m, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[1024];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        unsigned cls = (unsigned)strtoul(tab + 1, NULL, 10);
        if (cls == 0 || cls > 65535) continue;
        mn_set_class(m, mn_word_hash(line, strlen(line)), cls);
        n++;
    }
    fclose(f);
    return n;
}

typedef struct {
    mn_config cfg;
    const char *classes;
    const char *train[64];
    int ntrain;
    const char *test;
    const char *model;
    const char *out;
    const char *prompt;
    int n;
    double temp;
    unsigned long long seed;
    int is_static;
    int learn_gen;
    const char *pos[8];
    int npos;
} Opts;

static void parse(int argc, char **argv, Opts *o) {
    memset(o, 0, sizeof *o);
    mn_default_config(&o->cfg);
    o->n = 400;
    o->temp = 0.7;
    o->seed = 1;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value for %s\n", a), exit(1), ""))
        if (!strcmp(a, "--train"))
            o->train[o->ntrain++] = NEXT();
        else if (!strcmp(a, "--test"))
            o->test = NEXT();
        else if (!strcmp(a, "--tb"))
            o->cfg.table_bits = atoi(NEXT());
        else if (!strcmp(a, "--hb"))
            o->cfg.hist_bits = atoi(NEXT());
        else if (!strcmp(a, "--mb"))
            o->cfg.match_bits = atoi(NEXT());
        else if (!strcmp(a, "--cb"))
            o->cfg.class_bits = atoi(NEXT());
        else if (!strcmp(a, "--lr"))
            o->cfg.lr = (float)atof(NEXT());
        else if (!strcmp(a, "--lr0"))
            o->cfg.lr0 = (float)atof(NEXT());
        else if (!strcmp(a, "--lex"))
            o->cfg.lex_order = atoi(NEXT());
        else if (!strcmp(a, "--spaced"))
            o->cfg.lex_spaced = 1;
        else if (!strcmp(a, "--ctx"))
            o->cfg.ctx_mask = (uint32_t)strtoul(NEXT(), NULL, 16);
        else if (!strcmp(a, "--classes"))
            o->classes = NEXT();
        else if (!strcmp(a, "-m"))
            o->model = NEXT();
        else if (!strcmp(a, "-o"))
            o->out = NEXT();
        else if (!strcmp(a, "--prompt"))
            o->prompt = NEXT();
        else if (!strcmp(a, "-n"))
            o->n = atoi(NEXT());
        else if (!strcmp(a, "--temp"))
            o->temp = atof(NEXT());
        else if (!strcmp(a, "--seed"))
            o->seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--static"))
            o->is_static = 1;
        else if (!strcmp(a, "--learn"))
            o->learn_gen = 1;
        else if (o->npos < 8)
            o->pos[o->npos++] = a;
#undef NEXT
    }
    if (o->classes && o->cfg.class_bits == 0) o->cfg.class_bits = 20;
}

static mn_model *make_model(Opts *o) {
    mn_model *m = mn_new(&o->cfg);
    if (!m) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    if (o->classes) {
        int n = load_classes(m, o->classes);
        if (n < 0) {
            fprintf(stderr, "cannot read classes %s\n", o->classes);
            exit(1);
        }
        fprintf(stderr, "loaded %d word concepts\n", n);
    }
    return m;
}

static double train_files(mn_model *m, const char **files, int nf, size_t *total) {
    double bits = 0;
    *total = 0;
    for (int i = 0; i < nf; i++) {
        size_t n;
        uint8_t *b = read_file(files[i], &n);
        bits += mn_observe(m, b, n);
        *total += n;
        free(b);
    }
    return bits;
}

static void report_eval(const char *name, mn_model *m, const char *path, int is_static) {
    size_t n;
    uint8_t *b = read_file(path, &n);
    mn_set_learning(m, !is_static);
    mn_lex_stats(m, NULL, NULL, 1);
    double t0 = now_sec();
    double bits = mn_observe(m, b, n);
    double dt = now_sec() - t0;
    size_t tok = count_tokens(b, n);
    printf("%s %s: %zu bytes, %.4f bpc, %zu tokens, %.2f token-ppl, %.2f s (%.2f MB/s)%s\n", name,
           path, n, bits / n, tok, pow(2.0, bits / tok), dt, n / dt / 1e6,
           is_static ? " [static]" : " [online]");
    double lb;
    uint64_t ln;
    mn_lex_stats(m, &lb, &ln, 0);
    if (ln) printf("  lexicon alone: active on %.1f%% of bytes, %.4f bpc there\n", 100.0 * ln / n, lb / ln);
    printf("  memory after: %.1f MB\n", mn_memory(m) / 1048576.0);
    free(b);
}

static int cmd_bench(Opts *o) {
    if (!o->ntrain && !o->test) {
        fprintf(stderr, "bench needs --train and/or --test\n");
        return 1;
    }
    mn_model *m = make_model(o);
    printf("memory: %.1f MB, mixer inputs: %d\n", mn_memory(m) / 1048576.0, mn_num_inputs(m));
    if (o->ntrain) {
        size_t total;
        double t0 = now_sec();
        double bits = train_files(m, o->train, o->ntrain, &total);
        double dt = now_sec() - t0;
        printf("train: %zu bytes, %.4f bpc (online), %.2f s (%.2f MB/s)\n", total, bits / total, dt,
               total / dt / 1e6);
    }
    if (o->test) report_eval("test", m, o->test, o->is_static);
    mn_free(m);
    return 0;
}

static int cmd_train(Opts *o) {
    if (!o->out || o->npos == 0) {
        fprintf(stderr, "usage: mnemo train -o MODEL FILE...\n");
        return 1;
    }
    mn_model *m = make_model(o);
    size_t total;
    double t0 = now_sec();
    double bits = train_files(m, o->pos, o->npos, &total);
    double dt = now_sec() - t0;
    printf("trained on %zu bytes: %.4f bpc, %.2f s (%.2f MB/s), memory %.1f MB\n", total,
           bits / total, dt, total / dt / 1e6, mn_memory(m) / 1048576.0);
    if (mn_save(m, o->out) != 0) {
        fprintf(stderr, "cannot save %s\n", o->out);
        return 1;
    }
    mn_free(m);
    return 0;
}

static mn_model *load_or_die(const char *path) {
    mn_model *m = mn_load(path);
    if (!m) {
        fprintf(stderr, "cannot load model %s\n", path);
        exit(1);
    }
    return m;
}

static int cmd_eval(Opts *o) {
    if (!o->model || o->npos == 0) {
        fprintf(stderr, "usage: mnemo eval -m MODEL FILE\n");
        return 1;
    }
    mn_model *m = load_or_die(o->model);
    report_eval("eval", m, o->pos[0], o->is_static);
    mn_free(m);
    return 0;
}

static int cmd_gen(Opts *o) {
    if (!o->model) {
        fprintf(stderr, "usage: mnemo gen -m MODEL [--prompt TEXT]\n");
        return 1;
    }
    mn_model *m = load_or_die(o->model);
    mn_set_learning(m, o->learn_gen);
    uint64_t rng = o->seed * 0x9E3779B97F4A7C15ULL + 1;
    if (o->prompt) {
        mn_observe(m, (const uint8_t *)o->prompt, strlen(o->prompt));
        fputs(o->prompt, stdout);
    }
    for (int i = 0; i < o->n; i++) {
        int c = mn_sample_byte(m, o->temp, &rng);
        putchar(c);
    }
    putchar('\n');
    mn_free(m);
    return 0;
}

/* ---- arithmetic coder (carry-less, 32-bit) ---- */
typedef struct {
    uint32_t x1, x2, x;
    FILE *f;
} Coder;

static void enc_bit(Coder *c, int y, uint32_t p16) {
    uint32_t xmid = c->x1 + (uint32_t)(((uint64_t)(c->x2 - c->x1) * p16) >> 16);
    if (y)
        c->x2 = xmid;
    else
        c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) & 0xFF000000u) == 0) {
        putc((int)(c->x2 >> 24), c->f);
        c->x1 <<= 8;
        c->x2 = (c->x2 << 8) | 255u;
    }
}
static int dec_bit(Coder *c, uint32_t p16) {
    uint32_t xmid = c->x1 + (uint32_t)(((uint64_t)(c->x2 - c->x1) * p16) >> 16);
    int y = c->x <= xmid;
    if (y)
        c->x2 = xmid;
    else
        c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) & 0xFF000000u) == 0) {
        c->x1 <<= 8;
        c->x2 = (c->x2 << 8) | 255u;
        int ch = getc(c->f);
        c->x = (c->x << 8) | (uint32_t)(ch == EOF ? 0 : ch);
    }
    return y;
}

static int cmd_compress(Opts *o) {
    if (o->npos < 2) {
        fprintf(stderr, "usage: mnemo compress IN OUT\n");
        return 1;
    }
    size_t n;
    uint8_t *b = read_file(o->pos[0], &n);
    FILE *f = fopen(o->pos[1], "wb");
    if (!f) return 1;
    uint32_t hdr[8] = {0x335A4E4Du, (uint32_t)n, (uint32_t)((uint64_t)n >> 32),
                       (uint32_t)o->cfg.table_bits, (uint32_t)o->cfg.hist_bits,
                       (uint32_t)o->cfg.match_bits, (uint32_t)o->cfg.lex_order,
                       (uint32_t)o->cfg.lex_spaced};
    fwrite(hdr, 4, 8, f);
    mn_model *m = make_model(o);
    Coder c = {0, 0xFFFFFFFFu, 0, f};
    double t0 = now_sec();
    for (size_t i = 0; i < n; i++)
        for (int k = 7; k >= 0; k--) {
            int y = (b[i] >> k) & 1;
            enc_bit(&c, y, (uint32_t)mn_p16(m));
            mn_update(m, y);
        }
    for (int k = 0; k < 4; k++) {
        putc((int)(c.x1 >> 24), f);
        c.x1 <<= 8;
    }
    long out = ftell(f);
    fclose(f);
    double dt = now_sec() - t0;
    printf("%zu -> %ld bytes (%.4f bpc) in %.2f s\n", n, out, 8.0 * out / (n ? n : 1), dt);
    mn_free(m);
    free(b);
    return 0;
}

static int cmd_decompress(Opts *o) {
    if (o->npos < 2) {
        fprintf(stderr, "usage: mnemo decompress IN OUT\n");
        return 1;
    }
    FILE *f = fopen(o->pos[0], "rb");
    if (!f) return 1;
    uint32_t hdr[8];
    if (fread(hdr, 4, 8, f) != 8 || hdr[0] != 0x335A4E4Du) {
        fprintf(stderr, "bad archive\n");
        return 1;
    }
    size_t n = (size_t)hdr[1] | ((size_t)hdr[2] << 32);
    o->cfg.table_bits = (int)hdr[3];
    o->cfg.hist_bits = (int)hdr[4];
    o->cfg.match_bits = (int)hdr[5];
    o->cfg.lex_order = (int)hdr[6];
    o->cfg.lex_spaced = (int)hdr[7];
    mn_model *m = make_model(o);
    Coder c = {0, 0xFFFFFFFFu, 0, f};
    for (int k = 0; k < 4; k++) {
        int ch = getc(f);
        c.x = (c.x << 8) | (uint32_t)(ch == EOF ? 0 : ch);
    }
    FILE *g = fopen(o->pos[1], "wb");
    if (!g) return 1;
    for (size_t i = 0; i < n; i++) {
        int byte = 0;
        for (int k = 0; k < 8; k++) {
            int y = dec_bit(&c, (uint32_t)mn_p16(m));
            mn_update(m, y);
            byte = (byte << 1) | y;
        }
        putc(byte, g);
    }
    fclose(g);
    fclose(f);
    mn_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: mnemo bench|train|eval|gen|compress|decompress ...\n"
                "  bench --train FILE [--train FILE] --test FILE [--static]\n"
                "  train -o MODEL FILE...      eval -m MODEL FILE [--static]\n"
                "  gen -m MODEL [--prompt T] [-n N] [--temp T] [--seed S]\n"
                "  compress IN OUT             decompress IN OUT\n"
                "options: --tb --hb --mb --cb BITS, --lr X, --lr0 X, --classes FILE\n");
        return 1;
    }
    Opts o;
    parse(argc, argv, &o);
    const char *cmd = argv[1];
    if (!strcmp(cmd, "bench")) return cmd_bench(&o);
    if (!strcmp(cmd, "train")) return cmd_train(&o);
    if (!strcmp(cmd, "eval")) return cmd_eval(&o);
    if (!strcmp(cmd, "gen")) return cmd_gen(&o);
    if (!strcmp(cmd, "compress")) return cmd_compress(&o);
    if (!strcmp(cmd, "decompress")) return cmd_decompress(&o);
    fprintf(stderr, "unknown command %s\n", cmd);
    return 1;
}
