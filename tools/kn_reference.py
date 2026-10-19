"""Reference interpolated modified Kneser-Ney n-gram LM (word level).

Used to validate the online lexicon in src/lexicon.c against the published
PTB baseline (KN5 = 141.2, Mikolov et al. 2011).

  python3 tools/kn_reference.py data/ptb/ptb.train.txt data/ptb/ptb.test.txt [order] [--stream]

--stream treats the corpus as one token stream with '\\n' as a token (the
setting used by the char-level lexicon); otherwise each line is a sentence
with <s> ... </s>.
"""
import math
import sys
from collections import defaultdict


def tokens(path, stream):
    for line in open(path, encoding="utf-8"):
        ws = line.split()
        if stream:
            yield from ws
            yield "\n"
        else:
            yield ["<s>"] + ws + ["</s>"]


def train(path, order, stream):
    counts = [defaultdict(int) for _ in range(order + 1)]  # counts[k][(ctx..., w)]
    seqs = [list(tokens(path, True))] if stream else list(tokens(path, False))
    for seq in seqs:
        start = 0 if stream else 1
        for i in range(start, len(seq)):
            for k in range(1, order + 1):
                if i - k + 1 < 0:
                    break
                counts[k][tuple(seq[i - k + 1:i + 1])] += 1
    # continuation counts for lower orders
    cont = [defaultdict(int) for _ in range(order + 1)]
    for k in range(2, order + 1):
        for g in counts[k]:
            cont[k - 1][g[1:]] += 1
    tabs = [None] * (order + 1)
    for k in range(1, order + 1):
        tabs[k] = counts[k] if k == order else cont[k]
        # for sentence mode, n-grams starting with <s> are not left-extendable
        if not stream and k < order:
            for g, c in counts[k].items():
                if g[0] == "<s>":
                    tabs[k][g] = c
    model = []
    for k in range(1, order + 1):
        t = tabs[k]
        coc = defaultdict(int)
        for c in t.values():
            if c <= 4:
                coc[c] += 1
        n1, n2, n3, n4 = (coc[i] for i in (1, 2, 3, 4))
        Y = n1 / (n1 + 2 * n2)
        D = (1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3)
        tot = defaultdict(int)
        nn = defaultdict(lambda: [0, 0, 0])
        for g, c in t.items():
            tot[g[:-1]] += c
            nn[g[:-1]][min(c, 3) - 1] += 1
        model.append((t, D, tot, nn))
    return model


def prob(model, ctx, w, vocab_size):
    p = 1.0 / vocab_size
    for k in range(1, len(model) + 1):
        if k - 1 > len(ctx):
            break
        h = tuple(ctx[len(ctx) - (k - 1):]) if k > 1 else ()
        t, D, tot, nn = model[k - 1]
        T = tot.get(h, 0)
        if T == 0:
            continue
        c = t.get(h + (w,), 0)
        d = D[min(c, 3) - 1] if c else 0
        g = (D[0] * nn[h][0] + D[1] * nn[h][1] + D[2] * nn[h][2]) / T
        p = max(c - d, 0) / T + g * p
    return p


def main():
    tr, te = sys.argv[1], sys.argv[2]
    order = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 5
    stream = "--stream" in sys.argv
    model = train(tr, order, stream)
    V = len(model[0][0]) + 1
    bits, n = 0.0, 0
    if stream:
        seq = list(tokens(te, True))
        hist = list(tokens(tr, True))[-(order - 1):]
        for w in seq:
            bits -= math.log2(prob(model, hist[-(order - 1):], w, V))
            hist.append(w)
            n += 1
    else:
        for s in tokens(te, False):
            for i in range(1, len(s)):
                bits -= math.log2(prob(model, s[max(0, i - order + 1):i], s[i], V))
                n += 1
    print(f"order {order} {'stream' if stream else 'sentences'}: {n} tokens, ppl {2 ** (bits / n):.2f}")


if __name__ == "__main__":
    main()
