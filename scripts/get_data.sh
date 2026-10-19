#!/usr/bin/env bash
# Download the public text corpora used for benchmarks into data/ (git-ignored).
# Each corpus goes to its own directory. Sources are plain-text mirrors on GitHub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data"
fetch() {  # fetch URL OUT
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "have  $out"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "fetch $url"
  curl -fsSL --retry 4 --retry-delay 2 -o "$out.part" "$url" && mv "$out.part" "$out"
}
# Tiny Shakespeare (Karpathy char-rnn), 1.1 MB
fetch https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt "$D/shakespeare/input.txt"
# Penn Treebank (Mikolov preprocessing), word level; char-level runs use the same text
for s in train valid test; do
  fetch "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.$s.txt" "$D/ptb/ptb.$s.txt"
done
# WikiText-2 (tokenized), from the PyTorch examples repo
for s in train valid test; do
  fetch "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$s.txt" "$D/wikitext2/$s.txt"
done
# Project Gutenberg selections bundled by NLTK (public domain books, ~12 MB)
if [ ! -d "$D/gutenberg/gutenberg" ]; then
  fetch https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip "$D/gutenberg/gutenberg.zip"
  (cd "$D/gutenberg" && unzip -q -o gutenberg.zip)
fi
# WordNet 3.0 (Princeton; permissive WordNet license) and first-name lists, as
# packaged by NLTK -- general knowledge for `python3 -m mnemo feed --wordnet`
for pkg in wordnet names; do
  if [ ! -d "$D/nltk/$pkg" ]; then
    fetch "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$pkg.zip" "$D/nltk/$pkg.zip"
    (cd "$D/nltk" && unzip -q -o "$pkg.zip")
  fi
done
echo "done"; du -sh "$D"/*
