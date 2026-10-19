#!/usr/bin/env bash
# Reproduce the language-modelling numbers in README.md / RESEARCH.md.
# Usage: scripts/benchmark.sh [jobs]   (results in results/*.txt, summary on stdout)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
JOBS="${1:-4}"
make -s
[ -s data/ptb/ptb.train.txt ] || scripts/get_data.sh
mkdir -p results
B=./build/mnemo
PTB="--train data/ptb/ptb.train.txt --test data/ptb/ptb.test.txt --spaced"
WT2="--train data/wikitext2/train.txt --test data/wikitext2/test.txt --spaced"
SHK="--train data/shakespeare/train.txt --test data/shakespeare/val.txt"
python3 -c "
d=open('data/shakespeare/input.txt','rb').read(); k=int(len(d)*0.9)
open('data/shakespeare/train.txt','wb').write(d[:k]); open('data/shakespeare/val.txt','wb').write(d[k:])"
cat > results/jobs.txt <<JOBS
ptb_tiny        $B bench $PTB --tb 18 --hb 21 --mb 18 --lex 3
ptb_small       $B bench $PTB --tb 21 --hb 23 --mb 21 --lex 4
ptb_default     $B bench $PTB --tb 22 --hb 24 --mb 22 --lex 4
ptb_large       $B bench $PTB --tb 23 --hb 24 --mb 22 --lex 5
ptb_default_static $B bench $PTB --tb 22 --hb 24 --mb 22 --lex 4 --static
ptb_nolex       $B bench $PTB --tb 22 --hb 24 --mb 22 --lex 0
ptb_nolex_static $B bench $PTB --tb 22 --hb 24 --mb 22 --lex 0 --static
wt2_tiny        $B bench $WT2 --tb 18 --hb 24 --mb 18 --lex 3
wt2_small       $B bench $WT2 --tb 21 --hb 24 --mb 21 --lex 4
wt2_default     $B bench $WT2 --tb 22 --hb 24 --mb 22 --lex 4
wt2_default_static $B bench $WT2 --tb 22 --hb 24 --mb 22 --lex 4 --static
shk_tiny        $B bench $SHK --tb 18 --hb 21 --mb 18 --lex 3
shk_default     $B bench $SHK --tb 22 --hb 24 --mb 22 --lex 4
shk_default_static $B bench $SHK --tb 22 --hb 24 --mb 22 --lex 4 --static
shk_nolex_static $B bench $SHK --tb 22 --hb 24 --mb 22 --lex 0 --static
JOBS
while read -r name cmd; do echo "$name|$cmd"; done < results/jobs.txt |
  xargs -P "$JOBS" -I{} bash -c 'n="${1%%|*}"; c="${1#*|}"; $c > "results/$n.txt" 2>&1' _ {}
printf "%-20s %10s %10s %12s %10s %8s\n" run test_bpc token_ppl lexicon_bpc memory_MB train_s
for f in results/*.txt; do
  n=$(basename "$f" .txt); [ "$n" = jobs ] && continue
  bpc=$(grep -o '[0-9.]* bpc, [0-9]* tokens' "$f" | head -1 | awk '{print $1}')
  ppl=$(grep -o '[0-9.]* token-ppl' "$f" | awk '{print $1}')
  lex=$(grep -o 'there' "$f" >/dev/null && grep -o '[0-9.]* bpc there' "$f" | awk '{print $1}' || echo -)
  mem=$(grep -o 'memory after: [0-9.]*' "$f" | awk '{print $3}')
  tr=$(grep '^train' "$f" | grep -o '[0-9.]* s (' | awk '{print $1}')
  printf "%-20s %10s %10s %12s %10s %8s\n" "$n" "$bpc" "$ppl" "${lex:--}" "$mem" "$tr"
done
