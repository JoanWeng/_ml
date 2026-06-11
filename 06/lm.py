"""
Character-level interpolated n-gram language model
Non-transformer, pure Python implementation

Usage:
  python3 lm.py tw.txt
"""

import sys
import math
import random
from collections import defaultdict, Counter

# ── config ────────────────────────────────────────────────
MAX_N = 3           # max n-gram order
K = 0.5             # add-k smoothing for each order
MAX_GEN = 50        # max generation length
TEMPERATURE = 1.0
TOP_K = 5
SEED = 42
# ──────────────────────────────────────────────────────────

def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        chars = []
        for line in f:
            line = line.strip()
            if line:
                chars.extend(list(line))
                chars.append("<EOS>")
    return chars

class NGramLM:
    def __init__(self, max_n, k=0.5):
        self.max_n = max_n
        self.k = k
        # counts[n][context][char] for n-grams of order n (1 <= n <= max_n)
        self.counts = [defaultdict(Counter) for _ in range(max_n + 1)]
        self.unigram = Counter()
        self.vocab = set()

    def train(self, chars):
        for order in range(1, self.max_n + 1):
            for i in range(len(chars) - order):
                ctx = tuple(chars[i : i + order])
                nxt = chars[i + order]
                self.counts[order][ctx][nxt] += 1
                self.vocab.add(nxt)
                for c in ctx:
                    self.vocab.add(c)
        self.unigram = self.counts[1][tuple()]
        self.vocab_size = len(self.vocab)
        print(f"  vocab size: {self.vocab_size}")
        for n in range(1, self.max_n + 1):
            print(f"  {n}-gram contexts: {len(self.counts[n])}")

    def prob_order(self, order, ctx, char):
        if order == 0:
            return 1.0 / self.vocab_size
        c = self.counts[order].get(ctx, {})
        numer = c.get(char, 0) + self.k
        denom = sum(c.values()) + self.k * self.vocab_size
        return numer / denom

    def prob(self, ctx, char):
        ctx_len = len(ctx)
        result = 0.0
        for order in range(1, min(ctx_len, self.max_n) + 1):
            sub_ctx = ctx[-order:]
            pw = self.prob_order(order, sub_ctx, char)
            weight = 1.0 / min(ctx_len, self.max_n)
            result += weight * pw
        lower = 1.0 / self.vocab_size
        return result

    def predict_proba(self, ctx):
        probs = []
        for c in self.vocab:
            p = self.prob(ctx, c)
            if p > 0:
                probs.append((c, p))
        probs.sort(key=lambda x: -x[1])
        return probs

    def sample(self, ctx, temperature=1.0, top_k=0):
        probs = self.predict_proba(ctx)
        if not probs:
            return "<EOS>", []
        raw_probs = list(probs)
        if top_k and top_k < len(probs):
            probs = probs[:top_k]
        chars, pvals = zip(*probs)
        if temperature != 1.0:
            logits = [math.log(max(p, 1e-10)) / temperature for p in pvals]
            logits = [v - max(logits) for v in logits]
            exps = [math.exp(v) for v in logits]
            total = sum(exps)
            pvals = [e / total for e in exps]
        r = random.random()
        cum = 0.0
        for c, p in zip(chars, pvals):
            cum += p
            if r <= cum:
                return c, raw_probs[:5]
        return chars[-1], raw_probs[:5]

    def generate(self, prompt, max_len=MAX_GEN, temperature=TEMPERATURE,
                 top_k=TOP_K, verbose=False):
        seed = list(prompt)
        out = list(prompt)
        ctx = tuple(seed)

        if verbose:
            print(f"\n  prompt: {prompt}")
            print(f"  {'step':<4} {'char':<6} {'prob':<8} top-5")

        for step in range(max_len):
            char, top5 = self.sample(ctx, temperature, top_k)
            if verbose:
                s = " ".join(f"{c}({p:.0%})" for c, p in top5)
                print(f"  {step+1:<4} {char:<6} {top5[0][1]:.2%}  {s}")
            if char == "<EOS>":
                break
            out.append(char)
            ctx = ctx + (char,)
            if len(ctx) > self.max_n * 2:
                ctx = ctx[-(self.max_n * 2):]

        return "".join(out)


def interactive(lm):
    temp = TEMPERATURE
    topk = TOP_K
    maxlen = MAX_GEN
    verbose = False

    print("\n" + "=" * 54)
    print("  Interpolated N-gram Language Model")
    print("=" * 54)
    print(f"  max_n={MAX_N}  temp={temp}  topk={topk}  maxlen={maxlen}")
    print("  commands: /temp N  /topk N  /len N  /v  /demo  /help  q")
    print("=" * 54)

    demo(lm)

    while True:
        try:
            raw = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not raw:
            continue
        if raw.lower() == "q":
            print("bye!")
            break
        if raw == "/help":
            print("  /temp N  /topk N  /len N  /v  /demo  q")
            continue
        if raw == "/v":
            verbose = not verbose
            print(f"  verbose = {verbose}")
            continue
        if raw == "/demo":
            demo(lm)
            continue
        if raw.startswith("/"):
            try:
                cmd, val = raw.split()
                val = float(val)
                if cmd == "/temp":
                    temp = val
                elif cmd == "/topk":
                    topk = int(val)
                elif cmd == "/len":
                    maxlen = int(val)
                print(f"  set {cmd[1:]} = {val}")
            except:
                print("  bad command")
            continue

        out = lm.generate(raw, maxlen, temp, topk, verbose)
        if not verbose:
            print(f"  {out}")


DEMOS = ["小貓", "天上", "今天", "我", "春天", "山上"]

def demo(lm):
    print("\n  -- demo generation --")
    for p in DEMOS:
        out = lm.generate(p, max_len=20, temperature=0.8, top_k=5)
        print(f"  [{p}] {out}")
    print()


if __name__ == "__main__":
    random.seed(SEED)
    path = sys.argv[1] if len(sys.argv) > 1 else "tw.txt"
    print(f"[1/2] loading corpus: {path}")
    chars = load_corpus(path)
    print(f"  {len(chars)} chars total")
    print(f"[2/2] training interpolated {MAX_N}-gram model ...")
    lm = NGramLM(MAX_N, K)
    lm.train(chars)
    interactive(lm)
