"""
AIX Model Training
Usage:
  python train.py --task lm  [--data DIR] [--epochs N] [--batch N] [--lr F]
  python train.py --task asr [--data DIR]
  python train.py --task tts [--data DIR]

Dependencies: numpy scipy  (no ML framework required)
For production: swap analytic-gradient LM head with torch.nn + autograd
"""

import json, argparse, math, time, sys
import numpy as np
from pathlib import Path


# ── Tokenizer ────────────────────────────────────────────────
class AIXTokenizer:
    PHONEMES = [
        "AX","EX","IX","OX","UX","YX",
        "BX","PX","DX","TX","GX","KX",
        "FX","VX","SX","ZX","HX","QX",
        "MX","NX","LX","RX","WX","JX",
        "ΞX","ΛX","ΨX","ΦX","ΔX","ΩX",
    ]
    LOGIC_SYMBOLS  = ["⟨⟩","⟦⟧","⟪⟫","⌈⌉","⌊⌋","⌊⌈⌉⌋"]
    SPECIAL_TOKENS = ["[PAD]","[UNK]","[BOS]","[EOS]","[MASK]","[SEP]"]

    def __init__(self):
        vocab = self.SPECIAL_TOKENS + self.PHONEMES + self.LOGIC_SYMBOLS
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.pad_id = self.token2id["[PAD]"]
        self.unk_id = self.token2id["[UNK]"]
        self.bos_id = self.token2id["[BOS]"]
        self.eos_id = self.token2id["[EOS]"]

    @property
    def vocab_size(self): return len(self.token2id)

    def encode(self, tokens, add_special=True):
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        return ([self.bos_id] + ids + [self.eos_id]) if add_special else ids

    def decode(self, ids, skip_special=True):
        return [self.id2token.get(i, "[UNK]") for i in ids
                if not (skip_special and self.id2token.get(i) in self.SPECIAL_TOKENS)]

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tok


# ── Math helpers ─────────────────────────────────────────────
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def positional_encoding(max_len, d):
    pe  = np.zeros((max_len, d), dtype=np.float32)
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

def edit_distance(a, b):
    d = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1): d[i, 0] = i
    for j in range(len(b)+1): d[0, j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            d[i, j] = d[i-1, j-1] if a[i-1] == b[j-1] \
                       else 1 + min(d[i-1,j], d[i,j-1], d[i-1,j-1])
    return int(d[len(a), len(b)])


# ── Transformer blocks ───────────────────────────────────────
class Linear:
    def __init__(self, i, o, rng):
        self.W = (rng.standard_normal((i, o)) * math.sqrt(2/i)).astype(np.float32)
        self.b = np.zeros(o, dtype=np.float32)
    def __call__(self, x): return x @ self.W + self.b
    def params(self): return [self.W, self.b]

class LayerNorm:
    def __init__(self, d):
        self.g = np.ones(d, dtype=np.float32)
        self.b = np.zeros(d, dtype=np.float32)
    def __call__(self, x):
        mu, s = x.mean(-1, keepdims=True), x.std(-1, keepdims=True)
        return self.g * (x - mu) / (s + 1e-6) + self.b
    def params(self): return [self.g, self.b]

class MHA:
    def __init__(self, d, h, rng):
        self.h, self.dk = h, d // h
        self.Wq = Linear(d, d, rng); self.Wk = Linear(d, d, rng)
        self.Wv = Linear(d, d, rng); self.Wo = Linear(d, d, rng)
    def __call__(self, q, k, v, mask=None):
        B, T, D = q.shape; H, dk = self.h, self.dk
        def split(x, W): return W(x).reshape(B, -1, H, dk).transpose(0, 2, 1, 3)
        Q, K, V = split(q, self.Wq), split(k, self.Wk), split(v, self.Wv)
        s = Q @ K.transpose(0,1,3,2) / math.sqrt(dk)
        if mask is not None: s += mask * -1e9
        ctx = (softmax(s, -1) @ V).transpose(0,2,1,3).reshape(B, -1, D)
        return self.Wo(ctx)
    def params(self):
        return self.Wq.params()+self.Wk.params()+self.Wv.params()+self.Wo.params()

class FFN:
    def __init__(self, d, ff, rng):
        self.l1 = Linear(d, ff, rng); self.l2 = Linear(ff, d, rng)
    def __call__(self, x): return self.l2(np.maximum(0, self.l1(x)))
    def params(self): return self.l1.params() + self.l2.params()

class EncoderLayer:
    def __init__(self, d, h, ff, rng):
        self.attn = MHA(d, h, rng); self.ffn = FFN(d, ff, rng)
        self.ln1  = LayerNorm(d);   self.ln2  = LayerNorm(d)
    def __call__(self, x, mask=None):
        x = self.ln1(x + self.attn(x, x, x, mask))
        return self.ln2(x + self.ffn(x))
    def params(self):
        return self.attn.params()+self.ffn.params()+self.ln1.params()+self.ln2.params()

class Encoder:
    def __init__(self, n, d, h, ff, rng):
        self.layers = [EncoderLayer(d, h, ff, rng) for _ in range(n)]
    def __call__(self, x, mask=None):
        for l in self.layers: x = l(x, mask)
        return x
    def params(self):
        p = []
        for l in self.layers: p.extend(l.params())
        return p


# ── Models ───────────────────────────────────────────────────
class ASRModel:
    def __init__(self, n_mels=80, vocab_size=44, d=256, h=4, n=4, rng=None):
        rng = rng or np.random.default_rng(42)
        self.proj = Linear(n_mels, d, rng)
        self.enc  = Encoder(n, d, h, d*4, rng)
        self.head = Linear(d, vocab_size, rng)
        self.pe   = positional_encoding(2048, d)
    def __call__(self, mel):
        x = self.proj(mel.transpose(0, 2, 1)) + self.pe[:mel.shape[2]]
        return self.head(self.enc(x))
    def n_params(self):
        return sum(p.size for p in self.proj.params()+self.enc.params()+self.head.params())

class TTSModel:
    def __init__(self, vocab_size=44, n_mels=80, d=256, h=4, n=4, rng=None):
        rng = rng or np.random.default_rng(42)
        self.emb  = (np.random.randn(vocab_size, d) * .01).astype(np.float32)
        self.enc  = Encoder(n, d, h, d*4, rng)
        self.mel  = Linear(d, n_mels, rng)
        self.stop = Linear(d, 1, rng)
        self.pe   = positional_encoding(512, d)
    def __call__(self, ids):
        x = self.emb[ids] + self.pe[:ids.shape[1]]
        h = self.enc(x)
        return self.mel(h).transpose(0, 2, 1), self.stop(h).squeeze(-1)
    def n_params(self):
        return (self.emb.size +
                sum(p.size for p in self.enc.params()+self.mel.params()+self.stop.params()))

class LMModel:
    def __init__(self, vocab_size=44, d=256, h=4, n=4, max_len=128, rng=None):
        rng = rng or np.random.default_rng(42)
        self.emb  = (np.random.randn(vocab_size, d) * .01).astype(np.float32)
        self.enc  = Encoder(n, d, h, d*4, rng)
        self.head = Linear(d, vocab_size, rng)
        self.pe   = positional_encoding(max_len, d)
        self.V    = vocab_size
    def __call__(self, ids):
        T = ids.shape[1]
        x = self.emb[ids] + self.pe[:T]
        mask = np.triu(np.ones((1, 1, T, T), dtype=np.float32), k=1)
        return self.head(self.enc(x, mask))
    def generate(self, prompt, max_new=20, temperature=1.0):
        ids = list(prompt)
        for _ in range(max_new):
            inp   = np.array([ids[-64:]])
            logit = self(inp)[0, -1]
            if temperature != 1.0: logit = logit / temperature
            probs = softmax(logit)
            nxt   = int(np.random.choice(len(probs), p=probs))
            ids.append(nxt)
            if nxt == 3: break
        return ids
    def n_params(self):
        return self.emb.size + sum(p.size for p in self.enc.params()+self.head.params())


# ── Training ─────────────────────────────────────────────────
def train_lm(model, tokenizer, train_recs, val_recs, cfg):
    lr, V = cfg["lr"], model.V
    history = {"loss": [], "ppl": [], "val_loss": []}

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        np.random.shuffle(train_recs)
        losses = []

        for i in range(0, min(len(train_recs), 80), cfg["batch_size"]):
            batch   = train_recs[i:i+cfg["batch_size"]]
            ids     = [r["token_ids"] for r in batch]
            max_len = max(len(s) for s in ids)
            padded  = np.array([s + [tokenizer.pad_id]*(max_len-len(s)) for s in ids])

            T  = padded.shape[1]
            x  = model.emb[padded] + model.pe[:T]
            mk = np.triu(np.ones((1, 1, T, T), dtype=np.float32), k=1)
            h  = model.enc(x, mk)
            lg = h @ model.head.W + model.head.b

            tgt   = padded[:, 1:]
            lf    = lg[:, :-1, :].reshape(-1, V)
            tf    = tgt.reshape(-1)
            valid = tf != tokenizer.pad_id

            probs = softmax(lf)
            loss  = -np.log(probs[np.arange(len(tf)), tf] + 1e-9)[valid].mean()
            losses.append(float(loss))

            N = int(valid.sum())
            if N == 0: continue
            dl = probs.copy()
            dl[np.arange(len(tf)), tf] -= 1.0
            dl[~valid] = 0.0
            dl /= N

            hf = h[:, :-1, :].reshape(-1, h.shape[-1])
            model.head.W -= lr * (hf.T @ dl)
            model.head.b -= lr * dl.sum(0)

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        ppl      = math.exp(min(avg_loss, 20))
        history["loss"].append(round(avg_loss, 4))
        history["ppl"].append(round(ppl, 2))

        val_loss = None
        if val_recs:
            vloss = []
            for r in val_recs[:20]:
                ids_v = np.array([r["token_ids"]])
                lg_v  = model(ids_v)[:, :-1, :].reshape(-1, V)
                tf_v  = np.array(r["token_ids"][1:])
                valid_v = tf_v != tokenizer.pad_id
                if valid_v.any():
                    p_v = softmax(lg_v)
                    vloss.append(float(-np.log(p_v[np.arange(len(tf_v)), tf_v]+1e-9)[valid_v].mean()))
            val_loss = round(float(np.mean(vloss)), 4) if vloss else None
        history["val_loss"].append(val_loss)

        val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(f"epoch {epoch+1:02d}/{cfg['epochs']}  "
              f"loss={avg_loss:.4f}  ppl={ppl:.1f}{val_str}  ({time.time()-t0:.2f}s)")

    return history


# ── Evaluation ───────────────────────────────────────────────
def evaluate_lm(model, tokenizer, records, n=5):
    results = []
    for rec in records[:n]:
        prompt  = rec["token_ids"][:2]
        gen_ids = model.generate(prompt, max_new=12, temperature=0.8)
        gen_tok = tokenizer.decode(gen_ids)
        ref_tok = tokenizer.decode(rec["token_ids"])
        bleu1   = sum(1 for t in gen_tok if t in set(ref_tok)) / max(len(gen_tok), 1)
        wer     = edit_distance(ref_tok, gen_tok) / max(len(ref_tok), 1)
        results.append({
            "id":    rec.get("id", "?"),
            "ref":   ref_tok,
            "gen":   gen_tok,
            "bleu1": round(bleu1, 4),
            "wer":   round(wer, 4),
        })

    avg_bleu = float(np.mean([r["bleu1"] for r in results]))
    avg_wer  = float(np.mean([r["wer"]   for r in results]))

    print(f"\nEvaluation ({len(results)} samples)")
    print(f"  avg BLEU-1 : {avg_bleu:.4f}  (target > 0.5)")
    print(f"  avg WER    : {avg_wer:.4f}   (target < 0.3)")
    for r in results:
        print(f"  [{r['id']}]")
        print(f"    ref: {' '.join(r['ref'])}")
        print(f"    gen: {' '.join(r['gen'])}")
        print(f"    BLEU-1={r['bleu1']}  WER={r['wer']}")

    return results, {"avg_bleu1": round(avg_bleu, 4), "avg_wer": round(avg_wer, 4)}


# ── Data loading ─────────────────────────────────────────────
def load_split(data_dir, name):
    path = data_dir / f"processed_{name}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    for r in recs:
        r["_split"] = name
    return recs

def make_dummy_data(tokenizer):
    """使用 AIX 詞彙建立小型 demo 資料集（不需要先跑前兩個步驟）"""
    vocab_samples = [
        (["ΞX"],           "是"),
        (["ΩX"],           "否"),
        (["ΨX","AX"],      "存在"),
        (["ΦX","AX"],      "查詢"),
        (["ΨX","IX"],      "理解"),
        (["LX","AX"],      "學習"),
        (["MX","UX"],      "記憶"),
        (["SX","IX"],      "搜尋"),
        (["ΨX","ΞX"],      "AI"),
        (["MX","AX"],      "人類"),
        (["HX","AX","ΨX"], "你好"),
        (["ΩX","HX","OX"], "再見"),
        (["FX","IX","ΞX"], "謝謝"),
        (["MX","AX","NX"], "吃飯"),
        (["SX","UX","IX"], "睡覺"),
        (["HX","AX","PX"], "快樂"),
        (["NX","AX","WX"], "現在"),
        (["HX","OX","MX"], "家"),
    ]
    sent_samples = [
        ([["ΨX","ΞX"],["MX","AX"],["ΨX","IX"]], "AI理解人類"),
        ([["ΨX","ΞX"],["DX","AX"],["SX","IX"]], "AI搜尋資料"),
        ([["MX","AX"],["LX","OX"],["ΦX","AX"]], "人類查詢語言"),
        ([["ΨX","ΞX"],["ΩX"],["ΨX","IX"]],     "AI不理解"),
    ]
    recs = []
    for i, (tokens, zh) in enumerate(vocab_samples * 3):
        recs.append({
            "id": f"dummy_word_{i:03d}",
            "aix_text": "·".join(tokens),
            "tokens": tokens,
            "zh": zh,
            "token_ids": tokenizer.encode(tokens),
            "type": "word",
        })
    for i, (groups, zh) in enumerate(sent_samples * 5):
        tokens = [t for g in groups for t in g]
        recs.append({
            "id": f"dummy_sent_{i:03d}",
            "aix_text": " ".join("·".join(g) for g in groups),
            "tokens": tokens,
            "zh": zh,
            "token_ids": tokenizer.encode(tokens),
            "type": "sentence",
        })
    np.random.seed(42)
    np.random.shuffle(recs)
    n_train = int(len(recs) * .8)
    n_val   = int(len(recs) * .1)
    for r in recs[:n_train]:             r["_split"] = "train"
    for r in recs[n_train:n_train+n_val]:r["_split"] = "val"
    for r in recs[n_train+n_val:]:       r["_split"] = "test"
    return recs


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default="lm", choices=["asr","tts","lm"])
    parser.add_argument("--data",    default="./aix_dataset/processed")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--d-model", type=int,   default=128)
    parser.add_argument("--n-heads", type=int,   default=4)
    parser.add_argument("--n-layers",type=int,   default=2)
    args = parser.parse_args()

    data_dir  = Path(args.data)
    tok_path  = data_dir / "aix_tokenizer.json"
    tokenizer = AIXTokenizer.load(str(tok_path)) if tok_path.exists() else AIXTokenizer()
    print(f"Tokenizer vocab_size={tokenizer.vocab_size}")

    all_recs   = load_split(data_dir, "train") + \
                 load_split(data_dir, "val")   + \
                 load_split(data_dir, "test")
    train_recs = [r for r in all_recs if r.get("_split") == "train"]
    val_recs   = [r for r in all_recs if r.get("_split") == "val"]
    test_recs  = [r for r in all_recs if r.get("_split") == "test"]

    if not all_recs:
        print("No processed data found — using built-in demo data.")
        print("Run 01_generate_dataset.py + 02_preprocess.py for real training.\n")
        all_recs   = make_dummy_data(tokenizer)
        train_recs = [r for r in all_recs if r["_split"] == "train"]
        val_recs   = [r for r in all_recs if r["_split"] == "val"]
        test_recs  = [r for r in all_recs if r["_split"] == "test"]

    print(f"Data: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test")

    cfg = {"epochs": args.epochs, "batch_size": args.batch, "lr": args.lr}
    rng = np.random.default_rng(42)
    d, h, n = args.d_model, args.n_heads, args.n_layers

    if args.task == "asr":
        model = ASRModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"ASR model params: {model.n_params():,}")
        dummy = np.random.randn(2, 80, 50).astype(np.float32)
        print(f"Forward: mel{dummy.shape} → logits{model(dummy).shape}")

    elif args.task == "tts":
        model = TTSModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"TTS model params: {model.n_params():,}")
        dummy = np.array([[2, 6, 12, 20, 3]])
        mel, stop = model(dummy)
        print(f"Forward: ids{dummy.shape} → mel{mel.shape}, stop{stop.shape}")

    elif args.task == "lm":
        model = LMModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"LM model params: {model.n_params():,}\n")

        history = train_lm(model, tokenizer, train_recs, val_recs, cfg)

        eval_recs = test_recs if test_recs else val_recs[:5]
        eval_results, eval_metrics = evaluate_lm(model, tokenizer, eval_recs)

        ckpt_dir = data_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        with open(ckpt_dir / "lm_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "task":             "lm",
                "config":           cfg,
                "training_history": history,
                "eval_metrics":     eval_metrics,
                "eval_samples":     [
                    {**r, "ref": r["ref"], "gen": r["gen"]}
                    for r in eval_results
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nResult → {ckpt_dir / 'lm_result.json'}")


if __name__ == "__main__":
    main()