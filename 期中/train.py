"""
AIX Model Training
Usage:
  python 03_train.py --task lm  [--data DIR] [--epochs N] [--batch N] [--lr F]
  python 03_train.py --task asr [--data DIR]
  python 03_train.py --task tts [--data DIR]
"""

import json, argparse, math, time
import numpy as np
from pathlib import Path


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
        return [self.id2token.get(i,"[UNK]") for i in ids
                if not (skip_special and self.id2token.get(i) in self.SPECIAL_TOKENS)]

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k,v in data["id2token"].items()}
        return tok


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def positional_encoding(max_len, d):
    pe = np.zeros((max_len, d), dtype=np.float32)
    pos = np.arange(max_len)[:,None]
    div = np.exp(np.arange(0,d,2) * (-math.log(10000.0)/d))
    pe[:,0::2] = np.sin(pos*div)
    pe[:,1::2] = np.cos(pos*div)
    return pe


class Linear:
    def __init__(self, i, o, rng):
        self.W = (rng.standard_normal((i,o)) * math.sqrt(2/i)).astype(np.float32)
        self.b = np.zeros(o, dtype=np.float32)
    def __call__(self, x): return x @ self.W + self.b
    def params(self): return [self.W, self.b]

class LayerNorm:
    def __init__(self, d):
        self.g = np.ones(d, dtype=np.float32)
        self.b = np.zeros(d, dtype=np.float32)
    def __call__(self, x):
        mu = x.mean(-1,keepdims=True); s = x.std(-1,keepdims=True)
        return self.g*(x-mu)/(s+1e-6)+self.b
    def params(self): return [self.g, self.b]

class MHA:
    def __init__(self, d, h, rng):
        self.h, self.dk = h, d//h
        self.Wq = Linear(d,d,rng); self.Wk = Linear(d,d,rng)
        self.Wv = Linear(d,d,rng); self.Wo = Linear(d,d,rng)
    def __call__(self, q, k, v, mask=None):
        B,T,D = q.shape; H,dk = self.h, self.dk
        def split(x,W): return W(x).reshape(B,-1,H,dk).transpose(0,2,1,3)
        Q,K,V = split(q,self.Wq), split(k,self.Wk), split(v,self.Wv)
        s = Q @ K.transpose(0,1,3,2) / math.sqrt(dk)
        if mask is not None: s += mask*-1e9
        ctx = (softmax(s,-1) @ V).transpose(0,2,1,3).reshape(B,-1,D)
        return self.Wo(ctx)
    def params(self):
        return self.Wq.params()+self.Wk.params()+self.Wv.params()+self.Wo.params()

class FFN:
    def __init__(self, d, ff, rng):
        self.l1 = Linear(d,ff,rng); self.l2 = Linear(ff,d,rng)
    def __call__(self, x): return self.l2(np.maximum(0, self.l1(x)))
    def params(self): return self.l1.params()+self.l2.params()

class EncoderLayer:
    def __init__(self, d, h, ff, rng):
        self.attn = MHA(d,h,rng); self.ffn = FFN(d,ff,rng)
        self.ln1 = LayerNorm(d); self.ln2 = LayerNorm(d)
    def __call__(self, x, mask=None):
        x = self.ln1(x + self.attn(x,x,x,mask))
        return self.ln2(x + self.ffn(x))
    def params(self):
        return self.attn.params()+self.ffn.params()+self.ln1.params()+self.ln2.params()

class Encoder:
    def __init__(self, n, d, h, ff, rng):
        self.layers = [EncoderLayer(d,h,ff,rng) for _ in range(n)]
    def __call__(self, x, mask=None):
        for l in self.layers: x = l(x, mask)
        return x
    def params(self):
        p = []
        for l in self.layers: p.extend(l.params())
        return p


class ASRModel:
    def __init__(self, n_mels=80, vocab_size=44, d=256, h=4, n=4, rng=None):
        rng = rng or np.random.default_rng(42)
        self.proj = Linear(n_mels, d, rng)
        self.enc  = Encoder(n, d, h, d*4, rng)
        self.head = Linear(d, vocab_size, rng)
        self.pe   = positional_encoding(2048, d)
    def __call__(self, mel):
        x = self.proj(mel.transpose(0,2,1))
        x = x + self.pe[:x.shape[1]]
        return self.head(self.enc(x))
    def n_params(self):
        return sum(p.size for p in self.proj.params()+self.enc.params()+self.head.params())

class TTSModel:
    def __init__(self, vocab_size=44, n_mels=80, d=256, h=4, n=4, rng=None):
        rng = rng or np.random.default_rng(42)
        self.emb  = (np.random.randn(vocab_size,d)*.01).astype(np.float32)
        self.enc  = Encoder(n, d, h, d*4, rng)
        self.mel  = Linear(d, n_mels, rng)
        self.stop = Linear(d, 1, rng)
        self.pe   = positional_encoding(512, d)
    def __call__(self, ids):
        x = self.emb[ids] + self.pe[:ids.shape[1]]
        h = self.enc(x)
        return self.mel(h).transpose(0,2,1), self.stop(h).squeeze(-1)
    def n_params(self):
        return (self.emb.size +
                sum(p.size for p in self.enc.params()+self.mel.params()+self.stop.params()))

class LMModel:
    def __init__(self, vocab_size=44, d=256, h=4, n=4, max_len=128, rng=None):
        rng = rng or np.random.default_rng(42)
        self.emb    = (np.random.randn(vocab_size,d)*.01).astype(np.float32)
        self.enc    = Encoder(n, d, h, d*4, rng)
        self.head   = Linear(d, vocab_size, rng)
        self.pe     = positional_encoding(max_len, d)
        self.V      = vocab_size
    def __call__(self, ids):
        T = ids.shape[1]
        x = self.emb[ids] + self.pe[:T]
        mask = np.triu(np.ones((1,1,T,T),dtype=np.float32), k=1)
        return self.head(self.enc(x, mask))
    def generate(self, prompt, max_new=20, temperature=1.0):
        ids = list(prompt)
        for _ in range(max_new):
            inp    = np.array([ids[-64:]])
            logits = self(inp)[0,-1]
            if temperature != 1.0: logits = logits/temperature
            probs  = softmax(logits)
            nxt    = int(np.random.choice(len(probs), p=probs))
            ids.append(nxt)
            if nxt == 3: break
        return ids
    def n_params(self):
        return self.emb.size + sum(p.size for p in self.enc.params()+self.head.params())


def train_lm(model, tokenizer, records, cfg):
    lr = cfg["lr"]
    V  = model.V

    history = {"loss": [], "ppl": [], "val_loss": []}

    train_recs = [r for r in records if r.get("_split","train") == "train"]
    val_recs   = [r for r in records if r.get("_split") == "val"]
    if not train_recs:
        train_recs = records

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        np.random.shuffle(train_recs)
        losses = []

        for i in range(0, min(len(train_recs), 80), cfg["batch_size"]):
            batch   = train_recs[i:i+cfg["batch_size"]]
            ids     = [r["token_ids"] for r in batch]
            max_len = max(len(s) for s in ids)
            padded  = np.array([s+[tokenizer.pad_id]*(max_len-len(s)) for s in ids])

            x  = model.emb[padded] + model.pe[:padded.shape[1]]
            T  = x.shape[1]
            mk = np.triu(np.ones((1,1,T,T),dtype=np.float32), k=1)
            h  = model.enc(x, mk)
            logits = h @ model.head.W + model.head.b

            inp_t, tgt_t = padded[:,:-1], padded[:,1:]
            lg = logits[:,:-1,:]
            B, Ts, _ = lg.shape
            lf = lg.reshape(-1,V)
            tf = tgt_t.reshape(-1)
            valid = tf != tokenizer.pad_id

            probs  = softmax(lf)
            log_p  = np.log(probs+1e-9)
            loss   = -log_p[np.arange(len(tf)), tf][valid].mean()
            losses.append(float(loss))

            N = int(valid.sum())
            if N == 0: continue
            dl = probs.copy()
            dl[np.arange(len(tf)), tf] -= 1.0
            dl[~valid] = 0.0
            dl /= N

            hf = h[:,:-1,:].reshape(-1, h.shape[-1])
            model.head.W -= lr * (hf.T @ dl)
            model.head.b -= lr * dl.sum(0)

        avg_loss = float(np.mean(losses))
        ppl      = math.exp(min(avg_loss, 20))
        history["loss"].append(round(avg_loss, 4))
        history["ppl"].append(round(ppl, 2))

        val_loss = None
        if val_recs:
            vloss = []
            for r in val_recs[:20]:
                ids   = np.array([r["token_ids"]])
                lg_v  = model(ids)[:,:-1,:]
                tg_v  = np.array([r["token_ids"][1:]])
                lf_v  = lg_v.reshape(-1,V); tf_v = tg_v.reshape(-1)
                vld   = tf_v != tokenizer.pad_id
                if vld.any():
                    lp = np.log(softmax(lf_v)+1e-9)
                    vloss.append(float(-lp[np.arange(len(tf_v)),tf_v][vld].mean()))
            val_loss = round(float(np.mean(vloss)), 4) if vloss else None
            history["val_loss"].append(val_loss)

        elapsed = time.time()-t0
        val_str = f"  val_loss={val_loss:.4f}" if val_loss else ""
        print(f"epoch {epoch+1:02d}/{cfg['epochs']}  "
              f"loss={avg_loss:.4f}  ppl={ppl:.1f}{val_str}  ({elapsed:.2f}s)")

    return history


def evaluate_lm(model, tokenizer, records, n=5):
    results = []
    for rec in records[:n]:
        prompt   = rec["token_ids"][:2]
        gen_ids  = model.generate(prompt, max_new=12, temperature=0.8)
        gen_tok  = tokenizer.decode(gen_ids)
        ref_tok  = tokenizer.decode(rec["token_ids"])
        ref_set  = set(ref_tok)
        bleu1    = sum(1 for t in gen_tok if t in ref_set)/max(len(gen_tok),1)
        wer      = edit_distance(ref_tok, gen_tok)/max(len(ref_tok),1)
        results.append({
            "id": rec["id"], "ref": ref_tok, "gen": gen_tok,
            "bleu1": round(bleu1,4), "wer": round(wer,4),
        })
    avg_bleu = np.mean([r["bleu1"] for r in results])
    avg_wer  = np.mean([r["wer"]   for r in results])
    print(f"\nEvaluation ({len(results)} samples)")
    print(f"  avg BLEU-1 : {avg_bleu:.4f}  (target > 0.5)")
    print(f"  avg WER    : {avg_wer:.4f}   (target < 0.3)")
    for r in results:
        print(f"  ref: {' '.join(r['ref'])}")
        print(f"  gen: {' '.join(r['gen'])}")
        print(f"  BLEU-1={r['bleu1']}  WER={r['wer']}\n")
    return results, {"avg_bleu1": round(float(avg_bleu),4), "avg_wer": round(float(avg_wer),4)}

def edit_distance(a, b):
    d = np.zeros((len(a)+1,len(b)+1),dtype=int)
    for i in range(len(a)+1): d[i,0]=i
    for j in range(len(b)+1): d[0,j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i,j] = d[i-1,j-1] if a[i-1]==b[j-1] else 1+min(d[i-1,j],d[i,j-1],d[i-1,j-1])
    return d[len(a),len(b)]


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

    def load(name):
        p = data_dir / f"processed_{name}.json"
        if not p.exists(): return []
        with open(p, encoding="utf-8") as f:
            recs = json.load(f)
        for r in recs: r["_split"] = name
        return recs

    all_recs = load("train") + load("val") + load("test")
    train_n  = sum(1 for r in all_recs if r.get("_split")=="train")
    print(f"Data: {train_n} train, "
          f"{sum(1 for r in all_recs if r.get('_split')=='val')} val, "
          f"{sum(1 for r in all_recs if r.get('_split')=='test')} test")

    cfg = {"epochs": args.epochs, "batch_size": args.batch, "lr": args.lr}
    rng = np.random.default_rng(42)
    d, h, n = args.d_model, args.n_heads, args.n_layers

    if args.task == "asr":
        model = ASRModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"ASR model params: {model.n_params():,}")
        dummy = np.random.randn(2, 80, 50).astype(np.float32)
        print(f"Forward check: mel{dummy.shape} → logits{model(dummy).shape}")
        print("For full ASR training: use transformers.WhisperForConditionalGeneration")

    elif args.task == "tts":
        model = TTSModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"TTS model params: {model.n_params():,}")
        dummy = np.array([[2, 6, 12, 20, 3]])
        mel, stop = model(dummy)
        print(f"Forward check: ids{dummy.shape} → mel{mel.shape}, stop{stop.shape}")
        print("For full TTS training: use SpeechT5ForTextToSpeech or VITS")

    elif args.task == "lm":
        model = LMModel(vocab_size=tokenizer.vocab_size, d=d, h=h, n=n, rng=rng)
        print(f"LM model params: {model.n_params():,}\n")

        if not all_recs:
            dummy = [{"token_ids":[2,24+i%6,6+i%6,3],"_split":"train"} for i in range(40)]
            dummy += [{"token_ids":[2,24+i%4,8+i%4,3],"_split":"val"} for i in range(10)]
            all_recs = dummy

        history = train_lm(model, tokenizer, all_recs, cfg)

        test_recs = [r for r in all_recs if r.get("_split")=="test"]
        if not test_recs:
            test_recs = all_recs[:5]
        eval_results, eval_metrics = evaluate_lm(model, tokenizer, test_recs)

        ckpt_dir = data_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        out = {
            "task": "lm", "config": cfg,
            "training_history": history,
            "eval_metrics": eval_metrics,
            "eval_samples": eval_results,
        }
        with open(ckpt_dir / "lm_result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Result saved → {ckpt_dir / 'lm_result.json'}")

if __name__ == "__main__":
    main()