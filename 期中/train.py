"""
AIX Model Training
Usage:
  python train.py --task lm  [--data DIR] [--epochs N] [--batch N] [--lr F]
  python train.py --task asr [--data DIR] [--epochs N] [--lr F] [--d-model N]
  python train.py --task tts [--data DIR]

ASR backend auto-selects:
  - PyTorch (CTC Loss, full backprop) if torch is installed
  - NumPy   (frame-level CE, head-only) as fallback
"""

import json, argparse, math, time, sys
import numpy as np
from pathlib import Path

try:
    import importlib, sys
    _spec = importlib.util.find_spec("torch")
    if _spec is None:
        raise ImportError("torch not found")
    import torch
    import torch.nn as nn
    # Quick sanity check to catch broken installs
    _ = torch.zeros(1)
    TORCH = True
except Exception:
    TORCH = False


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


# ── Shared helpers ───────────────────────────────────────────
def edit_distance(a, b):
    d = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1): d[i,0] = i
    for j in range(len(b)+1): d[0,j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            d[i,j] = d[i-1,j-1] if a[i-1]==b[j-1] \
                     else 1+min(d[i-1,j], d[i,j-1], d[i-1,j-1])
    return int(d[len(a), len(b)])

def load_feat(rec):
    fp = rec.get("feat_path")
    if fp and Path(fp).exists():
        return np.load(fp).astype(np.float32)
    # Structured dummy: each token → unique frequency band, 8 frames each
    toks = rec["token_ids"]
    T, n_mels = len(toks)*8, 80
    mel = np.zeros((n_mels, T), dtype=np.float32)
    for i, tok in enumerate(toks):
        b = int((tok/42)*(n_mels-8))
        mel[b:b+8, i*8:(i+1)*8] = 1.0 + (tok%4)*0.25
    mel += np.random.randn(*mel.shape).astype(np.float32)*0.05
    return mel


# ════════════════════════════════════════════════════════════
# PyTorch ASR (full CTC, trains all layers)
# ════════════════════════════════════════════════════════════
if TORCH:
    class TorchASRModel(nn.Module):
        def __init__(self, n_mels=80, vocab_size=42, d_model=256,
                     n_heads=4, n_layers=4, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(n_mels, d_model)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.ctc_head = nn.Linear(d_model, vocab_size)
            self.dropout  = nn.Dropout(dropout)
            d = d_model
            max_len = 4096
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000)/d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

        def forward(self, mel):
            # mel: (B, n_mels, T)
            x = mel.transpose(1, 2)                 # (B, T, n_mels)
            x = self.dropout(self.input_proj(x))    # (B, T, d_model)
            x = x + self.pe[:, :x.size(1)]
            x = self.encoder(x)                     # (B, T, d_model)
            return self.ctc_head(x)                 # (B, T, vocab_size)

    def ctc_decode(log_probs, blank=0):
        """Greedy CTC decode. log_probs: (T, V)"""
        ids = log_probs.argmax(-1).tolist()
        out, prev = [], -1
        for i in ids:
            if i != blank and i != prev:
                out.append(i)
            prev = i
        return out

    def train_asr_torch(model, tokenizer, train_recs, val_recs, cfg):
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model    = model.to(device)
        ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"]*0.05
        )
        history = {"loss": [], "val_loss": [], "val_wer": []}

        print(f"Device: {device}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            model.train()
            np.random.shuffle(train_recs)
            losses = []

            for i in range(0, min(len(train_recs), 200), cfg["batch_size"]):
                batch = train_recs[i:i+cfg["batch_size"]]

                # Pad mel to same T in batch
                mels   = [torch.tensor(load_feat(r)) for r in batch]
                tgts   = [torch.tensor(r["token_ids"][1:-1], dtype=torch.long)
                          for r in batch]
                T_lens = [m.shape[1] for m in mels]
                T_max  = max(T_lens)
                mel_pad = torch.zeros(len(batch), 80, T_max)
                for j, m in enumerate(mels):
                    mel_pad[j, :, :m.shape[1]] = m
                mel_pad = mel_pad.to(device)

                logits     = model(mel_pad)            # (B, T, V)
                log_probs  = nn.functional.log_softmax(logits, dim=-1)
                log_probs_t = log_probs.permute(1, 0, 2)  # (T, B, V) for CTCLoss

                input_lengths  = torch.tensor(T_lens, dtype=torch.long)
                target_lengths = torch.tensor([len(t) for t in tgts], dtype=torch.long)
                targets_cat    = torch.cat(tgts)

                loss = ctc_loss(log_probs_t, targets_cat, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

            scheduler.step()
            avg_loss = float(np.mean(losses)) if losses else float("nan")
            history["loss"].append(round(avg_loss, 4))

            # Validation
            model.eval()
            val_losses, val_wers = [], []
            with torch.no_grad():
                for rec in (val_recs or train_recs)[:30]:
                    mel = torch.tensor(load_feat(rec)).unsqueeze(0).to(device)
                    tgt = rec["token_ids"][1:-1]
                    if not tgt: continue
                    logits   = model(mel)
                    log_p    = nn.functional.log_softmax(logits, dim=-1)
                    log_p_t  = log_p.permute(1, 0, 2)
                    tgt_t    = torch.tensor(tgt, dtype=torch.long)
                    vl = ctc_loss(log_p_t,
                                  tgt_t,
                                  torch.tensor([mel.shape[2]]),
                                  torch.tensor([len(tgt)]))
                    if torch.isfinite(vl):
                        val_losses.append(vl.item())
                    pred = ctc_decode(log_p[0].cpu().numpy())
                    val_wers.append(edit_distance(tgt, pred) / max(len(tgt), 1))

            vl  = round(float(np.mean(val_losses)), 4) if val_losses else None
            wer = round(float(np.mean(val_wers)),   4) if val_wers   else None
            history["val_loss"].append(vl)
            history["val_wer"].append(wer)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch+1:02d}/{cfg['epochs']}  loss={avg_loss:.4f}"
                  + (f"  val_loss={vl:.4f}" if vl  else "")
                  + (f"  val_WER={wer:.4f}" if wer else "")
                  + f"  lr={lr_now:.2e}  ({time.time()-t0:.2f}s)")

        return model, history

    def evaluate_asr_torch(model, tokenizer, records, n=10):
        device = next(model.parameters()).device
        model.eval()
        results, wers, exact = [], [], 0
        with torch.no_grad():
            for rec in records[:n]:
                mel  = torch.tensor(load_feat(rec)).unsqueeze(0).to(device)
                tgt  = rec["token_ids"][1:-1]
                if not tgt: continue
                log_p = nn.functional.log_softmax(model(mel), dim=-1)
                pred  = ctc_decode(log_p[0].cpu().numpy())
                wer   = edit_distance(tgt, pred) / max(len(tgt), 1)
                wers.append(wer)
                if pred == tgt: exact += 1
                ref_tok  = tokenizer.decode(tgt,  skip_special=False)
                pred_tok = tokenizer.decode(pred, skip_special=False)
                results.append({
                    "id": rec.get("id","?"), "zh": rec.get("zh",""),
                    "ref": ref_tok, "pred": pred_tok, "wer": round(wer,4),
                })
        avg_wer = float(np.mean(wers)) if wers else float("nan")
        print(f"\nASR Evaluation [CTC, {len(results)} samples]")
        print(f"  avg WER    : {avg_wer:.4f}")
        print(f"  exact match: {exact}/{len(results)} ({exact/max(len(results),1):.1%})")
        for r in results:
            mark = "✓" if r["pred"]==r["ref"] else "✗"
            print(f"  {mark} {r['zh']:6s}  ref=[{' '.join(r['ref'])}]")
            print(f"         pred=[{' '.join(r['pred'])}]  WER={r['wer']}")
        return results, {"avg_wer": round(avg_wer,4), "exact": exact/max(len(results),1)}


# ════════════════════════════════════════════════════════════
# NumPy fallback (frame-level CE, head only)
# ════════════════════════════════════════════════════════════
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def positional_encoding(max_len, d):
    pe  = np.zeros((max_len, d), dtype=np.float32)
    pos = np.arange(max_len)[:,None]
    div = np.exp(np.arange(0,d,2)*(-math.log(10000)/d))
    pe[:,0::2] = np.sin(pos*div); pe[:,1::2] = np.cos(pos*div)
    return pe

class NpLinear:
    def __init__(self, i, o, rng):
        self.W = (rng.standard_normal((i,o))*math.sqrt(2/i)).astype(np.float32)
        self.b = np.zeros(o, dtype=np.float32)
    def __call__(self, x): return x@self.W+self.b
    def params(self): return [self.W, self.b]

class NpLayerNorm:
    def __init__(self, d):
        self.g = np.ones(d,dtype=np.float32); self.b = np.zeros(d,dtype=np.float32)
    def __call__(self, x):
        mu,s = x.mean(-1,keepdims=True), x.std(-1,keepdims=True)
        return self.g*(x-mu)/(s+1e-6)+self.b
    def params(self): return [self.g, self.b]

class NpMHA:
    def __init__(self, d, h, rng):
        self.h, self.dk = h, d//h
        self.Wq=NpLinear(d,d,rng); self.Wk=NpLinear(d,d,rng)
        self.Wv=NpLinear(d,d,rng); self.Wo=NpLinear(d,d,rng)
    def __call__(self, x, mask=None):
        B,T,D = x.shape; H,dk = self.h, self.dk
        def sp(x,W): return W(x).reshape(B,-1,H,dk).transpose(0,2,1,3)
        Q,K,V = sp(x,self.Wq),sp(x,self.Wk),sp(x,self.Wv)
        s = Q@K.transpose(0,1,3,2)/math.sqrt(dk)
        if mask is not None: s += mask*-1e9
        return self.Wo((softmax(s,-1)@V).transpose(0,2,1,3).reshape(B,-1,D))
    def params(self):
        return self.Wq.params()+self.Wk.params()+self.Wv.params()+self.Wo.params()

class NpFFN:
    def __init__(self, d, ff, rng):
        self.l1=NpLinear(d,ff,rng); self.l2=NpLinear(ff,d,rng)
    def __call__(self, x): return self.l2(np.maximum(0,self.l1(x)))
    def params(self): return self.l1.params()+self.l2.params()

class NpEncoderLayer:
    def __init__(self, d, h, ff, rng):
        self.attn=NpMHA(d,h,rng); self.ffn=NpFFN(d,ff,rng)
        self.ln1=NpLayerNorm(d); self.ln2=NpLayerNorm(d)
    def __call__(self, x, mask=None):
        x=self.ln1(x+self.attn(x,mask=mask)); return self.ln2(x+self.ffn(x))
    def params(self):
        return self.attn.params()+self.ffn.params()+self.ln1.params()+self.ln2.params()

class NpEncoder:
    def __init__(self, n, d, h, ff, rng):
        self.layers=[NpEncoderLayer(d,h,ff,rng) for _ in range(n)]
    def __call__(self, x, mask=None):
        for l in self.layers: x=l(x,mask)
        return x
    def params(self):
        p=[]
        for l in self.layers: p.extend(l.params())
        return p

class NpASRModel:
    def __init__(self, n_mels=80, vocab_size=42, d=128, h=4, n=2, rng=None):
        rng = rng or np.random.default_rng(42)
        self.proj = NpLinear(n_mels, d, rng)
        self.enc  = NpEncoder(n, d, h, d*4, rng)
        self.head = NpLinear(d, vocab_size, rng)
        self.pe   = positional_encoding(2048, d)
        self.V    = vocab_size
    def forward(self, mel):
        T = mel.shape[2]
        x = self.proj(mel.transpose(0,2,1)) + self.pe[:T]
        return self.head(self.enc(x))
    def n_params(self):
        return sum(p.size for p in self.proj.params()+self.enc.params()+self.head.params())

def align_to_frames(token_ids, T):
    S = max(len(token_ids),1)
    fpk = max(1, T//S)
    labels = np.zeros(T, dtype=int)
    for i,tok in enumerate(token_ids):
        labels[min(i*fpk,T-1):min((i+1)*fpk,T)] = tok
    return labels

def greedy_decode_np(logits, max_len):
    ids = logits.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if int(i)!=prev: out.append(int(i))
        if len(out)>=max_len: break
        prev=int(i)
    return out

def train_asr_numpy(model, tokenizer, train_recs, val_recs, cfg):
    lr      = cfg["lr"]
    smooth  = 0.1
    V, n_ep = model.V, cfg["epochs"]
    m_W = np.zeros_like(model.head.W); v_W = np.zeros_like(model.head.W)
    m_b = np.zeros_like(model.head.b); v_b = np.zeros_like(model.head.b)
    b1,b2,eps_a = 0.9,0.999,1e-8
    step = 0
    history = {"loss":[],"val_loss":[],"val_wer":[]}
    print(f"[numpy fallback] params: {model.n_params():,}  (head-only Adam)\n")

    for epoch in range(n_ep):
        t0=time.time(); np.random.shuffle(train_recs); losses=[]
        lr_e = lr*0.5*(1+math.cos(math.pi*epoch/n_ep))

        for rec in train_recs[:80]:
            mel  = load_feat(rec)
            tgt  = rec["token_ids"]
            tgt_i= tgt[1:-1] if len(tgt)>2 else tgt
            if not tgt_i: continue
            S   = len(tgt_i)
            T_p = S*4
            if mel.shape[1]>T_p:
                chunk = mel.shape[1]/T_p
                mel   = np.stack([mel[:,int(i*chunk):int((i+1)*chunk)].mean(1)
                                  for i in range(T_p)], axis=1)
            T    = mel.shape[1]
            lbl  = align_to_frames(tgt_i, T)
            x    = model.proj(mel.T[None]) + model.pe[:T]
            h    = model.enc(x)[0]
            lg   = h@model.head.W+model.head.b
            pr   = softmax(lg)
            ce   = -np.log(pr[np.arange(T),lbl]+1e-9).mean()
            losses.append(float(ce))
            oh   = np.zeros_like(pr); oh[np.arange(T),lbl]=1.0
            dl   = ((1-smooth)*(pr-oh)+smooth*(pr-1/V))/T
            dW   = h.T@dl; db = dl.sum(0)
            step+=1
            for p,g,m,v in [(model.head.W,dW,m_W,v_W),(model.head.b,db,m_b,v_b)]:
                m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g**2
                mh=m/(1-b1**step); vh=v/(1-b2**step)
                p -= lr_e*mh/(np.sqrt(vh)+eps_a)

        avg=float(np.mean(losses)) if losses else float("nan")
        history["loss"].append(round(avg,4))

        vloss,vwers=[],[]
        for rec in (val_recs or train_recs)[:20]:
            mel=load_feat(rec); tgt=rec["token_ids"]
            tgt_i=tgt[1:-1] if len(tgt)>2 else tgt
            if not tgt_i: continue
            S=len(tgt_i); T_p=S*4
            if mel.shape[1]>T_p:
                chunk=mel.shape[1]/T_p
                mel=np.stack([mel[:,int(i*chunk):int((i+1)*chunk)].mean(1)
                              for i in range(T_p)],axis=1)
            T=mel.shape[1]; lbl=align_to_frames(tgt_i,T)
            x=model.proj(mel.T[None])+model.pe[:T]
            h=model.enc(x)[0]; lg=h@model.head.W+model.head.b; pr=softmax(lg)
            vloss.append(float(-np.log(pr[np.arange(T),lbl]+1e-9).mean()))
            pred=greedy_decode_np(lg,len(tgt_i))
            vwers.append(edit_distance(tgt_i,pred)/max(len(tgt_i),1))

        vl=round(float(np.mean(vloss)),4) if vloss else None
        wer=round(float(np.mean(vwers)),4) if vwers else None
        history["val_loss"].append(vl); history["val_wer"].append(wer)
        print(f"epoch {epoch+1:02d}/{n_ep}  loss={avg:.4f}"
              +(f"  val_loss={vl:.4f}" if vl else "")
              +(f"  val_WER={wer:.4f}" if wer else "")
              +f"  ({time.time()-t0:.2f}s)")
    return model, history

def evaluate_asr_numpy(model, tokenizer, records, n=10):
    results,wers,exact=[],[],0
    for rec in records[:n]:
        mel=load_feat(rec); tgt=rec["token_ids"]
        tgt_i=tgt[1:-1] if len(tgt)>2 else tgt
        if not tgt_i: continue
        S=len(tgt_i); T_p=S*4
        if mel.shape[1]>T_p:
            chunk=mel.shape[1]/T_p
            mel=np.stack([mel[:,int(i*chunk):int((i+1)*chunk)].mean(1)
                          for i in range(T_p)],axis=1)
        T=mel.shape[1]
        x=model.proj(mel.T[None])+model.pe[:T]
        h=model.enc(x)[0]; lg=h@model.head.W+model.head.b
        pred=greedy_decode_np(lg,len(tgt_i))
        wer=edit_distance(tgt_i,pred)/max(len(tgt_i),1)
        wers.append(wer)
        if pred==tgt_i: exact+=1
        results.append({"id":rec.get("id","?"),"zh":rec.get("zh",""),
                        "ref":tokenizer.decode(tgt_i,False),
                        "pred":tokenizer.decode(pred,False),"wer":round(wer,4)})
    avg=float(np.mean(wers)) if wers else float("nan")
    print(f"\nASR Evaluation [numpy, {len(results)} samples]")
    print(f"  avg WER: {avg:.4f}  (numpy head-only; use PyTorch for full backprop)")
    print(f"  exact  : {exact}/{len(results)}")
    for r in results:
        mark="✓" if r["pred"]==r["ref"] else "✗"
        print(f"  {mark} {r['zh']:6s}  ref=[{' '.join(r['ref'])}]  pred=[{' '.join(r['pred'])}]  WER={r['wer']}")
    return results,{"avg_wer":round(avg,4),"exact":exact/max(len(results),1)}


# ════════════════════════════════════════════════════════════
# LM (NumPy, unchanged)
# ════════════════════════════════════════════════════════════
class LMModel:
    def __init__(self, vocab_size=42, d=256, h=4, n=4, max_len=128, rng=None):
        rng = rng or np.random.default_rng(42)
        self.emb  = (np.random.randn(vocab_size,d)*.01).astype(np.float32)
        self.enc  = NpEncoder(n,d,h,d*4,rng)
        self.head = NpLinear(d,vocab_size,rng)
        self.pe   = positional_encoding(max_len,d)
        self.V    = vocab_size
    def __call__(self, ids):
        T=ids.shape[1]; x=self.emb[ids]+self.pe[:T]
        mask=np.triu(np.ones((1,1,T,T),dtype=np.float32),k=1)
        return self.head(self.enc(x,mask))
    def generate(self, prompt, max_new=20, temperature=1.0):
        ids=list(prompt)
        for _ in range(max_new):
            inp=np.array([ids[-64:]]); logit=self(inp)[0,-1]
            if temperature!=1.0: logit=logit/temperature
            probs=softmax(logit); nxt=int(np.random.choice(len(probs),p=probs))
            ids.append(nxt)
            if nxt==3: break
        return ids
    def n_params(self):
        return self.emb.size+sum(p.size for p in self.enc.params()+self.head.params())

def train_lm(model, tokenizer, train_recs, val_recs, cfg):
    lr,V = cfg["lr"],model.V
    history={"loss":[],"ppl":[],"val_loss":[]}
    for epoch in range(cfg["epochs"]):
        t0=time.time(); np.random.shuffle(train_recs); losses=[]
        for i in range(0,min(len(train_recs),80),cfg["batch_size"]):
            batch=train_recs[i:i+cfg["batch_size"]]
            ids=[r["token_ids"] for r in batch]
            ml=max(len(s) for s in ids)
            padded=np.array([s+[tokenizer.pad_id]*(ml-len(s)) for s in ids])
            T=padded.shape[1]; x=model.emb[padded]+model.pe[:T]
            mk=np.triu(np.ones((1,1,T,T),dtype=np.float32),k=1)
            h=model.enc(x,mk); lg=h@model.head.W+model.head.b
            tgt=padded[:,1:]; lf=lg[:,:-1,:].reshape(-1,V); tf=tgt.reshape(-1)
            valid=tf!=tokenizer.pad_id
            pr=softmax(lf); loss=-np.log(pr[np.arange(len(tf)),tf]+1e-9)[valid].mean()
            losses.append(float(loss))
            N=int(valid.sum())
            if N==0: continue
            dl=pr.copy(); dl[np.arange(len(tf)),tf]-=1.0; dl[~valid]=0.0; dl/=N
            hf=h[:,:-1,:].reshape(-1,h.shape[-1])
            model.head.W -= lr*(hf.T@dl); model.head.b -= lr*dl.sum(0)
        avg=float(np.mean(losses)) if losses else float("nan")
        ppl=math.exp(min(avg,20))
        history["loss"].append(round(avg,4)); history["ppl"].append(round(ppl,2))
        vloss=[]
        for r in (val_recs or [])[:20]:
            ids=np.array([r["token_ids"]]); lg_v=model(ids)[:,:-1,:].reshape(-1,V)
            tf_v=np.array(r["token_ids"][1:]); vld=tf_v!=tokenizer.pad_id
            if vld.any():
                p_v=softmax(lg_v)
                vloss.append(float(-np.log(p_v[np.arange(len(tf_v)),tf_v]+1e-9)[vld].mean()))
        vl=round(float(np.mean(vloss)),4) if vloss else None
        history["val_loss"].append(vl)
        print(f"epoch {epoch+1:02d}/{cfg['epochs']}  loss={avg:.4f}  ppl={ppl:.1f}"
              +(f"  val_loss={vl:.4f}" if vl else "")+f"  ({time.time()-t0:.2f}s)")
    return history

def evaluate_lm(model, tokenizer, records, n=5):
    results=[]
    for rec in records[:n]:
        prompt=rec["token_ids"][:2]
        gen=model.generate(prompt,max_new=12,temperature=0.8)
        gen_tok=tokenizer.decode(gen); ref_tok=tokenizer.decode(rec["token_ids"])
        bleu1=sum(1 for t in gen_tok if t in set(ref_tok))/max(len(gen_tok),1)
        wer=edit_distance(ref_tok,gen_tok)/max(len(ref_tok),1)
        results.append({"id":rec.get("id","?"),"ref":ref_tok,"gen":gen_tok,
                        "bleu1":round(bleu1,4),"wer":round(wer,4)})
    avg_bleu=float(np.mean([r["bleu1"] for r in results]))
    avg_wer =float(np.mean([r["wer"]   for r in results]))
    print(f"\nEvaluation ({len(results)} samples)")
    print(f"  avg BLEU-1: {avg_bleu:.4f}  avg WER: {avg_wer:.4f}")
    for r in results:
        print(f"  ref: {' '.join(r['ref'])}")
        print(f"  gen: {' '.join(r['gen'])}  BLEU={r['bleu1']} WER={r['wer']}\n")
    return results,{"avg_bleu1":round(avg_bleu,4),"avg_wer":round(avg_wer,4)}


# ════════════════════════════════════════════════════════════
# TTS stub
# ════════════════════════════════════════════════════════════
class TTSModel:
    def __init__(self, vocab_size=42, n_mels=80, d=256, h=4, n=4, rng=None):
        rng=rng or np.random.default_rng(42)
        self.emb =(np.random.randn(vocab_size,d)*.01).astype(np.float32)
        self.enc =NpEncoder(n,d,h,d*4,rng)
        self.mel =NpLinear(d,n_mels,rng); self.stop=NpLinear(d,1,rng)
        self.pe  =positional_encoding(512,d)
    def __call__(self, ids):
        x=self.emb[ids]+self.pe[:ids.shape[1]]; h=self.enc(x)
        return self.mel(h).transpose(0,2,1), self.stop(h).squeeze(-1)
    def n_params(self):
        return (self.emb.size+
                sum(p.size for p in self.enc.params()+self.mel.params()+self.stop.params()))


# ════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════
def load_split(data_dir, name):
    p=data_dir/f"processed_{name}.json"
    if not p.exists(): return []
    with open(p,encoding="utf-8") as f: recs=json.load(f)
    for r in recs: r["_split"]=name
    return recs

def make_dummy_data(tokenizer):
    samples=[
        (["ΞX"],"是"),(["ΩX"],"否"),(["ΨX","AX"],"存在"),(["ΦX","AX"],"查詢"),
        (["ΨX","IX"],"理解"),(["LX","AX"],"學習"),(["MX","UX"],"記憶"),
        (["SX","IX"],"搜尋"),(["ΨX","ΞX"],"AI"),(["MX","AX"],"人類"),
        (["HX","AX","ΨX"],"你好"),(["ΩX","HX","OX"],"再見"),
        (["FX","IX","ΞX"],"謝謝"),(["MX","AX","NX"],"吃飯"),
        (["SX","UX","IX"],"睡覺"),(["HX","AX","PX"],"快樂"),
        (["NX","AX","WX"],"現在"),(["HX","OX","MX"],"家"),
    ]
    recs=[]
    for rep in range(3):
        for i,(toks,zh) in enumerate(samples):
            recs.append({"id":f"dummy_{rep}_{i:03d}","aix_text":"·".join(toks),
                         "tokens":toks,"zh":zh,"token_ids":tokenizer.encode(toks),"type":"word"})
    for rep in range(5):
        for i,(groups,zh) in enumerate([
            ([["ΨX","ΞX"],["MX","AX"],["ΨX","IX"]],"AI理解人類"),
            ([["ΨX","ΞX"],["DX","AX"],["SX","IX"]],"AI搜尋資料"),
            ([["MX","AX"],["LX","OX"],["ΦX","AX"]],"人類查詢語言"),
            ([["ΨX","ΞX"],["ΩX"],["ΨX","IX"]],"AI不理解"),
        ]):
            toks=[t for g in groups for t in g]
            recs.append({"id":f"dummy_s{rep}_{i}","aix_text":" ".join("·".join(g) for g in groups),
                         "tokens":toks,"zh":zh,"token_ids":tokenizer.encode(toks),"type":"sentence"})
    np.random.seed(42); np.random.shuffle(recs)
    n_tr=int(len(recs)*.8); n_v=int(len(recs)*.1)
    for r in recs[:n_tr]:        r["_split"]="train"
    for r in recs[n_tr:n_tr+n_v]:r["_split"]="val"
    for r in recs[n_tr+n_v:]:    r["_split"]="test"
    return recs


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",    default="lm",choices=["asr","tts","lm"])
    parser.add_argument("--data",    default="./aix_dataset/processed")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--d-model", type=int,   default=128)
    parser.add_argument("--n-heads", type=int,   default=4)
    parser.add_argument("--n-layers",type=int,   default=2)
    args=parser.parse_args()

    data_dir =Path(args.data)
    tok_path =data_dir/"aix_tokenizer.json"
    tokenizer=AIXTokenizer.load(str(tok_path)) if tok_path.exists() else AIXTokenizer()
    print(f"Tokenizer vocab_size={tokenizer.vocab_size}")
    if TORCH: print("PyTorch available — ASR will use CTC Loss + full backprop")
    else:     print("PyTorch not found — ASR will use numpy frame-level CE fallback")

    all_recs  =load_split(data_dir,"train")+load_split(data_dir,"val")+load_split(data_dir,"test")
    if not all_recs:
        print("No processed data — using built-in demo data.")
        print("Run generate_dataset.py + preprocess.py for real training.\n")
        all_recs=make_dummy_data(tokenizer)

    train_recs=[r for r in all_recs if r.get("_split")=="train"]
    val_recs  =[r for r in all_recs if r.get("_split")=="val"]
    test_recs =[r for r in all_recs if r.get("_split")=="test"]
    print(f"Data: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test")

    cfg={"epochs":args.epochs,"batch_size":args.batch,"lr":args.lr}
    rng=np.random.default_rng(42)
    d,h,n=args.d_model,args.n_heads,args.n_layers

    ckpt_dir=data_dir/"checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    if args.task=="asr":
        if TORCH:
            model=TorchASRModel(vocab_size=tokenizer.vocab_size,
                                d_model=d,n_heads=h,n_layers=n)
            model,history=train_asr_torch(model,tokenizer,train_recs,val_recs,cfg)
            eval_recs=test_recs or val_recs[:10]
            results,metrics=evaluate_asr_torch(model,tokenizer,eval_recs)
        else:
            model=NpASRModel(vocab_size=tokenizer.vocab_size,d=d,h=h,n=n,rng=rng)
            model,history=train_asr_numpy(model,tokenizer,train_recs,val_recs,cfg)
            eval_recs=test_recs or val_recs[:10]
            results,metrics=evaluate_asr_numpy(model,tokenizer,eval_recs)
        with open(ckpt_dir/"asr_result.json","w",encoding="utf-8") as f:
            json.dump({"task":"asr","config":cfg,"backend":"torch" if TORCH else "numpy",
                       "training_history":history,"eval_metrics":metrics,
                       "eval_samples":results},f,ensure_ascii=False,indent=2)
        print(f"\nResult → {ckpt_dir/'asr_result.json'}")

    elif args.task=="tts":
        model=TTSModel(vocab_size=tokenizer.vocab_size,d=d,h=h,n=n,rng=rng)
        print(f"TTS model params: {model.n_params():,}")
        dummy=np.array([[2,6,12,20,3]])
        mel,stop=model(dummy)
        print(f"Forward: ids{dummy.shape} → mel{mel.shape}, stop{stop.shape}")

    elif args.task=="lm":
        model=LMModel(vocab_size=tokenizer.vocab_size,d=d,h=h,n=n,rng=rng)
        print(f"LM model params: {model.n_params():,}\n")
        history=train_lm(model,tokenizer,train_recs,val_recs,cfg)
        eval_recs=test_recs or val_recs[:5]
        results,metrics=evaluate_lm(model,tokenizer,eval_recs)
        with open(ckpt_dir/"lm_result.json","w",encoding="utf-8") as f:
            json.dump({"task":"lm","config":cfg,"training_history":history,
                       "eval_metrics":metrics,"eval_samples":results},f,ensure_ascii=False,indent=2)
        print(f"\nResult → {ckpt_dir/'lm_result.json'}")

if __name__=="__main__":
    main()