"""
AIX Preprocessing Pipeline
Usage: python preprocess.py [--dataset DIR] [--feature mel|mfcc]
Output: processed/{train,val,test}.json  +  aix_tokenizer.json
No external ML dependencies required.
"""

import os, json, csv, argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile


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

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()},
                "vocab_size": self.vocab_size,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tok


class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100, n_mels=80, n_mfcc=40,
                 hop_length=512, n_fft=2048, fmin=0, fmax=8000):
        self.sr, self.n_mels, self.n_mfcc = sample_rate, n_mels, n_mfcc
        self.hop, self.n_fft = hop_length, n_fft
        self.fmin, self.fmax = fmin, fmax

    def _load(self, path):
        sr, data = wavfile.read(path)
        data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
        return data.mean(axis=1) if data.ndim > 1 else data

    def _stft(self, wav):
        win = np.hanning(self.n_fft)
        return np.array([
            np.abs(np.fft.rfft(wav[i:i+self.n_fft] * win, n=self.n_fft))
            for i in range(0, len(wav) - self.n_fft, self.hop)
        ]).T

    def _mel_fb(self):
        hz2mel = lambda h: 2595 * np.log10(1 + h / 700)
        mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        pts = np.floor(
            (self.n_fft + 1) *
            mel2hz(np.linspace(hz2mel(self.fmin), hz2mel(self.fmax), self.n_mels + 2)) /
            self.sr
        ).astype(int)
        fb = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            for k in range(pts[m-1], pts[m]):
                fb[m-1, k] = (k - pts[m-1]) / (pts[m] - pts[m-1])
            for k in range(pts[m], pts[m+1]):
                fb[m-1, k] = (pts[m+1] - k) / (pts[m+1] - pts[m])
        return fb

    def mel(self, path):
        return np.log(self._mel_fb() @ self._stft(self._load(path)) + 1e-9).astype(np.float32)

    def mfcc(self, path):
        M = self.mel(path)
        n_mel = M.shape[0]
        return np.array([
            np.sum(M * np.cos(np.pi * i / n_mel * (np.arange(n_mel) + .5))[:, None], 0)
            for i in range(self.n_mfcc)
        ], dtype=np.float32)

    def extract(self, path, feature_type="mel"):
        wav = self._load(path)
        feat = self.mel(path) if feature_type == "mel" else self.mfcc(path)
        return feat, round(len(wav) / self.sr, 3)


def process_split(records, tokenizer, extractor, feature_type, out_dir, name):
    feat_dir = out_dir / "features" / name
    feat_dir.mkdir(parents=True, exist_ok=True)
    processed, failed = [], []

    for rec in records:
        try:
            feat, dur = extractor.extract(rec["wav_path"], feature_type)
            feat_path = feat_dir / f"{rec['id']}.npy"
            np.save(str(feat_path), feat)
            token_ids = tokenizer.encode(rec["tokens"])
            processed.append({
                **rec,
                "feat_path":    str(feat_path),
                "feat_shape":   list(feat.shape),
                "feature_type": feature_type,
                "token_ids":    token_ids,
                "seq_len":      len(token_ids),
            })
        except Exception as e:
            failed.append({"id": rec["id"], "error": str(e)})

    # JSON — 完整欄位，供 03_train.py 直接讀取
    with open(out_dir / f"processed_{name}.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    # CSV — 輕量版，方便用 Excel 或 pandas 查看
    if processed:
        csv_path = out_dir / f"{name}.csv"
        fields = ["id", "type", "aix_text", "zh",
                  "category", "augment", "duration_s", "seq_len", "feat_path"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(processed)

    if failed:
        with open(out_dir / f"failed_{name}.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    status = f"{len(processed)} ok"
    if failed: status += f", {len(failed)} failed"
    print(f"  {name}: {status}")
    return processed


def print_summary(splits, feature_type):
    all_recs = [r for rs in splits.values() for r in rs]
    if not all_recs:
        return
    shapes   = [r["feat_shape"] for r in all_recs]
    total_s  = sum(r["duration_s"] for r in all_recs)
    avg_frames = int(np.mean([s[1] for s in shapes]))
    print(f"\nSummary")
    print(f"  total samples : {len(all_recs)}")
    print(f"  total duration: {total_s:.1f}s")
    print(f"  feature type  : {feature_type}  shape[0]={shapes[0][0]}")
    print(f"  avg frames    : {avg_frames}")
    for name, recs in splits.items():
        print(f"  {name:5s}         : {len(recs)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./aix_dataset")
    parser.add_argument("--feature", default="mel", choices=["mel", "mfcc"])
    parser.add_argument("--n-mels",  type=int, default=80)
    parser.add_argument("--n-mfcc",  type=int, default=40)
    args = parser.parse_args()

    ds_dir  = Path(args.dataset)
    out_dir = ds_dir / "processed"
    out_dir.mkdir(exist_ok=True)

    tokenizer = AIXTokenizer()
    tokenizer.save(str(out_dir / "aix_tokenizer.json"))
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    extractor = AudioFeatureExtractor(n_mels=args.n_mels, n_mfcc=args.n_mfcc)

    splits = {}
    for name in ["train", "val", "test"]:
        path = ds_dir / "metadata" / f"{name}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        splits[name] = process_split(records, tokenizer, extractor, args.feature, out_dir, name)

    print_summary(splits, args.feature)
    print(f"\nNext: python 03_train.py --data {out_dir}")


if __name__ == "__main__":
    main()