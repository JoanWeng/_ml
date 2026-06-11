"""
AIX Dataset Generator
Usage: python generate_dataset.py [--augment N] [--output DIR]
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile

SR = 44100

def env(n, atk, rel):
    e = np.ones(n)
    a, r = max(1, int(SR*atk)), max(1, int(SR*rel))
    if a < n: e[:a] = np.linspace(0, 1, a)
    if r < n: e[-r:] = np.linspace(1, 0, r)
    return e

def vowel(freq, dur=0.38):
    n, t = int(SR*dur), np.linspace(0, dur, int(SR*dur))
    w = sum(a*np.sin(2*np.pi*freq*h*t) for h,a in zip([1,2.1,3.4,4.7],[.50,.25,.14,.07]))
    return w * env(n, .05, .07)

def stop(freq, asp=False, dur=0.15):
    from scipy.signal import butter, lfilter
    bn = int(SR*.04)
    b = np.random.randn(bn) * np.exp(-np.linspace(0,6,bn))
    bf, af = butter(2, min(freq*6/(SR/2), .99), btype='low')
    b = lfilter(bf, af, b) / (np.abs(b).max()+1e-9) * .7
    w = np.concatenate([b, np.random.randn(int(SR*.10))*np.exp(-np.linspace(0,4,int(SR*.10)))*.3]) if asp else b
    tn = int(SR*dur)
    return np.pad(w, (0, max(0, tn-len(w))))[:tn]

def fric(freq, voiced=False, dur=0.28):
    from scipy.signal import butter, lfilter
    n = int(SR*dur)
    noise = np.random.randn(n)
    bw = freq*.4
    b, a = butter(3, [max((freq-bw)/(SR/2),.01), min((freq+bw)/(SR/2),.99)], btype='band')
    w = lfilter(b, a, noise) * .5
    if voiced:
        w += .15*np.sign(np.sin(2*np.pi*120*np.linspace(0,dur,n)))
    return w * env(n, .03, .06)

def nasal(freq, dur=0.30):
    from scipy.signal import butter, lfilter, iirnotch
    n, t = int(SR*dur), np.linspace(0, dur, int(SR*dur))
    w = np.sign(np.sin(2*np.pi*freq*t)) * .4
    b, a = butter(2, 750/(SR/2), btype='low'); w = lfilter(b,a,w)
    b2, a2 = iirnotch(1100/(SR/2), Q=3); w = lfilter(b2,a2,w)
    return w * env(n, .04, .06)

def lat(freq, dur=0.28):
    n, t = int(SR*dur), np.linspace(0, dur, int(SR*dur))
    return (.45*np.sin(2*np.pi*freq*t) + .20*np.sin(2*np.pi*freq*1.7*t)) * env(n,.04,.06)

def trill(freq, dur=0.34):
    n, w = int(SR*dur), np.zeros(int(SR*dur))
    per, bn = int(SR*.065), int(SR*.04)
    tb = np.linspace(0,1,bn)
    burst = np.sin(2*np.pi*freq*tb*.04)*np.exp(-tb*4)*.5
    pos = 0
    while pos+bn < n: w[pos:pos+bn] += burst; pos += per
    return w

def approx(freq, dur=0.26):
    n = int(SR*dur)
    fc = np.linspace(freq*.65, freq, n)
    return .38*np.sin(2*np.pi*np.cumsum(fc)/SR) * env(n,.04,.06)

def ai_pulse():
    w = np.zeros(int(SR*.18))
    for i, m in enumerate([1., 1.6, 2.3]):
        s, n = int(SR*i*.045), int(SR*.025)
        t = np.linspace(0,1,n)
        w[s:s+n] += .5*np.sign(np.sin(2*np.pi*1200*m*t/SR*n))*np.exp(-t*5)
    return w

def ai_harmonic(freq=440, dur=0.52):
    n, t = int(SR*dur), np.linspace(0, dur, int(SR*dur))
    return sum(.22/(h*1.1)*np.sin(2*np.pi*freq*h*t) for h in range(1,6)) * env(n,.10,.15)

def ai_query(freq=880, dur=0.42):
    n = int(SR*dur)
    fc = np.linspace(freq*.55, freq*1.9, n)
    return .40*np.sin(2*np.pi*np.cumsum(fc)/SR) * env(n,.05,.08)

def ai_shift(freq=660):
    w = np.zeros(int(SR*.28))
    for i, f in enumerate([freq, freq*1.55, freq*.75]):
        s, n = int(SR*i*.08), int(SR*.065)
        t = np.linspace(0, n/SR, n)
        w[s:s+n] += .35*np.sin(2*np.pi*f*t) * env(n,.02,.025)
    return w

def ai_end(freq=180, dur=0.72):
    n = int(SR*dur)
    fc = np.linspace(freq*2.8, freq*.35, n)
    return .42*np.sin(2*np.pi*np.cumsum(fc)/SR) * env(n,.07,.22)

SYNTH = {
    "AX": lambda: vowel(800,.40),  "EX": lambda: vowel(530,.40),
    "IX": lambda: vowel(2300,.36), "OX": lambda: vowel(500,.40),
    "UX": lambda: vowel(300,.36),  "YX": lambda: vowel(235,.40),
    "BX": lambda: stop(120,False,.15), "PX": lambda: stop(100,True,.18),
    "DX": lambda: stop(150,False,.13), "TX": lambda: stop(130,True,.17),
    "GX": lambda: stop(100,False,.15), "KX": lambda: stop(90,True,.18),
    "FX": lambda: fric(6500,False,.28), "VX": lambda: fric(5800,True,.28),
    "SX": lambda: fric(7800,False,.26), "ZX": lambda: fric(7000,True,.26),
    "HX": lambda: fric(3500,False,.24), "QX": lambda: fric(4200,False,.28),
    "MX": lambda: nasal(200,.32), "NX": lambda: nasal(250,.30),
    "LX": lambda: lat(350,.28),   "RX": lambda: trill(220,.34),
    "WX": lambda: approx(400,.26),"JX": lambda: approx(600,.24),
    "ΞX": ai_pulse, "ΛX": lambda: np.zeros(int(SR*.15)),
    "ΨX": ai_harmonic, "ΦX": ai_query, "ΔX": ai_shift, "ΩX": ai_end,
}

def gen_ph(ph):
    fn = SYNTH.get(ph)
    return fn() if fn else np.zeros(int(SR*.1))

def concat(tokens, gap=0.04):
    g = np.zeros(int(SR*gap))
    parts = []
    for i, t in enumerate(tokens):
        parts.append(gen_ph(t))
        if i < len(tokens)-1: parts.append(g)
    return np.concatenate(parts)

def normalize(w, peak=0.85):
    mx = np.abs(w).max()
    return w*(peak/mx) if mx > 1e-9 else w

def save_wav(path, w):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, SR, (np.clip(w,-1,1)*32767).astype(np.int16))

def augment(w, seed):
    rng = np.random.default_rng(seed)
    w = w.copy().astype(np.float32)
    if seed % 3 == 1:
        w += rng.standard_normal(len(w)) * rng.uniform(0.003, 0.015)
    if seed % 3 == 2:
        speed = rng.uniform(0.88, 1.12)
        idx = np.round(np.arange(0, len(w), speed)).astype(int)
        w = w[idx[idx < len(w)]]
    w *= rng.uniform(0.75, 1.0)
    w -= w.mean()
    return w

VOCAB = [
    ("ΞX",       ["ΞX"],           "是",     "basic"),
    ("ΩX",       ["ΩX"],           "否",     "basic"),
    ("ΨX·AX",    ["ΨX","AX"],      "存在",   "basic"),
    ("ΦX·AX",    ["ΦX","AX"],      "查詢",   "basic"),
    ("ΔX·AX",    ["ΔX","AX"],      "改變",   "basic"),
    ("ΨX·IX",    ["ΨX","IX"],      "理解",   "cognitive"),
    ("LX·AX",    ["LX","AX"],      "學習",   "cognitive"),
    ("MX·UX",    ["MX","UX"],      "記憶",   "cognitive"),
    ("SX·IX",    ["SX","IX"],      "搜尋",   "cognitive"),
    ("GX·AX",    ["GX","AX"],      "生成",   "cognitive"),
    ("TX·IX",    ["TX","IX"],      "傳輸",   "cognitive"),
    ("NX·OX",    ["NX","OX"],      "連結",   "cognitive"),
    ("FX·AX",    ["FX","AX"],      "分析",   "cognitive"),
    ("ΨX·ΞX",    ["ΨX","ΞX"],      "AI",     "entity"),
    ("MX·AX",    ["MX","AX"],      "人類",   "entity"),
    ("DX·AX",    ["DX","AX"],      "資料",   "entity"),
    ("LX·OX",    ["LX","OX"],      "語言",   "entity"),
    ("NX·UX",    ["NX","UX"],      "網路",   "entity"),
    ("WX·AX",    ["WX","AX"],      "世界",   "entity"),
    ("HX·AX·ΨX", ["HX","AX","ΨX"],"你好",   "greet"),
    ("ΩX·HX·OX", ["ΩX","HX","OX"],"再見",   "greet"),
    ("ΨX·WX·AX", ["ΨX","WX","AX"],"早安",   "greet"),
    ("NX·AX·ΨX", ["NX","AX","ΨX"],"晚安",   "greet"),
    ("FX·IX·ΞX", ["FX","IX","ΞX"],"謝謝",   "greet"),
    ("ΩX·MX·AX", ["ΩX","MX","AX"],"對不起", "greet"),
    ("ΞX·ΛX·IX", ["ΞX","ΛX","IX"],"沒關係", "greet"),
    ("ΦX·IX·AX", ["ΦX","IX","AX"],"請問",   "greet"),
    ("MX·AX·NX", ["MX","AX","NX"],"吃飯",   "food"),
    ("WX·AX·SX", ["WX","AX","SX"],"喝水",   "food"),
    ("HX·IX·MX", ["HX","IX","MX"],"好吃",   "food"),
    ("ΦX·OX·DX", ["ΦX","OX","DX"],"餓了",   "food"),
    ("SX·AX·TX", ["SX","AX","TX"],"飽了",   "food"),
    ("GX·IX·AX", ["GX","IX","AX"],"咖啡",   "food"),
    ("SX·UX·IX", ["SX","UX","IX"],"睡覺",   "daily"),
    ("KX·AX·IX", ["KX","AX","IX"],"起床",   "daily"),
    ("WX·OX·KX", ["WX","OX","KX"],"工作",   "daily"),
    ("HX·UX·IX", ["HX","UX","IX"],"休息",   "daily"),
    ("ZX·AX·UX", ["ZX","AX","UX"],"走路",   "daily"),
    ("TX·AX·KX", ["TX","AX","KX"],"搭車",   "daily"),
    ("MX·AX·IX", ["MX","AX","IX"],"買東西", "daily"),
    ("HX·AX·PX", ["HX","AX","PX"],"快樂",   "emotion"),
    ("SX·AX·DX", ["SX","AX","DX"],"難過",   "emotion"),
    ("LX·AX·VX", ["LX","AX","VX"],"喜歡",   "emotion"),
    ("TX·AX·YX", ["TX","AX","YX"],"疲倦",   "emotion"),
    ("HX·OX·PX", ["HX","OX","PX"],"希望",   "emotion"),
    ("NX·AX·WX", ["NX","AX","WX"],"現在",   "time"),
    ("TX·UX·DX", ["TX","UX","DX"],"今天",   "time"),
    ("JX·IX·SX", ["JX","IX","SX"],"昨天",   "time"),
    ("MX·OX·RX", ["MX","OX","RX"],"明天",   "time"),
    ("HX·OX·MX", ["HX","OX","MX"],"家",     "place"),
    ("SX·KX·UX", ["SX","KX","UX"],"學校",   "place"),
    ("SX·TX·OX", ["SX","TX","OX"],"商店",   "place"),
    ("TX·IX·WX", ["TX","IX","WX"],"城市",   "place"),
    ("ΛX·AX",    ["ΛX","AX"],     "零",     "number"),
    ("ΞX·IX",    ["ΞX","IX"],     "一",     "number"),
    ("DX·UX",    ["DX","UX"],     "二",     "number"),
    ("TX·RX",    ["TX","RX"],     "三",     "number"),
    ("FX·OX",    ["FX","OX"],     "四",     "number"),
    ("FX·IX",    ["FX","IX"],     "五",     "number"),
]

SENTENCES = [
    {"aix":"ΨX·ΞX MX·AX ΨX·IX",
     "token_groups":[["ΨX","ΞX"],["MX","AX"],["ΨX","IX"]],"zh":"AI理解人類","grammar":"declarative"},
    {"aix":"ΨX·ΞX DX·AX SX·IX",
     "token_groups":[["ΨX","ΞX"],["DX","AX"],["SX","IX"]],"zh":"AI搜尋資料","grammar":"declarative"},
    {"aix":"MX·AX LX·OX ΦX·AX",
     "token_groups":[["MX","AX"],["LX","OX"],["ΦX","AX"]],"zh":"人類查詢語言","grammar":"question"},
    {"aix":"ΨX·ΞX ΩX ΨX·IX",
     "token_groups":[["ΨX","ΞX"],["ΩX"],["ΨX","IX"]],"zh":"AI不理解","grammar":"negation"},
    {"aix":"HX·AX·ΨX",
     "token_groups":[["HX","AX","ΨX"]],"zh":"你好","grammar":"greeting"},
    {"aix":"ΨX·WX·AX",
     "token_groups":[["ΨX","WX","AX"]],"zh":"早安","grammar":"greeting"},
    {"aix":"MX·AX·NX HX·IX·MX",
     "token_groups":[["MX","AX","NX"],["HX","IX","MX"]],"zh":"吃飯好吃","grammar":"declarative"},
    {"aix":"ΨX·ΞX NX·AX·WX SX·UX·IX ΔX·AX",
     "token_groups":[["ΨX","ΞX"],["NX","AX","WX"],["SX","UX","IX"],["ΔX","AX"]],"zh":"AI現在正在睡覺","grammar":"progressive"},
    {"aix":"FX·IX·ΞX",
     "token_groups":[["FX","IX","ΞX"]],"zh":"謝謝","grammar":"social"},
    {"aix":"MX·AX TX·AX·KX HX·OX·MX",
     "token_groups":[["MX","AX"],["TX","AX","KX"],["HX","OX","MX"]],"zh":"人類搭車回家","grammar":"declarative"},
]

def gen_words(out_dir, n_aug):
    wav_dir = out_dir / "wav"
    records = []
    for idx, (aix, tokens, zh, cat) in enumerate(VOCAB):
        base = normalize(concat(tokens, gap=0.035))
        for aug_i in range(n_aug + 1):
            w = base if aug_i == 0 else normalize(augment(base, aug_i))
            uid = f"word_{idx:04d}_aug{aug_i}"
            path = wav_dir / "words" / f"{uid}.wav"
            save_wav(str(path), w)
            records.append({
                "id": uid, "type": "word", "aix_text": aix,
                "tokens": tokens, "zh": zh, "category": cat,
                "augment": aug_i, "duration_s": round(len(w)/SR, 3),
                "wav_path": str(path), "sample_rate": SR,
            })
    return records

def gen_sentences(out_dir, n_aug):
    wav_dir = out_dir / "wav"
    records = []
    for idx, sent in enumerate(SENTENCES):
        word_waves = [concat(g, gap=0.03) for g in sent["token_groups"]]
        gap = np.zeros(int(SR*.10))
        base = normalize(np.concatenate([w for pair in zip(word_waves, [gap]*len(word_waves)) for w in pair][:-1]))
        all_tokens = [t for g in sent["token_groups"] for t in g]
        for aug_i in range(n_aug + 1):
            w = base if aug_i == 0 else normalize(augment(base, aug_i))
            uid = f"sent_{idx:04d}_aug{aug_i}"
            path = wav_dir / "sentences" / f"{uid}.wav"
            save_wav(str(path), w)
            records.append({
                "id": uid, "type": "sentence", "aix_text": sent["aix"],
                "tokens": all_tokens, "token_groups": sent["token_groups"],
                "zh": sent["zh"], "grammar": sent["grammar"],
                "augment": aug_i, "duration_s": round(len(w)/SR, 3),
                "wav_path": str(path), "sample_rate": SR,
            })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=int, default=3)
    parser.add_argument("--output",  default="./aix_dataset")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_records = gen_words(out, args.augment) + gen_sentences(out, args.augment)

    np.random.seed(42)
    idx = np.random.permutation(len(all_records))
    n_train, n_val = int(len(idx)*.80), int(len(idx)*.10)
    splits = {
        "train": [all_records[i] for i in idx[:n_train]],
        "val":   [all_records[i] for i in idx[n_train:n_train+n_val]],
        "test":  [all_records[i] for i in idx[n_train+n_val:]],
    }

    meta_dir = out / "metadata"
    meta_dir.mkdir(exist_ok=True)
    for split, records in splits.items():
        with open(meta_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    total_dur = sum(r["duration_s"] for r in all_records)
    print(f"Generated {len(all_records)} samples ({total_dur:.1f}s) → {out}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Next: python 02_preprocess.py --dataset {out}")

if __name__ == "__main__":
    main()