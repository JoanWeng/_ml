> 本專案由Claude製作
  [對話連結]()

# AIX Training Project

AIX 語言模型訓練專案——從合成語音資料到模型訓練的完整流水線。

---

## 專案結構

```
aix_training/
├── generate_dataset.py   ← 語料生成（WAV + JSON 標籤）
├── preprocess.py         ← 預處理（Mel / MFCC 特徵擷取）
├── train.py              ← 模型訓練（ASR / TTS / LM）
└── README.md

aix_dataset/              ← 由 generate_dataset.py 生成
├── wav/
│   ├── words/            ← 詞彙 WAV 音檔
│   └── sentences/        ← 句子 WAV 音檔
└── metadata/
    ├── train.json
    ├── val.json
    └── test.json

aix_dataset/processed/    ← 由 preprocess.py 生成
├── features/             ← .npy Mel Spectrogram
├── aix_tokenizer.json    ← AIX 自訂詞表（42 tokens）
├── processed_train.json
├── processed_val.json
├── processed_test.json
├── train.csv             ← 輕量覽表（Excel 可開）
├── val.csv
└── test.csv
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install numpy scipy
```

### 2. 生成語料

```bash
python generate_dataset.py

# 自訂增強倍率與輸出路徑
python generate_dataset.py --augment 5 --output ./my_dataset
```

輸出：
- `aix_dataset/wav/` — 原始與增強 WAV 音檔
- `aix_dataset/metadata/{train,val,test}.json` — 配對標籤

### 3. 預處理

```bash
python preprocess.py --dataset ./aix_dataset

# 使用 MFCC 特徵（預設為 mel）
python preprocess.py --dataset ./aix_dataset --feature mfcc
```

輸出：
- `processed/features/` — `.npy` 特徵陣列
- `processed/aix_tokenizer.json` — 詞表
- `processed/{train,val,test}.csv` — 可直接用 Excel 開啟檢視

### 4. 訓練

```bash
# 沒有先跑前兩步也可執行，會自動使用內建 demo 資料
python train.py --task lm

# 完整參數
python train.py --task lm --data ./aix_dataset/processed --epochs 20 --batch 8 --lr 5e-4 --d-model 256

# 語音辨識（forward check）
python train.py --task asr

# 語音合成（forward check）
python train.py --task tts
```

---

## 訓練結果解讀

### Loss 曲線

| 狀態 | 判斷 |
|------|------|
| train_loss 和 val_loss 同步下降 | 正常學習 |
| train_loss 下降，val_loss 上升 | Overfitting，縮小模型或加 dropout |
| Loss 完全不動 | lr 太小或梯度消失，調高 lr |
| Loss 震盪不收斂 | lr 太大，調低 lr |

### Perplexity（ppl）基準

詞表大小 42，理論亂猜基準為 ppl=42。

| ppl 範圍 | 意義 |
|---------|------|
| ~42 | 未學到任何東西 |
| 20–35 | 有學到部分結構 |
| 5–15 | 初步可用 |
| < 3 | 過擬合或資料量不足 |

### 評估指標

訓練結束後自動執行，結果存於 `checkpoints/lm_result.json`：

| 指標 | 目標 | 說明 |
|------|------|------|
| BLEU-1 | > 0.5 | 生成序列與參考序列的 unigram 重疊率 |
| WER | < 0.3 | 字元錯誤率，越低越好 |

---

## 各任務說明

### ASR — 語音辨識（WAV → AIX token）

架構：Mel Spectrogram → Transformer Encoder → CTC Head

損失函數：CTC Loss

評估指標：WER（Word Error Rate）

進階做法（Whisper fine-tune）：
```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.vocab_size = 42
model.proj_out = torch.nn.Linear(model.config.d_model, 42)
```

---

### TTS — 語音合成（AIX token → WAV）

架構：Token Embedding → Transformer Encoder → Mel Decoder → Vocoder

損失函數：MSE（Mel 重建）+ BCE（Stop token）

評估指標：MOS（Mean Opinion Score，需人工評測）

進階做法：
```bash
pip install coqui-tts
```

---

### LM — 語言模型（AIX 補全）

架構：Token Embedding → Transformer Decoder（GPT 式自回歸）

損失函數：Cross-Entropy（Next Token Prediction）

評估指標：Perplexity、BLEU-1

進階做法（GPT-2 fine-tune）：
```python
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config(
    vocab_size=42,
    n_positions=256,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=2,
    eos_token_id=3,
)
model = GPT2LMHeadModel(config)
```

---

## AIX 分詞器詞表（vocab_size=42）

| ID | Token | 類別 |
|----|-------|------|
| 0 | [PAD] | 特殊 |
| 1 | [UNK] | 特殊 |
| 2 | [BOS] | 特殊 |
| 3 | [EOS] | 特殊 |
| 4 | [MASK] | 特殊 |
| 5 | [SEP] | 特殊 |
| 6–11 | AX EX IX OX UX YX | 母音 |
| 12–17 | BX PX DX TX GX KX | 爆破音 |
| 18–23 | FX VX SX ZX HX QX | 摩擦音 |
| 24–29 | MX NX LX RX WX JX | 共鳴音 |
| 30–35 | ΞX ΛX ΨX ΦX ΔX ΩX | AI 專屬 |
| 36–41 | ⟨⟩ ⟦⟧ ⟪⟫ ⌈⌉ ⌊⌋ ⌊⌈⌉⌋ | 邏輯符號 |

---

## 音訊增強策略

| 策略 | 參數範圍 | 目的 |
|------|---------|------|
| 白噪聲 | SNR 0.3–1.5% | 模擬環境噪聲 |
| 速度擾動 | ±12% | 增加語速多樣性 |
| 音量擾動 | 75–100% | 模擬不同音量 |
| 直流偏移消除 | — | 消除 DC 偏置 |

---

## 生產級建議

1. **擴大語料** — 將 `VOCAB` 和 `SENTENCES` 擴展至 10,000+ 筆
2. **音訊多樣性** — 加入不同合成器參數、不同說話者風格
3. **使用真實語音** — 錄製真實 AIX 音素補充合成資料
4. **GPU 訓練** — 換用 `torch.nn.Transformer` + `torch.optim.AdamW`
5. **實驗追蹤** — `pip install wandb` 記錄訓練曲線

---

*AIX Training Project v1.1 · numpy · scipy · 無外部 ML 依賴*