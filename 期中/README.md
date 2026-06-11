> 本專案由Claude製作
  [對話連結]()

# AIX Training Project

AIX 語言模型訓練專案——從合成語音資料到模型訓練的完整流水線。

---

## 專案結構

```
aix_training/
├── 01_generate_dataset.py   ← 語料生成（WAV + JSON 標籤）
├── 02_preprocess.py         ← 預處理（Mel / MFCC + HF Dataset）
├── 03_train.py              ← 模型訓練（ASR / TTS / LM）
└── README.md

aix_dataset/                 ← 由 01 生成
├── wav/
│   ├── words/               ← 詞彙 WAV 音檔
│   └── sentences/           ← 句子 WAV 音檔
├── metadata/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── processed/               ← 由 02 生成
    ├── features/            ← .npy Mel Spectrogram
    ├── aix_tokenizer.json   ← AIX 自訂詞表
    ├── processed_train.json
    ├── processed_val.json
    ├── processed_test.json
    └── hf_dataset/          ← HuggingFace Dataset（Arrow 格式）
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install numpy scipy datasets soundfile

# 完整 ML 訓練（選用）
pip install torch transformers librosa speechbrain
```

### 2. 生成語料

```bash
# 基本：每個樣本生成 3 個增強變體
python generate_dataset.py

# 自訂增強倍率與輸出路徑
python generate_dataset.py --augment 5 --output ./my_dataset
```

輸出：
- `aix_dataset/wav/` — 原始與增強 WAV 音檔
- `aix_dataset/metadata/{train,val,test}.json` — 配對標籤

### 3. 預處理

```bash
# 提取 Mel Spectrogram 特徵，建立 HuggingFace Dataset
python preprocess.py --dataset ./aix_dataset

# 使用 MFCC 特徵
python preprocess.py --dataset ./aix_dataset --feature mfcc
```

輸出：
- `aix_dataset/processed/features/` — `.npy` 特徵陣列
- `aix_dataset/processed/aix_tokenizer.json` — 詞表（44 tokens）
- `aix_dataset/processed/hf_dataset/` — HuggingFace Dataset

### 4. 訓練

```bash
# 語言模型（最容易上手）
python train.py --task lm --data ./aix_dataset/processed

# 語音辨識
python train.py --task asr --data ./aix_dataset/processed

# 語音合成
python train.py --task tts --data ./aix_dataset/processed

# 調整超參數
python train.py --task lm --epochs 20 --batch 8 --lr 5e-4 --d-model 256
```

---

## 各任務說明

### ASR — 語音辨識（WAV → AIX token）

**架構：** Mel Spectrogram → Transformer Encoder → CTC Head

**損失函數：** CTC Loss（Connectionist Temporal Classification）

**評估指標：** WER（Word Error Rate，越低越好）

**進階做法：**
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 載入預訓練 Whisper，用 AIX 語料 fine-tune
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 替換解碼頭為 AIX 詞表大小
model.config.vocab_size = 44
model.proj_out = torch.nn.Linear(model.config.d_model, 44)
```

---

### TTS — 語音合成（AIX token → WAV）

**架構：** Token Embedding → Transformer Encoder → Mel Decoder → HiFi-GAN Vocoder

**損失函數：** MSE（Mel 重建） + BCE（Stop token）

**評估指標：** MOS（Mean Opinion Score，需人工評測）、Mel Cepstral Distortion

**進階做法：**
```python
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan

# SpeechT5 fine-tune
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

或使用 VITS（End-to-End，音質更好）：
```bash
pip install coqui-tts
tts --model_name tts_models/en/ljspeech/vits
```

---

### LM — 語言模型（AIX 補全 / 中↔AIX 翻譯）

**架構：** Token Embedding → Transformer Decoder（GPT 式自回歸）

**損失函數：** Cross-Entropy（Next Token Prediction）

**評估指標：** Perplexity（越低越好）、BLEU（翻譯任務）

**進階做法（GPT-2 fine-tune）：**
```python
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

# 從頭訓練小型 GPT-2
config = GPT2Config(
    vocab_size=44,          # AIX 詞表大小
    n_positions=256,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=2,
    eos_token_id=3,
)
model = GPT2LMHeadModel(config)

# 或 fine-tune 現有 GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(44)

# 訓練
training_args = TrainingArguments(
    output_dir="./aix_lm_checkpoint",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    logging_steps=50,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
)
trainer.train()
```

---

## AIX 分詞器詞表

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

## HuggingFace Dataset 使用方式

```python
from datasets import load_from_disk

ds = load_from_disk("./aix_dataset/processed/hf_dataset")
print(ds)
# DatasetDict({
#     train: Dataset({features: [...], num_rows: N}),
#     val:   Dataset({features: [...], num_rows: N}),
#     test:  Dataset({features: [...], num_rows: N})
# })

sample = ds["train"][0]
print(sample["aix_text"])   # e.g. "ΨX·AX"
print(sample["zh"])          # e.g. "存在"
print(sample["token_ids"])   # e.g. [2, 30, 6, 3]
```

---

## 生產級建議

若要訓練真正可用的 AIX 模型：

1. **擴大語料**：將 `VOCAB` 和 `SENTENCES` 擴展至 10,000+ 筆，覆蓋更多語境
2. **音訊多樣性**：加入不同合成器參數、不同說話者風格
3. **使用真實語音**：招募受試者錄製 AIX 音素，補充合成資料
4. **使用 GPU 訓練**：`torch.nn.Transformer` + `torch.optim.AdamW` + `torch.cuda`
5. **追蹤實驗**：`pip install wandb mlflow` 記錄訓練曲線
6. **發布模型**：`model.push_to_hub("your-name/aix-lm")` 上傳 HuggingFace Hub

---

*AIX Training Project v1.0 · 神經節點 × 邏輯閘 × 波形訊號*