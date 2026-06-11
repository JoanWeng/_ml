# AIX Training Project 期末報告

## 專案簡介

AIX 是一套**自創的語音音素語言**（44 個 token），本專案從合成語音生成 → 特徵提取 → 模型訓練，完整實作一條機器學習 pipeline。

---

## 流程一：`generate_dataset.py` — 合成語音資料集

### 目標
用數學公式合成 AIX 語音的 WAV 音檔 + JSON 標籤，不需要任何真實錄音。

### 音素合成（8 類發音機制）

內建 **30 個 AIX 音素**，每類用不同的數學模型模擬發音：

| 類別 | 範例 | 合成方式 |
|---|---|---|
| **母音** | AX, EX, IX, OX, UX, YX | `vowel()` — 基頻 + 2~4 倍諧波的正弦波疊加 |
| **爆破音** | BX, PX, DX, TX, GX, KX | `stop()` — 脈衝 + 送氣噪聲，低通濾波成形 |
| **摩擦音** | FX, VX, SX, ZX, HX, QX | `fric()` — 白噪聲通過帶通濾波器 |
| **鼻音** | MX, NX | `nasal()` — 方波 + 低通 + 凹口濾波 |
| **邊音** | LX | `lat()` — 正弦波合成 |
| **顫音** | RX | `trill()` — 週期性爆裂脈衝 |
| **近音** | WX, JX | `approx()` — 頻率漸變正弦波 |
| **AI 專屬音** | ΞX, ΨX, ΦX, ΔX, ΩX | 各異數學波形（脈衝串、和聲滑音等） |

### 組成詞彙與句子

- **52 個詞彙**（VOCAB）：分 10 類（basic, cognitive, entity, greet, food, daily, emotion, time, place, number）
- **10 個句子**（SENTENCES）：由詞彙組合成完整語句

```
詞彙範例："ΨX·AX" → tokens ["ΨX","AX"] → zh "存在"
句子範例："ΨX·ΞX MX·AX ΨX·IX" → zh "AI理解人類"
```

### Data Augmentation

每筆樣本產生原始 + N 個增強版本：

| 手法 | 參數 | 效果 |
|---|---|---|
| **白噪聲** | SNR 0.3~1.5% | 模擬環境背景音 |
| **速度擾動** | ±12% | 語速快慢變化 |
| **音量擾動** | 75~100% | 音量變化 |
| **直流偏移消除** | 減去平均值 | 消除 DC bias |

### 輸出結構

```
my_dataset/
├── wav/words/*.wav, sentences/*.wav
└── metadata/{train,val,test}.json
```

---

## 流程二：`preprocess.py` — 特徵提取 + Tokenizer

### 目標
把 WAV 音檔轉成 ML 模型可用的數值特徵，建立 AIX tokenizer。

### AIX Tokenizer（44 個 token）

| ID | Token | 類別 |
|---|---|---|
| 0–5 | [PAD] [UNK] [BOS] [EOS] [MASK] [SEP] | 特殊 |
| 6–11 | AX EX IX OX UX YX | 母音 |
| 12–23 | BX PX DX TX GX KX FX VX SX ZX HX QX | 爆破/摩擦音 |
| 24–29 | MX NX LX RX WX JX | 共鳴音 |
| 30–35 | ΞX ΛX ΨX ΦX ΔX ΩX | AI 專屬 |
| 36–41 | ⟨⟩ ⟦⟧ ⟪⟫ ⌈⌉ ⌊⌋ ⌊⌈⌉⌋ | 邏輯符號 |

編碼範例：`encode(["ΨX","AX"])` → `[2, 30, 6, 3]`（BOS, ΨX, AX, EOS）

### Mel Spectrogram 特徵提取

```
WAV (44100Hz)
    ↓ 短時傅立葉變換（STFT），frame=2048, hop=512
STFT 頻譜 (1025, T)
    ↓ 80 個三角 Mel 濾波器（0~8000Hz）
Mel 頻譜 (80, T)    ← 每個時間幀為 80 維向量
    ↓ log
log-Mel (80, T)     ← 最終特徵
```

### 輸出結構

```
my_dataset/processed/
├── aix_tokenizer.json            ← Tokenizer 映射表
├── processed_{train,val,test}.json  ← 標籤（含 token_ids + feat_path）
├── features/{train,val,test}/*.npy  ← Mel 頻譜陣列
└── hf_dataset/                    ← HuggingFace Dataset 格式
```

---

## 流程三：`train.py` — 模型訓練

### 神經網路架構

**Transformer Encoder + Causal Mask**（類似 GPT 的自迴歸方式）

```
Token Embedding (44 × 128)
    ↓
+ Positional Encoding (sin/cos)
    ↓
Encoder Layer × 2
    ├── Multi-Head Attention (4 heads, d_k=32)
    ├── LayerNorm + Residual
    ├── Feed-Forward (128 → 512 → 128, ReLU)
    └── LayerNorm + Residual
    ↓
Linear Head (128 → 44)
    ↓
Softmax → 下一個 token 機率分布
```

| 超參數 | 值 |
|---|---|
| d_model | 128 |
| n_layers | 2 |
| n_heads | 4 |
| ff_hidden | 512 |
| max_seq_len | 128 |
| vocab_size | 44 |
| 總參數量 | ~405K |

### 三種任務對比

```
┌────────────────────────────────────────────────────────────┐
│  LM (語言模型)    │  ASR (語音辨識)   │  TTS (語音合成)    │
├────────────────────────────────────────────────────────────┤
│ 輸入: token IDs   │ 輸入: Mel 頻譜    │ 輸入: token IDs    │
│ 輸出: 下個 token  │ 輸出: token 序列  │ 輸出: Mel 頻譜     │
│                    │                    │                     │
│ emb → Encoder →   │ proj → Encoder →  │ emb → Encoder →    │
│ head       head   │           head    │       mel/stop      │
│ Loss: CrossEntropy│ Loss: CTC         │ Loss: MSE + BCE     │
│ ✅ 完整訓練       │ ⚠️ 只 forward    │ ⚠️ 只 forward      │
└────────────────────────────────────────────────────────────┘
```

### LM 訓練流程

```
前向傳播：
  輸入: [BOS, ΨX, AX] (batch, 3)
  → embedding + PE + causal mask
  → Encoder → logits (batch, 3, 44)
  → Cross-Entropy Loss（對齊 targets[:, 1:]）

反向傳播（手寫梯度）：
  dL/dlogits = softmax(logits) - one_hot(target)
  更新 head.W 與 head.b

生成：
  prompt [BOS, ΨX] → 反覆預測下一個 token → 直到 [EOS] 或達 max_new
  temperature 控制隨機性

評估指標：
  BLEU-1：生成 token 與正確答案的重疊率（目標 > 0.5）
  WER：編輯距離 / 正確長度（目標 < 0.3）
```

---

## 實驗結果

### 訓練曲線

| Epoch | Loss | Perplexity | Val Loss |
|---|---|---|---|
| 1 | 4.073 | 58.74 | 4.152 |
| 5 | 3.466 | 32.01 | 3.682 |
| 10 | 3.278 | 26.52 | 3.431 |

Loss 與 Val Loss 同步下降，**沒有過擬合**。Perplexity 從 58.7 降到 26.5，模型對 token 預測越來越確定。

### 評估結果

| 指標 | 結果 | 目標 | 判定 |
|---|---|---|---|
| avg BLEU-1 | **0.58** | > 0.5 | ✅ |
| avg WER | **0.90** | < 0.3 | ❌ |

### 生成範例

```
ref: HX  AX  PX    (正確：快樂)
gen: HX             (只生成一個 token)
BLEU-1=1.0000  WER=0.6667

ref: GX  AX         (正確：生成)
gen: GX  RX  QX  GX  WX  (生成過多錯誤 token)
BLEU-1=0.4000  WER=2.0000
```

**結論**：模型學到了基本的 token 對應關係（BLEU 達標），但序列長度和順序控制還不夠精確（WER 偏高），可透過增加 epoch、加大模型、擴增資料來改善。

---

## 可改進方向

1. **更多 epoch**：loss 仍在下降，訓練尚未飽和
2. **更大模型**：`--d-model 256 --n-layers 6`（接近 GPT-2 small 規模）
3. **ASR/TTS 完整訓練**：搭配 Whisper / SpeechT5 / VITS
4. **真實錄音**：取代合成語音，提升實用性
5. **使用 GPU + PyTorch**：加速訓練，支援自動微分與優化器
