> 本專案由 opencode 撰寫

# 字元級插值 N-gram 語言模型

非 Transformer、非神經網路，純粹以統計方法實作的語言模型。

## 程式架構

```
lm.py
├── load_corpus()       # 讀取 tw.txt，逐字元切分，句尾插入 <EOS>
├── class NGramLM       # 核心模型
│   ├── train()         # 統計 1~3-gram 計數
│   ├── prob()          # 插值機率估計 (Jelinek-Mercer smoothing)
│   ├── predict_proba() # 給定 context，回傳所有字元的機率分布
│   ├── sample()        # temperature + top-k 抽樣
│   └── generate()      # 自回歸逐字生成 (<EOS> 停止)
├── interactive()       # 互動式命令列介面
└── demo()              # 預設示範生成
```

## 核心方法：統計式 N-gram 語言模型

**沒有使用任何神經網路。** 本模型是傳統的 n-gram 統計語言模型，核心是計數與平滑。

與 Transformer (GPT) 的差異：

|                | 本模型                 | Transformer (GPT)       |
| -------------- | ---------------------- | ----------------------- |
| 核心機制       | 計數 + 平滑            | 多頭自注意力            |
| 參數           | 無（純計數）           | 數百億個權重            |
| 訓練方式       | 統計字元出現次數       | 梯度下降 + backprop     |
| 上下文長度上限 | n（固定，此處為 3）    | 無限制（理論上）        |
| 所需資料量     | 很少（數 KB 即可）     | 極大（GB 級別）         |

**生成流程與 GPT 相同**（自回歸、temperature、top-k），只是「預測下一個字」的方式從神經網路換成了計數機率表。

## 程式碼解析

### 1. 語料載入 — `load_corpus()`

```python
def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        chars = []
        for line in f:
            line = line.strip()
            if line:
                chars.extend(list(line))
                chars.append("<EOS>")
    return chars
```

每一行是一句中文，逐字元拆開後在句尾插入 `<EOS>` 作為結束標記。
例如 `"小貓坐在桌上"` 變成 `['小','貓','坐','在','桌','上','<EOS>']`。

### 2. 訓練 — `NGramLM.train()`

```python
def train(self, chars):
    for order in range(1, self.max_n + 1):
        for i in range(len(chars) - order):
            ctx = tuple(chars[i : i + order])
            nxt = chars[i + order]
            self.counts[order][ctx][nxt] += 1
```

對每個 n-gram 階數 (1~3)，用滑動視窗掃過字元序列，統計 `(前 N 字) → (下一個字)` 的出現次數。
- `self.counts[1][('小',)]['貓'] += 1`（unigram context 其實是空的，此處示範 bigram）
- `self.counts[2][('小','貓')]['坐'] += 1`
- `self.counts[3][('小','貓','坐')]['在'] += 1`

### 3. 機率估計 — `prob()` 與 `prob_order()`

```python
def prob_order(self, order, ctx, char):
    c = self.counts[order].get(ctx, {})
    numer = c.get(char, 0) + self.k
    denom = sum(c.values()) + self.k * self.vocab_size
    return numer / denom

def prob(self, ctx, char):
    result = 0.0
    for order in range(1, min(len(ctx), self.max_n) + 1):
        sub_ctx = ctx[-order:]
        pw = self.prob_order(order, sub_ctx, char)
        weight = 1.0 / min(len(ctx), self.max_n)
        result += weight * pw
    return result
```

**Jelinek-Mercer 插值平滑**：最終機率 = 1/3 × trigram + 1/3 × bigram + 1/3 × unigram。

每個子機率使用 **add-k 平滑**避免未出現的 n-gram 機率為零：
```
P_addk(w | ctx) = (count(ctx, w) + k) / (sum(count(ctx, *)) + k * |V|)
```

### 4. 抽樣 — `sample()`

```python
def sample(self, ctx, temperature=1.0, top_k=0):
    probs = self.predict_proba(ctx)
    # top-k: 只保留機率最高的 k 個字
    if top_k and top_k < len(probs):
        probs = probs[:top_k]
    # temperature: 重新 scaling 機率分布
    if temperature != 1.0:
        logits = [math.log(max(p, 1e-10)) / temperature for p in pvals]
        exps = [math.exp(v - max(logits)) for v in logits]
        pvals = [e / sum(exps) for e in exps]
    # 依機率抽樣
    r = random.random()
    cum = 0.0
    for c, p in zip(chars, pvals):
        cum += p
        if r <= cum:
            return c, raw_probs[:5]
```

- **top-k**：只從機率最高的 k 個字中選，避免長尾雜訊。
- **temperature**：`t → 0` 趨近 greedy（最保守），`t → ∞` 趨近均勻分布（最有創意）。
- 最後用累積機率抽樣（inverse transform sampling）。

### 5. 自回歸生成 — `generate()`

```python
def generate(self, prompt, max_len, temperature, top_k, verbose):
    out = list(prompt)          # 保留完整輸出
    ctx = tuple(seed)           # 初始 context = prompt 的全部字元
    for step in range(max_len):
        char, top5 = self.sample(ctx, temperature, top_k)
        if char == "<EOS>":
            break
        out.append(char)
        ctx = ctx + (char,)     # 將新字加入 context（滑動視窗）
        if len(ctx) > self.max_n * 2:
            ctx = ctx[-(self.max_n * 2):]  # 限制 context 長度
    return "".join(out)
```

和 GPT 完全相同的自回歸邏輯：
1. 用目前 context 預測下一個字
2. 抽樣一個字
3. 接到序列尾端
4. 更新 context
5. 重複直到 `<EOS>` 或達到最大長度

## 訓練資料

語料：`tw.txt`（147 句中文短句，991 個字元）

訓練方式：
1. 將整份語料視為字元序列，句尾插入 `<EOS>`
2. 滑動視窗統計各階 n-gram 計數：
   - unigram：`P(字)`
   - bigram：`P(字 | 前1字)`
   - trigram：`P(字 | 前2字)`
3. 各階機率等權重插值，並以 add-k 平滑處理未出現的 n-gram

## 執行方式

```bash
python3 lm.py tw.txt
```

互動指令：
- `/temp N` — 調整 temperature（預設 1.0）
- `/topk N` — 調整 top-k（預設 5）
- `/len N`  — 調整最大生成長度（預設 50）
- `/v`      — 顯示每步機率細節
- `/demo`   — 示範生成
- `q`       — 離開

## 依賴

純 Python 3 標準函式庫，無任何外部套件。
