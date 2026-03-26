# gpt.py 說明文檔

## 概述

gpt.py 是一個從零實現的 mini GPT 模型，採用純 Python 編寫，僅使用標準庫。這個程式可以：

1. 讀取文字檔案作為訓練數據
2. 從頭訓練一個 Transformer 語言模型
3. 生成新的文字樣本

## 使用方式

```bash
python gpt.py <your_file.txt>
```

- 預設使用 `input.txt`
- 可自行指定其他文字檔案（每行一個名字/單詞）

---

## 詳細程式碼解釋

### 1. 初始化與數據處理（第 1-30 行）

```python
random.seed(42)  # 固定隨機種子，確保結果可重現

DATASET = 'input.txt'
if len(sys.argv) > 1:
    DATASET = sys.argv[1]  # 可透過命令列參數指定輸入檔案
```

這段程式碼處理檔案讀取：
- 固定隨機種子確保訓練可重現
- 支援命令列參數指定訓練檔案
- 讀取檔案後，將每行視為一個獨立的訓練樣本（通常是人名）

```python
docs = [l.strip() for l in open(DATASET, encoding='utf-8').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
```

- 讀取檔案並去除空白
- 隨機打亂順序（ Stochastic Gradient Descent 需要）

```python
all_chars = ''.join(docs)          # 合併所有字元
uchs = sorted(set(all_chars))       # 去重並排序
BOS = len(uchars)                   # BOS = Begin of Sequence (當作特殊token)
vocab_size = len(uchars) + 1         # 詞彙表大小 = 字元數 + BOS
```

這裡建立了字元級的詞彙表：
- 例如：如果資料是 "abc", "abd"，則 `uchars = ['a', 'b', 'c', 'd']`
- `BOS = 4`（新增加的特殊字元）
- `vocab_size = 5`

---

### 2. Value 類別：自動微分系統（第 32-97 行）

這是程式碼的核心 - 從零實現的自動微分（AutoDiff）系統，類似 PyTorch 的 Tensor。

#### 2.1 基本結構

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data       # 數值
        self.grad = 0          # 梯度（預設為0）
        self._children = children      # 子節點（依賴的 Value 物件）
        self._local_grads = local_grads # 對每個子節點的局部梯度
```

每個 `Value` 物件儲存：
- `data`: 實際數值
- `grad`: 梯度值
- `_children`: 依賴的其他 Value 物件（建構計算圖）
- `_local_grads`: 對每個子節點的局部梯度 ∂output/∂input

#### 2.2 運算子重載

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

加法的鏈規則：
- ∂(a+b)/∂a = 1
- ∂(a+b)/∂b = 1

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

乘法的鏈規則：
- ∂(a*b)/∂a = b
- ∂(a*b)/∂b = a

```python
def __pow__(self, other):
    return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))
```

指數函數的梯度（當 exponent 為常數）：
- ∂(x^n)/∂x = n * x^(n-1)

```python
def log(self):
    return Value(math.log(self.data), (self,), (1 / self.data,))

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

常見激活函數的梯度：
- log: ∂log(x)/∂x = 1/x
- exp: ∂exp(x)/∂x = exp(x)
- ReLU: ∂ReLU(x)/∂x = 1 if x > 0 else 0

#### 2.3 反向傳播

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1  # 輸出對自己的梯度 = 1

    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

反向傳播演算法：
1. **拓撲排序**：建立計算圖的依賴順序（從輸入到輸出）
2. **從輸出開始**：設定輸出節點的梯度為 1
3. **鏈式法則**：沿拓撲順序反向傳播梯度

---

### 3. 模型架構配置（第 99-121 行）

```python
n_embd = 8      # 嵌入維度：每個字元表示成 8 維向量
n_head = 2      # 注意力頭數：2 個頭
n_layer = 1     # Transformer 層數：1 層
block_size = 16 # 上下文長度：最多看 16 個字元
head_dim = n_embd // n_head  # 每個頭的維度 = 8 / 2 = 4
```

模型配置說明：
- `n_embd = 8`：向量維度很小，是簡化版 GPT
- `n_head = 2`：2 個注意力頭，每個頭關注不同特徵
- `n_layer = 1`：只有 1 層 Transformer（非常淺）
- `block_size = 16`：最長上下文為 16 個字元

#### 3.1 權重初始化

```python
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

使用高斯分佈初始化權重矩陣：
- `nout`: 輸出維度
- `nin`: 輸入維度
- `std = 0.08`: 標準差（小值使訓練更穩定）

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # Token Embedding 權重
    'wpe': matrix(block_size, n_embd),      # Position Embedding 權重
    'lm_head': matrix(vocab_size, n_embd)   # 輸出層權重
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output
    state_dict[f'layer{i}.mlp_fc1'] = matrix(2 * n_embd, n_embd)  # FFN 第一層（擴展維度）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 2 * n_embd)  # FFN 第二層（縮小維度）
```

所有模型參數：
- `wte` (Weight Token Embedding): 將 token ID 映射為 n_embd 維向量
- `wpe` (Weight Position Embedding): 將位置 ID 映射為 n_embd 維向量
- `attn_wq/wk/wv`: 計算 Q, K, V 的線性變換
- `attn_wo`: 注意力輸出投影
- `mlp_fc1/fc2`: 前饋網路的兩層（維度擴展 2 倍後再縮小）

---

### 4. 層與函數（第 125-176 行）

#### 4.1 線性層

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

線性變換（矩陣乘法）：
- `x`: 輸入向量 [nin]
- `w`: 權重矩組 [nout, nin]
- 輸出: [nout]

數學公式：y_i = Σ(w_ij * x_j)

#### 4.2 Softmax

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Softmax 函數：
- 數學公式：softmax(x_i) = exp(x_i) / Σexp(x_j)
- 減去最大值（數值穩定性技巧）：防止 exp 溢出

#### 4.3 RMS Norm

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

RMS Norm（Root Mean Square Normalization）：
- 數學公式：RMSNorm(x) = x / √(mean(x²) + ε)
- 比 Layer Norm 更簡單，只有縮放沒有偏移

#### 4.4 GPT 前向傳播

```python
def gpt(token_id, pos_id, keys, values):
    # 1. Embedding
    tok_emb = state_dict['wte'][token_id]    # [n_embd]
    pos_emb = state_dict['wpe'][pos_id]     # [n_embd]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # [n_embd]
    
    # 2. Pre-norm
    x = rmsnorm(x)
    
    # 3. Transformer 層
    for li in range(n_layer):
        x_residual = x
        
        # Self-Attention
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
        # 多頭注意力
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]
            
            # Attention scores
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5 
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # 注意力輸出
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(head_dim)]
            x_attn.extend(head_out)
        
        # 注意力輸出投影 + 殘差
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # FFN + 殘差
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    
    # 4. 輸出層
    logits = linear(x, state_dict['lm_head'])
    return logits
```

詳細流程：
1. **Token Embedding**: 將字元 ID 轉為向量
2. **Position Embedding**: 將位置轉為向量（讓模型知道位置資訊）
3. **RMS Norm**: 標準化
4. **Self-Attention**:
   - Q = x @ W_Q, K = x @ W_K, V = x @ W_V
   - Attention(Q,K,V) = softmax(QK^T / √d)V
   - 多頭注意力：每個頭獨立計算，最後拼接
5. **殘差連接**: x = Attention(x) + x
6. **FFN**: 兩層線性網路，中間有 ReLU 激活
7. **輸出層**: 將 hidden state 映射到詞彙表大小

---

### 5. 訓練（第 178-212 行）

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # Adam 第一動量估計
v = [0.0] * len(params)  # Adam 第二動量估計
```

Adam 優化器初始化：
- `m`: 梯度的一階動量（類似動量）
- `v`: 梯度的二階動量（類似 RMSProp）

```python
num_steps = 500
for step in range(num_steps):
    doc = docs[step % len(docs)]  # 循環取樣
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
```

訓練步驟：
1. 取一個訓練樣本（名字）
2. 轉為 token 序列，首尾加上 BOS
3. 計算可處理的 token 數量

```python
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)
    loss.backward()
```

每個位置的訓練：
- 輸入: token_id（在位置 pos_id）
- 目標: target_id（下一個 token）
- 計算 logits → softmax → 取目標 token 的機率
- Loss = -log(機率)（negative log-likelihood）

```python
    lr_t = learning_rate * (1 - step / num_steps)  # 學習率衰減
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

Adam 優化器更新規則：
1. m_t = β₁ * m_{t-1} + (1-β₁) * g  (梯度的一階估計)
2. v_t = β₂ * v_{t-1} + (1-β₂) * g²  (梯度的二階估計)
3. m̂ = m_t / (1-β₁^t)  (偏差校正)
4. v̂ = v_t / (1-β₂^t)  (偏差校正)
5. p = p - lr * m̂ / (√v̂ + ε)  (更新參數)

---

### 6. 推理生成（第 214-232 行）

```python
temperature = 0.5  # 溫度參數控制隨機性
```

- temperature > 1: 更多隨機性（創造性）
- temperature < 1: 更少隨機性（確定性）
- temperature = 0: 貪心選擇（always 選最高機率）

```python
for sample_idx in range(10):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS  # 從 BOS 開始
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        if token_id == BOS:
            break
        if token_id < len(uchars):
            sample.append(uchars[token_id])

    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
```

生成過程：
1. 初始化：keys/values 為空，從 BOS token 開始
2. 每個位置：
   - 計算 logits
   - 除以 temperature 後 softmax（ logits / temperature 讓分布更尖銳或更平滑）
   - 根據機率採樣下一個 token
3. 停止條件：選到 BOS 或達到 block_size
4. 輸出生成的字元序列

---

## 數據格式

輸入檔案應為純文字，每行一個獨立的訓練樣本（通常是人名或單詞）：

```
john
jane
bob
alice
...
```

模型會學習預測下一個字元。

---

## 完整訓練流程

1. **數據準備**: 讀入文字檔案，建立字元級詞彙表
2. **模型初始化**: 建立 Transformer 架構，隨機初始化權重
3. **前向傳播**: 輸入 token，計算預測下一個 token 的 logits
4. **計算 Loss**: -log(正確 token 的預測機率)
5. **反向傳播**: 使用自動微分計算梯度
6. **參數更新**: Adam 優化器
7. **重複 3-6**: 多次訓練
8. **生成**: 使用訓練好的模型生成新文字
