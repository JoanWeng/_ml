> 本專案由Claude製作
  [對話連結](https://claude.ai/share/73adcf6c-57f7-487c-ae30-28c57c60ebc5)

# nn0.py — 純 Python 自動微分引擎

> 從零實作的 autograd 框架，不依賴 NumPy 或任何深度學習函式庫。

---

## 概覽

`nn0.py` 是一個用純 Python 寫成的迷你機器學習引擎，包含：

- **`Value`** — 自動微分節點，支援前向運算與反向傳播
- **`Adam`** — Adam 優化器，含學習率線性衰減
- **`linear()`** — 矩陣乘法（全連接層）
- **`softmax()`** — 數值穩定的 softmax
- **`rmsnorm()`** — RMS Normalization
- **`gd()`** — 單步梯度下降封裝

## 依賴套件

| 檔案 | 需要安裝 |
|------|---------|
| `nn0.py` | 無，只需 Python 3.8+ 標準函式庫（僅用到 `math`）|
| `nn0_demo.html` | 無，直接用瀏覽器開啟 |
| `cartpole_rl.py` | `gymnasium`（提供 CartPole 模擬環境）|

```bash
# 只有要跑 cartpole_rl.py 才需要安裝
pip install gymnasium
```

---

## 檔案結構

```
.
├── nn0.py              # 核心引擎
├── cartpole_rl.py      # 範例：CartPole 強化學習
├── nn0_demo.html       # 範例：瀏覽器互動訓練視覺化
└── README.md
```

---

## 快速開始

### 基本用法：Value 自動微分

```python
from nn0 import Value

x = Value(2.0)
w = Value(-0.5)
b = Value(1.0)

z = (x * w + b).relu()   # 前向運算，自動建立計算圖
z.backward()              # 反向傳播，計算所有梯度

print(x.grad)  # ∂z/∂x
print(w.grad)  # ∂z/∂w
print(b.grad)  # ∂z/∂b
```

### 建立並訓練一個神經網路

```python
import random
from nn0 import Value, Adam, linear, softmax

def randn(): return (random.random() * 2 - 1) * 0.5

# 定義一個 2→4→1 的網路
W1 = [[Value(randn()) for _ in range(2)] for _ in range(4)]
b1 = [Value(0.0) for _ in range(4)]
W2 = [[Value(randn()) for _ in range(4)] for _ in range(1)]
b2 = [Value(0.0) for _ in range(1)]

params = [p for row in W1+W2 for p in row] + b1 + b2
optimizer = Adam(params, lr=0.01)

# 訓練一步
x = [Value(0.5), Value(-1.2)]
h = [sum(W1[i][j] * x[j] for j in range(2)) + b1[i] for i in range(4)]
h = [hi.relu() for hi in h]
out = sum(W2[0][i] * h[i] for i in range(4)) + b2[0]

loss = (out - Value(1.0)) ** 2   # MSE loss
loss.backward()
optimizer.step()
```

---

## 範例說明

### 1. 神經網路互動視覺化（`nn0_demo.html`）

直接用瀏覽器開啟，不需要任何伺服器。

**功能：**

| 分頁 | 內容 |
|------|------|
| 訓練實驗室 | XOR / 圓形 / 螺旋分類，即時顯示決策邊界與損失曲線 |
| 反向傳播動畫 | 計算圖視覺化，點擊節點查看 `data` 與 `grad` |
| 程式碼解析 | 帶語法高亮的核心邏輯，Adam vs SGD 對比圖 |

**可調整的超參數：**
- 隱藏層大小（2–16）
- 學習率（0.001–0.050）
- 批次大小（1–20）
- 優化器（Adam / SGD）

---

### 2. CartPole 強化學習（`cartpole_rl.py`）

用 REINFORCE 策略梯度演算法訓練虛擬小車撐住平衡竿。

**執行方式：**

```bash
# 基本訓練（文字輸出）
python cartpole_rl.py

# 自訂超參數
python cartpole_rl.py --episodes 1000 --lr 0.003 --gamma 0.99

# 訓練完後開視窗播放
python cartpole_rl.py --render

# 固定隨機種子（重現結果）
python cartpole_rl.py --seed 42
```

**演算法流程：**

```
① 策略網路推論動作機率  π(a | s)
② 按機率採樣動作，與環境互動
③ 收集一整個 episode 的獎勵
④ 計算折扣回報  G_t = Σ γᵏ · r_{t+k}
⑤ 損失  L = -Σ log π(aₜ | sₜ) · Gₜ
⑥ loss.backward()  →  optimizer.step()
```

**預期訓練結果：**

```
Episode    獎勵   近100平均
─────────────────────────────
    100      87      52.3
    200     215     143.6
    350     478     389.4
    412     500     476.2   ✓ SOLVED
```

通常在 300–500 個 episode 內，近百回合平均可突破 475 分（解決標準）。

---

## 核心概念

### Value 節點與計算圖

每個 `Value` 記錄三件事：當前值（`data`）、梯度（`grad`）、以及「我是由哪些節點經過什麼運算得來的」。
做運算時自動串接成有向無環圖（DAG），`backward()` 再沿圖反向傳播梯度。

```
x=2 ──┐
       ×  →  xw=-1  ──┐
w=-0.5─┘               +  →  xw+b=0  →  ReLU  →  z=0
b=1 ──────────────────┘

backward：z.grad=1 → ReLU截斷 → 梯度為0 → 不再往前傳
```

### Adam 優化器

相較於基本 SGD，Adam 維護兩組動量：

```
m ← β₁·m + (1-β₁)·g          # 一階矩：平滑梯度方向
v ← β₂·v + (1-β₂)·g²         # 二階矩：自適應縮放步長
θ ← θ - lr · m̂ / (√v̂ + ε)   # 偏差校正後更新
```

預設值 `β₁=0.85, β₂=0.99`，適合小批次訓練。學習率支援在 `step()` 時動態覆寫，方便實作線性衰減。

---

## 支援的運算

| 運算 | 方法 | 局部梯度 |
|------|------|---------|
| 加法 | `a + b` | ∂/∂a = 1，∂/∂b = 1 |
| 乘法 | `a * b` | ∂/∂a = b，∂/∂b = a |
| 次方 | `a ** n` | ∂/∂a = n·aⁿ⁻¹ |
| 自然對數 | `.log()` | ∂/∂a = 1/a |
| 指數 | `.exp()` | ∂/∂a = eᵃ |
| ReLU | `.relu()` | ∂/∂a = 1 if a>0 else 0 |

減法、除法、負號均由上述運算組合派生。

---

## 限制

- 純 Python 實作，速度遠不及 PyTorch / NumPy，適合學習用途
- 無 GPU 支援
- 無批次張量（每個 `Value` 只存一個純量）
- `CartPole` 範例訓練一次約需 2–5 分鐘（視硬體而定）