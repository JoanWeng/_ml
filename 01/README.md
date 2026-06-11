> 程式碼由Gemini製作
  [對話連結](https://gemini.google.com/share/beca4e42674c)

> README.md由opencode製作

# 爬山演算法解決旅行推銷員問題（TSP）

## 問題簡介

**旅行推銷員問題（Traveling Salesman Problem, TSP）**：給定 n 個城市及城市間的距離，尋找一條經過所有城市恰好一次並回到起點的最短路徑。

屬於 NP-hard 問題，當城市數量增加時，窮舉所有可能排列（n! 種）在計算上不可行，因此常使用啟發式演算法求得近似解。

## 爬山演算法（Hill Climbing）

爬山演算法是一種局部搜尋的啟發式演算法：

1. 從一個**初始解**出發
2. 每次產生一個**鄰居解**（微調當前解）
3. 若鄰居解**更優**則移動過去，否則停留在原地
4. 重複直到收斂（連續失敗次數達上限）

### 優點
- 實作簡單、直觀
- 收斂速度快

### 缺點
- 容易陷入**局部最佳解**，不一定能找到全域最佳解
- 對初始解敏感

## 程式碼解析

### 1. 距離計算

```python
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
```

歐幾里得距離公式，計算二維平面上兩點的直線距離。

### 2. TspSolution 類別

```python
class TspSolution:
    def __init__(self, cities, path=None):
        self.cities = cities  # 城市座標清單
        self.n = len(cities)
        if path is None:
            self.path = list(range(self.n))  # 初始解：0=>1=>2=>...=>n-1
        else:
            self.path = path
```

- `cities`：各城市的 (x, y) 座標列表
- `path`：城市索引的走訪順序，預設為 `[0, 1, 2, ..., n-1]`

### 3. 高度函數（適應度）

```python
def height(self):
    total_dist = 0
    for i in range(self.n):
        c1 = self.path[i]
        c2 = self.path[(i + 1) % self.n]
        total_dist += distance(self.cities[c1], self.cities[c2])
    return -total_dist
```

- 計算完整迴圈的**總距離**
- 回傳**負的總距離**作為「高度」——總距離越短，高度越高
- 爬山演算法目標：最大化高度（即最小化總距離）

### 4. 鄰居產生（2-opt 交換）

```python
def neighbor(self):
    new_path = list(self.path)
    i = random.randint(0, self.n - 3)
    j = random.randint(i + 2, self.n - 1)
    new_path[i+1:j+1] = reversed(new_path[i+1:j+1])
    return TspSolution(self.cities, new_path)
```

使用 **2-opt 移動**：隨機選取兩條邊 (a→b) 和 (c→d)，將 b 到 c 之間的路徑反轉，相當於交換原本的兩條邊為 (a→c) 和 (b→d)。

```
原始：... → a → b → ... → c → d → ...
2-opt：... → a → c → ... → b → d → ...
```

### 5. 爬山演算法主體

```python
def hillClimbing(s, maxGens, maxFails):
    fails = 0
    for gens in range(maxGens):
        snew = s.neighbor()
        if nheight >= sheight:  # 鄰居不差就移動
            s = snew
            fails = 0
        else:
            fails += 1
        if fails >= maxFails:   # 連續失敗過多則提前停止
            break
    return s
```

- `maxGens`：最大迭代次數
- `maxFails`：連續失敗次數上限，用於提前終止（避免平原期空轉）
- 允許 `nheight >= sheight`（包含相等），讓演算法能在平原持續探索

### 6. 主程式測試

```python
random.seed(42)
test_cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
initial_s = TspSolution(test_cities)
result = hillClimbing(initial_s, maxGens=10000, maxFails=500)
```

- 隨機生成 10 個城市，座標範圍 0~100
- 固定隨機種子 42 確保結果可重現
- 最大迭代 10000 次，連續 500 次未改善則停止

## 執行結果範例

```
start:  [1=>2=>3=>4=>5=>6=>7=>8=>9=>10=>1] Height: -3278.15
Gen 0: [1=>2=>3=>9=>8=>7=>6=>5=>4=>10=>1] Height: -2990.00
Gen 3: [1=>2=>10=>4=>5=>6=>7=>8=>9=>3=>1] Height: -2934.91
...
solution: [1=>2=>10=>9=>8=>7=>6=>5=>4=>3=>1] Height: -1919.58
```

## 延伸方向

| 方向 | 說明 |
|------|------|
| **模擬退火** | 以一定機率接受較差的解，跳出局部最佳 |
| **遺傳演算法** | 透過交配、突變等機制進行全域搜尋 |
| **禁忌搜尋** | 記錄近期嘗試過的解，避免繞回 |
| **蟻群演算法** | 模仿螞蟻費洛蒙路徑，適合 TSP |
| **3-opt / k-opt** | 更高階的邊交換策略，搜尋更廣 |