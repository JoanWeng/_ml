import random
import math

# 計算兩點間距離的輔助函數
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class TspSolution:
    def __init__(self, cities, path=None):
        self.cities = cities  # 城市座標清單 [(x,y), (x,y), ...]
        self.n = len(cities)
        if path is None:
            # 初始解：1=>2=>3=>...=>n
            self.path = list(range(self.n))
        else:
            self.path = path

    # 計算總距離並取負值作為高度 (距離越短，高度越高)
    def height(self):
        total_dist = 0
        for i in range(self.n):
            c1 = self.path[i]
            c2 = self.path[(i + 1) % self.n] # 回到起點，形成閉環
            total_dist += distance(self.cities[c1], self.cities[c2])
        return -total_dist

    # 取得鄰居：選擇兩個邊 (a,b) 與 (c,d) 進行交換 (2-opt move)
    def neighbor(self):
        new_path = list(self.path)
        # 修正：確保 i 至少留兩個位置給 j (i+2 <= n-1)
        # 因此 i 的最大值應該是 n - 3
        i = random.randint(0, self.n - 3)
        j = random.randint(i + 2, self.n - 1)
        
        # 2-opt 交換邏輯
        new_path[i+1:j+1] = reversed(new_path[i+1:j+1])
        
        return TspSolution(self.cities, new_path)

    def str(self):
        # 格式化輸出：1=>2=>3=>1
        p_str = "=>".join(str(c + 1) for c in self.path)
        return f"[{p_str}=>{self.path[0]+1}] Height: {self.height():.2f}"

def hillClimbing(s, maxGens, maxFails):
    print("start: ", s.str())
    fails = 0
    for gens in range(maxGens):
        snew = s.neighbor()
        sheight = s.height()
        nheight = snew.height()
        
        if (nheight >= sheight): # 如果鄰居距離更短 (高度更高)
            if nheight > sheight: # 只有真的進步才印出，避免平原期洗版
                print(f"Gen {gens}: {snew.str()}")
            s = snew
            fails = 0
        else:
            fails = fails + 1
            
        if (fails >= maxFails):
            print(f"Early stop at Gen {gens} due to maxFails")
            break
            
    print("solution: ", s.str())
    return s

# --- 測試程式碼 ---
if __name__ == "__main__":
    # 建立 10 個隨機城市
    random.seed(42) # 固定隨機種子以便觀察
    test_cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    
    initial_s = TspSolution(test_cities)
    # 執行爬山演算法
    # maxGens: 總嘗試次數, maxFails: 連續失敗幾次就放棄
    result = hillClimbing(initial_s, maxGens=10000, maxFails=500)