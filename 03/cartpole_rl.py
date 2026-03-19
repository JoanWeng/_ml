"""
cartpole_rl.py — 用 nn0.py 實作 CartPole 強化學習（REINFORCE 策略梯度）

依賴：
    pip install gymnasium

使用方式：
    python cartpole_rl.py            # 訓練 + 文字輸出
    python cartpole_rl.py --render   # 訓練完後視覺化播放

演算法：REINFORCE (Monte Carlo Policy Gradient)
  1. 用策略網路 π(a|s) 選動作
  2. 跑完一整個 episode，收集獎勵
  3. 計算折扣回報 G_t = Σ γ^k * r_{t+k}
  4. 損失 = -Σ log π(a_t|s_t) * G_t  (最大化期望回報)
  5. 反向傳播 + Adam 更新
"""

import sys
import math
import random
import argparse

# ──────────────────────────────────────────────
#  從 nn0.py 引入（請確保 nn0.py 在同目錄）
# ──────────────────────────────────────────────
try:
    from nn0 import Value, Adam, linear, softmax
except ImportError:
    print("[錯誤] 找不到 nn0.py，請確認它在同一目錄。")
    sys.exit(1)

try:
    import gymnasium as gym
except ImportError:
    print("[錯誤] 請先安裝 gymnasium：pip install gymnasium")
    sys.exit(1)


# ══════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════

def randn_small():
    """Box-Muller 常態分布亂數，標準差 0.1"""
    u1 = random.random() + 1e-12
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) * 0.1


def make_weights(nin, nout):
    """建立全連接層權重 (nout × nin) 與偏置 (nout,)"""
    w = [[Value(randn_small()) for _ in range(nin)] for _ in range(nout)]
    b = [Value(0.0) for _ in range(nout)]
    return w, b


def forward(x_list, layers):
    """
    前向傳播：x_list 為 float list，layers 為 [(w,b), ...]。
    除最後一層外均套 ReLU。
    """
    h = [Value(v) for v in x_list]
    for i, (w, b) in enumerate(layers):
        h = linear(h, w)
        h = [hi + bi for hi, bi in zip(h, b)]
        if i < len(layers) - 1:          # 隱藏層才做 ReLU
            h = [hi.relu() for hi in h]
    return h


def all_params(layers):
    """收集所有 Value 參數"""
    params = []
    for w, b in layers:
        for row in w:
            params.extend(row)
        params.extend(b)
    return params


def zero_grad(params):
    for p in params:
        p.grad = 0.0


# ══════════════════════════════════════════════
#  策略網路（Policy Network）
# ══════════════════════════════════════════════

class PolicyNet:
    """
    輸入：4 個狀態（位置、速度、角度、角速度）
    隱藏：兩層，各 32 個神經元，ReLU
    輸出：2 個動作的 logit（左推 / 右推）→ softmax 機率
    """
    def __init__(self, n_obs=4, hidden=32, n_act=2):
        self.layers = [
            make_weights(n_obs,  hidden),
            make_weights(hidden, hidden),
            make_weights(hidden, n_act),
        ]
        self.params = all_params(self.layers)

    def __call__(self, obs):
        """obs: list of float → list of Value (動作機率)"""
        logits = forward(obs, self.layers)
        return softmax(logits)


# ══════════════════════════════════════════════
#  REINFORCE 訓練
# ══════════════════════════════════════════════

def discount_returns(rewards, gamma=0.99):
    """
    計算每步的折扣累積回報 G_t。
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    並做標準化（減均值除標準差）穩定訓練。
    """
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    mean = sum(returns) / len(returns)
    var  = sum((g - mean) ** 2 for g in returns) / len(returns)
    std  = math.sqrt(var + 1e-8)
    return [(g - mean) / std for g in returns]


def run_episode(env, policy, render=False):
    """
    執行一個 episode，回傳：
      log_probs : [Value] — log π(a_t | s_t)
      rewards   : [float] — 每步獎勵
      total_r   : float   — 總獎勵（episode 長度）
    """
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        if render:
            env.render()

        # 策略網路推論
        probs = policy(list(obs))

        # 按機率採樣動作
        p0 = probs[0].data
        action = 0 if random.random() < p0 else 1

        # 儲存 log π(a|s)
        log_probs.append(probs[action].log())

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

    return log_probs, rewards, sum(rewards)


def train(n_episodes=800, lr=3e-3, gamma=0.99, print_every=20):
    """主訓練迴圈"""
    env = gym.make("CartPole-v1")
    policy = PolicyNet()
    optimizer = Adam(policy.params, lr=lr)

    history = []          # 每 episode 的總獎勵
    best_avg = 0.0

    print("=" * 60)
    print("  CartPole REINFORCE — nn0.py 強化學習範例")
    print("=" * 60)
    print(f"  參數量: {len(policy.params)}")
    print(f"  學習率: {lr}   折扣因子 γ: {gamma}")
    print(f"  訓練回合: {n_episodes}")
    print("=" * 60)
    print(f"{'Episode':>8}  {'獎勵':>6}  {'近100平均':>10}  {'最佳':>6}  進度")
    print("-" * 60)

    for ep in range(1, n_episodes + 1):
        # ── Forward：跑一整個 episode ──
        log_probs, rewards, total_r = run_episode(env, policy)

        # ── 計算折扣回報 ──
        returns = discount_returns(rewards, gamma)

        # ── 損失：-Σ log π(a_t) * G_t ──
        zero_grad(policy.params)
        loss = Value(0.0)
        for lp, g in zip(log_probs, returns):
            loss = loss + lp * (-g)          # 負號 → 梯度上升

        # ── Backward ──
        loss.backward()

        # ── Adam 更新（學習率線性衰減）──
        lr_t = lr * (1 - ep / n_episodes * 0.5)
        optimizer.step(lr_override=lr_t)

        history.append(total_r)

        # ── 統計與列印 ──
        avg100 = sum(history[-100:]) / min(len(history), 100)
        if avg100 > best_avg:
            best_avg = avg100

        bar_len = int(total_r / 5)           # 最大 100 格
        bar = "█" * min(bar_len, 40) + ("…" if bar_len > 40 else "")

        if ep % print_every == 0 or total_r >= 490:
            solved = "  ✓ SOLVED!" if avg100 >= 475 else ""
            print(f"  {ep:>6}    {total_r:>5.0f}     {avg100:>8.1f}    {best_avg:>5.1f}  {bar}{solved}")

        if avg100 >= 475 and ep >= 100:
            print(f"\n  🎉 連續 100 回合平均達 {avg100:.1f}，已解決 CartPole！（ep={ep}）")
            break

    env.close()
    print("\n" + "=" * 60)
    print(f"  訓練結束  |  最終 100 回合平均: {avg100:.1f}  |  最佳: {best_avg:.1f}")
    print("=" * 60)
    return policy, history


# ══════════════════════════════════════════════
#  視覺化播放（可選）
# ══════════════════════════════════════════════

def play(policy, n_eps=5):
    """訓練完後，用 render 模式播放幾個 episode 觀察效果。"""
    print(f"\n  播放 {n_eps} 個 episode（關閉視窗繼續）...")
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(1, n_eps + 1):
        _, _, total_r = run_episode(env, policy, render=False)
        print(f"  Play ep {ep}: 獎勵 = {total_r:.0f}")
    env.close()


# ══════════════════════════════════════════════
#  訓練曲線（純文字 sparkline）
# ══════════════════════════════════════════════

def print_curve(history, width=60):
    """用 ASCII 畫出獎勵曲線。"""
    # 降採樣到 width 個點
    n = len(history)
    step = max(1, n // width)
    sampled = [sum(history[i:i+step]) / step for i in range(0, n, step)][:width]

    maxv, minv = max(sampled), min(sampled)
    height = 10
    rows = []
    for row in range(height, -1, -1):
        threshold = minv + (maxv - minv) * row / height
        line = ""
        for v in sampled:
            line += "█" if v >= threshold else " "
        label = f"{threshold:5.0f} |" if row % 2 == 0 else "      |"
        rows.append(label + line)

    print("\n  訓練獎勵曲線")
    print("  " + "─" * (width + 7))
    for r in rows:
        print("  " + r)
    print("  " + "─" * (width + 7))
    print(f"  ep: 1{'─'*(width-8)}{len(history)}")


# ══════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole REINFORCE with nn0.py")
    parser.add_argument("--episodes", type=int, default=800,   help="訓練回合數")
    parser.add_argument("--lr",       type=float, default=3e-3, help="學習率")
    parser.add_argument("--gamma",    type=float, default=0.99, help="折扣因子")
    parser.add_argument("--render",   action="store_true",      help="訓練後視覺化播放")
    parser.add_argument("--seed",     type=int, default=42,     help="隨機種子")
    args = parser.parse_args()

    random.seed(args.seed)

    policy, history = train(
        n_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
    )

    print_curve(history)

    if args.render:
        play(policy)