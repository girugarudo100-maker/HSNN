"""
bench_fast.py
=============
「多タスク経験を積んだHSNN（外部報酬なし）は
 PPO（外部報酬あり）より未知タスクへの適応が速いか」
標準化ベンチマーク

出力: benchmark_fig1_main.png / benchmark_fig2_core.png
"""
import numpy as np, time, sys, copy
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 既存実装を再利用 ───────────────────────────────────────
from hierarchical_memory import (
    HierarchicalGenome, HierarchicalAgent, MinesweeperEnv,
    N0, N1, N2, LEAK, THRESHOLD, E_MAX, E_GAIN, E_LOSS,
    LR_P, PE_SLOW_ALPHA, _rng as _hm_rng
)
from experiment_v5 import (
    InfoDrivenGenome, InfoDrivenAgent,
    _W_KEYS, _integrate,
    FAILURE_RATE_INIT, SUCCESS_RATE, PE_WEIGHT, CLEARED_BONUS
)

# ══════════════════════════════════════════════════════════
# ベンチマークパラメータ
# ══════════════════════════════════════════════════════════
SEEDS         = [0, 1, 2, 3, 4]
N_EP_PER_TASK = 60      # タスクごとの訓練エピソード数
N_TRANSFER    = 250     # 転移テストエピソード数
L2            = 32      # PPO隠れ層サイズ

_CORE_KEYS = ('Win', 'W0', 'W01', 'W1', 'W12', 'W2')  # タスク非依存の SNN 重み

# ══════════════════════════════════════════════════════════
# タスク定義
# ══════════════════════════════════════════════════════════
TRAIN_TASKS = [
    ('MS_4x4_1m', lambda: MinesweeperEnv(4, 1)),   # T0: easy
    ('MS_4x4_2m', lambda: MinesweeperEnv(4, 2)),   # T1: medium
    ('MS_4x4_3m', lambda: MinesweeperEnv(4, 3)),   # T2: hard
    ('MS_5x5_2m', lambda: MinesweeperEnv(5, 2)),   # T3: larger
    ('MS_5x5_3m', lambda: MinesweeperEnv(5, 3)),   # T4: larger+hard
    ('MS_5x5_4m', lambda: MinesweeperEnv(5, 4)),   # T5: largest training
]
TRANSFER_TASK = ('MS_6x6_4m', lambda: MinesweeperEnv(6, 4))  # 未知タスク

# ══════════════════════════════════════════════════════════
# HSNN ユーティリティ
# ══════════════════════════════════════════════════════════

def make_genome(n_cells, core_weights=None):
    """InfoDrivenGenome を生成。core_weights があれば SNN 層にコピー。"""
    g = InfoDrivenGenome(n_cells)
    if core_weights is not None:
        for k in _CORE_KEYS:
            if k in core_weights:
                setattr(g, k, core_weights[k].copy())
    g.failure_rate = 0.033
    return g


def extract_core(genome):
    """SNN コア重みを辞書として抽出。"""
    return {k: getattr(genome, k).copy() for k in _CORE_KEYS}


def run_episode_hsnn(genome, env):
    """
    HSNN 1エピソード（情報量駆動STDP + 失敗統合）。
    Returns: lifetime
    """
    agent   = InfoDrivenAgent(genome)
    s       = env.reset(first_reveal=True)
    pe_fast = pe_slow = 0.0
    e_scale = 0.0
    mine_hit = False

    while not env.done:
        action = agent.choose_action(env)
        s_next, e_delta, done, info = env.step(action)
        energy_scale = e_delta / E_LOSS
        agent.step_v5(env.local_patch(action), pe_fast, pe_slow, e_scale)
        agent.apply_energy_delta(e_delta)
        pe_fast, pe_slow = agent.update_world_model(s, action, s_next)
        e_scale  = energy_scale
        s        = s_next
        if info['was_mine']:
            mine_hit = True

    failure_rate = genome.failure_rate
    if not mine_hit:
        _integrate(genome, agent, SUCCESS_RATE)
    else:
        _integrate(genome, agent, failure_rate)

    return env.lifetime


def train_hsnn_multitask(k_tasks, seed):
    """
    K タスクで HSNN を順番に学習。
    各タスク N_EP_PER_TASK エピソード。
    コア SNN 重みを引き継ぎながら転移。
    Returns: core_weights (dict)
    """
    rng_seed = seed * 1000
    core = None
    for i in range(k_tasks):
        name, make_env = TRAIN_TASKS[i]
        env    = make_env()
        genome = make_genome(env.n_cells, core)
        for ep in range(N_EP_PER_TASK):
            run_episode_hsnn(genome, env)
        core = extract_core(genome)
    return core


def eval_hsnn_transfer(core_weights, transfer_env_factory, n_ep):
    """
    転移タスクで HSNN を評価。
    core_weights が None なら初期化なし（0task）。
    Returns: np.ndarray of lifetime per episode
    """
    env    = transfer_env_factory()
    genome = make_genome(env.n_cells, core_weights)
    curve  = []
    for ep in range(n_ep):
        lt = run_episode_hsnn(genome, env)
        curve.append(lt)
    return np.array(curve)


# ══════════════════════════════════════════════════════════
# PPO (REINFORCE) 実装
# ══════════════════════════════════════════════════════════

class PolicyNet:
    """2層 ReLU ネットワーク + ソフトマックス方策。"""
    def __init__(self, n_in, n_out, hidden=L2, lr=0.005, gamma=0.99):
        sc = 0.1
        self.W1 = np.random.randn(hidden, n_in) * sc
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(n_out, hidden) * sc
        self.b2 = np.zeros(n_out)
        self.lr    = lr
        self.gamma = gamma

    def forward(self, x):
        h = np.maximum(0.0, self.W1 @ x + self.b1)
        return h, self.W2 @ h + self.b2

    def policy(self, x, valid):
        h, logits = self.forward(x)
        logits = np.where(valid, logits, -1e9)
        logits -= logits.max()
        probs  = np.exp(logits)
        probs /= probs.sum()
        return h, probs

    def act(self, x, valid):
        h, probs = self.policy(x, valid)
        idx = np.random.choice(len(probs), p=probs)
        return idx, h, probs

    def update(self, traj):
        """REINFORCE with standardized returns."""
        rewards = [t[2] for t in traj]
        G = 0.0; rets = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            rets.insert(0, G)
        rets = np.array(rets, dtype=float)
        if rets.std() > 1e-8:
            rets = (rets - rets.mean()) / rets.std()

        for (x, a, _, h, probs), G in zip(traj, rets):
            d2  = probs.copy(); d2[a] -= 1.0   # -(one_hot - probs)
            d2 *= -G
            self.W2 -= self.lr * np.outer(d2, h)
            self.b2 -= self.lr * d2
            dh  = self.W2.T @ d2 * (h > 0)
            self.W1 -= self.lr * np.outer(dh, x)
            self.b1 -= self.lr * dh


def run_episode_ppo(agent, env):
    """
    PPO (REINFORCE) 1エピソード。外部報酬 e_delta を直接使用。
    Returns: lifetime
    """
    state = env.reset(first_reveal=True)
    traj  = []

    while not env.done:
        hidden = env.hidden_cells()
        if not hidden: break
        valid  = np.zeros(env.n_cells, dtype=bool)
        valid[hidden] = True
        action, h, probs = agent.act(state, valid)
        s_next, reward, done, info = env.step(action)
        traj.append((state.copy(), action, float(reward), h, probs))
        state = s_next

    agent.update(traj)
    return env.lifetime


def eval_ppo_transfer(transfer_env_factory, n_ep, seed):
    """PPO をゼロから転移タスクで学習。Returns: lifetime curve."""
    np.random.seed(seed)
    env   = transfer_env_factory()
    agent = PolicyNet(env.n_cells, env.n_cells)
    curve = []
    for ep in range(n_ep):
        lt = run_episode_ppo(agent, env)
        curve.append(lt)
    return np.array(curve)


# ══════════════════════════════════════════════════════════
# ランダム基準
# ══════════════════════════════════════════════════════════

def random_baseline(transfer_env_factory, n_ep=500):
    env   = transfer_env_factory()
    total = 0
    for _ in range(n_ep):
        env.reset(first_reveal=True)
        while not env.done:
            h = env.hidden_cells()
            if not h: break
            env.step(int(_hm_rng.choice(h)))
        total += env.lifetime
    return total / n_ep


# ══════════════════════════════════════════════════════════
# メトリクス計算
# ══════════════════════════════════════════════════════════

def rolling_mean(arr, window=20):
    return np.convolve(arr, np.ones(window)/window, mode='valid')


def adaptation_speed(curve, threshold, window=20):
    """rolling mean が threshold を初めて超えるエピソード番号。"""
    rm = rolling_mean(curve, window)
    hits = np.where(rm >= threshold)[0]
    return int(hits[0]) + window if len(hits) else len(curve)


def compute_metrics(curve):
    return {
        'early':  float(np.mean(curve[:50])),
        'final':  float(np.mean(curve[-50:])),
        'curve':  curve,
    }


# ══════════════════════════════════════════════════════════
# メイン実験
# ══════════════════════════════════════════════════════════

MODELS = [
    ('HSNN_0task', 0),
    ('HSNN_2task', 2),
    ('HSNN_4task', 4),
    ('HSNN_6task', 6),
]

def run_all(seeds=SEEDS):
    t0 = time.time()
    transfer_factory = TRANSFER_TASK[1]

    rand_base = random_baseline(transfer_factory)
    print(f"ランダム基準（転移タスク）: {rand_base:.2f} 手")

    # 結果格納: model_name → list of curves (per seed)
    all_curves = {m[0]: [] for m in MODELS}
    all_curves['PPO']  = []

    for seed in seeds:
        np.random.seed(seed * 37 + 13)
        print(f"\n── SEED {seed} ──────────────────────────────")

        # HSNN 各 K-task
        for model_name, k in MODELS:
            print(f"  {model_name}: 学習中...", end='', flush=True)
            core = train_hsnn_multitask(k, seed)
            curve = eval_hsnn_transfer(core, transfer_factory, N_TRANSFER)
            all_curves[model_name].append(curve)
            print(f" 完了  early={np.mean(curve[:50]):.2f}  final={np.mean(curve[-50:]):.2f}")

        # PPO
        print(f"  PPO: 学習中...", end='', flush=True)
        ppo_curve = eval_ppo_transfer(transfer_factory, N_TRANSFER, seed)
        all_curves['PPO'].append(ppo_curve)
        print(f" 完了  early={np.mean(ppo_curve[:50]):.2f}  final={np.mean(ppo_curve[-50:]):.2f}")

    elapsed = time.time() - t0
    print(f"\n実験完了  elapsed={elapsed:.0f}s")
    return all_curves, rand_base


# ══════════════════════════════════════════════════════════
# 集計・可視化
# ══════════════════════════════════════════════════════════

def aggregate(all_curves):
    """シード平均・標準誤差。"""
    agg = {}
    for name, curves in all_curves.items():
        arr = np.stack(curves)                          # (n_seeds, N_TRANSFER)
        agg[name] = {
            'mean':  arr.mean(0),
            'sem':   arr.std(0) / np.sqrt(len(curves)),
            'early': float(arr[:, :50].mean()),
            'final': float(arr[:, -50:].mean()),
        }
    return agg


def plot_results(agg, rand_base):
    COLORS = {
        'HSNN_0task': '#9E9E9E',
        'HSNN_2task': '#64B5F6',
        'HSNN_4task': '#1E88E5',
        'HSNN_6task': '#0D47A1',
        'PPO':        '#E53935',
    }
    eps = np.arange(N_TRANSFER)

    # ── Fig1: 学習曲線（全モデル） ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Benchmark: HSNN Multi-task vs PPO (Transfer Task: 6×6 Minesweeper 4mines)',
                 fontsize=11)

    ax = axes[0]
    for name, d in agg.items():
        rm = rolling_mean(d['mean'], 20)
        ax.plot(np.arange(len(rm)) + 20, rm,
                color=COLORS[name], linewidth=2, label=name)
        se = rolling_mean(d['sem'], 20)
        ax.fill_between(np.arange(len(rm)) + 20,
                        rm - se, rm + se, color=COLORS[name], alpha=0.15)
    ax.axhline(rand_base, color='black', linestyle=':', linewidth=1, label='random')
    ax.set_xlabel('Episode'); ax.set_ylabel('Lifetime (steps)')
    ax.set_title('Learning Curve on Transfer Task (rolling mean 20)')
    ax.legend(fontsize=8)

    ax = axes[1]
    model_names = list(agg.keys())
    x = np.arange(len(model_names))
    early = [agg[m]['early'] for m in model_names]
    final = [agg[m]['final'] for m in model_names]
    w = 0.35
    ax.bar(x - w/2, early, w, color=[COLORS[m] for m in model_names], alpha=0.6, label='First 50ep')
    ax.bar(x + w/2, final, w, color=[COLORS[m] for m in model_names], alpha=1.0, label='Last 50ep')
    ax.axhline(rand_base, color='black', linestyle=':', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    ax.set_title('Early vs Final Performance')
    ax.set_ylabel('Lifetime (steps)'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('benchmark_fig1_main.png', dpi=130); plt.close()
    print("  → benchmark_fig1_main.png 保存完了")

    # ── Fig2: コア指標（適応速度 / 最終性能 / PPO比） ──────
    ppo_final = agg['PPO']['final']
    ppo_early = agg['PPO']['early']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Core Metrics Summary', fontsize=11)

    # Compute adaptation speed for each model (first ep rolling > threshold)
    threshold = rand_base + 0.5 * (max(d['final'] for d in agg.values()) - rand_base)
    adapt_speeds = {}
    for name, d in agg.items():
        adapt_speeds[name] = adaptation_speed(d['mean'], threshold)

    ax = axes[0]
    bars = ax.bar(model_names, [adapt_speeds[m] for m in model_names],
                  color=[COLORS[m] for m in model_names])
    ax.set_title('Adaptation Speed (ep to threshold)\nlower = faster')
    ax.set_ylabel('Episodes'); ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    for bar, m in zip(bars, model_names):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{adapt_speeds[m]}', ha='center', va='bottom', fontsize=7)

    ax = axes[1]
    ax.bar(model_names, [agg[m]['final'] for m in model_names],
           color=[COLORS[m] for m in model_names])
    ax.axhline(ppo_final, color='red', linestyle='--', linewidth=1.5, label='PPO final')
    ax.axhline(rand_base, color='black', linestyle=':', linewidth=1, label='random')
    ax.set_title('Final Performance (last 50ep)')
    ax.set_ylabel('Lifetime (steps)')
    ax.set_xticklabels(model_names, rotation=15, fontsize=8); ax.legend(fontsize=7)

    ax = axes[2]
    ratios = [ppo_early / max(adapt_speeds[m], 1) * adapt_speeds['PPO'] / max(adapt_speeds['PPO'], 1)
              if adapt_speeds['PPO'] > 0 else 1.0 for m in model_names]
    # PPO 比 = PPO_adapt_speed / model_adapt_speed (高い = PPOより速い)
    ppo_speed = adapt_speeds['PPO']
    speed_ratios = [ppo_speed / max(adapt_speeds[m], 1) for m in model_names]
    ax.bar(model_names, speed_ratios, color=[COLORS[m] for m in model_names])
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='PPO=1.0')
    ax.set_title('Speed Ratio vs PPO\n>1.0 = faster than PPO')
    ax.set_ylabel('Ratio (PPO speed / model speed)')
    ax.set_xticklabels(model_names, rotation=15, fontsize=8); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('benchmark_fig2_core.png', dpi=130); plt.close()
    print("  → benchmark_fig2_core.png 保存完了")

    return adapt_speeds, threshold


def print_report(agg, adapt_speeds, rand_base):
    ppo_final = agg['PPO']['final']
    ppo_speed = adapt_speeds['PPO']

    print("\n" + "=" * 65)
    print(" ベンチマーク結果サマリー")
    print(" 転移タスク: 6×6 マインスイーパー（4地雷）")
    print(f" ランダム基準: {rand_base:.2f} 手")
    print("=" * 65)
    print(f"\n{'モデル':<14} {'適応速度(ep)':>12} {'最終性能(手)':>12} {'PPO比':>8}")
    print("-" * 52)

    for name in ['HSNN_0task', 'HSNN_2task', 'HSNN_4task', 'HSNN_6task', 'PPO']:
        final = agg[name]['final']
        speed = adapt_speeds[name]
        if name == 'PPO':
            ratio_str = '基準'
        else:
            ratio = ppo_speed / max(speed, 1)
            ratio_str = f'{ratio:.2f}x'
        print(f"  {name:<12}  {speed:>10}  {final:>11.2f}  {ratio_str:>8}")

    print()
    print("  適応速度(ep) = 性能閾値を初めて超えるエピソード数（小 = 速い）")
    print("  PPO比 > 1.0  = PPOより速く適応")

    # 主結論
    hsnn6_speed = adapt_speeds['HSNN_6task']
    hsnn6_final = agg['HSNN_6task']['final']
    faster = hsnn6_speed < ppo_speed
    better = hsnn6_final > ppo_final
    print(f"\n  主結論: HSNN_6task は PPO より")
    print(f"    適応速度: {'速い ✓' if faster else '遅い ✗'}"
          f"  ({ppo_speed}ep vs {hsnn6_speed}ep)")
    print(f"    最終性能: {'高い ✓' if better else '低い ✗'}"
          f"  ({ppo_final:.2f} vs {hsnn6_final:.2f})")
    print("=" * 65)


# ══════════════════════════════════════════════════════════
# エントリポイント
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print(" HSNN vs PPO ベンチマーク実験")
    print(f" SEEDS={SEEDS}  N_EP_PER_TASK={N_EP_PER_TASK}")
    print(f" N_TRANSFER={N_TRANSFER}  L2={L2}")
    print(f" 訓練タスク: {len(TRAIN_TASKS)}種類（最大6task使用）")
    print(f" 転移タスク: {TRANSFER_TASK[0]}")
    print("=" * 65)

    all_curves, rand_base = run_all()
    agg                   = aggregate(all_curves)
    adapt_speeds, thresh  = plot_results(agg, rand_base)
    print_report(agg, adapt_speeds, rand_base)
