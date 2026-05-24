"""
minigrid_benchmark.py
=====================
「学習効率ベンチマーク：MiniGrid標準環境での θ初期構造の効果検証」
"""
import numpy as np
import time
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper

# ─────────────────────────────────────────────────────────────
# HSNN アーキテクチャ定数
# ─────────────────────────────────────────────────────────────
N0, N1, N2 = 80, 40, 20
N_CONCAT = N0 + N1 + N2   # 140
N_ACT_HSNN = 4             # forward / left / right / toggle

LEAK = 0.9; THRESHOLD = 1.0
E_INIT_N = 1.0; E_MAX_N = 2.0; E_FIRE = 0.05; E_DECAY = 0.005
E_LOSS_NORM = 30.0

TAU_PLUS  = [20.0, 30.0, 40.0]
TAU_MINUS = [20.0, 30.0, 40.0]
A_PLUS    = [0.01, 0.008, 0.005]
A_MINUS   = [0.012, 0.009, 0.006]

PROJ_DIM   = 32   # world model random projection
PPO_HIDDEN = 64
BLEND_RATE = 0.3

# ─────────────────────────────────────────────────────────────
# 実験パラメータ
# ─────────────────────────────────────────────────────────────
N_EPISODES     = 1000
SEEDS          = [0, 1, 2, 3, 4]
MAX_STEPS      = 300
ROLLING_WIN    = 50
THRESHOLD_RATE = 0.10
PRETRAIN_EP    = 80

# MiniGrid action mapping: HSNN 0-3 → MiniGrid 0-6
HSNN_TO_MG = [2, 0, 1, 5]   # forward, left-turn, right-turn, toggle

# Internal ΔE (外部報酬は使わない)
STEP_COST    = -0.1
WALL_COST    = -0.5
EXPLORE_GAIN = +1.0
GOAL_GAIN    = +10.0

# ─────────────────────────────────────────────────────────────
# θ パラメータ
# ─────────────────────────────────────────────────────────────
class Theta:
    def __init__(self, eta=0.0393, beta=2.190, gamma=2.039,
                 tau_trace=0.87, tau_wm=0.95):
        self.eta = eta; self.beta = beta; self.gamma = gamma
        self.tau_trace = tau_trace; self.tau_wm = tau_wm

BEST_THETA = Theta()

# ─────────────────────────────────────────────────────────────
# MiniGrid HSNN ゲノム
# ─────────────────────────────────────────────────────────────
class MiniHSNNGenome:
    def __init__(self, obs_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        # 入力 → L0: obs_dim 次元を直接受け取る
        self.Win  = rng.normal(0, 0.1,  (N0, obs_dim))
        self.W0   = rng.normal(0, 0.05, (N0, N0)); np.fill_diagonal(self.W0,  0.0)
        self.W01  = rng.normal(0, 0.05, (N1, N0))
        self.W1   = rng.normal(0, 0.05, (N1, N1)); np.fill_diagonal(self.W1,  0.0)
        self.W12  = rng.normal(0, 0.05, (N2, N1))
        self.W2   = rng.normal(0, 0.05, (N2, N2)); np.fill_diagonal(self.W2,  0.0)
        self.W_act = rng.normal(0, 0.01, (N_ACT_HSNN, N_CONCAT))
        # ワールドモデル用固定ランダム射影
        R = rng.normal(0, 1.0, (PROJ_DIM, obs_dim))
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        self.R = R / (norms + 1e-8)
        self.P = np.zeros((PROJ_DIM, PROJ_DIM + N_ACT_HSNN))

    def load_inter_weights(self, w_dict):
        """事前訓練した W0/W01/W1/W12/W2 を上書き（形状一致のみ）。"""
        for k, v in w_dict.items():
            if hasattr(self, k) and getattr(self, k).shape == v.shape:
                setattr(self, k, v.copy())

# ─────────────────────────────────────────────────────────────
# MiniGrid HSNN エージェント
# ─────────────────────────────────────────────────────────────
class MiniHSNNAgent:
    def __init__(self, genome: MiniHSNNGenome, theta: Theta):
        self.theta = theta
        for k in ('Win', 'W0', 'W01', 'W1', 'W12', 'W2', 'W_act', 'R', 'P'):
            setattr(self, k, getattr(genome, k).copy())
        self.V0 = np.zeros(N0); self.V1 = np.zeros(N1); self.V2 = np.zeros(N2)
        self.sp0 = np.zeros(N0); self.sp1 = np.zeros(N1); self.sp2 = np.zeros(N2)
        self.energies = np.full(N0 + N1 + N2, E_INIT_N)
        self.tr_pre   = [np.zeros(N0), np.zeros(N1), np.zeros(N2)]
        self.tr_post  = [np.zeros(N0), np.zeros(N1), np.zeros(N2)]
        self.pe_slow  = 0.0

    def choose_action(self):
        sp_all = np.concatenate([self.sp0, self.sp1, self.sp2])
        logits = self.W_act @ sp_all
        logits -= logits.max()
        probs = np.exp(logits); probs /= (probs.sum() + 1e-12)
        return int(np.random.choice(N_ACT_HSNN, p=probs))

    def step_net(self, obs, pe_l0, pe_l1, energy_scale):
        th = self.theta
        sp0p, sp1p, sp2p = self.sp0.copy(), self.sp1.copy(), self.sp2.copy()

        I0 = self.Win @ obs + self.W0 @ sp0p
        self.V0 = LEAK * self.V0 + I0
        f0 = self.V0 >= THRESHOLD; self.V0[f0] = 0.0; f0f = f0.astype(float)

        I1 = self.W01 @ sp0p + self.W1 @ sp1p
        self.V1 = LEAK * self.V1 + I1
        f1 = self.V1 >= THRESHOLD; self.V1[f1] = 0.0; f1f = f1.astype(float)

        I2 = self.W12 @ sp1p + self.W2 @ sp2p
        self.V2 = LEAK * self.V2 + I2
        f2 = self.V2 >= THRESHOLD; self.V2[f2] = 0.0; f2f = f2.astype(float)

        self.energies -= E_DECAY
        self.energies -= E_FIRE * np.concatenate([f0f, f1f, f2f])
        np.clip(self.energies, 0, E_MAX_N, out=self.energies)

        # θ制御 STDP（W0/W1/W2 のみ更新、W01/W12 は STDP 対象外）
        for li, fired, prev, pe_sig, W in [
            (0, f0f, sp0p, pe_l0 * th.beta, self.W0),
            (1, f1f, sp1p, pe_l0 * th.beta, self.W1),
            (2, f2f, sp2p, pe_l1,            self.W2),
        ]:
            dp = np.exp(-th.tau_trace / TAU_PLUS[li])
            dm = np.exp(-th.tau_trace / TAU_MINUS[li])
            self.tr_pre[li]  = self.tr_pre[li]  * dp + prev
            self.tr_post[li] = self.tr_post[li] * dm + fired
            mag = pe_sig * abs(energy_scale) * th.gamma
            if energy_scale >= 0:
                dW =  A_PLUS[li]  * mag * np.outer(fired, self.tr_pre[li])
            else:
                dW = -A_MINUS[li] * mag * np.outer(fired, self.tr_pre[li])
            W += dW; np.fill_diagonal(W, 0.0)

        dead = self.energies <= 0
        if dead.any():
            d0, d1, d2 = dead[:N0], dead[N0:N0+N1], dead[N0+N1:]
            for d, W, V, sp in [
                (d0, self.W0, self.V0, self.sp0),
                (d1, self.W1, self.V1, self.sp1),
                (d2, self.W2, self.V2, self.sp2),
            ]:
                if d.any():
                    nd = d.sum(); nw = W.shape[0]
                    W[d, :] = np.random.normal(0, 0.05, (nd, nw))
                    W[:, d] = np.random.normal(0, 0.05, (nw, nd))
                    np.fill_diagonal(W, 0.0)
                    V[d] = 0.0; sp[d] = 0.0
            self.energies[dead] = E_INIT_N

        self.sp0, self.sp1, self.sp2 = f0f, f1f, f2f

    def update_world_model(self, obs_t, action, obs_t1):
        th = self.theta
        proj_t  = self.R @ obs_t
        proj_t1 = self.R @ obs_t1
        a_oh = np.zeros(N_ACT_HSNN); a_oh[action] = 1.0
        x = np.concatenate([proj_t, a_oh])
        delta = proj_t1 - self.P @ x
        pe_fast = float(np.mean(delta ** 2))
        self.P += th.eta * np.outer(delta, x)
        wm_alpha = 1.0 - th.tau_wm
        self.pe_slow = (1 - wm_alpha) * self.pe_slow + wm_alpha * pe_fast
        return pe_fast, self.pe_slow

    def blend_to_genome(self, genome: MiniHSNNGenome):
        for k in ('W0', 'W01', 'W1', 'W12', 'W2', 'W_act', 'Win', 'P'):
            g_w = getattr(genome, k)
            a_w = getattr(self, k)
            g_w[:] = (1 - BLEND_RATE) * g_w + BLEND_RATE * a_w

# ─────────────────────────────────────────────────────────────
# HSNN エピソード実行
# ─────────────────────────────────────────────────────────────
def run_hsnn_ep(genome: MiniHSNNGenome, theta: Theta, env, update=True):
    obs, _ = env.reset()
    agent = MiniHSNNAgent(genome, theta)

    try:
        visited = {tuple(env.unwrapped.agent_pos)}
    except Exception:
        visited = set()

    energy = 100.0
    pe_fast = pe_slow = 0.0
    e_scale = 0.0
    reached_goal = False
    prev_obs = obs.copy()

    for _ in range(MAX_STEPS):
        hsnn_act = agent.choose_action()
        mg_act   = HSNN_TO_MG[hsnn_act]

        try:
            prev_pos = tuple(env.unwrapped.agent_pos)
        except Exception:
            prev_pos = None

        obs_next, ext_rew, terminated, truncated, _ = env.step(mg_act)

        # 内部 ΔE
        delta_e = STEP_COST
        try:
            curr_pos = tuple(env.unwrapped.agent_pos)
            if prev_pos is not None and curr_pos == prev_pos:
                delta_e += WALL_COST        # 壁衝突
            elif curr_pos not in visited:
                delta_e += EXPLORE_GAIN     # 新マス
            visited.add(curr_pos)
        except Exception:
            pass

        if terminated and ext_rew > 0:
            delta_e += GOAL_GAIN
            reached_goal = True

        energy  += delta_e
        e_scale  = delta_e / E_LOSS_NORM

        agent.step_net(obs, pe_fast, pe_slow, e_scale)
        pe_fast, pe_slow = agent.update_world_model(prev_obs, hsnn_act, obs_next)
        prev_obs = obs.copy()
        obs = obs_next

        if energy <= 0 or terminated or truncated:
            break

    if update:
        agent.blend_to_genome(genome)

    return reached_goal

# ─────────────────────────────────────────────────────────────
# PPO (REINFORCE)
# ─────────────────────────────────────────────────────────────
class PolicyNet:
    def __init__(self, n_in, n_out=7, hidden=PPO_HIDDEN, lr=0.001, gamma=0.99):
        sc = 0.1 / np.sqrt(max(n_in, 1))
        self.Wh = np.random.randn(hidden, n_in) * sc
        self.bh = np.zeros(hidden)
        self.Wo = np.random.randn(n_out, hidden) * 0.1
        self.bo = np.zeros(n_out)
        self.lr = lr; self.gamma = gamma; self.n_out = n_out

    def forward(self, x):
        h = np.maximum(0.0, self.Wh @ x + self.bh)
        return h, self.Wo @ h + self.bo

    def act(self, x):
        h, logits = self.forward(x)
        logits -= logits.max()
        probs = np.exp(logits); probs /= (probs.sum() + 1e-12)
        idx = int(np.random.choice(self.n_out, p=probs))
        return idx, h, probs

    def update(self, traj):
        if not traj:
            return
        rewards = [t[2] for t in traj]
        G = 0.0; rets = []
        for r in reversed(rewards):
            G = r + self.gamma * G; rets.insert(0, G)
        rets = np.array(rets, dtype=float)
        if rets.std() > 1e-8:
            rets = (rets - rets.mean()) / rets.std()
        for (x, a, _, h, probs), G in zip(traj, rets):
            d2 = probs.copy(); d2[a] -= 1.0; d2 *= -G
            self.Wo -= self.lr * np.outer(d2, h); self.bo -= self.lr * d2
            dh = self.Wo.T @ d2 * (h > 0)
            self.Wh -= self.lr * np.outer(dh, x); self.bh -= self.lr * dh

def run_ppo_ep(agent: PolicyNet, env):
    obs, _ = env.reset()
    traj = []
    reached_goal = False
    for _ in range(MAX_STEPS):
        act, h, probs = agent.act(obs)
        obs_next, rew, terminated, truncated, _ = env.step(act)
        traj.append((obs.copy(), act, float(rew), h, probs))
        obs = obs_next
        if terminated and rew > 0:
            reached_goal = True
        if terminated or truncated:
            break
    agent.update(traj)
    return reached_goal

# ─────────────────────────────────────────────────────────────
# ランダム基準
# ─────────────────────────────────────────────────────────────
def run_random_ep(env):
    obs, _ = env.reset()
    reached_goal = False
    for _ in range(MAX_STEPS):
        act = env.action_space.sample()
        obs, rew, terminated, truncated, _ = env.step(act)
        if terminated and rew > 0:
            reached_goal = True
        if terminated or truncated:
            break
    return reached_goal

# ─────────────────────────────────────────────────────────────
# 事前訓練（6タスク × PRETRAIN_EP）
# ─────────────────────────────────────────────────────────────
def pretrain_hsnn(seed):
    """6タスクで訓練し、W0/W01/W1/W12/W2 を返す。"""
    import importlib, os
    sys.path.insert(0, r'C:\Users\girug\Downloads')
    try:
        import acceleration_experiment as ae
        importlib.reload(ae)   # module-level rng を再初期化

        np.random.seed(seed * 7 + 3)

        TASKS = [
            ae.RiskGridEnv(),
            ae.TimeSeriesRiskEnv(grid=5, n_mines=4),
            ae.InvertedControlEnv(grid=5, n_mines=4),
            ae.ResourceManageEnv(),
            ae.SequenceMemoryEnv(grid=5, n_mines=5),
            ae.PredatorEvadeEnv(),
        ]

        # 事前訓練用 θ（acceleration_experiment の Theta と同一パラメータ）
        pre_theta = ae.Theta()  # uses best θ defaults

        genome = ae.InfoDrivenGenome(25)
        genome.failure_rate = 0.033

        for task_env in TASKS:
            for _ in range(PRETRAIN_EP):
                ae.run_episode_hsnn(genome, task_env, pre_theta)

        return {k: getattr(genome, k).copy()
                for k in ('W0', 'W01', 'W1', 'W12', 'W2')}

    except Exception as e:
        print(f"  [pretrain warning] {e} → cold start で代替")
        return {}

# ─────────────────────────────────────────────────────────────
# 統計・ローリング平均
# ─────────────────────────────────────────────────────────────
def rolling_avg(arr, win=ROLLING_WIN):
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        s = max(0, i - win + 1)
        out[i] = np.mean(arr[s:i+1])
    return out

def adaptation_speed(rolling, threshold=THRESHOLD_RATE):
    for i, v in enumerate(rolling):
        if v >= threshold:
            return i
    return N_EPISODES

def compute_stats(results):
    stats = {}
    for m, seed_curves in results.items():
        mat = np.array(seed_curves)            # (n_seeds, n_ep)
        mean_c = mat.mean(axis=0)
        sem_c  = mat.std(axis=0) / np.sqrt(len(seed_curves))
        roll   = rolling_avg(mean_c)
        speed  = adaptation_speed(roll)
        final  = mean_c[-100:].mean() * 100
        stats[m] = dict(mean=mean_c, sem=sem_c, rolling=roll,
                        speed=speed, final_rate=final)
    return stats

# ─────────────────────────────────────────────────────────────
# 図1: 学習曲線
# ─────────────────────────────────────────────────────────────
def plot_learning_curve(stats):
    fig, ax = plt.subplots(figsize=(10, 6))
    COLOR = {'HSNN': '#2196F3', 'HSNN_pre': '#4CAF50',
             'PPO':  '#FF5722', 'Random':   '#9E9E9E'}
    LABEL = {'HSNN': 'HSNN (cold θ)', 'HSNN_pre': 'HSNN_pretrained',
             'PPO':  'PPO',           'Random':   'Random'}
    x = np.arange(N_EPISODES)
    for m, st in stats.items():
        ax.plot(x, st['rolling'], label=LABEL[m], color=COLOR[m], lw=2)
        lo = rolling_avg(np.maximum(st['mean'] - st['sem'], 0))
        hi = rolling_avg(np.minimum(st['mean'] + st['sem'], 1))
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.15)
    ax.axvline(500, color='black', ls='--', lw=1, label='ep=500')
    ax.axhline(THRESHOLD_RATE, color='gray', ls=':', lw=1.2,
               label=f'{THRESHOLD_RATE*100:.0f}%閾値')
    ax.set_xlabel('エピソード'); ax.set_ylabel('ゴール到達率 (rolling 50ep)')
    ax.set_title('MiniGrid-FourRooms 学習曲線: θ構造の効果')
    ax.legend(loc='upper left'); ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig('minigrid_fig1_learning_curve.png', dpi=150)
    plt.close()
    print("  -> minigrid_fig1_learning_curve.png")

# ─────────────────────────────────────────────────────────────
# 図2: 適応速度バーチャート
# ─────────────────────────────────────────────────────────────
def plot_adaptation_speed(stats):
    fig, ax = plt.subplots(figsize=(8, 5))
    COLOR = {'HSNN': '#2196F3', 'HSNN_pre': '#4CAF50',
             'PPO':  '#FF5722', 'Random':   '#9E9E9E'}
    LABEL = {'HSNN': 'HSNN\n(cold θ)', 'HSNN_pre': 'HSNN\npretrained',
             'PPO':  'PPO',            'Random':   'Random'}
    models = ['HSNN', 'HSNN_pre', 'PPO', 'Random']
    speeds = [stats[m]['speed'] for m in models]
    cols   = [COLOR[m] for m in models]
    bars = ax.bar(range(len(models)), speeds, color=cols, edgecolor='white')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABEL[m] for m in models])
    ax.set_ylabel('10%閾値到達エピソード数')
    ax.set_title('適応速度比較（少ないほど速い）')
    ax.axhline(N_EPISODES, color='red', ls='--', lw=1, alpha=0.6, label='未到達')
    for bar, spd in zip(bars, speeds):
        lbl = str(spd) if spd < N_EPISODES else 'N/A'
        ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height() + 15, N_EPISODES*0.95),
                lbl, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend(); ax.set_ylim(0, N_EPISODES * 1.12)
    plt.tight_layout()
    plt.savefig('minigrid_fig2_adaptation_speed.png', dpi=150)
    plt.close()
    print("  -> minigrid_fig2_adaptation_speed.png")

# ─────────────────────────────────────────────────────────────
# 図3: 前半/後半比較
# ─────────────────────────────────────────────────────────────
def plot_before_after(stats):
    fig, ax = plt.subplots(figsize=(9, 5))
    COLOR = {'HSNN': '#2196F3', 'HSNN_pre': '#4CAF50',
             'PPO':  '#FF5722', 'Random':   '#9E9E9E'}
    LABEL = {'HSNN': 'HSNN', 'HSNN_pre': 'HSNN_pre',
             'PPO':  'PPO',  'Random':   'Random'}
    models = ['HSNN', 'HSNN_pre', 'PPO', 'Random']
    x = np.arange(len(models)); w = 0.35
    first  = [stats[m]['mean'][:500].mean() * 100 for m in models]
    second = [stats[m]['mean'][500:].mean() * 100 for m in models]
    ax.bar(x - w/2, first,  w, label='前半 (ep 0-499)',
           color=[COLOR[m] for m in models], alpha=0.85)
    ax.bar(x + w/2, second, w, label='後半 (ep 500-999)',
           color=[COLOR[m] for m in models], alpha=0.45, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models])
    ax.set_ylabel('平均ゴール到達率 (%)'); ax.set_title('前半・後半の学習進捗比較')
    ax.legend()
    plt.tight_layout()
    plt.savefig('minigrid_fig3_before_after.png', dpi=150)
    plt.close()
    print("  -> minigrid_fig3_before_after.png")

# ─────────────────────────────────────────────────────────────
# 結果レポート
# ─────────────────────────────────────────────────────────────
def print_report(stats):
    print("\n=================================================================")
    print(" 実験サマリー")
    print("=================================================================")
    print(f"  {'モデル':20s} {'適応速度(ep)':>12s} {'最終到達率(%)':>14s}")
    print("  " + "-" * 50)
    for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random'):
        st = stats[m]
        spd = str(st['speed']) if st['speed'] < N_EPISODES else f"{N_EPISODES}(未到達)"
        print(f"  {m:20s} {spd:>12s} {st['final_rate']:>13.1f}%")

    q1 = stats['HSNN']['speed'] < stats['PPO']['speed']
    q2 = stats['HSNN_pre']['speed'] < stats['HSNN']['speed']
    q3 = stats['PPO']['speed'] < stats['Random']['speed']

    print(f"\n Q1: HSNNはPPOより少ないepで10%閾値に到達したか")
    print(f"   HSNN={stats['HSNN']['speed']}ep  PPO={stats['PPO']['speed']}ep")
    print(f"   → {'YES ✓ 学習効率が高い' if q1 else 'NO ✗'}")

    print(f"\n Q2: HSNN_pretrainedはHSNN_coldより速いか")
    print(f"   HSNN_pre={stats['HSNN_pre']['speed']}ep  HSNN_cold={stats['HSNN']['speed']}ep")
    print(f"   → {'YES ✓ 事前経験がMiniGridに転移した' if q2 else 'NO ✗'}")

    print(f"\n Q3: PPOはRandomより有意に速いか")
    print(f"   PPO={stats['PPO']['speed']}ep  Random={stats['Random']['speed']}ep")
    print(f"   → {'YES ✓ PPO実装は正常' if q3 else 'NO ✗ PPO実装に問題がある可能性'}")

    n_pass = sum([q1, q2, q3])
    print(f"\n 総合: {n_pass}/3 PASS")
    print("=================================================================")

# ─────────────────────────────────────────────────────────────
# メイン実験
# ─────────────────────────────────────────────────────────────
def run_experiment():
    t0 = time.time()

    # obs_dim 取得
    _env_tmp = FlatObsWrapper(gym.make('MiniGrid-FourRooms-v0'))
    obs_dim = _env_tmp.observation_space.shape[0]
    _env_tmp.close()
    print(f"obs_dim = {obs_dim}")

    print("=" * 65)
    print(" 「学習効率ベンチマーク：MiniGrid標準環境での θ初期構造の効果検証」")
    print(f" SEEDS={SEEDS}  N_EPISODES={N_EPISODES}  MAX_STEPS={MAX_STEPS}")
    print("=" * 65)

    results = {m: [] for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random')}

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n── SEED {seed} ──────────────────────────────────────────────")

        # 事前訓練（6タスク）
        print("  [事前訓練]", flush=True)
        pretrained_w = pretrain_hsnn(seed)
        print(f"  pretrain完了: {list(pretrained_w.keys())}")

        # 環境（モデルごとに独立）
        envs = {m: FlatObsWrapper(gym.make('MiniGrid-FourRooms-v0'))
                for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random')}

        # エージェント初期化
        genome_cold = MiniHSNNGenome(obs_dim, seed=seed)
        genome_pre  = MiniHSNNGenome(obs_dim, seed=seed)
        genome_pre.load_inter_weights(pretrained_w)
        ppo_agent   = PolicyNet(obs_dim)

        curves = {m: [] for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random')}

        for ep in range(N_EPISODES):
            curves['HSNN'].append(    float(run_hsnn_ep(genome_cold, BEST_THETA, envs['HSNN'])))
            curves['HSNN_pre'].append(float(run_hsnn_ep(genome_pre,  BEST_THETA, envs['HSNN_pre'])))
            curves['PPO'].append(     float(run_ppo_ep(ppo_agent, envs['PPO'])))
            curves['Random'].append(  float(run_random_ep(envs['Random'])))

            if (ep + 1) % 200 == 0:
                for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random'):
                    r50 = np.mean(curves[m][-50:]) * 100
                    print(f"  ep{ep+1:4d} {m:10s}: 直近50ep到達率={r50:.1f}%", flush=True)

        for env in envs.values():
            env.close()

        for m in ('HSNN', 'HSNN_pre', 'PPO', 'Random'):
            arr = np.array(curves[m])
            results[m].append(arr)
            roll = rolling_avg(arr)
            spd  = adaptation_speed(roll)
            final = arr[-100:].mean() * 100
            print(f"  {m:12s}: 適応速度={spd:4d}ep  最終到達率={final:.1f}%")

    elapsed = time.time() - t0
    print(f"\n実験完了 elapsed={elapsed:.0f}s")
    return results


if __name__ == '__main__':
    results = run_experiment()
    stats   = compute_stats(results)

    print("\n=== 図生成中 ===")
    plot_learning_curve(stats)
    plot_adaptation_speed(stats)
    plot_before_after(stats)
    print_report(stats)
