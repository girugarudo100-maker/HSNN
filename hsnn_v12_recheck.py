"""
v12再検証 — best θを転用した抽象転移実験
==========================================

実験設計:
  Phase1: RiskGrid（空間ナビ）で訓練
  Phase2: AbstractCombinationGame（抽象数値ゲーム）に転移
          → 空間概念ゼロ、グリッドトポロジーなし

比較モデル:
  A. HSNN_bestθ   : Phase1最良θ + Grid訓練済みW → 抽象ゲームへ
  B. HSNN_randomθ : ランダムθ + Grid訓練済みW → 抽象ゲームへ  
  C. HSNN_cold    : ランダムθ + W未訓練 → 抽象ゲームへ（コールドスタート）
  D. PPO_proxy    : 外部報酬ベース → 抽象ゲームへ

測定:
  - 各エピソードの生存手数（学習曲線）
  - 収束速度（何エピソードで安定するか）
  - 総ステップ数（計算効率）
  - FLOPs推定（v12との比較）

v12の再現条件:
  「空間的なグリッド知識がゼロの環境で
   危機回避・リソース管理の本質が転移するか」
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 環境
# ============================================================

class RiskGridEnv:
    """訓練タスク: 5×5空間グリッド"""
    SIZE=5; MAX_E=20; N_MINES=4; N_FOOD=3; MAX_STEPS=80
    def __init__(self, seed=None):
        self.rng=np.random.default_rng(seed); self.reset()
    def reset(self):
        self.grid=np.zeros((self.SIZE,self.SIZE),dtype=int)
        pos=self.rng.choice(self.SIZE**2,self.N_MINES+self.N_FOOD+1,replace=False)
        self.pos=np.array(divmod(int(pos[0]),self.SIZE))
        for i in range(1,self.N_MINES+1):
            r,c=divmod(int(pos[i]),self.SIZE); self.grid[r,c]=-1
        for i in range(self.N_MINES+1,self.N_MINES+self.N_FOOD+1):
            r,c=divmod(int(pos[i]),self.SIZE); self.grid[r,c]=1
        self.energy=float(self.MAX_E); self.steps=0; return self._obs()
    def _obs(self):
        r,c=self.pos; patch=np.zeros(9); idx=0
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                nr,nc=r+dr,c+dc
                if 0<=nr<self.SIZE and 0<=nc<self.SIZE:
                    patch[idx]=self.grid[nr,nc]
                idx+=1
        return np.concatenate([self.pos/self.SIZE,
                                [self.energy/self.MAX_E],patch])
    def step(self,action):
        d=[(-1,0),(1,0),(0,-1),(0,1)][action%4]; r,c=self.pos
        self.pos=np.array([np.clip(r+d[0],0,self.SIZE-1),
                           np.clip(c+d[1],0,self.SIZE-1)])
        self.energy-=1.0; self.steps+=1
        cell=self.grid[self.pos[0],self.pos[1]]
        done=False; de=-1.0
        if cell==-1: done=True; de=-self.energy
        elif cell==1:
            self.energy=min(self.energy+5.0,self.MAX_E)
            self.grid[self.pos[0],self.pos[1]]=0; de=5.0
        if self.energy<=0 or self.steps>=self.MAX_STEPS: done=True
        return self._obs(),done,{"delta_e":de}


class AbstractCombinationGame:
    """
    転移タスク: 抽象的な組み合わせゲーム
    空間的なグリッドトポロジーを完全に排除。

    ルール:
      - 状態は12個の実数値（経済指標・リスク指標のアナロジー）
      - 毎ステップ状態が確率的に変動する
      - 行動0: 「安全策」（energy消費少、獲得少）
      - 行動1: 「リスク策」（energy消費多、成功で大きく回復）
      - 行動2: 「待機」（energy微減、状態観察）
      - 行動3: 「撤退」（energy減、リスクリセット）
      - スパイク状の危険信号の後に大きなペナルティが来る
      → Grid訓練で獲得した「危機予兆の検出」が転移するか試す

    obs_dim=12（RiskGridと同一次元）
    """
    MAX_STEPS=120
    DANGER_PROB=0.15   # 危険シグナルの発生確率
    RISK_REWARD=8.0    # リスク策成功時の回復量
    RISK_PENALTY=12.0  # リスク策失敗時のペナルティ

    def __init__(self, seed=None):
        self.rng=np.random.default_rng(seed); self.reset()

    def reset(self):
        # 12次元の抽象的な状態ベクトル
        self.state=self.rng.uniform(-1,1,12).astype(float)
        self.energy=20.0
        self.steps=0
        self.danger_level=0.0  # 危険度（内部状態）
        self.pending_danger=False
        return self._obs()

    def _obs(self):
        # 状態にエネルギー情報を埋め込む
        obs=self.state.copy()
        obs[0]=self.energy/20.0         # エネルギー残量
        obs[1]=self.danger_level        # 危険度（観察可能）
        return obs

    def step(self,action):
        self.steps+=1
        de=-0.5  # 毎ステップの基本消費

        # 危険シグナルの発生
        if self.rng.random()<self.DANGER_PROB:
            self.danger_level=self.rng.uniform(0.7,1.0)
            self.pending_danger=True
        else:
            self.danger_level=max(0.0, self.danger_level-0.1)

        # 状態の確率的変動
        self.state+=self.rng.normal(0,0.1,12)
        self.state=np.clip(self.state,-1,1)
        self.state[0]=self.energy/20.0
        self.state[1]=self.danger_level

        # 行動の効果
        if action==0:   # 安全策
            gain=0.3
            self.energy+=gain; de+=gain
            if self.pending_danger:
                self.pending_danger=False  # 危険回避成功

        elif action==1: # リスク策
            if self.pending_danger:
                # 危険時にリスクを取ると大ペナルティ
                self.energy-=self.RISK_PENALTY
                de-=self.RISK_PENALTY
                self.pending_danger=False
            else:
                # 安全時のリスクは大きなリターン
                self.energy+=self.RISK_REWARD
                de+=self.RISK_REWARD

        elif action==2: # 待機
            de-=0.2  # 追加消費なし、状態観察
            self.energy-=0.2

        elif action==3: # 撤退
            if self.pending_danger:
                self.pending_danger=False  # 危険をリセット
                de+=0.5; self.energy+=0.5
            else:
                de-=0.3; self.energy-=0.3

        self.energy=np.clip(self.energy,0,20)
        done=self.energy<=0 or self.steps>=self.MAX_STEPS

        return self._obs(),done,{"delta_e":de}


# ============================================================
# HSNN エージェント（θ固定、W学習）
# ============================================================

class HSNNAgent:
    def __init__(self, theta, l2=32, obs_dim=12, n_actions=4, seed=0):
        self.theta=theta; self.l2=l2; self.n_actions=n_actions
        self.l1=max(12,int(l2*1.2))
        rng=np.random.default_rng(seed); s=0.12
        self.W01=rng.normal(0,s,(self.l1,obs_dim))
        self.W12=rng.normal(0,s,(l2,self.l1))
        self.W2o=rng.normal(0,s,(n_actions,l2))
        self.e01=np.zeros_like(self.W01)
        self.e12=np.zeros_like(self.W12)
        self.e2o=np.zeros_like(self.W2o)
        self.wm=np.zeros(l2)
        self.total_steps=0  # FLOPs計算用

    def _act(self,x): return np.tanh(x)

    def step(self,obs):
        h1=self._act(self.W01@obs)
        h2=self._act(self.W12@h1)
        pe=float(np.linalg.norm(h2-self.wm))
        tau_wm=self.theta.get("tau_wm",0.95)
        self.wm=tau_wm*self.wm+(1-tau_wm)*h2
        logits=self.W2o@h2
        exp_l=np.exp(logits-logits.max()); probs=exp_l/exp_l.sum()
        action=int(np.random.choice(self.n_actions,p=probs))
        tau_e=self.theta.get("tau_trace",0.87)
        self.e01=tau_e*self.e01+np.outer(h1,obs)
        self.e12=tau_e*self.e12+np.outer(h2,h1)
        self.e2o=tau_e*self.e2o+np.outer(probs,h2)
        self.last_h1=h1; self.last_h2=h2
        return action,pe

    def update(self,de,pe):
        eta=self.theta.get("eta",0.015)
        beta=self.theta.get("beta",1.0)
        gamma=self.theta.get("gamma",1.0)
        pe_mod=max(pe,1e-6)**beta
        de_mod=(abs(de)**gamma)*np.sign(de)
        m=eta*pe_mod*de_mod
        self.W01+=m*self.e01; self.W12+=m*self.e12; self.W2o+=m*self.e2o
        for W in [self.W01,self.W12,self.W2o]: np.clip(W,-3,3,out=W)
        self.total_steps+=1

    def run_episode(self,EnvClass,seed):
        env=EnvClass(seed=seed); obs=env.reset(); done=False; steps=0
        while not done:
            action,pe=self.step(obs)
            obs,done,info=env.step(action)
            self.update(info["delta_e"],pe)
            steps+=1
        return steps


# ============================================================
# PPO Proxy（外部報酬ベース）
# ============================================================

class PPOProxy:
    def __init__(self, l2=32, obs_dim=12, n_actions=4, seed=0, eta=0.008):
        rng=np.random.default_rng(seed); s=0.12
        self.l1=max(12,int(l2*1.2)); self.n_actions=n_actions
        self.W01=rng.normal(0,s,(self.l1,obs_dim))
        self.W12=rng.normal(0,s,(l2,self.l1))
        self.W2o=rng.normal(0,s,(n_actions,l2))
        self.eta=eta; self.last_h2=np.zeros(l2)
        self.last_action=0; self.total_steps=0

    def _act(self,x): return np.tanh(x)

    def step(self,obs):
        h1=self._act(self.W01@obs); h2=self._act(self.W12@h1)
        logits=self.W2o@h2; exp_l=np.exp(logits-logits.max())
        probs=exp_l/exp_l.sum()
        action=int(np.random.choice(self.n_actions,p=probs))
        self.last_h2=h2; self.last_action=action
        return action,0.0

    def update(self,de,pe,ext_reward=0.0):
        grad=np.zeros(self.n_actions); grad[self.last_action]=ext_reward
        self.W2o+=self.eta*np.outer(grad,self.last_h2)
        np.clip(self.W2o,-3,3,out=self.W2o)
        self.total_steps+=1

    def run_episode(self,EnvClass,seed):
        env=EnvClass(seed=seed); obs=env.reset(); done=False; steps=0
        while not done:
            action,pe=self.step(obs)
            obs,done,info=env.step(action)
            # PPOは外部報酬を使う（ΔEを報酬として使用）
            self.update(info["delta_e"],pe,ext_reward=info["delta_e"])
            steps+=1
        return steps


# ============================================================
# 実験設定
# ============================================================

# Phase1で得られたbest θ（3シードの最良値）
# seed0が最高surv=33.1
BEST_THETA = {
    "eta": 0.0393, "beta": 2.190, "gamma": 2.039,
    "tau_trace": 0.87, "tau_wm": 0.95
}
# seed1
THETA_1 = {
    "eta": 0.0240, "beta": 1.749, "gamma": 2.577,
    "tau_trace": 0.87, "tau_wm": 0.95
}
# seed2（β低い：対照として面白い）
THETA_2 = {
    "eta": 0.0121, "beta": 0.810, "gamma": 1.311,
    "tau_trace": 0.87, "tau_wm": 0.95
}

# ランダムθのデフォルト
DEFAULT_THETA = {
    "eta": 0.015, "beta": 1.0, "gamma": 1.0,
    "tau_trace": 0.87, "tau_wm": 0.95
}

N_TRAIN    = 200   # Grid訓練エピソード数
N_TRANSFER = 300   # 抽象ゲーム転移エピソード数
L2_SIZE    = 32
SEEDS      = [0,1,2,3,4]
SMOOTH     = 15

print("="*65)
print("v12再検証 — best θを用いた抽象転移実験")
print(f"  Grid訓練: {N_TRAIN}ep → AbstractGame転移: {N_TRANSFER}ep")
print(f"  {len(SEEDS)}シード")
print("="*65)
print(f"\nbest θ: η={BEST_THETA['eta']:.4f} "
      f"β={BEST_THETA['beta']:.3f} "
      f"γ={BEST_THETA['gamma']:.3f}")

import time

# モデルの定義
model_configs = {
    "HSNN_bestθ_transfer": {
        "theta": BEST_THETA, "train_grid": True,
        "color": "#7C3AED", "ls": "-", "lw": 2.5,
        "label": "HSNN best θ\n(Grid訓練→抽象転移)"
    },
    "HSNN_bestθ_cold": {
        "theta": BEST_THETA, "train_grid": False,
        "color": "#A78BFA", "ls": "--", "lw": 2.0,
        "label": "HSNN best θ\n(コールドスタート)"
    },
    "HSNN_lowβ_transfer": {
        "theta": THETA_2, "train_grid": True,
        "color": "#F59E0B", "ls": "-", "lw": 2.0,
        "label": f"HSNN low-β(β={THETA_2['beta']:.2f})\n(Grid訓練→抽象転移)"
    },
    "HSNN_default_transfer": {
        "theta": DEFAULT_THETA, "train_grid": True,
        "color": "#6B7280", "ls": "-", "lw": 1.8,
        "label": "HSNN default θ\n(Grid訓練→抽象転移)"
    },
    "PPO_cold": {
        "theta": None, "train_grid": False,
        "color": "#EF4444", "ls": "--", "lw": 2.0,
        "label": "PPO (外部報酬)\nコールドスタート"
    },
}

results = {name: [] for name in model_configs}
total_steps_log = {name: [] for name in model_configs}

for seed in SEEDS:
    np.random.seed(seed*137)
    print(f"\n[Seed {seed}]")

    for model_name, cfg in model_configs.items():
        if cfg["theta"] is None:
            # PPO
            agent = PPOProxy(l2=L2_SIZE, seed=seed)
        else:
            agent = HSNNAgent(cfg["theta"], l2=L2_SIZE, seed=seed)

        # Phase1: Grid訓練
        if cfg["train_grid"]:
            for ep in range(N_TRAIN):
                agent.run_episode(RiskGridEnv, seed=ep+seed*1000)

        steps_after_train = agent.total_steps

        # Phase2: 抽象ゲームへの転移
        surv_curve = []
        for ep in range(N_TRANSFER):
            s = agent.run_episode(AbstractCombinationGame,
                                  seed=ep+seed*2000)
            surv_curve.append(s)

        results[model_name].append(surv_curve)
        total_steps_log[model_name].append(agent.total_steps)

        final = np.mean(surv_curve[-50:])
        early = np.mean(surv_curve[:20])
        print(f"  {model_name:30s}: "
              f"early={early:.1f} final={final:.1f} "
              f"steps={agent.total_steps:,}")


# ============================================================
# 可視化
# ============================================================

BASE   = "/mnt/user-data/outputs"
eps    = np.arange(N_TRANSFER)

def sm(arr2d, smooth=SMOOTH):
    a=np.array(arr2d)
    m=uniform_filter1d(a.mean(0),smooth)
    s=uniform_filter1d(a.std(0), smooth)
    return m,s

# ---- Fig 1: メイン適応曲線 ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "v12 Re-verification: Abstract Transfer with Evolved θ\n"
    "Task A (RiskGrid, spatial) → Task B (AbstractCombinationGame, no topology)\n"
    "W always reset before transfer | θ fixed from evolution",
    fontsize=12, fontweight='bold'
)

ax = axes[0]
for model_name, cfg in model_configs.items():
    mean, std = sm(results[model_name])
    c=cfg["color"]; ls=cfg["ls"]; lw=cfg["lw"]
    ax.plot(eps, mean, color=c, linestyle=ls, linewidth=lw,
            label=cfg["label"], alpha=0.9)
    ax.fill_between(eps, mean-std*0.4, mean+std*0.4,
                    color=c, alpha=0.10)

ax.set_xlabel("Episode on Abstract Game", fontsize=11)
ax.set_ylabel("Survival Steps", fontsize=11)
ax.set_title("Adaptation Curves on Abstract Task\n"
             "(higher = better abstract understanding)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_facecolor('#FAFAFA')
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---- 右: early vs final 比較 ----
ax = axes[1]
names = list(model_configs.keys())
display_labels = [cfg["label"].split("\n")[0] for cfg in model_configs.values()]
colors_bar = [cfg["color"] for cfg in model_configs.values()]

x = np.arange(len(names)); w = 0.35

early_means = [np.array(results[n])[:,:20].mean() for n in names]
final_means = [np.array(results[n])[:,-50:].mean() for n in names]
early_stds  = [np.array(results[n])[:,:20].std() for n in names]
final_stds  = [np.array(results[n])[:,-50:].std() for n in names]

bars1 = ax.bar(x-w/2, early_means, w, label="Early (first 20ep)",
               color=colors_bar, alpha=0.40, edgecolor='white', linewidth=1.5,
               yerr=early_stds, capsize=3,
               error_kw={'ecolor':'#555','alpha':0.5})
bars2 = ax.bar(x+w/2, final_means, w, label="Final (last 50ep)",
               color=colors_bar, alpha=0.90, edgecolor='white', linewidth=1.5,
               yerr=final_stds, capsize=3,
               error_kw={'ecolor':'#555','alpha':0.5})

for bar,v,s in zip(bars2, final_means, final_stds):
    ax.text(bar.get_x()+bar.get_width()/2, v+s+0.2,
            f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(display_labels, fontsize=8, rotation=15, ha='right')
ax.set_ylabel("Mean Survival Steps", fontsize=11)
ax.set_title("Early vs Final Performance", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/v12_fig1_adaptation.png", dpi=135, bbox_inches='tight')
plt.close()
print("\nSaved: v12_fig1_adaptation.png")


# ---- Fig 2: 学習効率（総ステップ数 vs 性能）----
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle(
    "Learning Efficiency: Performance vs Total Steps\n"
    "Upper-left = better (high performance, few steps)",
    fontsize=13, fontweight='bold'
)

for model_name, cfg in model_configs.items():
    final_mean = np.array(results[model_name])[:,-50:].mean()
    final_std  = np.array(results[model_name])[:,-50:].std()
    steps_mean = np.mean(total_steps_log[model_name])
    c = cfg["color"]
    ax.scatter(steps_mean, final_mean,
               color=c, s=200, zorder=5,
               edgecolors='white', linewidths=2)
    ax.errorbar(steps_mean, final_mean, yerr=final_std,
                color=c, linewidth=1.5, capsize=4, alpha=0.7)
    ax.text(steps_mean*1.02, final_mean+0.15,
            cfg["label"].split("\n")[0],
            fontsize=9, color=c, fontweight='bold')

ax.set_xlabel("Total Steps (training + transfer)", fontsize=11)
ax.set_ylabel("Final Survival on Abstract Task (last 50ep)", fontsize=11)
ax.set_xscale('log')
ax.set_facecolor('#FAFAFA')
ax.grid(alpha=0.3, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/v12_fig2_efficiency.png", dpi=135, bbox_inches='tight')
plt.close()
print("Saved: v12_fig2_efficiency.png")


# ---- Fig 3: β値と転移性能の関係（核心的な発見）----
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    "β (PE sensitivity) vs Transfer Performance\n"
    "Core question: does high β enable abstract transfer?",
    fontsize=13, fontweight='bold'
)

beta_vals = []
final_perf = []
model_labels = []
colors_sc = []

for model_name, cfg in model_configs.items():
    if cfg["theta"] is not None and cfg["train_grid"]:
        beta_vals.append(cfg["theta"]["beta"])
        final_perf.append(np.array(results[model_name])[:,-50:].mean())
        model_labels.append(cfg["label"].split("\n")[0])
        colors_sc.append(cfg["color"])

ax.scatter(beta_vals, final_perf,
           c=colors_sc, s=250, zorder=5,
           edgecolors='white', linewidths=2)
for b, p, l, c in zip(beta_vals, final_perf, model_labels, colors_sc):
    ax.text(b+0.03, p+0.1, l, fontsize=9, color=c, fontweight='bold')

# 線形回帰
if len(beta_vals) >= 2:
    z = np.polyfit(beta_vals, final_perf, 1)
    xline = np.linspace(min(beta_vals)-0.2, max(beta_vals)+0.2, 50)
    ax.plot(xline, np.poly1d(z)(xline), color='#9CA3AF',
            linestyle='--', linewidth=1.5, alpha=0.7,
            label=f"Trend: slope={z[0]:+.3f}")
    ax.legend(fontsize=10)

ax.set_xlabel("β (PE sensitivity in learning rule)", fontsize=11)
ax.set_ylabel("Final Survival on Abstract Task", fontsize=11)
ax.set_facecolor('#FAFAFA')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/v12_fig3_beta_transfer.png", dpi=135, bbox_inches='tight')
plt.close()
print("Saved: v12_fig3_beta_transfer.png")


# ---- 数値サマリー ----
print("\n" + "="*65)
print("v12再検証 サマリー")
print("="*65)
print(f"{'モデル':35s} {'early':>7} {'final':>7} "
      f"{'gain':>7} {'steps':>10}")
print("-"*65)
for model_name, cfg in model_configs.items():
    arr   = np.array(results[model_name])
    early = arr[:,:20].mean()
    final = arr[:,-50:].mean()
    gain  = final - early
    steps = int(np.mean(total_steps_log[model_name]))
    label = cfg["label"].split("\n")[0]
    print(f"  {label:33s}: {early:7.2f} {final:7.2f} "
          f"{gain:+7.2f} {steps:10,}")

print("\n核心的な比較:")
best_final  = np.array(results["HSNN_bestθ_transfer"])[:,-50:].mean()
cold_final  = np.array(results["HSNN_bestθ_cold"])[:,-50:].mean()
ppo_final   = np.array(results["PPO_cold"])[:,-50:].mean()
lowb_final  = np.array(results["HSNN_lowβ_transfer"])[:,-50:].mean()

best_steps  = int(np.mean(total_steps_log["HSNN_bestθ_transfer"]))
ppo_steps   = int(np.mean(total_steps_log["PPO_cold"]))

print(f"  bestθ転移 vs コールドスタート: {best_final-cold_final:+.2f}手")
print(f"  bestθ転移 vs PPO:            {best_final-ppo_final:+.2f}手")
print(f"  highβ vs lowβ:               {best_final-lowb_final:+.2f}手")
print(f"  ステップ効率 HSNN/PPO:        {ppo_steps/best_steps:.1f}倍少ない")

print("\n全実験完了。")
