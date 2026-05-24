"""
hsnn_theta_extended.py
======================
「自律的な内発的動機付けの進化：何を報酬とみなすかをAI自身が決める」
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
# HSNN 定数
# ─────────────────────────────────────────────────────────────
N0, N1, N2 = 80, 40, 20
N_CONCAT   = N0 + N1 + N2   # 140
N_ACT_HSNN = 4
LEAK = 0.9; THRESHOLD = 1.0
E_INIT_N = 1.0; E_MAX_N = 2.0; E_FIRE = 0.05; E_DECAY = 0.005
E_LOSS_NORM = 30.0
TAU_PLUS  = [20.0, 30.0, 40.0]
TAU_MINUS = [20.0, 30.0, 40.0]
A_PLUS    = [0.01, 0.008, 0.005]
A_MINUS   = [0.012, 0.009, 0.006]
PROJ_DIM   = 32
PPO_HIDDEN = 64
BLEND_RATE = 0.3
HSNN_TO_MG = [2, 0, 1, 5]   # forward / left / right / toggle
W_GOAL     = 10.0            # ゴール報酬（固定）

# ─────────────────────────────────────────────────────────────
# 実験パラメータ
# ─────────────────────────────────────────────────────────────
N_POP      = 20
N_GEN      = 200
N_ELITE    = 5
N_EVAL     = 20
EARLY_STOP = 30
N_COMP_EP  = 300
MAX_STEPS  = 300
SEEDS      = [0, 1, 2, 3, 4]
ROLLING_WIN = 30
THRESH_RATE = 0.10

# ─────────────────────────────────────────────────────────────
# ThetaExtended
# ─────────────────────────────────────────────────────────────
BOUNDS = {
    'eta':       (0.005, 0.12),
    'beta':      (0.3,   4.0),
    'gamma':     (0.3,   4.0),
    'tau_trace': (0.4,   1.8),
    'tau_wm':    (0.80,  0.99),
    'w_explore': (0.0,   3.0),
    'w_wall':    (-2.0,  0.0),
    'w_step':    (-1.0,  0.0),
}
KEYS = list(BOUNDS.keys())

class ThetaExtended:
    def __init__(self, eta=0.0393, beta=2.190, gamma=2.039,
                 tau_trace=0.87, tau_wm=0.95,
                 w_explore=1.0, w_wall=-0.5, w_step=-0.1):
        self.eta=eta; self.beta=beta; self.gamma=gamma
        self.tau_trace=tau_trace; self.tau_wm=tau_wm
        self.w_explore=w_explore; self.w_wall=w_wall; self.w_step=w_step

    def as_array(self):
        return np.array([getattr(self, k) for k in KEYS])

    @classmethod
    def from_array(cls, arr):
        t = cls.__new__(cls)
        for k, v in zip(KEYS, arr):
            setattr(t, k, float(v))
        return t

    def clip(self):
        for k, (lo, hi) in BOUNDS.items():
            setattr(self, k, float(np.clip(getattr(self, k), lo, hi)))
        return self

    def copy(self):
        return ThetaExtended.from_array(self.as_array())

    def __repr__(self):
        return (f"ThetaExt(eta={self.eta:.4f} beta={self.beta:.3f} "
                f"gamma={self.gamma:.3f} tau_tr={self.tau_trace:.3f} "
                f"tau_wm={self.tau_wm:.3f} | "
                f"w_exp={self.w_explore:.3f} w_wall={self.w_wall:.3f} "
                f"w_step={self.w_step:.3f})")

# 従来best θ + 固定ΔE設計（比較用）
FIXED_THETA = ThetaExtended(w_explore=1.0, w_wall=-0.5, w_step=-0.1)

# ─────────────────────────────────────────────────────────────
# MiniGrid HSNN ゲノム
# ─────────────────────────────────────────────────────────────
class MiniHSNNGenome:
    def __init__(self, obs_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.Win  = rng.normal(0, 0.1,  (N0, obs_dim))
        self.W0   = rng.normal(0, 0.05, (N0, N0)); np.fill_diagonal(self.W0,  0.0)
        self.W01  = rng.normal(0, 0.05, (N1, N0))
        self.W1   = rng.normal(0, 0.05, (N1, N1)); np.fill_diagonal(self.W1,  0.0)
        self.W12  = rng.normal(0, 0.05, (N2, N1))
        self.W2   = rng.normal(0, 0.05, (N2, N2)); np.fill_diagonal(self.W2,  0.0)
        self.W_act = rng.normal(0, 0.01, (N_ACT_HSNN, N_CONCAT))
        R = rng.normal(0, 1.0, (PROJ_DIM, obs_dim))
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        self.R = R / (norms + 1e-8)
        self.P = np.zeros((PROJ_DIM, PROJ_DIM + N_ACT_HSNN))

# ─────────────────────────────────────────────────────────────
# MiniGrid HSNN エージェント
# ─────────────────────────────────────────────────────────────
class MiniHSNNAgent:
    def __init__(self, genome: MiniHSNNGenome, theta: ThetaExtended):
        self.theta = theta
        for k in ('Win','W0','W01','W1','W12','W2','W_act','R','P'):
            setattr(self, k, getattr(genome, k).copy())
        self.V0=np.zeros(N0); self.V1=np.zeros(N1); self.V2=np.zeros(N2)
        self.sp0=np.zeros(N0); self.sp1=np.zeros(N1); self.sp2=np.zeros(N2)
        self.energies = np.full(N0+N1+N2, E_INIT_N)
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
        self.V0 = LEAK*self.V0+I0; f0=self.V0>=THRESHOLD; self.V0[f0]=0.0; f0f=f0.astype(float)
        I1 = self.W01 @ sp0p + self.W1 @ sp1p
        self.V1 = LEAK*self.V1+I1; f1=self.V1>=THRESHOLD; self.V1[f1]=0.0; f1f=f1.astype(float)
        I2 = self.W12 @ sp1p + self.W2 @ sp2p
        self.V2 = LEAK*self.V2+I2; f2=self.V2>=THRESHOLD; self.V2[f2]=0.0; f2f=f2.astype(float)
        self.energies -= E_DECAY
        self.energies -= E_FIRE * np.concatenate([f0f,f1f,f2f])
        np.clip(self.energies, 0, E_MAX_N, out=self.energies)
        for li, fired, prev, pe_sig, W in [
            (0, f0f, sp0p, pe_l0*th.beta, self.W0),
            (1, f1f, sp1p, pe_l0*th.beta, self.W1),
            (2, f2f, sp2p, pe_l1,          self.W2),
        ]:
            dp = np.exp(-th.tau_trace/TAU_PLUS[li])
            dm = np.exp(-th.tau_trace/TAU_MINUS[li])
            self.tr_pre[li]  = self.tr_pre[li]*dp  + prev
            self.tr_post[li] = self.tr_post[li]*dm + fired
            mag = pe_sig * abs(energy_scale) * th.gamma
            dW = (A_PLUS[li] if energy_scale>=0 else -A_MINUS[li]) * mag * np.outer(fired, self.tr_pre[li])
            W += dW; np.fill_diagonal(W, 0.0)
        dead = self.energies <= 0
        if dead.any():
            d0,d1,d2 = dead[:N0], dead[N0:N0+N1], dead[N0+N1:]
            for d,W,V,sp in [(d0,self.W0,self.V0,self.sp0),
                              (d1,self.W1,self.V1,self.sp1),
                              (d2,self.W2,self.V2,self.sp2)]:
                if d.any():
                    nd=d.sum(); nw=W.shape[0]
                    W[d,:]=np.random.normal(0,0.05,(nd,nw))
                    W[:,d]=np.random.normal(0,0.05,(nw,nd))
                    np.fill_diagonal(W,0.0); V[d]=0.0; sp[d]=0.0
            self.energies[dead] = E_INIT_N
        self.sp0, self.sp1, self.sp2 = f0f, f1f, f2f

    def update_world_model(self, obs_t, action, obs_t1):
        th = self.theta
        proj_t  = self.R @ obs_t
        proj_t1 = self.R @ obs_t1
        a_oh = np.zeros(N_ACT_HSNN); a_oh[action]=1.0
        x = np.concatenate([proj_t, a_oh])
        delta = proj_t1 - self.P @ x
        pe_fast = float(np.mean(delta**2))
        self.P += th.eta * np.outer(delta, x)
        wm_a = 1.0 - th.tau_wm
        self.pe_slow = (1-wm_a)*self.pe_slow + wm_a*pe_fast
        return pe_fast, self.pe_slow

    def blend_to_genome(self, genome: MiniHSNNGenome):
        for k in ('W0','W01','W1','W12','W2','W_act','Win','P'):
            g_w=getattr(genome,k); a_w=getattr(self,k)
            g_w[:] = (1-BLEND_RATE)*g_w + BLEND_RATE*a_w

# ─────────────────────────────────────────────────────────────
# HSNN エピソード実行（ThetaExtended）
# ─────────────────────────────────────────────────────────────
def run_hsnn_ep(genome: MiniHSNNGenome, theta: ThetaExtended, env, update=True):
    obs, _ = env.reset()
    agent  = MiniHSNNAgent(genome, theta)
    try:    visited = {tuple(env.unwrapped.agent_pos)}
    except: visited = set()
    energy = 100.0
    pe_fast = pe_slow = e_scale = 0.0
    reached = False
    prev_obs = obs.copy()
    for _ in range(MAX_STEPS):
        hsnn_act = agent.choose_action()
        mg_act   = HSNN_TO_MG[hsnn_act]
        try:    prev_pos = tuple(env.unwrapped.agent_pos)
        except: prev_pos = None
        obs_next, ext_rew, terminated, truncated, _ = env.step(mg_act)
        delta_e = theta.w_step
        try:
            curr_pos = tuple(env.unwrapped.agent_pos)
            if   prev_pos is not None and curr_pos == prev_pos: delta_e += theta.w_wall
            elif curr_pos not in visited:                        delta_e += theta.w_explore
            visited.add(curr_pos)
        except: pass
        if terminated and ext_rew > 0:
            delta_e += W_GOAL; reached = True
        energy  += delta_e
        e_scale  = delta_e / E_LOSS_NORM
        agent.step_net(obs, pe_fast, pe_slow, e_scale)
        pe_fast, pe_slow = agent.update_world_model(prev_obs, hsnn_act, obs_next)
        prev_obs = obs.copy(); obs = obs_next
        if energy <= 0 or terminated or truncated: break
    if update:
        agent.blend_to_genome(genome)
    return reached

# ─────────────────────────────────────────────────────────────
# PPO (REINFORCE) + Random
# ─────────────────────────────────────────────────────────────
class PolicyNet:
    def __init__(self, n_in, n_out=7, hidden=PPO_HIDDEN, lr=0.001, gamma=0.99):
        sc = 0.1/np.sqrt(max(n_in,1))
        self.Wh=np.random.randn(hidden,n_in)*sc; self.bh=np.zeros(hidden)
        self.Wo=np.random.randn(n_out,hidden)*0.1; self.bo=np.zeros(n_out)
        self.lr=lr; self.gamma=gamma; self.n_out=n_out
    def forward(self,x):
        h=np.maximum(0.0,self.Wh@x+self.bh); return h, self.Wo@h+self.bo
    def act(self,x):
        h,logits=self.forward(x); logits-=logits.max()
        probs=np.exp(logits); probs/=(probs.sum()+1e-12)
        return int(np.random.choice(self.n_out,p=probs)), h, probs
    def update(self,traj):
        if not traj: return
        rewards=[t[2] for t in traj]; G=0.0; rets=[]
        for r in reversed(rewards): G=r+self.gamma*G; rets.insert(0,G)
        rets=np.array(rets,dtype=float)
        if rets.std()>1e-8: rets=(rets-rets.mean())/rets.std()
        for (x,a,_,h,probs),G in zip(traj,rets):
            d2=probs.copy(); d2[a]-=1.0; d2*=-G
            self.Wo-=self.lr*np.outer(d2,h); self.bo-=self.lr*d2
            dh=self.Wo.T@d2*(h>0)
            self.Wh-=self.lr*np.outer(dh,x); self.bh-=self.lr*dh

def run_ppo_ep(agent:PolicyNet, env):
    obs,_=env.reset(); traj=[]; reached=False
    for _ in range(MAX_STEPS):
        act,h,probs=agent.act(obs)
        obs_next,rew,terminated,truncated,_=env.step(act)
        traj.append((obs.copy(),act,float(rew),h,probs)); obs=obs_next
        if terminated and rew>0: reached=True
        if terminated or truncated: break
    agent.update(traj); return reached

def run_random_ep(env):
    obs,_=env.reset(); reached=False
    for _ in range(MAX_STEPS):
        obs,rew,terminated,truncated,_=env.step(env.action_space.sample())
        if terminated and rew>0: reached=True
        if terminated or truncated: break
    return reached

# ─────────────────────────────────────────────────────────────
# GA 操作関数
# ─────────────────────────────────────────────────────────────
def _random_theta(rng):
    t = ThetaExtended()
    t.eta       = float(np.clip(rng.normal(0.0393,0.010), *BOUNDS['eta']))
    t.beta      = float(np.clip(rng.normal(2.190, 0.30),  *BOUNDS['beta']))
    t.gamma     = float(np.clip(rng.normal(2.039, 0.30),  *BOUNDS['gamma']))
    t.tau_trace = float(np.clip(rng.normal(0.870, 0.05),  *BOUNDS['tau_trace']))
    t.tau_wm    = float(np.clip(rng.normal(0.950, 0.03),  *BOUNDS['tau_wm']))
    t.w_explore = float(rng.uniform(*BOUNDS['w_explore']))
    t.w_wall    = float(rng.uniform(*BOUNDS['w_wall']))
    t.w_step    = float(rng.uniform(*BOUNDS['w_step']))
    return t

def _tournament(pop, fits, k, rng):
    idx  = rng.choice(len(pop), k, replace=False)
    best = idx[np.argmax([fits[i] for i in idx])]
    return pop[best].copy()

def _crossover(p1:ThetaExtended, p2:ThetaExtended, rng):
    a1=p1.as_array(); a2=p2.as_array()
    mask=rng.random(len(KEYS))<0.5
    return ThetaExtended.from_array(np.where(mask, a1, a2))

def _mutate(t:ThetaExtended, rng):
    arr=t.as_array()
    for i,(k,(lo,hi)) in enumerate(BOUNDS.items()):
        arr[i] += rng.normal(0, 0.05*(hi-lo))
    return ThetaExtended.from_array(arr).clip()

# ─────────────────────────────────────────────────────────────
# 適応度評価
# ─────────────────────────────────────────────────────────────
def evaluate_theta(theta:ThetaExtended, obs_dim:int, env, genome_seed:int):
    genome = MiniHSNNGenome(obs_dim, seed=genome_seed)
    goals  = sum(run_hsnn_ep(genome, theta, env, update=True) for _ in range(N_EVAL))
    return goals

# ─────────────────────────────────────────────────────────────
# 進化メインループ
# ─────────────────────────────────────────────────────────────
def evolve(obs_dim):
    rng = np.random.default_rng(42)
    pop = [_random_theta(rng) for _ in range(N_POP)]

    fit_hist   = {'best':[], 'mean':[], 'worst':[]}
    theta_hist = {k: {'mean':[], 'std':[]} for k in ('w_explore','w_wall','w_step')}

    best_fit   = -1
    no_improve = 0
    best_theta = pop[0].copy()

    eval_env = FlatObsWrapper(gym.make('MiniGrid-FourRooms-v0'))

    for gen in range(N_GEN):
        fits = [evaluate_theta(t, obs_dim, eval_env, gen*N_POP+i)
                for i, t in enumerate(pop)]

        order = np.argsort(fits)[::-1]
        pop   = [pop[o] for o in order]
        fits  = [fits[o] for o in order]

        fit_hist['best'].append(fits[0])
        fit_hist['mean'].append(float(np.mean(fits)))
        fit_hist['worst'].append(fits[-1])

        top5 = pop[:N_ELITE]
        for k in ('w_explore','w_wall','w_step'):
            vals = [getattr(t, k) for t in top5]
            theta_hist[k]['mean'].append(float(np.mean(vals)))
            theta_hist[k]['std'].append(float(np.std(vals)))

        if fits[0] > best_fit:
            best_fit   = fits[0]
            best_theta = pop[0].copy()
            no_improve = 0
        else:
            no_improve += 1

        if gen % 20 == 0 or gen < 3:
            print(f"  Gen {gen:3d}: best={fits[0]}/{N_EVAL} "
                  f"mean={np.mean(fits):.1f} | {pop[0]}", flush=True)

        if no_improve >= EARLY_STOP:
            print(f"  早期終了 gen={gen}  ({EARLY_STOP}世代改善なし)")
            break

        # 次世代生成
        next_pop = [t.copy() for t in pop[:N_ELITE]]
        while len(next_pop) < N_POP:
            p1    = _tournament(pop, fits, k=3, rng=rng)
            p2    = _tournament(pop, fits, k=3, rng=rng)
            child = _mutate(_crossover(p1, p2, rng), rng)
            next_pop.append(child)
        pop = next_pop

    eval_env.close()
    n_gen = len(fit_hist['best'])
    print(f"\n  進化完了: {n_gen}世代  best_fit={best_fit}/{N_EVAL}")
    print(f"  最良θ: {best_theta}")
    return best_theta, fit_hist, theta_hist

# ─────────────────────────────────────────────────────────────
# 比較実験
# ─────────────────────────────────────────────────────────────
def rolling_avg(arr, win=None):
    if win is None: win = ROLLING_WIN
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        s = max(0, i-win+1); out[i] = np.mean(arr[s:i+1])
    return out

def adaptation_speed(roll, threshold=THRESH_RATE):
    for i, v in enumerate(roll):
        if v >= threshold: return i
    return N_COMP_EP

def run_comparison(best_theta:ThetaExtended, obs_dim:int):
    results = {m:[] for m in ('HSNN_evolved','HSNN_fixed','PPO','Random')}
    for seed in SEEDS:
        np.random.seed(seed)
        print(f"  SEED {seed}: ", end='', flush=True)
        envs = {m: FlatObsWrapper(gym.make('MiniGrid-FourRooms-v0'))
                for m in ('HSNN_evolved','HSNN_fixed','PPO','Random')}
        g_evo = MiniHSNNGenome(obs_dim, seed=seed)
        g_fix = MiniHSNNGenome(obs_dim, seed=seed)
        ppo   = PolicyNet(obs_dim)
        curves = {m:[] for m in ('HSNN_evolved','HSNN_fixed','PPO','Random')}
        for ep in range(N_COMP_EP):
            curves['HSNN_evolved'].append(float(run_hsnn_ep(g_evo, best_theta,  envs['HSNN_evolved'])))
            curves['HSNN_fixed'].append(  float(run_hsnn_ep(g_fix, FIXED_THETA, envs['HSNN_fixed'])))
            curves['PPO'].append(         float(run_ppo_ep(ppo,  envs['PPO'])))
            curves['Random'].append(      float(run_random_ep(envs['Random'])))
        for env in envs.values(): env.close()
        for m, c in curves.items():
            arr = np.array(c); results[m].append(arr)
            print(f"{m}={arr[-50:].mean()*100:.1f}%  ", end='', flush=True)
        print()
    return results

def compute_stats(results):
    stats = {}
    for m, seed_curves in results.items():
        mat   = np.array(seed_curves)
        mean_c= mat.mean(axis=0); sem_c=mat.std(axis=0)/np.sqrt(len(seed_curves))
        roll  = rolling_avg(mean_c)
        stats[m] = dict(mean=mean_c, sem=sem_c, rolling=roll,
                        speed=adaptation_speed(roll),
                        final_rate=mean_c[-50:].mean()*100)
    return stats

# ─────────────────────────────────────────────────────────────
# 図1: 適応度の世代推移
# ─────────────────────────────────────────────────────────────
def plot_fitness(fit_hist):
    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(fit_hist['best']))
    ax.plot(x, fit_hist['best'],  label='Best',  color='#2196F3', lw=2)
    ax.plot(x, fit_hist['mean'],  label='Mean',  color='#FF9800', lw=1.5, ls='--')
    ax.plot(x, fit_hist['worst'], label='Worst', color='#9E9E9E', lw=1,   ls=':')
    ax.axhline(N_EVAL*0.1, color='red', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('Generation'); ax.set_ylabel(f'Goal count / {N_EVAL} ep')
    ax.set_title('Fitness over Generations — intrinsic motivation GA')
    ax.legend(); ax.set_ylim(0, N_EVAL)
    plt.tight_layout()
    plt.savefig('theta_ext_fig1_fitness.png', dpi=150); plt.close()
    print("  -> theta_ext_fig1_fitness.png")

# ─────────────────────────────────────────────────────────────
# 図2: 内発的動機付けの収束（最重要）
# ─────────────────────────────────────────────────────────────
def plot_intrinsic(theta_hist):
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    params = [
        ('w_explore', 'w_explore (new cell bonus)',  '#4CAF50', BOUNDS['w_explore']),
        ('w_wall',    'w_wall (wall penalty)',        '#FF5722', BOUNDS['w_wall']),
        ('w_step',    'w_step (step cost)',           '#2196F3', BOUNDS['w_step']),
    ]
    x = np.arange(len(theta_hist['w_explore']['mean']))
    for ax, (k, title, color, (lo, hi)) in zip(axes, params):
        m = np.array(theta_hist[k]['mean'])
        s = np.array(theta_hist[k]['std'])
        ax.plot(x, m, color=color, lw=2, label='top-5 mean')
        ax.fill_between(x, m-s, m+s, color=color, alpha=0.25, label='+/-std')
        ax.axhline(lo, color='gray', ls=':', lw=0.8)
        ax.axhline(hi, color='gray', ls=':', lw=0.8)
        ax.axhline(0,  color='black', lw=0.5, alpha=0.4)
        ax.set_xlabel('Generation'); ax.set_title(title)
        ax.set_ylim(lo-0.15, hi+0.15); ax.legend(fontsize=8)
    plt.suptitle('Convergence of Intrinsic Motivation Parameters (top-5 elite, mean +/- std)')
    plt.tight_layout()
    plt.savefig('theta_ext_fig2_intrinsic.png', dpi=150, bbox_inches='tight'); plt.close()
    print("  -> theta_ext_fig2_intrinsic.png")

# ─────────────────────────────────────────────────────────────
# 図3: 最終性能比較
# ─────────────────────────────────────────────────────────────
def plot_performance(stats):
    fig, ax = plt.subplots(figsize=(10,6))
    COLOR = {'HSNN_evolved':'#4CAF50','HSNN_fixed':'#2196F3',
             'PPO':'#FF5722','Random':'#9E9E9E'}
    LABEL = {'HSNN_evolved':'HSNN evolved theta',
             'HSNN_fixed':  'HSNN fixed dE (best theta)',
             'PPO':         'PPO (external reward)',
             'Random':      'Random'}
    x = np.arange(N_COMP_EP)
    for m, st in stats.items():
        ax.plot(x, st['rolling'], label=LABEL[m], color=COLOR[m], lw=2)
        lo = rolling_avg(np.maximum(st['mean']-st['sem'],0))
        hi = rolling_avg(np.minimum(st['mean']+st['sem'],1))
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.15)
    ax.axhline(THRESH_RATE, color='gray', ls=':', lw=1.2, label='10% threshold')
    ax.set_xlabel('Episode'); ax.set_ylabel(f'Goal reach rate (rolling {ROLLING_WIN}ep)')
    ax.set_title('Final Performance: evolved vs fixed intrinsic motivation')
    ax.legend(loc='upper left'); ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig('theta_ext_fig3_performance.png', dpi=150); plt.close()
    print("  -> theta_ext_fig3_performance.png")

# ─────────────────────────────────────────────────────────────
# 図4: 収束したθ_extendedの値
# ─────────────────────────────────────────────────────────────
def plot_theta_values(best_theta:ThetaExtended):
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    w = 0.35
    # 左: 従来θ
    t_keys = ['eta','beta','gamma','tau_trace','tau_wm']
    evo_t  = [best_theta.eta, best_theta.beta, best_theta.gamma,
               best_theta.tau_trace, best_theta.tau_wm]
    fix_t  = [0.0393, 2.190, 2.039, 0.87, 0.95]
    x = np.arange(len(t_keys))
    axes[0].bar(x-w/2, fix_t, w, label='fixed best theta', color='#2196F3', alpha=0.75)
    axes[0].bar(x+w/2, evo_t, w, label='evolved theta',    color='#4CAF50', alpha=0.75)
    axes[0].set_xticks(x); axes[0].set_xticklabels(t_keys)
    axes[0].set_title('Traditional theta: evolved vs fixed')
    axes[0].legend()
    # 右: 内発的動機付け
    i_keys  = ['w_explore','w_wall','w_step']
    evo_i   = [best_theta.w_explore, best_theta.w_wall, best_theta.w_step]
    fix_i   = [1.0, -0.5, -0.1]
    x2 = np.arange(len(i_keys))
    axes[1].bar(x2-w/2, fix_i, w, label='fixed design', color='#2196F3', alpha=0.75)
    axes[1].bar(x2+w/2, evo_i, w, label='evolved',      color='#4CAF50', alpha=0.75)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(i_keys)
    axes[1].set_title('Intrinsic motivation: evolved vs fixed design')
    axes[1].legend()
    plt.suptitle('Evolved ThetaExtended vs Human-designed defaults')
    plt.tight_layout()
    plt.savefig('theta_ext_fig4_theta_values.png', dpi=150); plt.close()
    print("  -> theta_ext_fig4_theta_values.png")

# ─────────────────────────────────────────────────────────────
# 結果レポート
# ─────────────────────────────────────────────────────────────
def print_report(best_theta:ThetaExtended, stats):
    print("\n=================================================================")
    print(" 「自律的な内発的動機付けの進化」実験サマリー")
    print("=================================================================")
    print("\n 収束したθ_extended:")
    print(f"   eta={best_theta.eta:.4f}  beta={best_theta.beta:.3f}  "
          f"gamma={best_theta.gamma:.3f}  tau_trace={best_theta.tau_trace:.3f}  "
          f"tau_wm={best_theta.tau_wm:.3f}")
    print(f"   w_explore={best_theta.w_explore:.3f}  "
          f"w_wall={best_theta.w_wall:.3f}  "
          f"w_step={best_theta.w_step:.3f}")
    print()
    print(f"  {'Model':20s} {'Speed(ep)':>10s} {'Final reach(%)':>15s}")
    print("  " + "-"*48)
    for m in ('HSNN_evolved','HSNN_fixed','PPO','Random'):
        st  = stats[m]
        spd = str(st['speed']) if st['speed']<N_COMP_EP else f"{N_COMP_EP}(N/A)"
        print(f"  {m:20s} {spd:>10s} {st['final_rate']:>14.1f}%")

    q1 = best_theta.w_explore > 0.1
    q2 = stats['HSNN_evolved']['final_rate'] > stats['HSNN_fixed']['final_rate']
    q3_e = best_theta.w_explore > 0
    q3_w = best_theta.w_wall    < 0
    q3_s = best_theta.w_step    < 0
    q3   = q3_e and q3_w and q3_s

    print(f"\n Q1: w_exploreが0より大きい値に収束したか")
    print(f"   w_explore = {best_theta.w_explore:.3f}")
    print(f"   → {'YES ✓ 探索ボーナスが自然に生まれた' if q1 else 'NO ✗ 探索ボーナスは生まれなかった'}")

    print(f"\n Q2: 進化θが固定ΔE設計より到達率が高いか")
    print(f"   evolved={stats['HSNN_evolved']['final_rate']:.1f}%  "
          f"fixed={stats['HSNN_fixed']['final_rate']:.1f}%")
    print(f"   → {'YES ✓ 自律設計が人間設計を上回った' if q2 else 'NO ✗'}")

    print(f"\n Q3: 収束したw_explore/w_wall/w_stepは合理的か")
    print(f"   w_explore>0: {'YES' if q3_e else 'NO'} ({best_theta.w_explore:.3f})")
    print(f"   w_wall<0:    {'YES' if q3_w else 'NO'} ({best_theta.w_wall:.3f})")
    print(f"   w_step<0:    {'YES' if q3_s else 'NO'} ({best_theta.w_step:.3f})")
    print(f"   → {'YES ✓ 直感的に合理的な動機付けに収束' if q3 else 'NO ✗ 一部が非合理的'}")

    n_pass = sum([q1, q2, q3])
    print(f"\n 総合: {n_pass}/3 PASS")
    print("=================================================================")

# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    _tmp = FlatObsWrapper(gym.make('MiniGrid-FourRooms-v0'))
    obs_dim = _tmp.observation_space.shape[0]; _tmp.close()
    print(f"obs_dim = {obs_dim}")
    print("="*65)
    print(" 「自律的な内発的動機付けの進化」実験開始")
    print(f" N_POP={N_POP} N_GEN={N_GEN} N_EVAL={N_EVAL} SEEDS={SEEDS}")
    print("="*65)

    print("\n=== Phase 1: θ_extended 進化 ===")
    best_theta, fit_hist, theta_hist = evolve(obs_dim)

    print("\n=== Phase 2: 最終比較実験 ===")
    results = run_comparison(best_theta, obs_dim)
    stats   = compute_stats(results)

    print(f"\n実験完了 elapsed={time.time()-t0:.0f}s")
    print("\n=== 図生成中 ===")
    plot_fitness(fit_hist)
    plot_intrinsic(theta_hist)
    plot_performance(stats)
    plot_theta_values(best_theta)
    print_report(best_theta, stats)
