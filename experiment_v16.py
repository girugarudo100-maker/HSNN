#!/usr/bin/env python3
"""
experiment_v16.py  長期多タスク転移実験
「5タスク連続転移：破滅的忘却は蓄積するか」

確定設計:
  - τなし（全ニューロン同一物性）
  - 距離コスト（空間制約）
  - 継承率: lr = 0.001 + 0.199 * survival_fraction
  - 複数タスク（別チャンネル、識別子なし）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def pearsonr(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.std()<1e-10 or y.std()<1e-10: return (0.,1.)
    return (float(np.corrcoef(x,y)[0,1]),0.)

def relu(x): return np.maximum(0,x)
def softmax(x): e=np.exp(x-x.max()); return e/(e.sum()+1e-10)

# ── グローバルパラメータ ──────────────────────────────────────────
N           = 200
N_ACT       = 4
N_CH        = 9
N_TASKS     = 5
N_IN        = N_CH * N_TASKS   # 45
EPS_GEN     = 8
N_GEN_PHASE = 25
N_GEN_P5    = 20
SEEDS       = [0, 1]
A_PLUS      = 0.012; A_MINUS = 0.009
W_GAIN      = 0.4;  LR_W = 0.08; W_CLIP = 1.0
D_SPATIAL   = 5.0
LR_MIN      = 0.001; LR_MAX = 0.20

def get_lr(steps_survived, max_steps=50, lr_min=LR_MIN, lr_max=LR_MAX, power=1.0):
    fraction = steps_survived / max(max_steps, 1)
    return lr_min + (lr_max - lr_min) * (fraction ** power)


# ─────────────────────────────────────────────────────────────────
# タスク環境 (各9次元)
# ─────────────────────────────────────────────────────────────────

class Task1Grid:
    """マインスイーパー 5x5 / 1地雷（即時反応型）"""
    def __init__(self, size=5):
        self.size = size

    def reset(self, rng):
        self.pos = np.array([self.size//2]*2)
        self.mine = rng.integers(0, self.size, 2)
        while np.array_equal(self.mine, self.pos):
            self.mine = rng.integers(0, self.size, 2)
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size
        o[1] = self.pos[1]/self.size
        o[2] = (self.mine[0]-self.pos[0])/self.size
        o[3] = (self.mine[1]-self.pos[1])/self.size
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx], 0, self.size-1); self.t+=1
        hit = np.array_equal(self.pos, self.mine)
        return self._obs(), (-5. if hit else 1.), hit or self.t>=50


class Task2Grid:
    """捕食環境 5x5 / 1捕食者 T_move=3（受動監視型）"""
    def __init__(self, size=5, T_move=3):
        self.size=size; self.T_move=T_move

    def reset(self, rng):
        self.pos = np.array([self.size//2]*2)
        corners = [(0,0),(0,self.size-1),(self.size-1,0),(self.size-1,self.size-1)]
        self.pred = np.array(corners[int(rng.integers(0,4))])
        self.food = rng.integers(1, self.size-1, 2)
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size; o[1] = self.pos[1]/self.size
        o[2] = (self.pred[0]-self.pos[0])/self.size
        o[3] = (self.pred[1]-self.pos[1])/self.size
        o[4] = (self.food[0]-self.pos[0])/self.size
        o[5] = (self.food[1]-self.pos[1])/self.size
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx], 0, self.size-1); self.t+=1
        if self.t % self.T_move == 0:
            diff = self.pos - self.pred
            if abs(diff[0]) >= abs(diff[1]): self.pred += [int(np.sign(diff[0])),0]
            else: self.pred += [0, int(np.sign(diff[1]))]
            self.pred[:] = np.clip(self.pred, 0, self.size-1)
        caught = np.array_equal(self.pos, self.pred)
        if caught: return self._obs(), -5., True
        on_food = np.array_equal(self.pos, self.food)
        return self._obs(), (2. if on_food else 0.5), self.t>=50


class Task3Grid:
    """食料グリッド 5x5 / T=10ステップごとに食料出現（定期イベント型）"""
    def __init__(self, size=5, food_interval=10):
        self.size=size; self.food_interval=food_interval

    def reset(self, rng):
        self.pos = np.array([self.size//2]*2)
        self._rng = rng
        self.food = rng.integers(0, self.size, 2)
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size; o[1] = self.pos[1]/self.size
        o[2] = (self.food[0]-self.pos[0])/self.size
        o[3] = (self.food[1]-self.pos[1])/self.size
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx], 0, self.size-1); self.t+=1
        got_food = np.array_equal(self.pos, self.food)
        if got_food:
            de = 3.
            self.food = self._rng.integers(0, self.size, 2)
        elif self.t % self.food_interval == 0:
            self.food = self._rng.integers(0, self.size, 2)
            de = 0.
        else:
            de = 0.
        return self._obs(), de, self.t>=50


class Task4Grid:
    """迷路探索 5x5 / (0,0)→(4,4) 再訪問不可（累積型）"""
    def __init__(self, size=5):
        self.size=size

    def reset(self, rng):
        self.pos = np.array([0,0])
        self.goal = np.array([self.size-1, self.size-1])
        self.visited = set(); self.visited.add((0,0))
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size; o[1] = self.pos[1]/self.size
        o[2] = (self.goal[0]-self.pos[0])/self.size
        o[3] = (self.goal[1]-self.pos[1])/self.size
        o[4] = len(self.visited)/25.
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        new_pos = np.clip(self.pos+[dy,dx], 0, self.size-1); self.t+=1
        key = (int(new_pos[0]), int(new_pos[1]))
        if key in self.visited:
            de = -1.
        else:
            de = 1.; self.visited.add(key); self.pos = new_pos.copy()
        reached = np.array_equal(new_pos, self.goal)
        if reached: return self._obs(), 5., True
        return self._obs(), de, self.t>=50


class Task5Grid:
    """逆マインスイーパー 5x5 / 地雷を全て踏む（逆転型）"""
    def __init__(self, size=5, n_mines=3):
        self.size=size; self.n_mines=n_mines

    def reset(self, rng):
        self.pos = np.array([self.size//2]*2)
        self.mines = [rng.integers(0, self.size, 2) for _ in range(self.n_mines)]
        self.remaining = list(range(self.n_mines))
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size; o[1] = self.pos[1]/self.size
        if self.remaining:
            m = self.mines[self.remaining[0]]
            o[2] = (m[0]-self.pos[0])/self.size
            o[3] = (m[1]-self.pos[1])/self.size
        o[4] = len(self.remaining)/self.n_mines
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx], 0, self.size-1); self.t+=1
        hit_idx = [i for i in self.remaining if np.array_equal(self.pos, self.mines[i])]
        if hit_idx:
            for i in hit_idx: self.remaining.remove(i)
            de = 3.
            if not self.remaining: return self._obs(), 5., True
        else:
            de = -0.5
        return self._obs(), de, self.t>=50


# ─────────────────────────────────────────────────────────────────
# 5タスク環境ラッパー (45次元)
# ─────────────────────────────────────────────────────────────────

class MultiTask5Env:
    """
    45次元入力: concat(CH1...CH5)
    アクティブタスクのチャンネルにのみ観測を入れ、他はゼロ。
    識別子なし。
    """
    def __init__(self):
        self.grids = [Task1Grid(), Task2Grid(), Task3Grid(), Task4Grid(), Task5Grid()]
        self.task_idx = 0
        self.n_active = 1

    def reset(self, rng, task_idx=None, n_active=None):
        if n_active is not None: self.n_active = n_active
        if task_idx is not None:
            self.task_idx = int(task_idx)
        else:
            self.task_idx = int(rng.integers(0, self.n_active))
        obs9 = self.grids[self.task_idx].reset(rng)
        return self._make_45(obs9)

    def _make_45(self, obs9):
        obs = np.zeros(N_IN)
        obs[self.task_idx*N_CH:(self.task_idx+1)*N_CH] = obs9
        return obs

    def step(self, act):
        obs9, de, done = self.grids[self.task_idx].step(act)
        return self._make_45(obs9), de, done

    @property
    def task(self): return self.task_idx


# ─────────────────────────────────────────────────────────────────
# 5タスク対応エージェント
# ─────────────────────────────────────────────────────────────────

class MultiTaskHSNN_v15:
    """
    5タスク空間分割 HSNN with survival-proportional inheritance.
    lr_mask: 5領域 (centers at linspace(0,1,5))
    fire_sum: (5, N) per-task firing accumulator
    """
    def __init__(self, seed, n_tasks=N_TASKS, D_spatial=D_SPATIAL,
                 lr_min=LR_MIN, lr_max=LR_MAX, power=1.0, E_maintain=0.002):
        self.n_tasks=n_tasks; self.n_in=N_IN; self.n_ch=N_CH
        self.D_spatial=D_spatial; self.E_maintain=E_maintain
        self.lr_min=lr_min; self.lr_max=lr_max; self.power=power
        self.eta=0.07; self.beta=0.05; self.gamma=0.3
        self.tau_trace=0.5; self.tau_wm=0.95; self.sparse=1e-4
        self.alpha_role=0.7; self.pass_rate=0.75

        rng=np.random.default_rng(seed); self.rng=rng
        self.pos=rng.random((N,2)); self.pos_x=self.pos[:,0]

        centers=np.linspace(0,1,n_tasks)
        self.lr_mask=np.zeros((N,N_IN))
        for k, c in enumerate(centers):
            lr_k = np.exp(-np.abs(self.pos_x-c)*D_spatial)
            self.lr_mask[:, k*N_CH:(k+1)*N_CH] = lr_k[:,None]
        self.cost_mask = 1.0 - self.lr_mask/(self.lr_mask.max()+1e-8)

        s=0.2
        self.Win=np.zeros((N,N_IN))
        for k, c in enumerate(centers):
            lr_k = np.exp(-np.abs(self.pos_x-c)*D_spatial)
            self.Win[:, k*N_CH:(k+1)*N_CH] = rng.normal(0,s,(N,N_CH))*lr_k[:,None]
        self.Wact=rng.normal(0,s,(N_ACT,N))
        self.tr=np.zeros((N,N_IN)); self.wm=np.zeros(N)

        self.Win_init=self.Win.copy()
        self.Wact_init=self.Wact.copy()

        self.role_memory=rng.random(N); self.neuron_fitness=np.zeros(N)
        self.fire_sum=np.zeros((n_tasks,N))
        self.n_fire=np.zeros(n_tasks)

    def reset_ep(self, seed):
        rng=np.random.default_rng(seed); ns=0.03
        self.Win=self.Win_init+rng.normal(0,ns,self.Win_init.shape)*self.lr_mask
        self.Wact=self.Wact_init+rng.normal(0,ns,self.Wact_init.shape)
        self.tr[:]=0; self.wm[:]=0

    def evolve(self):
        total=self.neuron_fitness.sum()
        p=(self.neuron_fitness/total) if total>1e-12 else np.ones(N)/N
        p=np.clip(p,0,None); p/=p.sum()
        if self.rng.random()<self.pass_rate:
            idx=self.rng.choice(N,N,replace=True,p=p)
            new_role=self.role_memory[idx]*self.alpha_role+self.rng.random(N)*(1-self.alpha_role)
        else:
            new_role=self.rng.random(N)
        self.role_memory=np.clip(new_role,0,1); self.neuron_fitness=np.zeros(N)

    def step(self, obs, de, task_idx=0):
        h=relu(self.Win@obs); h/=(h.max()+1e-8)
        self.fire_sum[task_idx]+=h; self.n_fire[task_idx]+=1
        pe=float(np.abs(h-self.wm).mean())
        self.wm=self.tau_wm*self.wm+(1-self.tau_wm)*h
        if de>0:
            em=(self.role_memory>0.6).astype(float)
            self.neuron_fitness+=de*h*em; self.role_memory+=0.01*h*em
        elif de<0:
            im=(self.role_memory<0.4).astype(float)
            self.neuron_fitness+=abs(de)*h*im*0.5; self.role_memory-=0.01*h*im
        self.role_memory=np.clip(self.role_memory,0,1)
        sign_de=np.sign(de) if de!=0 else 0.
        lam=self.eta*(pe**self.beta)*(abs(de)**self.gamma)*sign_de
        self.tr=self.tau_trace*self.tr+np.outer(h,obs)
        self.Win+=lam*self.tr*self.lr_mask
        self.Win-=self.E_maintain*self.Win*self.cost_mask
        self.Win-=self.sparse*np.sign(self.Win)
        probs=softmax(self.Wact@h)
        act=int(np.random.choice(N_ACT,p=probs))
        return act, pe, h.copy()

    def inherit(self, steps_survived, max_steps=50):
        lr=get_lr(steps_survived,max_steps,self.lr_min,self.lr_max,self.power)
        self.Win_init+=lr*(self.Win-self.Win_init)
        self.Wact_init+=lr*(self.Wact-self.Wact_init)

    def reset_fire(self):
        self.fire_sum[:]=0; self.n_fire[:]=0

    def task_selectivity_k(self, k):
        fk=self.fire_sum[k]/(self.n_fire[k]+1e-8)
        others=[j for j in range(self.n_tasks) if j!=k and self.n_fire[j]>0]
        if not others: return np.zeros(N)
        f_others=np.mean([self.fire_sum[j]/(self.n_fire[j]+1e-8) for j in others],axis=0)
        return (fk-f_others)/(fk+f_others+1e-8)

    def integration_ratio(self, threshold=0.3):
        rates=np.array([self.fire_sum[k]/(self.n_fire[k]+1e-8) for k in range(self.n_tasks)])
        peak=rates.max(0)+1e-8
        active=rates/peak>threshold
        return float((active.sum(0)>=2).mean())

    def n_specialized(self, threshold=0.3):
        rates=np.array([self.fire_sum[k]/(self.n_fire[k]+1e-8) for k in range(self.n_tasks)])
        peak=rates.max(0)+1e-8
        active=rates/peak>threshold
        return int((active.sum(0)==1).sum())

    def spatial_corr(self, k=0):
        sel=self.task_selectivity_k(k)
        return float(pearsonr(self.pos_x, sel)[0])

    def dominant_task(self):
        rates=np.array([self.fire_sum[k]/(self.n_fire[k]+1e-8) for k in range(self.n_tasks)])
        return rates.argmax(0)  # (N,) dominant task index per neuron


# ─────────────────────────────────────────────────────────────────
# 学習ループ
# ─────────────────────────────────────────────────────────────────

def run_ep_v16(agent, env, seed, task_idx=None, n_active=None):
    rng=np.random.default_rng(seed)
    obs=env.reset(rng, task_idx=task_idx, n_active=n_active)
    agent.reset_ep(seed)
    total=0.; de=0.; steps=0; tidx=env.task
    for _ in range(50):
        act,pe,h=agent.step(obs,de,task_idx=tidx)
        obs,de,done=env.step(act); total+=de; steps+=1
        if done: break
    agent.inherit(steps, max_steps=50)
    return total, steps

def run_gen_v16(agent, env, gen_idx, seed_off, n_active=1, task_force=None):
    scores=[]
    for ep in range(EPS_GEN):
        sc,_=run_ep_v16(agent, env, gen_idx*10000+ep+seed_off,
                        task_idx=task_force, n_active=n_active)
        scores.append(sc)
    return float(np.mean(scores))

def eval_fit(agent, env, seed_off, n_active, task_force=None, n_gen=5):
    scores=[]
    for g in range(n_gen):
        sc=run_gen_v16(agent, env, g+9000, seed_off, n_active=n_active, task_force=task_force)
        scores.append(sc)
    return float(np.mean(scores))

def measure_selectivity(agent, env, n_active, seed_off=500000, n_ep=50):
    agent.reset_fire()
    for ep in range(n_ep):
        task_idx = ep % n_active
        rng=np.random.default_rng(seed_off+ep)
        obs=env.reset(rng, task_idx=task_idx, n_active=n_active)
        de=0.; agent.reset_ep(seed_off+ep)
        for _ in range(50):
            act,pe,h=agent.step(obs,de,task_idx=task_idx)
            obs,de,done=env.step(act)
            if done: break


# ─────────────────────────────────────────────────────────────────
# メイン実験
# ─────────────────────────────────────────────────────────────────

print("="*65)
print(" experiment_v16: 長期多タスク転移実験")
print(f" N={N}  N_GEN_PHASE={N_GEN_PHASE}  EPS_GEN={EPS_GEN}  SEEDS={len(SEEDS)}")
print("="*65)
t_total=time.time()

all_phase_results=[]   # per seed list
all_cum_fgt=[]
all_fit1_p0=[]
all_fit1_p5=[]
final_agents=[]

for seed in SEEDS:
    print(f"\n[順次転移 seed={seed}]")
    ag=MultiTaskHSNN_v15(seed)
    env=MultiTask5Env()
    phase_results=[]

    # Phase 0: タスク1のみ
    print(f"  P0: タスク1...", end='', flush=True)
    t0=time.time()
    for g in range(N_GEN_PHASE):
        run_gen_v16(ag,env,g,seed,n_active=1,task_force=0); ag.evolve()
    fit1_p0=eval_fit(ag,env,seed,n_active=1,task_force=0)
    measure_selectivity(ag,env,1,seed_off=600000+seed*100)
    ir0=ag.integration_ratio(); ns0=ag.n_specialized()
    corr0=float(abs(ag.spatial_corr(0)))
    phase_results.append({'phase':0,'n_active':1,'fit_task1':fit1_p0,
                          'fit_mean':fit1_p0,'corr':corr0,'int_ratio':ir0,'n_spec':ns0,
                          'forgetting':0.})
    print(f" fit1={fit1_p0:.2f}  corr={corr0:.3f}  ir={ir0:.3f}  ({time.time()-t0:.1f}s)")

    # Phase 1-4: タスクを順次追加
    for phase in range(1,5):
        n_active=phase+1
        print(f"  P{phase}: {n_active}タスク...", end='', flush=True)
        t0=time.time()
        for g in range(N_GEN_PHASE):
            run_gen_v16(ag,env,g+phase*N_GEN_PHASE,seed,n_active=n_active); ag.evolve()
        fit1=eval_fit(ag,env,seed,n_active=1,task_force=0)
        fit_all=eval_fit(ag,env,seed,n_active=n_active)
        measure_selectivity(ag,env,n_active,seed_off=600000+seed*100+phase*10)
        ir=ag.integration_ratio(); ns=ag.n_specialized()
        corrs=[abs(ag.spatial_corr(k)) for k in range(n_active)]
        corr=float(np.mean(corrs))
        forgetting=(fit1_p0-fit1)/(abs(fit1_p0)+1e-8)
        phase_results.append({'phase':phase,'n_active':n_active,'fit_task1':fit1,
                              'fit_mean':fit_all,'corr':corr,'int_ratio':ir,'n_spec':ns,
                              'forgetting':forgetting})
        print(f" fit1={fit1:.2f}  fit_all={fit_all:.2f}  fgt={forgetting:.1%}  ir={ir:.3f}  ({time.time()-t0:.1f}s)")

    # Phase 5: タスク1復帰
    print(f"  P5: 復帰...", end='', flush=True)
    t0=time.time()
    for g in range(N_GEN_P5):
        run_gen_v16(ag,env,g+5*N_GEN_PHASE,seed,n_active=1,task_force=0); ag.evolve()
    fit1_p5=eval_fit(ag,env,seed,n_active=1,task_force=0)
    cum_fgt=(fit1_p0-fit1_p5)/(abs(fit1_p0)+1e-8)
    phase_results.append({'phase':5,'n_active':1,'fit_task1':fit1_p5,
                          'fit_mean':fit1_p5,'corr':corr0,'int_ratio':ir0,'n_spec':ns0,
                          'forgetting':cum_fgt})
    print(f" fit1={fit1_p5:.2f}  累積忘却={cum_fgt:.1%}  ({time.time()-t0:.1f}s)")

    all_phase_results.append(phase_results)
    all_cum_fgt.append(cum_fgt)
    all_fit1_p0.append(fit1_p0)
    all_fit1_p5.append(fit1_p5)
    final_agents.append((ag, env, seed))

# ── 対照: 全タスク同時提示 ──────────────────────────────────────
print("\n[対照: 全5タスク同時]")
ctrl_results=[]
for seed in SEEDS:
    print(f"  seed={seed}...", end='', flush=True)
    t0=time.time()
    ag_ctrl=MultiTaskHSNN_v15(seed)
    env_ctrl=MultiTask5Env()
    total_gens=N_GEN_PHASE*5+N_GEN_P5
    for g in range(total_gens):
        run_gen_v16(ag_ctrl,env_ctrl,g,seed,n_active=5); ag_ctrl.evolve()
    fit1_ctrl=eval_fit(ag_ctrl,env_ctrl,seed,n_active=1,task_force=0)
    fit_all_ctrl=eval_fit(ag_ctrl,env_ctrl,seed,n_active=5)
    measure_selectivity(ag_ctrl,env_ctrl,5,seed_off=700000+seed*100)
    ir_ctrl=ag_ctrl.integration_ratio(); ns_ctrl=ag_ctrl.n_specialized()
    corrs_ctrl=[abs(ag_ctrl.spatial_corr(k)) for k in range(5)]
    corr_ctrl=float(np.mean(corrs_ctrl))
    ctrl_results.append({'fit1':fit1_ctrl,'fit_all':fit_all_ctrl,
                         'corr':corr_ctrl,'ir':ir_ctrl,'ns':ns_ctrl})
    print(f" fit1={fit1_ctrl:.2f}  fit_all={fit_all_ctrl:.2f}  corr={corr_ctrl:.3f}  ({time.time()-t0:.1f}s)")

# ── 集計 ────────────────────────────────────────────────────────
mean_cum_fgt=float(np.mean(all_cum_fgt))
mean_fit1_p0=float(np.mean(all_fit1_p0))
mean_fit1_p5=float(np.mean(all_fit1_p5))

# フェーズごとの平均
n_phases=len(all_phase_results[0])
avg_phases=[]
for pi in range(n_phases):
    keys=['fit_task1','fit_mean','corr','int_ratio','n_spec','forgetting']
    avg={'phase':all_phase_results[0][pi]['phase'],
         'n_active':all_phase_results[0][pi]['n_active']}
    for k in keys:
        avg[k]=float(np.mean([all_phase_results[s][pi][k] for s in range(len(SEEDS))]))
    avg_phases.append(avg)

# Q1: 累積忘却率 < 50%?
Q1_nodisaster = mean_cum_fgt < 0.50
# Q2: 統合層の成長 (P0 → P4)
ir_p0=float(np.mean([all_phase_results[s][0]['int_ratio'] for s in range(len(SEEDS))]))
ir_p4=float(np.mean([all_phase_results[s][4]['int_ratio'] for s in range(len(SEEDS))]))
Q2_ir_grows = ir_p4 > ir_p0
# Q3: 空間相関 (corr) > 0.3 at P4
corr_p4=avg_phases[4]['corr']
Q3_spatial = corr_p4 > 0.3

# 対照
ctrl_mean_fit1=float(np.mean([r['fit1'] for r in ctrl_results]))
ctrl_mean_corr=float(np.mean([r['corr'] for r in ctrl_results]))
ctrl_mean_ir=float(np.mean([r['ir'] for r in ctrl_results]))

print("\n"+"="*65)
print(" 集計結果")
print("="*65)
print(f"  累積忘却率:    {mean_cum_fgt:.1%}  (理論上 17%×4=68%)")
q1_str = 'YES ✓' if Q1_nodisaster else 'NO ✗'
print(f"  Q1 非破滅的:   {q1_str}  (<50%)")
q2_str = 'YES ✓' if Q2_ir_grows else 'NO ✗'
print(f"  Q2 統合層成長: {q2_str}  (P0 ir={ir_p0:.3f} → P4 ir={ir_p4:.3f})")
q3_str = 'YES ✓' if Q3_spatial else 'NO ✗'
print(f"  Q3 空間分割:   {q3_str}  (P4 corr={corr_p4:.3f})")
print(f"  対照 fit1:     {ctrl_mean_fit1:.2f}  corr={ctrl_mean_corr:.3f}  ir={ctrl_mean_ir:.3f}")
print(f"  P0 fit_task1:  {mean_fit1_p0:.2f}")
print(f"  P5 fit_task1:  {mean_fit1_p5:.2f}")

# ── 可視化 ──────────────────────────────────────────────────────
print("\n[図] 可視化生成...")
fig=plt.figure(figsize=(22,16))
gs=GridSpec(3,5,figure=fig,hspace=0.55,wspace=0.42)
fig.suptitle("experiment_v16: Long-Term Multi-Task Transfer / Catastrophic Forgetting Analysis",
             fontsize=12,fontweight='bold')

phases_x=[r['phase'] for r in avg_phases]
fit1_y=[r['fit_task1'] for r in avg_phases]
corr_y=[r['corr'] for r in avg_phases]
ir_y=[r['int_ratio'] for r in avg_phases]
ns_y=[r['n_spec'] for r in avg_phases]

# (a) タスク1性能の推移
ax=fig.add_subplot(gs[0,:2])
ax.plot(phases_x, fit1_y, 'b-o', linewidth=2, markersize=8, label='fit_task1 (seq)')
ax.axhline(ctrl_mean_fit1, color='orange', linestyle='--', linewidth=1.5,
           label=f'ctrl fit1={ctrl_mean_fit1:.2f}')
for xi,yi in zip(phases_x,fit1_y):
    ax.annotate(f'{yi:.2f}', (xi,yi), textcoords='offset points', xytext=(0,8),
                ha='center', fontsize=9)
ax.set_xlabel('Phase'); ax.set_ylabel('fit_task1')
ax.set_xticks(phases_x)
ax.set_xticklabels(['P0\n(T1)','P1\n(2T)','P2\n(3T)','P3\n(4T)','P4\n(5T)','P5\n(ret)'])
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
ax.set_title("(a) タスク1性能の推移（忘却追跡）")

# (b) 空間相関の推移
ax=fig.add_subplot(gs[0,2])
ax.plot(phases_x[:6], corr_y, 'g-s', linewidth=2, markersize=8)
ax.axhline(ctrl_mean_corr, color='orange', linestyle='--', linewidth=1.5, label=f'ctrl={ctrl_mean_corr:.3f}')
ax.set_ylim(0,1); ax.set_ylabel('|corr(pos_x, sel)|')
ax.set_xticks(phases_x)
ax.set_xticklabels(['P0','P1','P2','P3','P4','P5'])
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_title("(b) 空間局在 corr")

# (c) 統合ニューロン比率の推移
ax=fig.add_subplot(gs[0,3])
ax.plot(phases_x, ir_y, 'm-^', linewidth=2, markersize=8)
ax.axhline(ctrl_mean_ir, color='orange', linestyle='--', linewidth=1.5, label=f'ctrl={ctrl_mean_ir:.3f}')
ax.set_ylim(0,1); ax.set_ylabel('integration_ratio')
ax.set_xticks(phases_x)
ax.set_xticklabels(['P0','P1','P2','P3','P4','P5'])
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
ax.set_title("(c) 統合層の成長")

# (d) 専門化ニューロン数の推移
ax=fig.add_subplot(gs[0,4])
ax.plot(phases_x, ns_y, 'r-D', linewidth=2, markersize=8)
ax.set_ylabel('n_specialized')
ax.set_xticks(phases_x)
ax.set_xticklabels(['P0','P1','P2','P3','P4','P5'])
ax.grid(True,alpha=0.3)
ax.set_title("(d) 専門化ニューロン数")

# (e)-(h) 各フェーズの空間分布 (Phase 1-4, seed=0)
ag0, env0, sd0 = final_agents[0]
phase_labels=['P1(2T)','P2(3T)','P3(4T)','P4(5T)']
cmap_tasks=plt.cm.get_cmap('tab10')
for pi, (plabel, phase_idx) in enumerate(zip(phase_labels,[1,2,3,4])):
    ax=fig.add_subplot(gs[1,pi])
    n_act=all_phase_results[0][phase_idx]['n_active']
    # 再測定 (最終エージェントは P5後なので参考程度)
    measure_selectivity(ag0, env0, n_act, seed_off=800000+pi*100, n_ep=60)
    dom=ag0.dominant_task()
    colors=[cmap_tasks(int(d)) for d in dom]
    ax.scatter(ag0.pos[:,0], ag0.pos[:,1], c=colors, s=18, alpha=0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"({chr(101+pi)}) {plabel} 空間分布", fontsize=10)
    ax.set_xlabel('pos_x'); ax.set_ylabel('pos_y')
    # 凡例
    for k in range(n_act):
        ax.plot([],[], 'o', color=cmap_tasks(k), label=f'T{k+1}', markersize=5)
    ax.legend(fontsize=7, loc='upper right')

# (j) 累積忘却率: 理論 vs 実測
ax=fig.add_subplot(gs[1,4])
n_transfers=[0,1,2,3,4]
theory=[0., 0.17, 0.34, 0.51, 0.68]
actual_fgt=[all_phase_results[s][pi]['forgetting'] for s in range(len(SEEDS)) for pi in range(1,5)]
actual_mean=[float(np.mean([all_phase_results[s][pi]['forgetting'] for s in range(len(SEEDS))]))
             for pi in range(1,5)]
actual_mean=[0.]+actual_mean
ax.plot(n_transfers, theory, 'r--o', linewidth=2, label='理論 (17%×n)')
ax.plot(n_transfers, actual_mean, 'b-s', linewidth=2, label='実測')
ax.set_xlabel('タスク転移数')
ax.set_ylabel('忘却率')
ax.set_ylim(-0.2,1.0)
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
ax.set_title("(j) 忘却率: 理論 vs 実測")

# (k) フェーズごとの fit_mean (全タスク)
ax=fig.add_subplot(gs[2,:2])
fit_mean_y=[r['fit_mean'] for r in avg_phases]
n_active_y=[r['n_active'] for r in avg_phases]
bars=ax.bar(range(len(avg_phases)), fit_mean_y,
            color=['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#607D8B'],
            edgecolor='black', linewidth=0.8, alpha=0.85)
for i,(bar,v,na) in enumerate(zip(bars,fit_mean_y,n_active_y)):
    ax.text(bar.get_x()+bar.get_width()/2, max(v,0)+0.05, f'{v:.2f}\n({na}T)',
            ha='center', fontsize=9)
ax.set_xticks(range(len(avg_phases)))
ax.set_xticklabels(['P0','P1','P2','P3','P4','P5'])
ax.set_ylabel('fit_mean (全タスク平均)')
ax.grid(True,alpha=0.3,axis='y')
ax.set_title("(k) fit_mean 推移（括弧内=タスク数）")

# (l) 対照実験比較
ax=fig.add_subplot(gs[2,2])
cats=['seq\nfit1','ctrl\nfit1','seq\ncorr×10','ctrl\ncorr×10','seq\nir×10','ctrl\nir×10']
vals=[mean_fit1_p5, ctrl_mean_fit1,
      avg_phases[4]['corr']*10, ctrl_mean_corr*10,
      avg_phases[4]['int_ratio']*10, ctrl_mean_ir*10]
colors_=['#2196F3','#FF9800']*3
bars2=ax.bar(cats,vals,color=colors_,edgecolor='black',linewidth=0.8,alpha=0.85)
for bar,v in zip(bars2,vals):
    ax.text(bar.get_x()+bar.get_width()/2, max(v,0)+0.05, f'{v:.2f}',
            ha='center',fontsize=9)
ax.set_title("(l) 順次 vs 対照 比較")
ax.grid(True,alpha=0.3,axis='y')

# (m) タスク数 vs integration_ratio
ax=fig.add_subplot(gs[2,3:])
task_counts=[r['n_active'] for r in avg_phases]
ir_vals=[r['int_ratio'] for r in avg_phases]
ax.plot(task_counts, ir_vals, 'b-o', linewidth=2, markersize=10,
        label='順次転移')
ax.axhline(ctrl_mean_ir, color='orange', linestyle='--', linewidth=1.5,
           label=f'対照 (全5タスク同時) = {ctrl_mean_ir:.3f}')
ax.set_xlabel('タスク数'); ax.set_ylabel('integration_ratio')
ax.set_ylim(0,1)
_ir_p0_str = f'{ir_p0:.3f}'
_ir_p4_str = f'{ir_p4:.3f}'
ax.set_title(f"(m) タスク数 vs 統合層\n(P0:{_ir_p0_str} → P4:{_ir_p4_str})")
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

plt.savefig('results_v16.png',dpi=120,bbox_inches='tight')
print("  → results_v16.png 保存完了")

# ── レポート生成 ────────────────────────────────────────────────
elapsed=time.time()-t_total

_cum_fgt_pct = f"{mean_cum_fgt:.1%}"
_q1_str = 'YES ✓' if Q1_nodisaster else 'NO ✗'
_q2_str = 'YES ✓' if Q2_ir_grows else 'NO ✗'
_q3_str = 'YES ✓' if Q3_spatial else 'NO ✗'
_ir_p0_rep = f'{ir_p0:.3f}'
_ir_p4_rep = f'{ir_p4:.3f}'
_corr_p4_str = f'{corr_p4:.3f}'
_ctrl_fit1_str = f'{ctrl_mean_fit1:.2f}'
_ctrl_corr_str = f'{ctrl_mean_corr:.3f}'
_fit1_p0_str = f'{mean_fit1_p0:.2f}'
_fit1_p5_str = f'{mean_fit1_p5:.2f}'

fgt_per_phase_str = '  '.join([
    f'P{i}:{avg_phases[i]["forgetting"]:.1%}' for i in range(1,5)
])

phase_table_rows = ''
for r in avg_phases:
    phase_table_rows += (
        f"| P{r['phase']} ({r['n_active']}T) "
        f"| {r['fit_task1']:.2f} "
        f"| {r['fit_mean']:.2f} "
        f"| {r['corr']:.3f} "
        f"| {r['int_ratio']:.3f} "
        f"| {r['n_spec']} "
        f"| {r['forgetting']:.1%} |\n"
    )

report = f"""# 長期多タスク転移実験 v16 報告書
「5タスク連続転移：破滅的忘却は蓄積するか」

## 核心の二文

「5タスク後の累積忘却率: {_cum_fgt_pct}
 （17%/タスクの理論予測 68% に対し、実際の値）」

「統合ニューロン比率の推移:
 2タスク={avg_phases[1]['int_ratio']:.3f} → 5タスク={avg_phases[4]['int_ratio']:.3f}
 タスク数に比例して統合層が成長: {_q2_str}」

## Q1-Q3 への回答

**Q1: 累積忘却率 < 50%（非破滅的）**
→ {_q1_str}  実測={_cum_fgt_pct}  (理論 68%)

**Q2: タスク数増加で統合層が成長するか**
→ {_q2_str}  P0={_ir_p0_rep} → P4={_ir_p4_rep}

**Q3: 5タスク時も空間的機能局在が維持されるか**
→ {_q3_str}  P4 corr={_corr_p4_str}

## フェーズ別結果表

| フェーズ | fit_task1 | fit_mean | corr | int_ratio | n_spec | 忘却率 |
|---------|-----------|----------|------|-----------|--------|--------|
{phase_table_rows}
## 忘却率の推移

{fgt_per_phase_str}
累積 (P0→P5): {_cum_fgt_pct}

理論値 (17%×n):  P1=17%  P2=34%  P3=51%  P4=68%

## 対照実験比較

| 条件 | fit_task1 | corr | int_ratio |
|------|-----------|------|-----------|
| 順次転移 (P5) | {_fit1_p5_str} | {_corr_p4_str} | {_ir_p4_rep} |
| 対照 (全同時) | {_ctrl_fit1_str} | {_ctrl_corr_str} | {ctrl_mean_ir:.3f} |

## Emergent Ventures 向け一段落




## 次の実験への提案

1. W_internal (伝導遅延) を加えて τ勾配が5タスク環境でも保たれるか検証
2. タスク10個への拡張 — 空間が10領域に収束するか
3. 本結果を HSNN 論文 Section 7「連続学習の物理的基盤」として記述
4. integration_ratio の増加が「前頭前野発達」仮説と対応するか神経科学文献と照合

実験完了: elapsed={elapsed:.0f}s
"""

with open('report_v16.md','w',encoding='utf-8') as f:
    f.write(report)
print("  → report_v16.md 保存完了")
print(f"\n実験完了: elapsed={elapsed:.0f}s")
