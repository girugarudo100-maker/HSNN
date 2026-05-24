"""
多タスク課程実験 — タスク数を増やすと癖が抜けるか
=====================================================

仮説:
  訓練タスク数が増えるほど、環境固有の「癖」が薄まり
  全タスクに共通する「本質的な学習構造」だけが残る
  → 未知の抽象タスクへの転移性能が向上する

実験設計:
  訓練タスク数: 1, 2, 4, 6（段階的に増加）
  各条件で best θ を使用（PE感度β=2.19）
  転移先: AbstractCombinationGame（訓練中に一度も見ない）
  
  各タスクで N_EP エピソード訓練 → Wをそのまま引き継いで次のタスクへ
  → Wに蓄積された「癖」が多タスクでどう変化するか

タスク群:
  T1: RiskGrid         (5×5空間, 地雷)
  T2: TimeSeriesRisk   (時系列スパイク検出)
  T3: InvertedControl  (行動反転空間)
  T4: ResourceManage   (複数リソースのバランス管理)
  T5: SequenceMemory   (パターン記憶・再現)
  T6: PredatorEvade    (捕食者回避、動的障害物)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 環境群（6種類）
# ============================================================

class RiskGridEnv:
    """T1: 空間ナビゲーション"""
    SIZE=5; MAX_E=20; N_MINES=4; N_FOOD=3; MAX_STEPS=80
    def __init__(self,seed=None):
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
                if 0<=nr<self.SIZE and 0<=nc<self.SIZE: patch[idx]=self.grid[nr,nc]
                idx+=1
        return np.concatenate([self.pos/self.SIZE,[self.energy/self.MAX_E],patch])
    def step(self,action):
        d=[(-1,0),(1,0),(0,-1),(0,1)][action%4]; r,c=self.pos
        self.pos=np.array([np.clip(r+d[0],0,self.SIZE-1),np.clip(c+d[1],0,self.SIZE-1)])
        self.energy-=1.0; self.steps+=1
        cell=self.grid[self.pos[0],self.pos[1]]; done=False; de=-1.0
        if cell==-1: done=True; de=-self.energy
        elif cell==1:
            self.energy=min(self.energy+5.0,self.MAX_E)
            self.grid[self.pos[0],self.pos[1]]=0; de=5.0
        if self.energy<=0 or self.steps>=self.MAX_STEPS: done=True
        return self._obs(),done,{"delta_e":de}

class TimeSeriesEnv:
    """T2: 時系列スパイク検出"""
    WINDOW=8; SPIKE_PROB=0.20; DRAIN=0.8
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed); self.reset()
    def reset(self):
        self.energy=20.0; self.buf=self.rng.normal(0,0.3,self.WINDOW).tolist()
        self.steps=0; return self._obs()
    def _obs(self):
        obs=np.array(self.buf[-self.WINDOW:],dtype=float)
        return np.pad(np.append(obs,self.energy/20.0),(0,12-9))
    def step(self,action):
        is_spike=self.rng.random()<self.SPIKE_PROB
        val=(float(self.rng.choice([-1,1]))*self.rng.uniform(2.5,4.0) if is_spike else self.rng.normal(0,0.3))
        self.buf.append(val); self.steps+=1
        self.energy-=self.DRAIN; de=-self.DRAIN
        if is_spike:
            if action==0: pen=abs(val)*1.8; self.energy-=pen; de-=pen
            else: self.energy+=0.5; de+=0.5
        else:
            if action==0: self.energy+=0.4; de+=0.4
        self.energy=np.clip(self.energy,0,20); done=self.energy<=0 or self.steps>=50
        return self._obs(),done,{"delta_e":de}

class InvertedEnv:
    """T3: 行動反転（Grid訓練の癖を矯正）"""
    def __init__(self,seed=None):
        self.base=RiskGridEnv(seed=seed); self.amap={0:1,1:0,2:3,3:2}
    def reset(self):
        obs=self.base.reset(); self.base.energy=20.0; return obs
    def step(self,action):
        obs,done,info=self.base.step(self.amap[action%4])
        if not done:
            self.base.energy-=0.3; info["delta_e"]-=0.3
            if self.base.energy<=0: done=True
        return obs,done,info

class ResourceManageEnv:
    """
    T4: 複数リソース管理
    3種類のリソース（food/water/energy）を同時に管理する。
    空間概念なし、時系列概念なし。
    obs: [food, water, energy, risk_signal, ...padding]
    action: 0=食料補充 1=水補充 2=エネルギー補充 3=待機
    """
    MAX_STEPS=100
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed); self.reset()
    def reset(self):
        self.food=self.rng.uniform(0.3,0.7)
        self.water=self.rng.uniform(0.3,0.7)
        self.energy_res=self.rng.uniform(0.3,0.7)
        self.steps=0; return self._obs()
    def _obs(self):
        risk=float(min(self.food,self.water,self.energy_res)<0.2)
        obs=np.array([self.food,self.water,self.energy_res,risk,
                      self.food*self.water,self.water*self.energy_res,
                      self.energy_res*self.food,
                      float(self.steps/self.MAX_STEPS),0,0,0,0])
        return obs
    def step(self,action):
        # 毎ステップ全リソースが減少
        self.food   -=self.rng.uniform(0.02,0.06)
        self.water  -=self.rng.uniform(0.02,0.06)
        self.energy_res-=self.rng.uniform(0.02,0.06)
        self.steps+=1; de=0.0
        if action==0: gain=self.rng.uniform(0.1,0.3); self.food+=gain; de+=gain
        elif action==1: gain=self.rng.uniform(0.1,0.3); self.water+=gain; de+=gain
        elif action==2: gain=self.rng.uniform(0.1,0.3); self.energy_res+=gain; de+=gain
        elif action==3: de-=0.05  # 待機コスト
        self.food=np.clip(self.food,0,1)
        self.water=np.clip(self.water,0,1)
        self.energy_res=np.clip(self.energy_res,0,1)
        alive=self.food>0 and self.water>0 and self.energy_res>0
        done=not alive or self.steps>=self.MAX_STEPS
        if not alive: de-=5.0
        return self._obs(),done,{"delta_e":de}

class SequenceMemoryEnv:
    """
    T5: パターン記憶・予測
    隠されたシーケンスパターンを学習し、次の値を予測する。
    obs: 最近8ステップのシグナル + エネルギー + 予測精度
    action: 0=高予測 1=低予測 2=中予測 3=棄権
    """
    WINDOW=8; MAX_STEPS=80
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
        self.pattern=self.rng.uniform(-1,1,4)  # 隠されたパターン
        self.reset()
    def reset(self):
        self.energy=20.0; self.t=0
        self.buf=[float(self.pattern[i%4]+self.rng.normal(0,0.2))
                  for i in range(self.WINDOW)]
        self.steps=0; return self._obs()
    def _obs(self):
        obs=np.array(self.buf[-self.WINDOW:],dtype=float)
        obs=np.append(obs,self.energy/20.0)
        return np.pad(obs,(0,12-len(obs)))
    def step(self,action):
        self.t+=1; self.steps+=1
        true_val=self.pattern[self.t%4]+self.rng.normal(0,0.15)
        self.buf.append(true_val)
        # 行動0:高予測, 1:低予測, 2:中, 3:棄権
        thresholds={0:0.3, 1:-0.3, 2:0.0, 3:None}
        de=-0.3
        if action==3: de-=0.1  # 棄権は小損失
        else:
            pred=thresholds[action]
            error=abs(true_val-pred)
            if error<0.4: gain=1.0-error; self.energy+=gain; de+=gain
            else: pen=error*0.5; self.energy-=pen; de-=pen
        self.energy=np.clip(self.energy,0,20)
        done=self.energy<=0 or self.steps>=self.MAX_STEPS
        return self._obs(),done,{"delta_e":de}

class PredatorEvadeEnv:
    """
    T6: 捕食者回避（動的障害物）
    4×4グリッドに捕食者が動き回る。距離を保ちながら食料を取る。
    obs: エージェント位置(2) + 捕食者位置(2) + 距離(1) + エネルギー(1) + 食料マップ(6)
    """
    SIZE=4; MAX_E=20; MAX_STEPS=80
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed); self.reset()
    def reset(self):
        self.pos=np.array([0,0],dtype=int)
        self.pred=np.array([self.SIZE-1,self.SIZE-1],dtype=int)
        # 食料を3箇所に配置
        self.food=set()
        while len(self.food)<3:
            r,c=self.rng.integers(0,self.SIZE,2)
            if tuple([r,c])!=tuple(self.pos) and tuple([r,c])!=tuple(self.pred):
                self.food.add((int(r),int(c)))
        self.energy=float(self.MAX_E); self.steps=0; return self._obs()
    def _obs(self):
        dist=float(np.linalg.norm(self.pos-self.pred))/self.SIZE
        food_map=np.zeros(6)
        for i,(r,c) in enumerate(list(self.food)[:6]):
            food_map[i]=1.0
        return np.array([self.pos[0]/self.SIZE, self.pos[1]/self.SIZE,
                          self.pred[0]/self.SIZE, self.pred[1]/self.SIZE,
                          dist, self.energy/self.MAX_E,
                          *food_map[:6]])
    def step(self,action):
        d=[(-1,0),(1,0),(0,-1),(0,1)][action%4]
        self.pos=np.array([np.clip(self.pos[0]+d[0],0,self.SIZE-1),
                            np.clip(self.pos[1]+d[1],0,self.SIZE-1)])
        # 捕食者がランダムに移動
        pd=[(-1,0),(1,0),(0,-1),(0,1)][self.rng.integers(4)]
        self.pred=np.array([np.clip(self.pred[0]+pd[0],0,self.SIZE-1),
                             np.clip(self.pred[1]+pd[1],0,self.SIZE-1)])
        self.energy-=1.0; self.steps+=1; de=-1.0
        # 捕食者に捕まった
        if np.array_equal(self.pos,self.pred):
            return self._obs(),True,{"delta_e":-self.energy}
        # 食料取得
        pos_t=tuple(self.pos.tolist())
        if pos_t in self.food:
            self.food.remove(pos_t); self.energy+=5.0; de+=5.0
        done=self.energy<=0 or self.steps>=self.MAX_STEPS or len(self.food)==0
        return self._obs(),done,{"delta_e":de}

class AbstractCombinationGame:
    """転移テスト（訓練中に一度も見ない）"""
    MAX_STEPS=120; DANGER_PROB=0.15
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed); self.reset()
    def reset(self):
        self.state=self.rng.uniform(-1,1,12).astype(float)
        self.energy=20.0; self.steps=0; self.danger=0.0; self.pending=False
        return self._obs()
    def _obs(self):
        obs=self.state.copy(); obs[0]=self.energy/20.0; obs[1]=self.danger; return obs
    def step(self,action):
        self.steps+=1; de=-0.5
        if self.rng.random()<self.DANGER_PROB:
            self.danger=self.rng.uniform(0.7,1.0); self.pending=True
        else:
            self.danger=max(0.0,self.danger-0.1)
        self.state+=self.rng.normal(0,0.1,12); self.state=np.clip(self.state,-1,1)
        self.state[0]=self.energy/20.0; self.state[1]=self.danger
        if action==0:
            gain=0.3; self.energy+=gain; de+=gain
            if self.pending: self.pending=False
        elif action==1:
            if self.pending: self.energy-=12.0; de-=12.0; self.pending=False
            else: self.energy+=8.0; de+=8.0
        elif action==2: self.energy-=0.2; de-=0.2
        elif action==3:
            if self.pending: self.pending=False; self.energy+=0.5; de+=0.5
            else: self.energy-=0.3; de-=0.3
        self.energy=np.clip(self.energy,0,20)
        done=self.energy<=0 or self.steps>=self.MAX_STEPS
        return self._obs(),done,{"delta_e":de}

# ============================================================
# タスクスケジュール
# ============================================================

ALL_TASKS = [
    ("T1_RiskGrid",   RiskGridEnv),
    ("T2_TimeSeries", TimeSeriesEnv),
    ("T3_Inverted",   InvertedEnv),
    ("T4_Resource",   ResourceManageEnv),
    ("T5_Sequence",   SequenceMemoryEnv),
    ("T6_Predator",   PredatorEvadeEnv),
]

CURRICULA = {
    "1タスク\n(Grid)":         [ALL_TASKS[0]],
    "2タスク\n(+TimeSeries)":  ALL_TASKS[:2],
    "4タスク\n(+Inv+Resource)":ALL_TASKS[:4],
    "6タスク\n(全種)":         ALL_TASKS[:6],
}

# ============================================================
# HSNN エージェント
# ============================================================

BEST_THETA = {
    "eta":0.0393,"beta":2.190,"gamma":2.039,
    "tau_trace":0.87,"tau_wm":0.95
}

class HSNNAgent:
    def __init__(self,theta,l2=32,obs_dim=12,n_actions=4,seed=0):
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
    def _act(self,x): return np.tanh(x)
    def step(self,obs):
        h1=self._act(self.W01@obs); h2=self._act(self.W12@h1)
        pe=float(np.linalg.norm(h2-self.wm))
        self.wm=self.theta["tau_wm"]*self.wm+(1-self.theta["tau_wm"])*h2
        logits=self.W2o@h2; exp_l=np.exp(logits-logits.max()); probs=exp_l/exp_l.sum()
        action=int(np.random.choice(self.n_actions,p=probs))
        tau_e=self.theta["tau_trace"]
        self.e01=tau_e*self.e01+np.outer(h1,obs)
        self.e12=tau_e*self.e12+np.outer(h2,h1)
        self.e2o=tau_e*self.e2o+np.outer(probs,h2)
        return action,pe
    def update(self,de,pe):
        pe_mod=max(pe,1e-6)**self.theta["beta"]
        de_mod=(abs(de)**self.theta["gamma"])*np.sign(de)
        m=self.theta["eta"]*pe_mod*de_mod
        self.W01+=m*self.e01; self.W12+=m*self.e12; self.W2o+=m*self.e2o
        for W in [self.W01,self.W12,self.W2o]: np.clip(W,-3,3,out=W)
    def run_episode(self,EnvClass,seed):
        env=EnvClass(seed=seed); obs=env.reset(); done=False; steps=0
        while not done:
            action,pe=self.step(obs); obs,done,info=env.step(action)
            self.update(info["delta_e"],pe); steps+=1
        return steps

# ============================================================
# 実験実行
# ============================================================

N_EP_PER_TASK = 150   # タスクごとの訓練エピソード数
N_TRANSFER    = 200   # 転移テストエピソード数
L2            = 32
SEEDS         = [0,1,2,3,4]
SMOOTH        = 12

print("="*65)
print("多タスク課程実験 — タスク数が増えると癖が抜けるか")
print(f"  タスクごと{N_EP_PER_TASK}ep × 転移{N_TRANSFER}ep × {len(SEEDS)}シード")
print("="*65)

results = {}   # curriculum_name → [seeds × transfer_ep]
task_surv= {}  # 各タスクでの訓練性能

import time
t0=time.time()

for curr_name, task_list in CURRICULA.items():
    print(f"\n[{curr_name.replace(chr(10),' ')}] {len(task_list)}タスク訓練")
    seeds_transfer = []

    for seed in SEEDS:
        np.random.seed(seed*137)
        agent = HSNNAgent(BEST_THETA, l2=L2, seed=seed)

        # 各タスクを順番に訓練（Wを引き継ぐ）
        for t_idx,(t_name,EnvClass) in enumerate(task_list):
            for ep in range(N_EP_PER_TASK):
                agent.run_episode(EnvClass, seed=ep+seed*1000+t_idx*500)

        # 転移テスト（AbstractCombinationGame）
        surv_curve = []
        for ep in range(N_TRANSFER):
            s = agent.run_episode(AbstractCombinationGame,
                                  seed=ep+seed*9999)
            surv_curve.append(s)
        seeds_transfer.append(surv_curve)

        final=np.mean(surv_curve[-50:])
        early=np.mean(surv_curve[:20])
        print(f"  Seed{seed}: early={early:.1f} final={final:.1f} "
              f"gain={final-early:+.1f}")

    results[curr_name] = seeds_transfer
    arr=np.array(seeds_transfer)
    print(f"  → 平均: early={arr[:,:20].mean():.1f} "
          f"final={arr[:,-50:].mean():.1f} "
          f"gain={arr[:,-50:].mean()-arr[:,:20].mean():+.1f}")

print(f"\n総実行時間: {time.time()-t0:.1f}s")

# ============================================================
# 可視化
# ============================================================

BASE   = "/mnt/user-data/outputs"
eps    = np.arange(N_TRANSFER)
COLORS = {
    "1タスク\n(Grid)":          "#EF4444",
    "2タスク\n(+TimeSeries)":   "#F59E0B",
    "4タスク\n(+Inv+Resource)": "#2563EB",
    "6タスク\n(全種)":          "#7C3AED",
}

def sm(arr2d,smooth=SMOOTH):
    a=np.array(arr2d)
    return uniform_filter1d(a.mean(0),smooth), uniform_filter1d(a.std(0),smooth)

# ---- Fig 1: 適応曲線（タスク数別）----
fig,axes=plt.subplots(1,2,figsize=(18,7))
fig.suptitle(
    "Multi-Task Curriculum: Does More Tasks = Better Abstract Transfer?\n"
    "All models use best θ (β=2.19) | W accumulated across tasks | Transfer to unseen AbstractGame",
    fontsize=12,fontweight='bold'
)

ax=axes[0]
for curr_name,color in COLORS.items():
    mean,std=sm(results[curr_name])
    n_tasks=len(CURRICULA[curr_name])
    ax.plot(eps,mean,color=color,linewidth=2.5,
            label=f"{curr_name.replace(chr(10),' ')} ({n_tasks}タスク)")
    ax.fill_between(eps,mean-std*0.4,mean+std*0.4,color=color,alpha=0.12)

ax.set_xlabel("Episode on Abstract Task (unseen)",fontsize=11)
ax.set_ylabel("Survival Steps",fontsize=11)
ax.set_title("Adaptation Curves by Curriculum Size",fontsize=11,fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA'); ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ---- 右: タスク数 vs 転移性能の関係 ----
ax=axes[1]
n_tasks_list=[]
early_list=[]; final_list=[]
early_std_list=[]; final_std_list=[]
colors_sc=[]

for curr_name in CURRICULA:
    arr=np.array(results[curr_name])
    n=len(CURRICULA[curr_name])
    n_tasks_list.append(n)
    early_list.append(float(arr[:,:20].mean()))
    final_list.append(float(arr[:,-50:].mean()))
    early_std_list.append(float(arr[:,:20].std()))
    final_std_list.append(float(arr[:,-50:].std()))
    colors_sc.append(COLORS[curr_name])

ax.scatter(n_tasks_list,early_list,
           c=colors_sc,s=200,marker='o',
           edgecolors='white',linewidths=2,
           label='Early (first 20ep)',alpha=0.5,zorder=5)
ax.scatter(n_tasks_list,final_list,
           c=colors_sc,s=300,marker='*',
           edgecolors='white',linewidths=1.5,
           label='Final (last 50ep)',zorder=5)
for n,e,f,c in zip(n_tasks_list,early_list,final_list,colors_sc):
    ax.annotate(f"{f:.1f}",xy=(n,f),
                xytext=(n+0.05,f+0.5),fontsize=10,
                color=c,fontweight='bold')
    ax.plot([n,n],[e,f],color=c,linewidth=1.5,alpha=0.5)

# トレンドライン（final）
if len(n_tasks_list)>=2:
    z=np.polyfit(n_tasks_list,final_list,1)
    xline=np.linspace(0.5,6.5,50)
    ax.plot(xline,np.poly1d(z)(xline),color='#9CA3AF',
            linestyle='--',linewidth=1.5,alpha=0.7,
            label=f"Trend: {z[0]:+.2f}/task")

ax.set_xlabel("Number of Training Tasks",fontsize=11)
ax.set_ylabel("Survival on Abstract Task",fontsize=11)
ax.set_title("Core Question: More Tasks → Better Transfer?\n"
             "★=final, ○=early",
             fontsize=11,fontweight='bold')
ax.set_xticks([1,2,4,6])
ax.legend(fontsize=9)
ax.set_facecolor('#FAFAFA'); ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/curriculum_fig1_main.png",dpi=135,bbox_inches='tight')
plt.close()
print("\nSaved: curriculum_fig1_main.png")


# ---- Fig 2: 学習改善量（early→finalのgain）----
fig,ax=plt.subplots(figsize=(12,6))
fig.suptitle(
    "Learning Gain on Abstract Task: Does Curriculum Size Reduce 'Habit Bias'?\n"
    "Positive = still improving during transfer | Negative = peaked early (habit stuck)",
    fontsize=12,fontweight='bold'
)

gains=[f-e for e,f in zip(early_list,final_list)]
bars=ax.bar([f"{n}タスク" for n in n_tasks_list],
            gains,
            color=colors_sc,alpha=0.85,
            edgecolor='white',linewidth=2,width=0.5)
ax.axhline(0,color='black',linewidth=1)
for bar,v,n in zip(bars,gains,n_tasks_list):
    ax.text(bar.get_x()+bar.get_width()/2,
            v+(0.2 if v>=0 else -0.5),
            f"{v:+.2f}",ha='center',fontsize=12,fontweight='bold')

ax.set_ylabel("Gain (final − early survival)",fontsize=11)
ax.set_xlabel("Number of training tasks",fontsize=11)
ax.set_title("Positive = more abstract learning | Negative = habit dominates",
             fontsize=11,fontweight='bold')
ax.set_facecolor('#FAFAFA'); ax.grid(axis='y',alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/curriculum_fig2_gain.png",dpi=135,bbox_inches='tight')
plt.close()
print("Saved: curriculum_fig2_gain.png")


# ---- 数値サマリー ----
print("\n"+"="*65)
print("多タスク課程実験 サマリー")
print("="*65)
print(f"{'課程':25s} {'early':>7} {'final':>7} {'gain':>7}")
print("-"*50)
for curr_name,n in zip(CURRICULA.keys(),[1,2,4,6]):
    arr=np.array(results[curr_name])
    e=arr[:,:20].mean(); f=arr[:,-50:].mean()
    print(f"  {curr_name.replace(chr(10),' '):23s}: {e:7.2f} {f:7.2f} {f-e:+7.2f}")

print("\n核心的な問いへの答え:")
arr_1=np.array(results["1タスク\n(Grid)"])
arr_6=np.array(results["6タスク\n(全種)"])
diff=arr_6[:,-50:].mean()-arr_1[:,-50:].mean()
gain_1=arr_1[:,-50:].mean()-arr_1[:,:20].mean()
gain_6=arr_6[:,-50:].mean()-arr_6[:,:20].mean()
print(f"  6タスク − 1タスク（最終性能）: {diff:+.2f}手")
print(f"  1タスクのgain: {gain_1:+.2f}手")
print(f"  6タスクのgain: {gain_6:+.2f}手")
if diff>1.0:
    print("  → タスク数が増えると転移性能が向上する（仮説支持）")
elif diff>0:
    print("  → 弱い優位性（より多くの試行が必要）")
else:
    print("  → タスク数では改善せず（別の要因を検討）")
if gain_6>gain_1:
    print(f"  → 6タスク訓練の方がgainが大きい: 抽象タスクで学習が続いている")

print("\n全実験完了。")
