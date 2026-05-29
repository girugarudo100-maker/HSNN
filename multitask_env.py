"""
multitask_env.py  複数タスク統合環境

18次元デュアルチャンネル入力:
  CH_A (dim 0-8):  タスクAがアクティブ時 = 実際の観測、非アクティブ時 = zeros
  CH_B (dim 9-17): タスクBがアクティブ時 = 実際の観測、非アクティブ時 = zeros

「今どちらのタスクか」という識別子は渡さない。
ニューロンは入力の統計的構造から自律的に判断する。
"""
import numpy as np

N_CH  = 9   # 各チャンネルの次元数
N_IN  = 18  # 全入力次元数 = N_CH * 2

# ── タスクA: リスク回避 (行動→即座の結果) ─────────────────────────
class TaskAGrid:
    """
    5×5グリッド。地雷を踏むとエピソード終了（de=-5）。
    安全なマスへの移動: de=+1。
    行動と結果が1対1対応する「即座フィードバック」タスク。
    """
    def __init__(self, size=5, n_mines=3):
        self.size=size; self.n_mines=n_mines

    def reset(self, rng):
        self.pos   = np.array([self.size//2]*2)
        self.mines = [rng.integers(0, self.size, 2) for _ in range(self.n_mines)]
        self.t = 0
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size
        o[1] = self.pos[1]/self.size
        for i, m in enumerate(self.mines[:3]):
            o[2+i*2] = (m[0]-self.pos[0])/self.size
            o[3+i*2] = (m[1]-self.pos[1])/self.size
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx],0,self.size-1); self.t+=1
        hit = any(np.array_equal(self.pos,m) for m in self.mines)
        return self._obs(), (-5. if hit else 1.), hit or self.t>=50


# ── タスクB: 捕食者回避 (環境が自律的に変化) ──────────────────────
class TaskBGrid:
    """
    5×5グリッド。捕食者2体がT_move ステップごとに追跡移動する。
    食料マスにいると +de_food/step、捕食者に接触すると de=-5。
    行動と結果が1対1対応しない「非同期フィードバック」タスク。
    """
    def __init__(self, size=5, n_pred=2, n_food=2, T_move=3):
        self.size=size; self.n_pred=n_pred
        self.n_food=n_food; self.T_move=T_move

    def reset(self, rng):
        self.pos   = np.array([self.size//2]*2)
        corners = [(0,0),(0,self.size-1),(self.size-1,0),(self.size-1,self.size-1)]
        idxs    = rng.choice(len(corners), self.n_pred, replace=False)
        self.preds = [np.array(corners[i]) for i in idxs]
        self.foods = [rng.integers(1, self.size-1, 2) for _ in range(self.n_food)]
        self.t = 0; self._rng = rng
        return self._obs()

    def _obs(self):
        o = np.zeros(N_CH)
        o[0] = self.pos[0]/self.size
        o[1] = self.pos[1]/self.size
        if self.preds:
            o[2] = (self.preds[0][0]-self.pos[0])/self.size
            o[3] = (self.preds[0][1]-self.pos[1])/self.size
        if len(self.preds) > 1:
            o[4] = (self.preds[1][0]-self.pos[0])/self.size
            o[5] = (self.preds[1][1]-self.pos[1])/self.size
        if self.foods:
            o[6] = (self.foods[0][0]-self.pos[0])/self.size
            o[7] = (self.foods[0][1]-self.pos[1])/self.size
        o[8] = self.t/50.
        return o.clip(-1,1)

    def step(self, act):
        dy,dx=[(-1,0),(1,0),(0,-1),(0,1)][act%4]
        self.pos = np.clip(self.pos+[dy,dx],0,self.size-1); self.t+=1

        # 捕食者がT_moveステップごとに追跡
        if self.t % self.T_move == 0:
            for p in self.preds:
                diff = self.pos - p
                if abs(diff[0]) >= abs(diff[1]):
                    p += [np.sign(diff[0]), 0]
                else:
                    p += [0, np.sign(diff[1])]
                p[:] = np.clip(p, 0, self.size-1)

        caught = any(np.array_equal(self.pos, p) for p in self.preds)
        if caught:
            return self._obs(), -5., True
        on_food = any(np.array_equal(self.pos, f) for f in self.foods)
        de = 2. if on_food else 0.5
        return self._obs(), de, self.t>=50


# ── 複合タスク環境 ───────────────────────────────────────────────
class MultiTaskEnv:
    """
    50/50でタスクAとBをエピソード単位で切り替える。
    識別子なし: どちらのチャンネルが来るかをニューロン自身が判断。

    obs = concat(ch_A, ch_B)  # 18次元
      タスクA時: ch_A = 実際の観測 (9D), ch_B = zeros
      タスクB時: ch_A = zeros,         ch_B = 実際の観測 (9D)
    """
    def __init__(self, T_move=3):
        self.env_A = TaskAGrid()
        self.env_B = TaskBGrid(T_move=T_move)
        self.task  = None

    def reset(self, rng, task=None):
        self.task = task if task is not None else ('A' if rng.random() < 0.5 else 'B')
        if self.task == 'A':
            obs9 = self.env_A.reset(rng)
        else:
            obs9 = self.env_B.reset(rng)
        return self._make_18(obs9)

    def _make_18(self, obs9):
        if self.task == 'A':
            return np.concatenate([obs9, np.zeros(N_CH)])
        else:
            return np.concatenate([np.zeros(N_CH), obs9])

    def step(self, act):
        if self.task == 'A':
            obs9, de, done = self.env_A.step(act)
        else:
            obs9, de, done = self.env_B.step(act)
        return self._make_18(obs9), de, done

    def switch_task(self, new_task, rng):
        """エピソード中にタスクを切り替えて新タスクの初期観測を返す (Q5用)"""
        old_task = self.task
        self.task = new_task
        if new_task == 'A':
            obs9 = self.env_A.reset(rng)
        else:
            obs9 = self.env_B.reset(rng)
        return old_task, self._make_18(obs9)


class MultiTaskEnvWithID(MultiTaskEnv):
    """
    識別子あり版 (Q4比較用):
    obs = concat(ch_A, ch_B, [task_id])  # 19次元
    task_id: タスクA=1.0, タスクB=0.0
    """
    def _make_obs(self, obs9):
        base18 = super()._make_18(obs9)
        task_id = np.array([1.0 if self.task=='A' else 0.0])
        return np.concatenate([base18, task_id])

    def reset(self, rng, task=None):
        self.task = task if task is not None else ('A' if rng.random() < 0.5 else 'B')
        if self.task == 'A':
            obs9 = self.env_A.reset(rng)
        else:
            obs9 = self.env_B.reset(rng)
        return self._make_obs(obs9)

    def step(self, act):
        if self.task == 'A':
            obs9, de, done = self.env_A.step(act)
        else:
            obs9, de, done = self.env_B.step(act)
        return self._make_obs(obs9), de, done
