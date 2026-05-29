# HSNN — Hierarchical Spiking Neural Network

**Spontaneous Functional Localization from Physical Constraints**

> 200 identical neurons. One constraint. No reward signal.  
> Functional localization, temporal hierarchy, and catastrophic-forgetting resistance emerge on their own.

---

## Key Results

| Finding | Value | Experiment |
|---------|-------|------------|
| Spatial selectivity correlation (2 tasks) | **\|r\| = 0.972** | v15 baseline |
| Correlation collapse without spatial cost | \|r\| = 0.070 (Δ = 0.902) | v17 ablation C1 |
| Spatial distance ↔ effective τ | **r = 0.954** | v15 delay model |
| Cumulative forgetting after 5 sequential tasks | **−14%** (improvement) | v16 |
| Integration neurons at 5 tasks | **60.8%** of population | v16 |
| Best inheritance: proportional vs. binary | **17.49 vs. 13.14** fit | v17 Figure 6 |

---

## Core Mechanism

The only constraint imposed on the network is a **spatial maintenance cost** on input weights:

$$m_{ik} = \exp\!\left(-|x_i - c_k| \cdot D_s\right)$$

$$\Delta W_{in} \mathrel{-}= E_m \cdot W_{in} \cdot (1 - m)$$

A neuron far from a task's spatial center pays a metabolic penalty to maintain connections to it.  
No task labels. No predefined areas. No explicit time constants.

Learning rule (PE×ΔE):

$$\lambda = \eta \cdot \text{PE}^\beta \cdot |\Delta E|^\gamma \cdot \text{sign}(\Delta E)$$

Survival-proportional inheritance (template update at episode end):

$$lr = lr_{\min} + (lr_{\max} - lr_{\min}) \cdot \frac{t_{\text{survive}}}{t_{\max}}$$

---

## Experiments

### v15 — Spatial HSNN Baseline
Establishes that spatial cost → functional localization. Discovers optimal hyperparameters (lr_max = 0.20, D_spatial = 5.0). Demonstrates temporal hierarchy via distance-based delays.

```bash
python experiment_v15.py
# → results_v15.png, report_v15.md
```

### v16 — Long-term Multi-task Transfer (Figure for Section 3.6)
Sequential 5-task curriculum (Phases 0→5). Measures cumulative forgetting and integration neuron growth.

```bash
python experiment_v16.py
# → results_v16.png, report_v16.md
```

| Phase | Active tasks | fit_task1 | \|corr\| | integration_ratio | forgetting |
|-------|-------------|-----------|---------|------------------|-----------|
| 0 | 1 | 28.39 | 0.000 | 0.000 | 0.0% |
| 1 | 2 | 27.43 | 0.440 | 0.022 | 3.7% |
| 2 | 3 | 29.27 | 0.571 | 0.255 | −3.1% |
| 3 | 4 | 27.45 | 0.329 | 0.292 | 3.8% |
| 4 | 5 | 32.89 | 0.417 | 0.608 | −15.8% |
| 5 (return) | 1 | **32.45** | — | — | **−14.0%** |

### v17 — Causal Validation (Figures 5, 6, 7)
Three sub-experiments proving causality via intervention.

```bash
python experiment_v17.py
# → results_v17.png, report_v17.md
```

**Figure 5 — Task count is a necessary condition:**

| Tasks | \|corr\| | integration_ratio |
|-------|---------|------------------|
| 1 | 0.000 | 0.000 |
| 2 | 0.864 | 0.145 |
| 3 | 0.659 | 0.292 |
| 5 | 0.451 | 0.600 |

**Figure 6 — Inheritance function comparison (2 tasks, D_s=3.0, 40 gens):**

| Condition | fit_mean |
|-----------|---------|
| B0 None | 15.02 |
| B1 Fixed (lr=0.033) | 15.18 |
| B2 Binary | 13.14 |
| **B3 Proportional (lr_max=0.20)** | **17.49** |

**Figure 7 — Ablation (5 tasks, 30 gens):**

| Condition | fit | \|corr\| | forgetting |
|-----------|-----|---------|-----------|
| C0 Full constraints (baseline) | −0.13 | 0.474 | −3.8% |
| C1 No spatial cost | 2.45 | **0.070** | −2.7% |
| C2 Binary inheritance | 1.03 | 0.473 | −1.5% |
| C3 No inheritance | 0.99 | 0.462 | **+8.7%** |
| C4 With task identifier | 3.26 | 0.080 | −14.1% |
| C5 No constraints | 0.61 | 0.061 | 2.5% |

---

## Project Structure

```
.
├── README.md
│
├── # Core environments
├── multitask_env.py          # Task1-5Grid, MultiTaskEnv, MultiTaskEnvWithID
│
├── # Experiments (chronological)
├── experiment_v15.py         # Spatial HSNN baseline & hyperparameter search
├── experiment_v16.py         # 5-task sequential transfer (catastrophic forgetting)
├── experiment_v17.py         # Causal validation (Figures 5, 6, 7)
│
├── # Generated reports
├── report_v15.md
├── report_v16.md
├── report_v17.md
│
├── # Generated figures
├── results_v15.png
├── results_v16.png
├── results_v17.png
│
└── # Paper draft
    ├── draft_paper_v1.md     # English (submitted for review)
    └── draft_paper_v1_ja.md  # Japanese translation
```

---

## Requirements

```bash
pip install numpy matplotlib scipy
```

No deep learning framework required. Pure NumPy implementation.

---

## Reproducing All Results

```bash
# Full pipeline (~2 min total)
python experiment_v15.py   # ~30s  — baseline + hyperparameter search
python experiment_v16.py   # ~25s  — 5-task sequential curriculum
python experiment_v17.py   # ~41s  — causal validation (Figures 5, 6, 7)
```

Each script writes a `report_vN.md` and `results_vN.png` to the working directory.

---

## Design Principles (Inviolable)

1. **No external reward** — only PE (prediction error) and ΔE (metabolic change)
2. **PE×ΔE learning rule only** — no backpropagation
3. **Compression bias** — spatial cost as the sole structural constraint
4. **Dual timescale** — within-episode plasticity + cross-episode inheritance
5. **Diversity curriculum** — heterogeneous tasks, no task identifiers
6. **Humanist philosophy** — structure from physics, not from design

---

## Paper

- English draft: [`draft_paper_v1.md`](draft_paper_v1.md)
- Japanese translation: [`draft_paper_v1_ja.md`](draft_paper_v1_ja.md)

Title: *Spontaneous Functional Localization from Physical Constraints: Spatial Cost, Temporal Hierarchy, and Continuous Learning in a Self-Organizing Spiking Neural Network*

---

## Author

吉岡 / Yoshioka  
Nihon University, Faculty of Philosophy

---

## License

MIT
