[README.md](https://github.com/user-attachments/files/28193383/README.md)
# HSNN — Hierarchical Spiking Neural Network

**Meaning Without Reward: Spontaneous Semantic
Grounding via Metabolic Constraints**

## Overview

HSNN learns exclusively from two internal signals:

- **PE** (prediction error): how surprising is this?
- **ΔE** (metabolic energy change): how did this affect survival?

No external reward. No labeled data.
The synaptic update rule:

```math
dW = η · PE(t)^β · |ΔE(t)|^γ · sign(ΔE(t)) · e
```

## Key Results

| Experiment | Result |
|------------|--------|
| Ablation | Removing ΔE reduces transfer accuracy by 58% |
| Multi-task transfer | 5.85× faster adaptation than PPO |
| MiniGrid-FourRooms | 6.8% goal rate (PPO: 1.6%), no external reward |
| Autonomous reward design | w_explore=2.067 emerges from terminal condition only |

## Files

| File | Experiment |
|------|------------|
| `hsnn_v12_recheck.py` | Ablation study (Experiment 1) |
| `hsnn_curriculum.py` | Multi-task transfer (Experiment 2) |
| `minigrid_benchmark.py` | MiniGrid benchmark (Experiment 3) |
| `hsnn_theta_extended.py` | Autonomous reward design (Experiment 4) |
| `bench_fast.py` | HSNN vs PPO comparison |

## Requirements

```bash
pip install numpy matplotlib scipy
pip install minigrid gymnasium
```

## Quick Start

```bash
# Ablation study
python hsnn_v12_recheck.py

# Multi-task transfer
python hsnn_curriculum.py

# MiniGrid benchmark
python minigrid_benchmark.py

# Autonomous reward design
python hsnn_theta_extended.py
```

## Best θ Parameters

Optimized via population-based evolution:

| Parameter | Value | Description |
|-----------|-------|-------------|
| η | 0.0393 | Learning rate |
| β | 2.190 | PE sensitivity |
| γ | 2.039 | ΔE sensitivity |
| τ_trace | 0.87 | Memory length |
| τ_wm | 0.95 | World model stability |

## Paper

Preprint: [arxiv — coming 2026-05-31]

## Author

吉岡 / Yoshioka
Nihon University, Faculty of Philosophy

## License

MIT
