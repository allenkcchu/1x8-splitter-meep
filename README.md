# 1×8 MMI Power Splitter — Inverse Design

Inverse design of a single-stage 1×8 MMI (Multimode Interference) power splitter using Meep FDTD + adjoint topology optimization, targeting a SiN 800nm photonic platform (λ = 1550 nm).

---

## Motivation

Conventional 1×8 splitting requires a 3-stage binary tree of 1×2 MMIs. This project investigates whether a single-stage MMI, shaped by inverse design, can achieve equal 8-way power splitting with competitive efficiency — eliminating two stages of cascaded insertion loss.

---

## Platform

| Parameter | Value |
|-----------|-------|
| Core material | Si₃N₄ (SiN) |
| Cladding | SiO₂ |
| SiN thickness | 800 nm |
| Wavelength | 1550 nm |
| SiN refractive index | 1.996 |
| SiO₂ refractive index | 1.444 |
| TE0 effective index (EIM) | 1.8836 |
| Single-mode cutoff width | 1.282 µm |
| Waveguide width | 1.0 µm |

---

## Method

### Simulation: 2D FDTD + Effective Index Method

The 3D SiN slab waveguide is reduced to a 2D problem via EIM. The TE0 slab mode effective index (n_eff = 1.8836) is used as the 2D core index. All simulations run with [Meep 1.25](https://github.com/NanoComp/meep).

### Topology Optimization

- **Tool:** `meep.adjoint` — computes exact gradients via adjoint FDTD (1 forward + 1 adjoint per iteration)
- **Optimizer:** Adam (lr = 0.02, β₁ = 0.9, β₂ = 0.999)
- **Filter:** Gaussian (σ = 300 nm) for Stage 1; Conic (R = 300 nm) for Stage 2
- **Binarization:** tanh projection with beta schedule 4 → 8 → 16 → 32 → 64

### Objective Function

Stage 1 uses a uniformity-penalized objective:

```
J = total_power × (1 − α × norm_CV²)

where norm_CV² = Σ(Pᵢ − mean)² / (N × mean²)   [dimensionless]
```

- α schedule: 0 (iter 0–19) → 0.05 (20–49) → 0.12 (50–99)
- Constraint: α < 1/(N−1) ≈ 0.143 to keep J > 0

### Geometry Co-optimization

Three geometry parameters are jointly optimized alongside topology using central-difference finite differences (every 8 iterations, lr = 0.005):

| Parameter | Bounds (W=14 µm) |
|-----------|-----------------|
| `port_pitch` | [1.20, 1.83] µm |
| `wg_w_in` | [0.80, 1.50] µm |
| `wg_w_out` | [0.80, 1.50] µm |

---

## Results

### Stage 1 v1 — Baseline (L=30 µm, W=14 µm, pure J maximization)

> 100 iterations, Gaussian filter σ=300 nm

| Metric | Value |
|--------|-------|
| Total transmission | **86.7%** |
| Mean per-port T | 10.7% (ideal: 12.5%) |
| Port-to-port std | 8.52% |
| Max imbalance | 23.3% |

Per-port breakdown:

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.125 | 24.6% |
| 2 | −4.375 |  1.4% |
| 3 | −2.625 | 14.3% |
| 4 | −0.875 |  4.4% |
| 5 | +0.875 |  7.3% |
| 6 | +2.625 |  9.6% |
| 7 | +4.375 |  1.3% |
| 8 | +6.125 | 22.8% |

Ports 1 and 8 dominate; ports 2 and 7 are nearly zero — pure power maximization does not produce uniform splitting.

---

### Stage 1 v2 — Uniformity Penalty + Geometry Optimization (L=30 µm, W=14 µm)

> 100 iterations, same topology starting point, α schedule added

| Metric | v1 | v2 |
|--------|----|----|
| Total T | 86.7% | **86.0%** |
| Port std | 8.52% | **6.67%** (↓22%) |
| Max imbalance | 23.3% | 22.6% |

Per-port breakdown:

| Port | T v1 (%) | T v2 (%) |
|------|----------|----------|
| 1 | 24.6% | 13.1% |
| 2 |  1.4% |  3.1% |
| 3 | 14.3% | 11.5% |
| 4 |  4.4% |  9.9% |
| 5 |  7.3% |  7.8% |
| 6 |  9.6% | 11.3% |
| 7 |  1.3% |  3.1% |
| 8 | 22.8% | 25.7% |

The penalty suppressed port 1 dominance but port 8 increased. Geometry FD gradients were negligible — the 1.0 µm width / 1.75 µm pitch is already near-optimal for the current objective. Root cause: a 30 µm MMI is too short for uniform 8-way splitting.

**Optimized geometry (v2):**

| Parameter | Initial | Final |
|-----------|---------|-------|
| port_pitch | 1.750 µm | 1.798 µm |
| wg_w_in | 1.000 µm | 1.009 µm |
| wg_w_out | 1.000 µm | 0.997 µm |

---

### Parametric Study — MMI Length & Width Sweep

Motivated by the 30 µm length bottleneck, three additional variants are being optimized in parallel:

| Variant | mmi_L | mmi_W | Design vars | Status | Total T | Std |
|---------|-------|-------|-------------|--------|---------|-----|
| L30W14 | 30 µm | 14 µm | 240×112 | ✅ Done | 86.0% | 6.67% |
| L40W14 | 40 µm | 14 µm | 320×112 | 🔄 Running | — | — |
| L40W16 | 40 µm | 16 µm | 320×128 | 🔄 Running | — | — |
| L50W14 | 50 µm | 14 µm | 400×112 | 🔄 Running | — | — |

Each variant runs `scripts/stage1_param.py --mmi_L {L} --mmi_W {W}` on WSL2 Ubuntu (conda env `pmp`) and saves results to `stage1_L{L}W{W}/`.

---

## Repo Structure

```
meep_1x8_progress/
├── scripts/
│   ├── stage1_optimize.py       Stage 1 v1 standalone script
│   ├── stage1_v2_optimize.py    Stage 1 v2 standalone script (L30W14)
│   ├── stage1_param.py          Parametric script (--mmi_L --mmi_W)
│   ├── stage2_refine.py         Stage 2 conic filter script
│   ├── eval_transmission.py     Transmission evaluation (v1)
│   └── eval_v2.py               Transmission evaluation (v2)
├── stage1_L30W14/               v1 results (baseline, 30×14 µm)
├── stage1_v2/                   v2 results (L30W14 + uniformity penalty)
├── stage1_L40W14/               Parametric run (in progress)
├── stage1_L40W16/               Parametric run (in progress)
├── stage1_L50W14/               Parametric run (in progress)
├── stage2/                      Stage 2 (pending best Stage 1 result)
├── meep_1x8_design_log.md       Full design log (Chinese)
├── meep_1x8_splitter_v2.ipynb   Stage 1 v2 notebook (executed)
├── meep_1x8_stage2.ipynb        Stage 2 notebook (ready)
└── run_*.sh                     WSL launch scripts for each variant
```

---

## Next Steps

- [ ] Collect parametric study results (L40W14, L40W16, L50W14)
- [ ] Select best variant → run Stage 2 (conic filter warm start, 50 iterations)
- [ ] GDS export via KLayout / GDSFactory, DRC check (min feature ≥ 300 nm)
- [ ] Broadband optimization (1520–1580 nm)

---

## Dependencies

```
meep >= 1.25
meep.adjoint
numpy
scipy
autograd
matplotlib
```

Tested on WSL2 Ubuntu with conda environment `pmp`.
