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

The 3D SiN slab waveguide is reduced to a 2D problem via EIM. The TE0 slab mode effective index (n_eff = 1.8836) is used as the 2D core index. All simulations run with [Meep 1.25](https://github.com/NanoComp/meep) on WSL2 Ubuntu.

### Optimization Pipeline

```
Stage 1 (100 iter)                    Stage 2 (50 iter)
─────────────────────────────────     ──────────────────────────────
Gaussian filter σ=300nm               Conic filter R=300nm
Adam lr=0.02                          Adam lr=0.01
beta schedule: 4→8→16→32             beta schedule: 32→48→64
J = total × (1 − α·CV²)              J = total × (1 − α·CV²)
α: 0 → 0.05 → 0.12                   α = 0.12 (fixed)
Joint geometry FD (every 8 iter)      warm start from Stage 1
```

### Objective Function

```
J = total_power × (1 − α × norm_CV²)

norm_CV² = Σ(Pᵢ − mean)² / (N × mean²)   [dimensionless, scale-invariant]
```

α < 1/(N−1) ≈ 0.143 ensures J > 0 throughout optimization.

### Geometry Co-optimization

Three parameters jointly optimized with central-difference FD gradients:

| Parameter | Description |
|-----------|-------------|
| `port_pitch` | Output port center-to-center spacing |
| `wg_w_in` | Input waveguide width |
| `wg_w_out` | Output waveguide width |

---

## Results

### Stage 1 — Parametric Study (7 variants)

All variants use the same objective and optimizer. Results after 100 iterations:

| Variant | Total T | Std | Imbalance | Design vars |
|---------|---------|-----|-----------|-------------|
| L20W14 | **91.5%** | 7.16% | 22.32% | 160×112 |
| L20W18 | 89.4% | 5.83% | 18.70% | 160×144 |
| L30W14 | 86.0% | 6.67% | 22.62% | 240×112 |
| L20W16 | 85.8% | 5.46% | 16.62% | 160×128 |
| L40W16 | 78.8% | 5.40% | 16.48% | 320×128 |
| L40W14 | 72.7% | 4.16% | 12.69% | 320×112 |
| L50W14 | 57.0% | 3.06% | 8.78% | 400×112 |

**Key finding:** Longer MMI does not improve performance — the apparent imbalance reduction in L40/L50 comes from power being lost (lower total T), not genuinely uniform splitting. The optimizer exploits the longer cavity to concentrate power in fewer ports. Stage 1 total T is the key metric for selecting a Stage 2 warm start.

---

### Stage 2 — Conic Filter Refinement

Stage 2 applies a conic filter (R=300nm) for fabrication-constrained binarization, warm-started from Stage 1:

| Variant | S1 Total T | S2 Total T | S1 Imbalance | S2 Imbalance | Gray fraction |
|---------|-----------|-----------|-------------|-------------|---------------|
| L20W14 | 91.5% | **95.9%** | 22.32% | **18.35%** | 4.6% |
| L20W18 | 89.4% | **97.0%** | 18.70% | 19.14% | 4.0% |
| L50W14 | 57.0% | 90.2% | 8.78% | 16.63% | 8.1% |

Stage 2 consistently improves total T through better binarization. L20W14 shows improvement in both T and imbalance; L20W18 achieves the highest total T (97.0%) but imbalance is unchanged. L50W14 shows the largest absolute gain (+33 pp T) but the imbalance worsens because the optimizer can no longer suppress the edge-port dominance pattern once the design is binarized.

The persistent edge-port dominance (Port 1/8 high, Port 2/7 low) is a topology-level limitation not addressable by Stage 2 refinement alone.

---

### Per-port Breakdown — Best Results

**Stage 2 L20W14** (Total T = 95.9%, Imbalance = 18.35%)

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.129 | 19.16% |
| 2 | −4.378 |  2.95% |
| 3 | −2.627 | 14.79% |
| 4 | −0.876 |  9.23% |
| 5 | +0.876 | 10.66% |
| 6 | +2.627 | 13.64% |
| 7 | +4.378 |  3.55% |
| 8 | +6.129 | 21.30% |

**Stage 2 L20W18** (Total T = 97.0%, Imbalance = 19.14%)

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.233 | 21.75% |
| 2 | −4.452 |  2.61% |
| 3 | −2.671 | 14.02% |
| 4 | −0.890 | 11.43% |
| 5 | +0.890 | 11.03% |
| 6 | +2.671 | 14.03% |
| 7 | +4.452 |  2.86% |
| 8 | +6.233 | 18.67% |

**Stage 2 L50W14** (Total T = 90.2%, Imbalance = 16.63%)

| Port | y (µm) | T (%) |
|------|--------|-------|
| 1 | −6.234 | 17.62% |
| 2 | −4.453 |  2.14% |
| 3 | −2.672 | 12.85% |
| 4 | −0.891 |  9.95% |
| 5 | +0.891 | 13.44% |
| 6 | +2.672 | 11.85% |
| 7 | +4.453 |  2.79% |
| 8 | +6.234 | 18.76% |

---

## Repo Structure

```
meep_1x8_progress/
├── scripts/
│   ├── stage1_param.py          Stage 1 parametric optimization (--mmi_L --mmi_W --warmstart --out)
│   ├── stage2_param.py          Stage 2 conic refinement (--variant --resume --out)
│   ├── eval_param.py            Stage 1 transmission eval (--variant)
│   ├── eval_stage2.py           Stage 2 transmission eval (--variant --base --out)
│   ├── print_eval.py            All-variant comparison printout
│   └── stage2_refine.py         Stage 2 legacy script (reference)
├── launchers/                   WSL launch scripts (.sh)
├── meep/                        Simulation results
│   ├── stage1_L20W14/           ✅ Total T=91.5%  Imbalance=22.32%
│   ├── stage1_L20W16/           ✅ Total T=85.8%  Imbalance=16.62%
│   ├── stage1_L20W18/           ✅ Total T=89.4%  Imbalance=18.70%
│   ├── stage1_L30W14/           ✅ Total T=86.0%  Imbalance=22.62%
│   ├── stage1_L40W14/           ✅ Total T=72.7%  Imbalance=12.69%
│   ├── stage1_L40W16/           ✅ Total T=78.8%  Imbalance=16.48%
│   ├── stage1_L50W14/           ✅ Total T=57.0%  Imbalance=8.78%
│   ├── stage2_L20W14/           ✅ Total T=95.9%  Imbalance=18.35%
│   ├── stage2_L20W18/           ✅ Total T=97.0%  Imbalance=19.14%
│   └── stage2_L50W14_r1/        ✅ Total T=90.2%  Imbalance=16.63%
└── meep_1x8_design_log.md       Full design log with methods and findings (Chinese)
```

---

## Next Steps

- [ ] New design concept targeting edge-port uniformity issue (Port 2/7 starvation)
- [ ] GDS export via KLayout / GDSFactory, DRC check (min feature ≥ 300 nm)
- [ ] Broadband optimization (1520–1580 nm)

---

## Dependencies

```
# Simulation (WSL2 Ubuntu, conda env: pmp)
meep >= 1.25
meep.adjoint
numpy / scipy / autograd / matplotlib

# Evaluation & scripts (Windows, conda env: photon)
ceviche
numpy / scipy / matplotlib
```
