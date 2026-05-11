"""
1x8 MMI Splitter — Stage 1: Adjoint Topology Optimization
Platform: SiN 800nm platform (SiO2 cladding, lambda=1550nm)
Method:   2D FDTD with Effective Index Method + Adam optimizer
Filter:   Gaussian, sigma=300nm
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter
import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

mp.verbosity(0)

OUT = '/mnt/c/Users/Hemera/Projects/meep_1x8_progress'
os.makedirs(OUT, exist_ok=True)

# ── Platform parameters ────────────────────────────────────────────────────────
lam0 = 1.55; n_SiN = 1.996; n_SiO2 = 1.444; t_SiN = 0.8
k0 = 2 * np.pi / lam0

def slab_eq(neff):
    ky = k0 * np.sqrt(n_SiN**2 - neff**2)
    gm = k0 * np.sqrt(neff**2 - n_SiO2**2)
    return np.tan(ky * t_SiN / 2) - gm / ky

n_eff_2d = brentq(slab_eq, 1.82, 1.99)
Delta_n  = np.sqrt(n_eff_2d**2 - n_SiO2**2)
print(f'Slab TE0 n_eff = {n_eff_2d:.4f}')
print(f'SM cutoff      = {lam0/Delta_n:.3f} um')

# ── Layout ─────────────────────────────────────────────────────────────────────
N_out = 8; wg_w = 1.0; port_pitch = 1.75
mmi_W = 14.0; mmi_L = 30.0; wg_ext = 4.0; pml_th = 2.0; resolution = 15
port_ys = np.array([i - (N_out-1)/2 for i in range(N_out)]) * port_pitch
sx = mmi_L + 2*wg_ext + 2*pml_th
sy = mmi_W + 2*pml_th
des_res = 8; Nx_des = int(mmi_L*des_res); Ny_des = int(mmi_W*des_res)
filter_sig = 0.3 * des_res   # ~300 nm

SiN_mat  = mp.Medium(index=n_eff_2d)
SiO2_mat = mp.Medium(index=n_SiO2)
freq0    = 1.0 / lam0

print(f'FDTD grid:   {int(sx*resolution)} x {int(sy*resolution)}')
print(f'Design vars: {Nx_des} x {Ny_des} = {Nx_des*Ny_des}')

# ── Simulation ─────────────────────────────────────────────────────────────────
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx_des, Ny_des), SiO2_mat, SiN_mat,
    weights=np.ones((Ny_des, Nx_des)) * 0.5, do_averaging=True,
)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(mmi_L, mmi_W))
)
geometry = [
    mp.Block(size=mp.Vector3(wg_ext+pml_th, wg_w),
             center=mp.Vector3(-(mmi_L/2+(wg_ext+pml_th)/2), 0), material=SiN_mat),
    mp.Block(size=mp.Vector3(mmi_L, mmi_W), center=mp.Vector3(), material=design_variables),
]
for y in port_ys:
    geometry.append(mp.Block(size=mp.Vector3(wg_ext+pml_th, wg_w),
                             center=mp.Vector3(mmi_L/2+(wg_ext+pml_th)/2, y), material=SiN_mat))

sources = [mp.EigenModeSource(
    mp.GaussianSource(freq0, fwidth=0.05*freq0),
    size=mp.Vector3(0, wg_w*4),
    center=mp.Vector3(-(mmi_L/2+wg_ext*0.5), 0),
    eig_match_freq=True, eig_parity=mp.ODD_Z,
)]
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy), boundary_layers=[mp.PML(pml_th)],
    geometry=geometry, sources=sources,
    default_material=SiO2_mat, resolution=resolution,
)
mon_x = mmi_L/2 + wg_ext*0.5
ob_list = [
    mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(mon_x, y), size=mp.Vector3(0, wg_w*4)),
        1, eig_parity=mp.ODD_Z, forward=True)
    for y in port_ys
]

def J(*args):
    return npa.sum([npa.abs(npa.squeeze(a))**2 for a in args])

opt = mpa.OptimizationProblem(
    simulation=sim, objective_functions=J,
    objective_arguments=ob_list, design_regions=[design_region], frequencies=[freq0],
)

# ── Filter + projection ────────────────────────────────────────────────────────
def sigmoid_project(x2d, beta, eta=0.5):
    xf  = gaussian_filter(x2d, sigma=filter_sig)
    num = np.tanh(beta*eta) + np.tanh(beta*(xf - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return num / den

def backprop_gradient(x2d, dJ_flat, beta, eta=0.5):
    # Chain rule: dJ/dx = G * (dsig ⊙ dJ/drho)  [G is self-adjoint]
    xf   = gaussian_filter(x2d, sigma=filter_sig)
    dsig = beta * (1 - np.tanh(beta*(xf - eta))**2)
    dsig /= np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return gaussian_filter(dsig * dJ_flat.reshape(Ny_des, Nx_des), sigma=filter_sig).flatten()

def get_beta(i):
    if i < 30: return 4
    if i < 55: return 8
    if i < 80: return 16
    return 32

# ── Progress saver ─────────────────────────────────────────────────────────────
def save_progress(i, x, hist):
    x2d = sigmoid_project(x.reshape(Ny_des, Nx_des), get_beta(i))
    ext = [-mmi_L/2, mmi_L/2, -mmi_W/2, mmi_W/2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(x2d, origin='lower', cmap='RdBu_r', vmin=0, vmax=1, extent=ext)
    axes[0].set_title(f'Design density  (iter {i})')
    axes[0].set_xlabel('x (um)'); axes[0].set_ylabel('y (um)')
    for y in port_ys:
        axes[0].axhline(y, color='yellow', lw=0.6, ls='--', alpha=0.7)
    axes[1].plot(hist, 'b-', lw=1.5)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('J')
    axes[1].set_title(f'Convergence  J={hist[-1]:.1f}'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/progress_iter{i:03d}.png', dpi=110, bbox_inches='tight')
    plt.savefig(f'{OUT}/progress_latest.png',       dpi=110, bbox_inches='tight')
    plt.close(fig)
    with open(f'{OUT}/log.json', 'w') as f:
        json.dump({'iter': i, 'J_history': hist, 'beta': get_beta(i)}, f)
    np.save(f'{OUT}/x_latest.npy', x)

# ── Adam optimizer ─────────────────────────────────────────────────────────────
n_iters = 100; lr = 0.02; save_every = 5
b1, b2, eps_a = 0.9, 0.999, 1e-8

x    = np.ones(Nx_des * Ny_des) * 0.5
m_ax = np.zeros_like(x)
v_ax = np.zeros_like(x)
hist = []

print(f'\nStarting Stage 1: {n_iters} iters x 2 FDTD')

for i in range(n_iters):
    beta = get_beta(i)
    x2d  = x.reshape(Ny_des, Nx_des)
    xp   = sigmoid_project(x2d, beta).flatten()

    f_val, dJ = opt([xp])
    grad = backprop_gradient(x2d, np.asarray(dJ).flatten(), beta)

    m_ax = b1*m_ax + (1-b1)*grad
    v_ax = b2*v_ax + (1-b2)*grad**2
    m_h  = m_ax / (1 - b1**(i+1))
    v_h  = v_ax / (1 - b2**(i+1))
    x    = np.clip(x + lr * m_h / (np.sqrt(v_h) + eps_a), 0, 1)

    hist.append(float(f_val))
    print(f'  Iter {i:3d} | beta={beta:2d} | J={float(f_val):.2f}')

    if i % save_every == 0 or i == n_iters - 1:
        save_progress(i, x, hist)

np.save(f'{OUT}/x_final.npy', x)
print(f'\nDone. J: {hist[0]:.2f} -> {hist[-1]:.2f}')
print(f'Results: {OUT}')
