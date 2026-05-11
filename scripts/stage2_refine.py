"""
1x8 MMI Splitter — Stage 2: Conic Filter Refinement
Platform: SiN 800nm platform (SiO2 cladding, lambda=1550nm)
Method:   Warm start from Stage 1, conic filter R=300nm for fabrication constraints
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
from scipy.optimize import brentq
from scipy.ndimage import convolve
import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

mp.verbosity(0)

OUT  = '/mnt/c/Users/Hemera/Projects/meep_1x8_progress'
OUT2 = f'{OUT}/stage2'
os.makedirs(OUT2, exist_ok=True)

# ── Platform parameters ────────────────────────────────────────────────────────
lam0 = 1.55; n_SiN = 1.996; n_SiO2 = 1.444; t_SiN = 0.8
k0 = 2 * np.pi / lam0

def slab_eq(neff):
    ky = k0 * np.sqrt(n_SiN**2 - neff**2)
    gm = k0 * np.sqrt(neff**2 - n_SiO2**2)
    return np.tan(ky * t_SiN / 2) - gm / ky

n_eff_2d = brentq(slab_eq, 1.82, 1.99)
print(f'Slab TE0 n_eff = {n_eff_2d:.4f}')

# ── Layout (same as Stage 1) ───────────────────────────────────────────────────
N_out = 8; wg_w = 1.0; port_pitch = 1.75
mmi_W = 14.0; mmi_L = 30.0; wg_ext = 4.0; pml_th = 2.0; resolution = 15
port_ys = np.array([i - (N_out-1)/2 for i in range(N_out)]) * port_pitch
sx = mmi_L + 2*wg_ext + 2*pml_th
sy = mmi_W + 2*pml_th
des_res = 8; Nx_des = int(mmi_L*des_res); Ny_des = int(mmi_W*des_res)

SiN_mat  = mp.Medium(index=n_eff_2d)
SiO2_mat = mp.Medium(index=n_SiO2)
freq0    = 1.0 / lam0

# ── Conic filter ───────────────────────────────────────────────────────────────
conic_radius_um = 0.3                    # µm — minimum feature size target
conic_radius_px = conic_radius_um * des_res

def make_conic_kernel(radius_px):
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    kernel = np.maximum(0.0, 1.0 - np.sqrt(xx**2 + yy**2) / radius_px)
    return kernel / kernel.sum()

conic_kernel = make_conic_kernel(conic_radius_px)
print(f'Conic filter: R={conic_radius_um*1000:.0f}nm, kernel {conic_kernel.shape}')

def conic_filter(x2d):
    return convolve(x2d, conic_kernel, mode='reflect')

def sigmoid_project(x2d, beta, eta=0.5):
    xf  = conic_filter(x2d)
    num = np.tanh(beta*eta) + np.tanh(beta*(xf - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return num / den

def backprop_gradient(x2d, dJ_flat, beta, eta=0.5):
    # Chain rule: dJ/dx = C * (dsig ⊙ dJ/drho)  [C is self-adjoint]
    xf   = conic_filter(x2d)
    dsig = beta * (1 - np.tanh(beta*(xf - eta))**2)
    dsig /= np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return convolve(dsig * dJ_flat.reshape(Ny_des, Nx_des), conic_kernel, mode='reflect').flatten()

def get_beta(i):
    if i < 20: return 32
    if i < 35: return 48
    return 64

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

# ── Progress saver ─────────────────────────────────────────────────────────────
def save_progress(i, x, hist):
    x2d = sigmoid_project(x.reshape(Ny_des, Nx_des), get_beta(i))
    ext = [-mmi_L/2, mmi_L/2, -mmi_W/2, mmi_W/2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(x2d, origin='lower', cmap='RdBu_r', vmin=0, vmax=1, extent=ext)
    axes[0].set_title(f'Stage 2 design (iter {i})')
    axes[0].set_xlabel('x (um)'); axes[0].set_ylabel('y (um)')
    for y in port_ys:
        axes[0].axhline(y, color='yellow', lw=0.6, ls='--', alpha=0.7)
    axes[1].plot(hist, 'r-', lw=1.5)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('J')
    axes[1].set_title(f'Convergence  J={hist[-1]:.1f}'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT2}/progress_iter{i:03d}.png', dpi=110, bbox_inches='tight')
    plt.savefig(f'{OUT2}/progress_latest.png',       dpi=110, bbox_inches='tight')
    plt.close(fig)
    with open(f'{OUT2}/log.json', 'w') as f:
        json.dump({'iter': i, 'J_history': hist, 'beta': get_beta(i)}, f)
    np.save(f'{OUT2}/x_latest.npy', x)

# ── Load Stage 1 warm start ────────────────────────────────────────────────────
x = np.load(f'{OUT}/x_final.npy')
print(f'Stage 1 result loaded: shape={x.shape}')

# ── Adam optimizer ─────────────────────────────────────────────────────────────
n_iters = 50; lr = 0.01; save_every = 5
b1, b2, eps_a = 0.9, 0.999, 1e-8

m_ax = np.zeros_like(x)
v_ax = np.zeros_like(x)
hist = []

print(f'\nStarting Stage 2: {n_iters} iters x 2 FDTD | conic R={conic_radius_um*1000:.0f}nm')

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

np.save(f'{OUT2}/x_final_s2.npy', x)
print(f'\nDone. J: {hist[0]:.2f} -> {hist[-1]:.2f}')
print(f'Results: {OUT2}')

# ── Transmission evaluation ────────────────────────────────────────────────────
x_final  = sigmoid_project(x.reshape(Ny_des, Nx_des), beta=64).flatten()
x_binary = (x_final > 0.5).astype(float)
design_variables.update_weights(x_binary.reshape(Ny_des, Nx_des))

sim_eval = mp.Simulation(
    cell_size=mp.Vector3(sx, sy), boundary_layers=[mp.PML(pml_th)],
    geometry=geometry, sources=sources,
    default_material=SiO2_mat, resolution=resolution,
)
flux_in = sim_eval.add_flux(
    freq0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(-(mmi_L/2+wg_ext*0.5), 0), size=mp.Vector3(0, wg_w*3))
)
flux_outs = [sim_eval.add_flux(
    freq0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(mon_x, y), size=mp.Vector3(0, wg_w*3)))
    for y in port_ys
]
sim_eval.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(mon_x, port_ys[0]), 1e-4))

P_in   = mp.get_fluxes(flux_in)[0]
P_outs = [mp.get_fluxes(m)[0] for m in flux_outs]

print(f'\nInput power: {P_in:.4f}')
print('Per-port transmission:')
for i, (y, P) in enumerate(zip(port_ys, P_outs)):
    T = P / P_in
    print(f'  Port {i+1} (y={y:+.2f}um): T={T:.4f} ({100*T:.1f}%)')
total_T = sum(P_outs) / P_in
print(f'\nTotal T: {100*total_T:.1f}%  |  Ideal/port: {100/N_out:.1f}%')

summary = {
    'total_transmission': float(total_T),
    'per_port': [float(P/P_in) for P in P_outs],
    'port_ys': port_ys.tolist(),
}
with open(f'{OUT2}/transmission_s2.json', 'w') as f:
    json.dump(summary, f, indent=2)
