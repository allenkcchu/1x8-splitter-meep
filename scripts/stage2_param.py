"""
1x8 MMI Splitter — Stage 2: Conic Filter Refinement
Warm start from Stage 1 variant, conic filter R=300nm.
Usage: python stage2_param.py --variant L20W14
"""

import warnings; warnings.filterwarnings('ignore')
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.ndimage import convolve
from autograd import numpy as npa
import meep as mp
import meep.adjoint as mpa

mp.verbosity(0)

parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True, help='e.g. L20W14')
parser.add_argument('--n_iters', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.12, help='uniformity penalty weight')
parser.add_argument('--resume', action='store_true', help='Resume from x_latest.npy in OUT2 (or --warmstart path)')
parser.add_argument('--warmstart', default=None, help='Path to .npy for warm start (overrides default resume path)')
parser.add_argument('--base', default='/mnt/c/Users/Hemera/Projects/meep_1x8_progress', help='Base directory')
parser.add_argument('--out', default=None, help='Override output directory (OUT2)')
parser.add_argument('--start_iter', type=int, default=0, help='Iteration offset for beta schedule and progress numbering')
args = parser.parse_args()

BASE  = args.base
OUTV1 = f'{BASE}/stage1_{args.variant}'
OUT2  = args.out if args.out else f'{BASE}/stage2_{args.variant}'
os.makedirs(OUT2, exist_ok=True)
print(f'=== Stage 2: variant={args.variant}  out={OUT2} ===')

# ── Load Stage 1 config + log ──────────────────────────────────────────────────
with open(f'{OUTV1}/config.json') as f:
    cfg = json.load(f)
with open(f'{OUTV1}/log.json') as f:
    log = json.load(f)

mmi_L      = log['mmi_L']
mmi_W      = log['mmi_W']
geo        = log['geo']
port_ys    = np.array(log['port_ys'])
N_out      = cfg['N_out']
Nx_des     = cfg['Nx_des']
Ny_des     = cfg['Ny_des']
resolution = cfg['resolution']
wg_ext     = cfg['wg_ext']
pml_th     = cfg['pml_th']
n_eff_2d   = cfg['n_eff_2d']
freq0      = cfg['freq0']
wg_w_in    = geo['wg_w_in']
wg_w_out   = geo['wg_w_out']

print(f'mmi={mmi_L}x{mmi_W} µm  n_eff={n_eff_2d:.4f}')
print(f'geo: {geo}')
print(f'port_ys: {np.round(port_ys, 3)}')

sx = mmi_L + 2*wg_ext + 2*pml_th
sy = mmi_W + 2*pml_th
SiN_mat  = mp.Medium(index=n_eff_2d)
SiO2_mat = mp.Medium(index=1.444)
mon_x    = mmi_L/2 + wg_ext*0.5

# ── Conic filter ───────────────────────────────────────────────────────────────
des_res        = cfg['des_res']
conic_R_um     = 0.3
conic_R_px     = conic_R_um * des_res

def make_conic_kernel(r_px):
    r = int(np.ceil(r_px))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    k = np.maximum(0.0, 1.0 - np.sqrt(xx**2 + yy**2) / r_px)
    return k / k.sum()

conic_kernel = make_conic_kernel(conic_R_px)
print(f'Conic filter: R={conic_R_um*1000:.0f}nm  kernel {conic_kernel.shape}')

def conic_filter(x2d):
    return convolve(x2d, conic_kernel, mode='reflect')

def sigmoid_project(x2d, beta, eta=0.5):
    xf = conic_filter(x2d)
    return (np.tanh(beta*eta) + np.tanh(beta*(xf - eta))) / \
           (np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta)))

def backprop_gradient(x2d, dJ_flat, beta, eta=0.5):
    xf   = conic_filter(x2d)
    dsig = beta * (1 - np.tanh(beta*(xf - eta))**2)
    dsig /= np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return convolve(dsig * dJ_flat.reshape(Ny_des, Nx_des), conic_kernel, mode='reflect').flatten()

def get_beta(i):
    if i < 20: return 32
    if i < 35: return 48
    return 64

# ── Simulation setup ───────────────────────────────────────────────────────────
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx_des, Ny_des), SiO2_mat, SiN_mat,
    weights=np.ones((Ny_des, Nx_des)) * 0.5, do_averaging=True,
)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(mmi_L, mmi_W))
)
geometry = [
    mp.Block(size=mp.Vector3(wg_ext+pml_th, wg_w_in),
             center=mp.Vector3(-(mmi_L/2+(wg_ext+pml_th)/2), 0), material=SiN_mat),
    mp.Block(size=mp.Vector3(mmi_L, mmi_W), center=mp.Vector3(), material=design_variables),
]
for y in port_ys:
    geometry.append(mp.Block(size=mp.Vector3(wg_ext+pml_th, wg_w_out),
                             center=mp.Vector3(mmi_L/2+(wg_ext+pml_th)/2, y), material=SiN_mat))

sources = [mp.EigenModeSource(
    mp.GaussianSource(freq0, fwidth=0.05*freq0),
    size=mp.Vector3(0, wg_w_in*4),
    center=mp.Vector3(-(mmi_L/2+wg_ext*0.5), 0),
    eig_match_freq=True, eig_parity=mp.ODD_Z,
)]
sim = mp.Simulation(
    cell_size=mp.Vector3(sx, sy), boundary_layers=[mp.PML(pml_th)],
    geometry=geometry, sources=sources,
    default_material=SiO2_mat, resolution=resolution,
)
ob_list = [
    mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(mon_x, y), size=mp.Vector3(0, wg_w_out*4)),
        1, eig_parity=mp.ODD_Z, forward=True)
    for y in port_ys
]

# J = total * (1 - alpha * norm_CV^2)  — same objective as Stage 1
def J(*args):
    powers = npa.array([npa.abs(npa.squeeze(a))**2 for a in args])
    total  = npa.sum(powers)
    mean   = total / N_out
    cv2    = npa.sum((powers - mean)**2) / N_out / (mean**2 + 1e-12)
    return total * (1.0 - args.alpha * cv2)

# Closure to capture alpha
def make_J(alpha):
    def _J(*args):
        powers = npa.array([npa.abs(npa.squeeze(a))**2 for a in args])
        total  = npa.sum(powers)
        mean   = total / N_out
        cv2    = npa.sum((powers - mean)**2) / N_out / (mean**2 + 1e-12)
        return total * (1.0 - alpha * cv2)
    return _J

opt = mpa.OptimizationProblem(
    simulation=sim, objective_functions=make_J(args.alpha),
    objective_arguments=ob_list, design_regions=[design_region], frequencies=[freq0],
)

# ── Progress saver ─────────────────────────────────────────────────────────────
def save_progress(abs_i, x, hist):
    x2d = sigmoid_project(x.reshape(Ny_des, Nx_des), get_beta(abs_i))
    ext = [-mmi_L/2, mmi_L/2, -mmi_W/2, mmi_W/2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(x2d, origin='lower', cmap='RdBu_r', vmin=0, vmax=1, extent=ext)
    axes[0].set_title(f'Stage 2 [{args.variant}] iter {abs_i}')
    axes[0].set_xlabel('x (µm)'); axes[0].set_ylabel('y (µm)')
    for y in port_ys:
        axes[0].axhline(y, color='yellow', lw=0.6, ls='--', alpha=0.7)
    axes[1].plot(hist, 'r-', lw=1.5)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('J')
    axes[1].set_title(f'J={hist[-1]:.2f}'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT2}/progress_iter{abs_i:03d}.png', dpi=110, bbox_inches='tight')
    plt.savefig(f'{OUT2}/progress_latest.png',           dpi=110, bbox_inches='tight')
    plt.close(fig)
    with open(f'{OUT2}/log.json', 'w') as f:
        json.dump({'iter': abs_i, 'J_history': hist, 'beta': get_beta(abs_i),
                   'alpha': args.alpha, 'variant': args.variant}, f)
    np.save(f'{OUT2}/x_latest.npy', x)

# ── Load warm start ────────────────────────────────────────────────────────────
if args.resume:
    src = args.warmstart if args.warmstart else f'{BASE}/stage2_{args.variant}/x_latest.npy'
    x = np.load(src)
    hist = []
    print(f'Warm start (Stage 2 restart): {src}  shape={x.shape}')
else:
    x = np.load(f'{OUTV1}/x_final.npy')
    hist = []
    print(f'Warm start loaded: {OUTV1}/x_final.npy  shape={x.shape}')

# ── Adam optimizer ─────────────────────────────────────────────────────────────
n_iters = args.n_iters; lr = args.lr
b1, b2, eps_a = 0.9, 0.999, 1e-8
m_ax = np.zeros_like(x); v_ax = np.zeros_like(x)

print(f'\nStarting Stage 2: {n_iters} iters | conic R={conic_R_um*1000:.0f}nm | alpha={args.alpha} | lr={lr} | start_iter={args.start_iter}')

for i in range(n_iters):
    abs_i = args.start_iter + i
    beta  = get_beta(abs_i)
    x2d   = x.reshape(Ny_des, Nx_des)
    xp    = sigmoid_project(x2d, beta).flatten()

    f_val, dJ = opt([xp])
    grad = backprop_gradient(x2d, np.asarray(dJ).flatten(), beta)

    m_ax = b1*m_ax + (1-b1)*grad
    v_ax = b2*v_ax + (1-b2)*grad**2
    m_h  = m_ax / (1 - b1**(i+1))
    v_h  = v_ax / (1 - b2**(i+1))
    x    = np.clip(x + lr * m_h / (np.sqrt(v_h) + eps_a), 0, 1)

    hist.append(float(f_val))
    print(f'  [S2/{args.variant}] Iter {abs_i:3d} | beta={beta:2d} | alpha={args.alpha} | J={float(f_val):.4f}', flush=True)

    if i % 5 == 0 or i == n_iters - 1:
        save_progress(abs_i, x, hist)

np.save(f'{OUT2}/x_final_s2.npy', x)
print(f'\nDone. J: {hist[0]:.4f} → {hist[-1]:.4f}')
print(f'Results: {OUT2}')
