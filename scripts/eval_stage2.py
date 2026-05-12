"""
Transmission evaluation for Stage 2 variants.
Reads layout config from stage1_<variant>/, loads x_final_s2.npy from stage2_<variant>/.
Usage: python eval_stage2.py --variant L20W14
"""

import warnings; warnings.filterwarnings('ignore')
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import meep as mp

mp.verbosity(0)

parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True)
args = parser.parse_args()

BASE  = '/mnt/c/Users/Hemera/Projects/meep_1x8_progress'
OUTV1 = f'{BASE}/stage1_{args.variant}'
OUT2  = f'{BASE}/stage2_{args.variant}'

with open(f'{OUTV1}/config.json') as f:
    cfg = json.load(f)
with open(f'{OUTV1}/log.json') as f:
    log = json.load(f)

mmi_L      = log['mmi_L'];  mmi_W  = log['mmi_W']
geo        = log['geo'];    port_ys = np.array(log['port_ys'])
N_out      = cfg['N_out'];  Nx_des  = cfg['Nx_des'];  Ny_des = cfg['Ny_des']
resolution = cfg['resolution']
wg_ext     = cfg['wg_ext']; pml_th  = cfg['pml_th']
n_eff_2d   = cfg['n_eff_2d']; freq0 = cfg['freq0']
wg_w_in    = geo['wg_w_in']; wg_w_out = geo['wg_w_out']
port_pitch = geo['port_pitch']

print(f'Stage 2 eval: {args.variant}  mmi={mmi_L}x{mmi_W} µm')
print(f'Geo: {geo}')

SiN_mat  = mp.Medium(index=n_eff_2d)
SiO2_mat = mp.Medium(index=1.444)
sx = mmi_L + 2*wg_ext + 2*pml_th
sy = mmi_W + 2*pml_th
src_x    = -(mmi_L/2 + wg_ext*0.5)
mon_in_x = src_x + 1.0
mon_out_x = mmi_L/2 + wg_ext*0.5

# ── Binarize Stage 2 design ────────────────────────────────────────────────────
from scipy.ndimage import convolve

des_res    = cfg['des_res']
conic_R_px = 0.3 * des_res

def make_conic_kernel(r_px):
    r = int(np.ceil(r_px))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    k = np.maximum(0.0, 1.0 - np.sqrt(xx**2 + yy**2) / r_px)
    return k / k.sum()

conic_kernel = make_conic_kernel(conic_R_px)

x = np.load(f'{OUT2}/x_final_s2.npy')
x2d = x.reshape(Ny_des, Nx_des)
xf  = convolve(x2d, conic_kernel, mode='reflect')
beta = 64; eta = 0.5
x_soft = (np.tanh(beta*eta) + np.tanh(beta*(xf - eta))) / \
         (np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta)))
x_bin  = (x_soft > 0.5).astype(float)
gray_frac = float(np.mean((x_soft > 0.05) & (x_soft < 0.95)))
print(f'Gray fraction (5-95%): {100*gray_frac:.1f}%')

# ── Geometry ───────────────────────────────────────────────────────────────────
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx_des, Ny_des), SiO2_mat, SiN_mat,
    weights=x_bin, do_averaging=True)
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
    size=mp.Vector3(0, wg_w_in*4), center=mp.Vector3(src_x, 0),
    eig_match_freq=True, eig_parity=mp.ODD_Z)]

# ── Simulation ─────────────────────────────────────────────────────────────────
print('\nRunning FDTD...', flush=True)
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), boundary_layers=[mp.PML(pml_th)],
                    geometry=geometry, sources=sources,
                    default_material=SiO2_mat, resolution=resolution)

mon_in = sim.add_flux(freq0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(mon_in_x, 0), size=mp.Vector3(0, wg_w_in*4)))

mon_w = min(wg_w_out*1.4, port_pitch*0.75)
mon_ports = [sim.add_flux(freq0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(mon_out_x, y), size=mp.Vector3(0, mon_w)))
    for y in port_ys]

mon_total = sim.add_flux(freq0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(mon_out_x, 0), size=mp.Vector3(0, sy-2*pml_th)))

sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(mon_out_x, port_ys[0]), 1e-4))
print('Done.')

# ── Results ────────────────────────────────────────────────────────────────────
P_in      = mp.get_fluxes(mon_in)[0]
P_ports   = [mp.get_fluxes(m)[0] for m in mon_ports]
P_tot_out = mp.get_fluxes(mon_total)[0]

Ts       = [P/P_in for P in P_ports]
T_arr    = np.array(Ts)
total_T  = P_tot_out / P_in
port_sum = sum(Ts)

print(f'\n{"="*52}')
print(f'[Stage 2 / {args.variant}] Transmission Results')
print(f'{"="*52}')
print(f'Ideal/port = {100/N_out:.2f}%\n')
for i, (y, T) in enumerate(zip(port_ys, Ts)):
    bar = '█' * max(0, int(T*160))
    print(f'  Port {i+1} (y={y:+.3f}µm): {100*T:6.2f}%  {bar}')
print(f'\n  Sum of port T     : {100*port_sum:.2f}%')
print(f'  Full cross-sec T  : {100*total_T:.2f}%')
print(f'  Mean/port         : {100*T_arr.mean():.2f}%')
print(f'  Std dev           : {100*T_arr.std():.2f}%')
print(f'  Max imbalance     : {100*(T_arr.max()-T_arr.min()):.2f}%')
print(f'{"="*52}')

# ── Compare with Stage 1 ───────────────────────────────────────────────────────
s1_path = f'{OUTV1}/transmission_eval.json'
if os.path.exists(s1_path):
    s1 = json.load(open(s1_path))
    T1 = np.array(s1['per_port_transmission'])
    print(f'\nStage 1 → Stage 2 comparison:')
    print(f'  Total T  : {100*s1["total_transmission_fullwidth"]:.1f}% → {100*total_T:.1f}%')
    print(f'  Std      : {100*T1.std():.2f}% → {100*T_arr.std():.2f}%')
    print(f'  Imbalance: {100*(T1.max()-T1.min()):.2f}% → {100*(T_arr.max()-T_arr.min()):.2f}%')

# ── Figures ────────────────────────────────────────────────────────────────────
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy), component=mp.Ez)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(x_bin, origin='lower', cmap='binary_r',
               extent=[-mmi_L/2, mmi_L/2, -mmi_W/2, mmi_W/2])
axes[0].set_title(f'Stage 2 [{args.variant}] — Binary\npitch={port_pitch:.3f}  w_in={wg_w_in:.3f}  w_out={wg_w_out:.3f}')
axes[0].set_xlabel('x (µm)'); axes[0].set_ylabel('y (µm)')
for y in port_ys:
    axes[0].axhline(y, color='cyan', lw=0.5, ls='--', alpha=0.6)

ext_sim = [-sx/2, sx/2, -sy/2, sy/2]
vmax = np.percentile(np.abs(ez_data), 99)
axes[1].imshow(ez_data.T, origin='lower', cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, extent=ext_sim, aspect='auto')
axes[1].set_title('Ez field'); axes[1].set_xlabel('x (µm)')
axes[1].axvline(-mmi_L/2, color='white', lw=0.5, ls=':')
axes[1].axvline(mmi_L/2,  color='white', lw=0.5, ls=':')

bars = axes[2].bar(range(1, N_out+1), [100*t for t in Ts], color='steelblue', alpha=0.8, label='Stage 2')
if os.path.exists(s1_path):
    s1 = json.load(open(s1_path))
    axes[2].plot(range(1, N_out+1), [100*t for t in s1['per_port_transmission']],
                 'r--o', ms=4, lw=1.2, label='Stage 1')
axes[2].axhline(100/N_out, color='gray', ls=':', lw=1.5, label=f'Ideal {100/N_out:.2f}%')
axes[2].set_xlabel('Port'); axes[2].set_ylabel('Transmission (%)')
axes[2].set_title(f'Stage 2: total={100*total_T:.1f}%  std={100*T_arr.std():.2f}%')
axes[2].set_xticks(range(1, N_out+1)); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
fig_path = f'{OUT2}/transmission_eval.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'\nFigure: {fig_path}')

# ── JSON ───────────────────────────────────────────────────────────────────────
result = {
    'variant': args.variant, 'stage': 2,
    'mmi_L': mmi_L, 'mmi_W': mmi_W, 'geo': geo,
    'port_ys_um': port_ys.tolist(),
    'P_in': float(P_in), 'P_total_out': float(P_tot_out),
    'total_transmission_fullwidth': float(total_T),
    'per_port_transmission': [float(t) for t in Ts],
    'port_sum_transmission': float(port_sum),
    'mean_T': float(T_arr.mean()), 'std_T': float(T_arr.std()),
    'max_imbalance': float(T_arr.max() - T_arr.min()),
    'gray_fraction': gray_frac,
}
json_path = f'{OUT2}/transmission_eval.json'
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'JSON:   {json_path}')
