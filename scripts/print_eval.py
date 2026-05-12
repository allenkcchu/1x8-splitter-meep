import json, numpy as np, os

BASE = '/mnt/c/Users/Hemera/Projects/meep_1x8_progress'

variants = [
    ('L20W14', f'{BASE}/stage1_L20W14/transmission_eval.json'),
    ('L20W16', f'{BASE}/stage1_L20W16/transmission_eval.json'),
    ('L20W18', f'{BASE}/stage1_L20W18/transmission_eval.json'),
    ('L30W14', f'{BASE}/stage1_L30W14/transmission_eval_v2.json'),
    ('L40W14', f'{BASE}/stage1_L40W14/transmission_eval.json'),
    ('L40W16', f'{BASE}/stage1_L40W16/transmission_eval.json'),
    ('L50W14', f'{BASE}/stage1_L50W14/transmission_eval.json'),
]

print(f'{"Variant":<10} {"Total T":>8} {"Mean":>7} {"Std":>7} {"Imbalance":>10}')
print('-' * 48)
for name, path in variants:
    if not os.path.exists(path):
        print(f'{name:<10}  (no eval yet)')
        continue
    d = json.load(open(path))
    T = np.array(d['per_port_transmission'])
    total = d['total_transmission_fullwidth']
    print(f'{name:<10} {100*total:>7.1f}% {100*T.mean():>6.2f}% {100*T.std():>6.2f}% {100*(T.max()-T.min()):>9.2f}%')

print()
for name, path in variants:
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    T = np.array(d['per_port_transmission'])
    total = d['total_transmission_fullwidth']
    print(f'[{name}]  Total={100*total:.1f}%  Std={100*T.std():.2f}%  Imbalance={100*(T.max()-T.min()):.2f}%')
    for i, t in enumerate(T):
        bar = '█' * max(0, int(t * 160))
        print(f'    Port {i+1}: {100*t:5.2f}%  {bar}')
    print()
