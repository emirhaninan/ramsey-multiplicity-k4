# Independent sampled check of the monochromatic-K4 density of a step graphon (equal block weights).
# c4 = t(K4,W) + t(K4,1-W); per block i: f_i = [tr((D_i W)^3) + tr((D_i' W')^3)] / m^3, c4 = mean_i f_i.
import gzip, json, numpy as np, sys, time
d = json.load(gzip.open(sys.argv[1])); nsamp = int(sys.argv[2])
assert d['schema'] == 'rational-step-graphon-v1'
bw = d['block_weights']; m = len(bw); assert len(set(map(str, bw))) == 1, 'unequal weights'
D = d['edge_probability_denominator']
R = np.array(d['red_probability_numerators'], dtype=np.float64); assert R.shape == (m, m)
assert (R == R.T).all() and R.min() >= 0 and R.max() <= D
W = R / D; del R
B = 1.0 - W
rng = np.random.default_rng(12345); rows = rng.choice(m, nsamp, replace=False)
vals = []
for c, i in enumerate(rows):
    t = time.time(); s = 0.0
    for M in (W, B):
        X = M * M[i][:, None]          # D_i M
        s += np.einsum('ij,ji->', X @ X, X)
    vals.append(s / m**3)
    print(c, i, f'{vals[-1]:.15f}', f'{time.time()-t:.1f}s', flush=True)
v = np.array(vals); mean = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
claimed = 73070278155526479765345578606645339095483962096083909137 / 2424455836135488232949728695296387575646833376892026880000
print('sample mean', f'{mean:.13f}', '+-', f'{se:.2e}', ' claimed', f'{claimed:.13f}', ' diff/se', (mean-claimed)/se)
print('previous record SK2026 0.030138879313256; margin', 0.030138879313256-claimed, '= ', (0.030138879313256-mean)/se, 'se')
