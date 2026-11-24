"""Independent float64 check: dense evaluation over every fine row, sharing no code with k4struct.
c4 = (1/m^4) sum_i [tr((D_i W)^3) + tr((D_i (1-W))^3)].   usage: dense_float_check.py witness.json.gz"""
import gzip, json, sys, time, numpy as np
d = json.load(gzip.open(sys.argv[1])); D = d['edge_probability_denominator']
W = np.array(d['red_probability_numerators'], dtype=np.float64) / D; m = W.shape[0]
tot = 0.0; t0 = time.time()
for M in (W, 1.0 - W):
    for i in range(m):
        A = M * M[i][None, :]           # A_jk = M_jk M_ik
        tot += float(np.einsum('jk,kj->', A @ A, A))   # tr(A^3) = sum_{jkl} M_ij M_ik M_il M_jk M_kl M_lj
        if i % 500 == 0: print(i, time.time() - t0, flush=True)
print(f'c4 ~ {tot / m ** 4:.15f}')
