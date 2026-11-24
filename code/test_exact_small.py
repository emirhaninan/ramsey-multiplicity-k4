"""Brute-force check of the structured evaluator (float and mod p) on random small structured graphons."""
import itertools, numpy as np, sys
from k4struct import Struct, from_dense
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def make(n, q, D, edges):
    C = rng.integers(0, 2, (n, n)) * D; C = np.triu(C, 1); C = C + C.T   # constant blocks 0/D, diag 0
    R = np.kron(C, np.ones((q, q), dtype=np.int64))
    for a, b in edges:
        X = rng.integers(0, D + 1, (q, q)); R[a*q:(a+1)*q, b*q:(b+1)*q] = X; R[b*q:(b+1)*q, a*q:(a+1)*q] = X.T
    return R
def brute(R):
    m = R.shape[0]; T = 0
    for i, j, k, l in itertools.product(range(m), repeat=4):
        T += int(R[i,j]*R[i,k]*R[i,l]*R[j,k]*R[j,l]*R[k,l])
    return T
for trial, (n, q, edges) in enumerate([(6, 3, [(i, (i+1) % 6) for i in range(6)]),
                                       (4, 3, [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]),   # K4: triangles of fractional blocks
                                       (7, 2, [(i, (i+1) % 7) for i in range(7)] + [(i, (i+3) % 7) for i in range(7)])]):
    D = 7; R = make(n, q, D, edges); m = n*q
    for M in (R, D - R):
        exact = brute(M)
        for p in (1000003 % (1 << 20) or 999983, 524287):
            Cc, nb, F = from_dense(M, q, dtype=np.int64); st = Struct(Cc % p, nb, F % p, mod=p)
            t = 0
            for a0 in range(n):
                V = st.rows(a0, list(range(q))); t = (t + int(np.sum(V * st.S_rows(V, a0)) % p)) % p
            assert t == exact % p, (trial, p, t, exact % p)
        Cc, nb, F = from_dense(M.astype(float), q); st = Struct(Cc, nb, F)
        tf = sum(float(np.sum(st.rows(a0, list(range(q))) * st.S_rows(st.rows(a0, list(range(q))), a0))) for a0 in range(n))
        assert abs(tf - exact) <= 1e-9 * exact
    print('trial', trial, 'ok', exact)
print('ALL OK')
