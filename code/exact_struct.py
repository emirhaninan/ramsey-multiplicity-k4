#!/usr/bin/env python3
"""Exact monochromatic-K4 density of an equal-weight rational step graphon given as a dense integer
table R (m x m, values R_ij / D), using the coarse-block structure (blocks of size q constant except on
'fractional' coarse pairs).  Every fine row i is summed (no symmetry assumed):
    numerator N = sum_i [ sum_{j,k,l} R_ij R_ik R_il R_jk R_jl R_kl ] + same for B = D - R,
    c4 = N / (m^4 D^6).
N is computed modulo primes p < 2^20 in int64 (all intermediate sums stay < 2^63) and combined by CRT,
with one extra prime as a consistency check.
usage: exact_struct.py witness.json.gz [q] [procs]"""
import gzip, json, sys, time, os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from fractions import Fraction
from math import prod
import numpy as np, multiprocessing as mp
from k4struct import Struct, from_dense

def primes_below(n, k):
    out, x = [], n - 1
    while len(out) < k:
        if all(x % d for d in range(2, int(x ** .5) + 1)): out.append(x)
        x -= 1
    return out

_S = {}
def _init(R, D, q, p):
    for key, M in (('r', R), ('b', D - R)):
        Cc, nb, F = from_dense(M, q, dtype=np.int64)
        _S[key] = Struct(Cc % p, nb, F % p, mod=p)
    _S['p'] = p

def _row(a0):
    p = _S['p']; t = 0
    for key in ('r', 'b'):
        st = _S[key]; V = st.rows(a0, list(range(st.q)))
        S = st.S_rows(V, a0)
        t = (t + int(np.sum((V * S) % p) % p)) % p
    return t

def numerator_mod(R, D, q, p, procs):
    with mp.get_context('fork').Pool(procs, initializer=_init, initargs=(R, D, q, p)) as pool:
        return sum(pool.map(_row, range(R.shape[0] // q))) % p

def main():
    path = sys.argv[1]; q = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = json.load(gzip.open(path) if path.endswith('.gz') else open(path))
    D = int(d['edge_probability_denominator'])
    R = np.array(d['red_probability_numerators'], dtype=np.int64); m = R.shape[0]
    assert np.array_equal(R, R.T) and R.min() >= 0 and R.max() <= D and m % q == 0
    assert len(set(Fraction(x) for x in d['block_weights'])) == 1, 'uniform weights only'
    # the structural assumption is checked by from_dense (each coarse block constant or 'fractional')
    bound = 2 * m ** 4 * D ** 6
    P = primes_below(1 << 20, 40); need = []
    for p in P:
        need.append(p)
        if prod(need) > 2 * bound: break
    extra = P[len(need)]
    res = []
    for p in need + [extra]:
        t0 = time.time(); r = numerator_mod(R, D, q, p, procs); res.append(r)
        print(f'  p={p}: {r} ({time.time()-t0:.0f}s)', flush=True)
    M = prod(need); N = 0
    for r, p in zip(res, need):
        Mi = M // p; N = (N + r * Mi * pow(Mi, -1, p)) % M
    assert 0 <= N <= bound and N % extra == res[-1], 'CRT / extra-prime check failed'
    c4 = Fraction(N, m ** 4 * D ** 6)
    print(f'm={m} D={D} primes={len(need)}+1')
    print(f'c4 = {c4.numerator}/{c4.denominator}')
    print(f'c4 = {float(c4):.16f}')
    return c4

if __name__ == '__main__':
    main()
