"""Structured evaluator for monochromatic-K4 density of step graphons of the form
   W[(a,s),(b,t)] = Cc[a,b]          if (a,b) is a 'constant' coarse pair,
                  = F[a,k][s,t]      if b = nb[a,k] is a 'fractional' partner of a,
with n coarse blocks of size q (all n*q fine blocks of equal weight).
For a fine row i=(a0,s0) with V = W[i] (shape n x q) we compute S[a,s] = (W D_V W D_V W)[(a,s),(a,s)]
by splitting each closed walk a->c->b->a into constant/fractional edges (8 cases).
Then sum_i sum_j V_j S_j = sum_{i,j,k,l} W_ij W_ik W_il W_jk W_jl W_kl, and dT/dW_ij = 12 S_i[j] (i != j).
The same code works over the integers mod p (int64, p < 2^20 keeps all sums exact) for certification."""
import numpy as np

class Struct:
    def __init__(self, Cc, nb, F, mod=None):
        self.Cc, self.nb, self.F, self.mod = Cc, nb, F, mod
        n, K = nb.shape
        self.n, self.K, self.q = n, K, F.shape[-1]
        # position of a in nb[c]: rev[a,k] = index k' with nb[nb[a,k],k'] = a
        self.rev = np.array([[list(nb[nb[a, k]]).index(a) for k in range(K)] for a in range(n)])
        # C between neighbours: Cnn[a,k,l] = Cc[nb[a,k], nb[a,l]]
        self.Cnn = Cc[nb[:, :, None], nb[:, None, :]]
        # Cn2[a,k,l] = Cc[nb[nb[a,k], l], a]
        self.Cn2 = Cc[nb[nb], np.arange(n)[:, None, None]]
        self.Cnb = Cc[nb]                          # (n,K,n): Cc[nb[a,k], x]
        # closed walks a -> c=nb[a,k] -> b=nb[c,j] -> a with (b,a) fractional
        W = []
        S = set(map(tuple, [(a, c) for a in range(n) for c in nb[a]]))
        pos = {(a, int(nb[a, k])): k for a in range(n) for k in range(K)}
        for a in range(n):
            for k in range(K):
                c = int(nb[a, k])
                for j in range(K):
                    b = int(nb[c, j])
                    if (b, a) in S:
                        W.append((a, k, c, j, b, pos[(a, b)]))
        self.walks = np.array(W, dtype=np.int64).reshape(-1, 6)
        a, k, c, j, b, kb = self.walks.T
        # T0_w = diag(F_ac F_cb F_ba), used when V is constant on blocks c and b
        P = self.red(np.matmul(F[a, k], F[c, j]))
        self.T0 = self.red((P * F[a, kb]).sum(-1))
        # walks whose middle vertices c or b are fractional partners of a0
        self.touch = []
        for a0 in range(n):
            N = set(map(int, nb[a0]))
            self.touch.append(np.array([w for w in range(len(W)) if c[w] in N or b[w] in N], dtype=np.int64))

    def red(self, x):
        return x % self.mod if self.mod else x

    def mm(self, eq, *ops):
        r = np.einsum(eq, *ops, optimize=True)
        return r % self.mod if self.mod else r

    def S_rows(self, V, a0):
        """V: (B,n,q) rows of fine blocks (a0, s); returns S: (B,n,q)."""
        Cc, nb, F, K = self.Cc, self.nb, self.F, self.K
        md = self.mod
        Vn = V[:, nb]                                          # (B,n,K,q)  V[nb[a,k]]
        vs = self.red(V.sum(-1))                               # (B,n)
        fv = self.mm('akst,bakt->baks', F, Vn)                 # F_{a,c} V_c
        # H = C diag(vs) C
        CV = self.red(Cc[None] * vs[:, None, :])
        N = nb[a0]
        vs0 = self.red(Cc[a0] * self.q)                        # vs on constant blocks (0 on N)
        H0 = self.red((Cc * vs0[None, :]) @ Cc)
        CN = Cc[N]                                             # (K,n)
        H = self.red(H0[None] + self.mm('kx,bk,ky->bxy', CN, vs[:, N], CN)) if not self.mod else \
            self.red(H0[None] + self.mm('kx,bky->bxy', CN, self.red(vs[:, N, None] * CN[None])))
        # EEE[a] = sum_c H[a,c] vs[c] C[c,a] = sum_c H[a,c] CV[a,c] (C symmetric)
        EEE = self.red((H * CV).sum(-1))
        # GEE[a,s] = sum_k fv[a,k,s] H[nb[a,k], a]
        Hg = H[:, nb, np.arange(self.n)[:, None]]              # (B,n,K): H[nb[a,k], a]
        GEE = self.mm('baks,bak->bas', fv, Hg)
        # EGE[a] = sum_{c,k} C[a,c] q[c,k] C[nb[c,k],a],  q[c,k] = V_c . fv[c,k]
        qq = self.mm('bcs,bcks->bck', V, fv)
        Y = self.mm('bck,cka->bca', qq, self.Cnb)              # (B,n(c),n(a))
        EGE = self.red((Cc[None] * np.swapaxes(Y, 1, 2)).sum(-1))  # sum_c C[a,c] Y[c,a]
        # GGE[a,s] = sum_k (F_{a,c} (V_c * z[a,k]))[s], z[a,k] = sum_l C[nb[c,l],a] fv[c,l]
        fvc = fv[:, nb]                                        # (B,n,K,K,q)  fv[nb[a,k], l]
        z = self.mm('akl,baklu->baku', self.Cn2, fvc)
        GGE = self.mm('akst,bakt->bas', F, self.red(Vn * z))
        # GEG[a,s] = sum_{k,l} C[nb[a,k],nb[a,l]] fv[a,k,s] fv[a,l,s]
        u = self.mm('akl,bals->baks', self.Cnn, fv)
        GEG = self.mm('baks,baks->bas', fv, u)
        # GGG over walks (a,k,c,j,b,kb): diag(F_ac D_c F_cb D_b F_ba)[s]
        al = Cc[a0]
        a, k, c, j, b, kb = self.walks.T
        A0 = np.zeros((self.n, self.q), dtype=V.dtype)
        np.add.at(A0, a, self.red(self.red(al[c] * al[b])[:, None] * self.T0))
        w = self.walks[self.touch[a0]]
        a, k, c, j, b, kb = w.T
        M1 = self.red(F[a, k][None] * V[:, c][:, :, None, :])     # (B,Wk,q,q)  F_ac D_c
        M2 = self.red(F[c, j][None] * V[:, b][:, :, None, :])     # F_cb D_b
        P = self.red(np.matmul(M1, M2))
        g = self.red((P * F[a, kb][None]).sum(-1))                # rowsum(P o F_ab)  (F_ba^T = F_ab)
        GGG = np.zeros_like(GEE) + A0[None]
        np.add.at(GGG, (slice(None), a), g)
        GGG = self.red(GGG)
        S = EEE[:, :, None] + 2 * GEE + EGE[:, :, None] + 2 * GGE + GEG + GGG
        return self.red(S)

    def rows(self, a0, s_list):
        """the fine rows (a0, s) for s in s_list, shape (B,n,q)"""
        B = len(s_list)
        V = np.repeat(np.broadcast_to(self.Cc[a0][None, :, None], (1, self.n, self.q)), B, 0).copy()
        V[:, self.nb[a0]] = np.transpose(self.F[a0][:, s_list, :], (1, 0, 2))
        return V


def from_dense(R, q, mod=None, dtype=np.float64):
    """R: dense m x m (values), split into n x n coarse blocks of size q."""
    m = R.shape[0]; n = m // q
    C4 = R.reshape(n, q, n, q).transpose(0, 2, 1, 3)
    nc = C4.max(axis=(2, 3)) != C4.min(axis=(2, 3))
    deg = nc.sum(1); K = deg.max(); assert (deg == K).all()
    nb = np.array([np.nonzero(nc[a])[0] for a in range(n)])
    Cc = np.where(nc, 0, C4[:, :, 0, 0]).astype(dtype)
    F = C4[np.arange(n)[:, None], nb].astype(dtype)          # (n,K,q,q)
    return Cc, nb, F
