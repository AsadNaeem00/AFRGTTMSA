"""
Simulation study: effect of machine-translation errors in multilingual sentiment features
on GNN sanction-impact prediction. All data are SYNTHETIC. Pure numpy/scipy/sklearn.
"""
import numpy as np, json, sys
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

N = 500
SEEDS = list(range(40))
REPS = 5
RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
SIG_L = 0.30                    # lexical-drift std of a translation error (assumed)
PHI = 0.25                      # share of translation errors that reverse polarity (negation/antonym errors) (assumed)
SIG_D = 0.20                    # scoring noise of direct multilingual scoring, no translation (assumed)
C_FN, C_FP = 5.0, 1.0           # scenario costs
DRIFT_SCALE_MIT, POL_KEEP_MIT = 0.5, 0.3   # reasoning MT: halves drift magnitude; leaves 30% of polarity errors (assumed)


def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def make_world(seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, (N, 2))
    g = rng.lognormal(0, 1, N)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    base = np.sqrt(np.outer(g, g)) / (1 + 5 * d)
    lo, hi = 1e-4, 10
    for _ in range(40):  # calibrate to mean degree ~10
        k = np.sqrt(lo * hi)
        P = np.clip(k * base, 0, 1); np.fill_diagonal(P, 0)
        deg = P.sum(1).mean()
        if deg > 10: hi = k
        else: lo = k
    U = rng.uniform(0, 1, (N, N)); U = np.triu(U, 1); U = U + U.T
    Aun = (U < P).astype(float); Aun = np.triu(Aun, 1); Aun = Aun + Aun.T
    W = Aun * (np.outer(g, g) / (1 + 5 * d))
    W = W / (W.max() + 1e-12) * 5 + Aun * 0.05
    A = W + np.eye(N)
    dinv = 1 / np.sqrt(A.sum(1))
    Ahat = dinv[:, None] * A * dinv[None]
    # row-normalised neighbour operator (no self loops) for imputation
    rs = W.sum(1, keepdims=True); Pn = np.where(rs > 0, W / np.maximum(rs, 1e-12), 0)
    # features
    dlang = np.clip(0.5 + 0.7 * (pos[:, 1] - 0.5) + rng.normal(0, 0.15, N), 0, 1)   # share of coverage in low-resource languages
    x1 = (np.log(g) - np.log(g).mean()) / np.log(g).std()
    x2 = 0.5 * x1 + rng.normal(0, 1, N)
    x3 = rng.beta(2, 5, N); x3 = (x3 - x3.mean()) / x3.std()
    align = 1.5 * (pos[:, 0] - 0.5) * 2 + rng.normal(0, 0.7, N)
    eps = rng.normal(0, 1, N)
    tau = np.tanh(0.5 * (Ahat @ align) + 0.9 * eps)          # latent true sentiment
    s_clean = np.clip(tau + rng.normal(0, 0.10, N), -1, 1)    # clean pipeline output
    z = 0.6 * x3 + 0.3 * x1 + 0.3 * align - 1.5 * (Ahat @ tau) + rng.normal(0, 0.5, N)
    y = (z > np.quantile(z, 0.65)).astype(float)
    E = g * rng.lognormal(0, 0.5, N)                           # exposure
    idx = rng.permutation(N)
    tr, va, te = idx[:int(.6 * N)], idx[int(.6 * N):int(.8 * N)], idx[int(.8 * N):]
    deg_un = Aun.sum(1)
    return dict(Ahat=Ahat, Pn=Pn, W=W, X0=np.c_[x1, x2, x3, align], s=s_clean, y=y, E=E,
                tr=tr, va=va, te=te, deg=deg_un, dlang=dlang, mult=0.4 + 1.2 * dlang)


def standardise(X, mu=None, sd=None):
    if mu is None: mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


class GCN:
    def __init__(self, F, H=16, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(2 / (F + H)), (F, H)); self.b1 = np.zeros(H)
        self.W2 = r.normal(0, np.sqrt(2 / (H + 1)), (H, 1)); self.b2 = np.zeros(1)

    def fwd(self, Ahat, X):
        AX = Ahat @ X; Z1 = AX @ self.W1 + self.b1; H1 = np.maximum(Z1, 0)
        AH = Ahat @ H1; Z2 = AH @ self.W2 + self.b2
        return sigmoid(Z2[:, 0]), (AX, Z1, H1, AH)

    def fit(self, Ahat, X, y, tr, va, epochs=300, lr=0.02, wd=5e-4):
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        best, bestp = 1e9, None
        for t in range(1, epochs + 1):
            p, (AX, Z1, H1, AH) = self.fwd(Ahat, X)
            dZ2 = np.zeros((N, 1)); dZ2[tr, 0] = (p[tr] - y[tr]) / len(tr)
            gW2 = AH.T @ dZ2 + wd * self.W2; gb2 = dZ2.sum(0)
            dH1 = Ahat @ dZ2 @ self.W2.T; dZ1 = dH1 * (Z1 > 0)
            gW1 = AX.T @ dZ1 + wd * self.W1; gb1 = dZ1.sum(0)
            for i, (pp, gg) in enumerate(zip(params, [gW1, gb1, gW2, gb2])):
                m[i] = 0.9 * m[i] + 0.1 * gg; v[i] = 0.999 * v[i] + 0.001 * gg * gg
                pp -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            vl = -np.mean(y[va] * np.log(p[va] + 1e-9) + (1 - y[va]) * np.log(1 - p[va] + 1e-9))
            if vl < best:
                best = vl; bestp = [q.copy() for q in params]
        self.W1, self.b1, self.W2, self.b2 = bestp
        return self

    def predict(self, Ahat, X): return self.fwd(Ahat, X)[0]




# ---------- translation-error process ----------
def corrupt(s, w, rho, rng, kind='mix', uniform=False, drift_scale=1.0, pol_keep=1.0, sel=None):
    n = len(s)
    r = np.full(n, rho) if uniform else np.clip(rho * w['mult'], 0, 1)
    if sel is not None: r = np.where(sel, 1.0, 0.0)
    phi = {'mix': PHI, 'lex': 0.0, 'pol': 1.0}[kind]
    ev = rng.uniform(size=n) < r
    is_pol = ev & (rng.uniform(size=n) < phi)
    kept_pol = is_pol & (rng.uniform(size=n) < pol_keep)       # polarity errors that survive the polarity check
    is_lex = ev & ~is_pol
    out = s.copy()
    out[kept_pol] = -out[kept_pol]
    out[is_lex] += rng.normal(0, SIG_L * drift_scale, is_lex.sum())
    return np.clip(out, -1, 1)


def energy_matched(s, w, rho, rng, energy):
    return np.clip(s + rng.normal(0, np.sqrt(max(energy, 0)), len(s)), -1, 1)


def mean_energy(s, w, rho, rng, n=30):
    return float(np.mean([np.mean((corrupt(s, w, rho, rng) - s) ** 2) for _ in range(n)]))


# ---------- metrics ----------
def metrics(p, y, te, E):
    yt, pt = y[te], p[te]
    auc = roc_auc_score(yt, pt)
    pred = (pt > 0.5).astype(float)
    f1 = f1_score(yt, pred, average='macro')
    miss = (yt == 1) & (pred == 0); fa = (yt == 0) & (pred == 1)
    loss = np.sum(E[te] * (C_FN * miss + C_FP * fa)) / E[te].sum()
    return auc, f1, loss


def tercile_auc(p, y, te, d):
    q1, q2 = np.quantile(d, [1 / 3, 2 / 3]); out = []
    for m in [d[te] <= q1, d[te] >= q2]:   # low-difficulty (high-resource languages), high-difficulty
        yy, pp = y[te][m], p[te][m]
        out.append(roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan)
    return out


def make_X(w, s): return np.c_[w['X0'], s]


def run():
    res = {k: [] for k in ['clean', 'curves', 'terc', 'blast', 'mit', 'conc', 'sgc', 'energy']}
    for seed in SEEDS:
        w = make_world(seed); rng = np.random.default_rng(1000 + seed)
        y, tr, va, te, E, Ahat = w['y'], w['tr'], w['va'], w['te'], w['E'], w['Ahat']
        Xc_full = make_X(w, w['s']); _, mu, sd = standardise(Xc_full[tr])
        Xs = lambda s: standardise(make_X(w, s), mu, sd)[0]
        Xn = standardise(w['X0'], mu[:4], sd[:4])[0]
        lr0 = LogisticRegression(max_iter=1000).fit(Xn[tr], y[tr]); p0 = lr0.predict_proba(Xn)[:, 1]
        g1 = GCN(4, seed=seed).fit(Ahat, Xn, y, tr, va)
        g2 = GCN(5, seed=seed).fit(Ahat, Xs(w['s']), y, tr, va)
        p1 = g1.predict(Ahat, Xn); p2 = g2.predict(Ahat, Xs(w['s']))
        direct = np.mean([metrics(g2.predict(Ahat, Xs(np.clip(w['s'] + rng.normal(0, SIG_D, N), -1, 1))), y, te, E) for _ in range(REPS * 2)], 0)
        res['clean'].append(dict(seed=seed, B0=metrics(p0, y, te, E), B1=metrics(p1, y, te, E), B2=metrics(p2, y, te, E), B3=direct.tolist()))
        # degradation curves
        for rho in RATES:
            row = dict(seed=seed, rho=rho); terc = dict(seed=seed, rho=rho)
            en = mean_energy(w['s'], w, rho, rng) if rho > 0 else 0.0
            specs = {'lex': dict(kind='lex'), 'pol': dict(kind='pol'), 'mix': dict(kind='mix'), 'mixu': dict(kind='mix', uniform=True),
                     'mit': dict(kind='mix', drift_scale=DRIFT_SCALE_MIT, pol_keep=POL_KEEP_MIT), 'energy': None}
            for name, kw in specs.items():
                vals, tv = [], []
                for r in range(REPS):
                    sc = energy_matched(w['s'], w, rho, rng, en) if kw is None else corrupt(w['s'], w, rho, rng, **kw)
                    p = g2.predict(Ahat, Xs(sc)); vals.append(metrics(p, y, te, E))
                    if name in ('mix', 'mixu'): tv.append(tercile_auc(p, y, te, w['dlang']))
                row[name] = np.mean(vals, 0).tolist()
                if tv: terc[name] = np.nanmean(tv, 0).tolist()
            res['curves'].append(row); res['terc'].append(terc)
            if rho > 0: res['energy'].append(dict(seed=seed, rho=rho, energy=en))
        # mitigation ablation at rho = 0.2 and 0.4
        stages = {'none': {}, 'drift_only': dict(drift_scale=DRIFT_SCALE_MIT), 'polarity_only': dict(pol_keep=POL_KEEP_MIT),
                  'reasoning_mt': dict(drift_scale=DRIFT_SCALE_MIT, pol_keep=POL_KEEP_MIT)}
        for keep in [0.5, 0.1]:
            stages[f'reasoning_mt_remove{int(round((1 - keep) * 100))}'] = dict(drift_scale=DRIFT_SCALE_MIT, pol_keep=keep)
        for rho in [0.2, 0.4]:
            for nm, kw in stages.items():
                vals = [metrics(g2.predict(Ahat, Xs(corrupt(w['s'], w, rho, rng, kind='mix', **kw))), y, te, E) for _ in range(REPS)]
                res['mit'].append(dict(seed=seed, rho=rho, stage=nm, m=np.mean(vals, 0).tolist()))
        # blast radius: corrupt one node at a time
        base_cls = (g2.predict(Ahat, Xs(w['s'])) > 0.5); br = {'pol': [], 'lex': []}
        for i in range(N):
            sel = np.zeros(N, bool); sel[i] = True
            for nm in ['pol', 'lex']:
                sc = corrupt(w['s'], w, 0, rng, kind=nm, sel=sel)
                br[nm].append(int(np.sum((g2.predict(Ahat, Xs(sc)) > 0.5) != base_cls)))
        res['blast'].append(dict(seed=seed, deg=w['deg'].tolist(), E=E.tolist(), **br))
        # concentration: polarity errors on 10% of countries, random vs top exposure
        k = int(0.1 * N); a0, f0, l0 = metrics(p2, y, te, E)
        for mode in ['random', 'top_exposure']:
            vals = []
            for r in range(REPS):
                sel = np.zeros(N, bool); sel[(rng.choice(N, k, replace=False) if mode == 'random' else np.argsort(-E)[:k])] = True
                vals.append(metrics(g2.predict(Ahat, Xs(corrupt(w['s'], w, 0, rng, kind='pol', sel=sel))), y, te, E))
            m = np.mean(vals, 0)
            res['conc'].append(dict(seed=seed, mode=mode, auc_drop=a0 - m[0], loss_inc=m[2] - l0, rel_auc=(a0 - m[0]) / a0, rel_loss=(m[2] - l0) / max(l0, 1e-9)))
        # depth ablation (simplified GCN = logistic regression on A^K X)
        for K in [0, 1, 2, 3]:
            def prop(X):
                Z = X
                for _ in range(K): Z = Ahat @ Z
                return Z
            clf = LogisticRegression(max_iter=1000).fit(prop(Xs(w['s']))[tr], y[tr])
            a_clean = roc_auc_score(y[te], clf.predict_proba(prop(Xs(w['s'])))[te, 1]); ds = {}
            for nm in ['mix', 'pol']:
                ds[nm] = a_clean - float(np.mean([roc_auc_score(y[te], clf.predict_proba(prop(Xs(corrupt(w['s'], w, 0.2, rng, kind=nm))))[te, 1]) for _ in range(REPS)]))
            res['sgc'].append(dict(seed=seed, K=K, clean=a_clean, **ds))
        print('seed', seed, 'done', file=sys.stderr)
    return res


if __name__ == '__main__':
    json.dump(run(), open('/home/claude/raw_results2.json', 'w'))
