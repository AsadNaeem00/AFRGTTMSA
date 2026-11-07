import json, numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('/home/claude/raw_results2.json'))
rng = np.random.default_rng(7); n = len(R['clean']); out = {}
def ci(x, B=4000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; b = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
rates = sorted({c['rho'] for c in R['curves']})
types = ['lex', 'pol', 'mix', 'mixu', 'mit', 'energy']
curve = {t: {r: [] for r in rates} for t in types}
for c in R['curves']:
    for t in types: curve[t][c['rho']].append(c[t])
mc = {t: {r: np.mean(curve[t][r], 0) for r in rates} for t in types}
out['mean_curve'] = {t: {str(r): mc[t][r].tolist() for r in rates} for t in types}
# clean
out['t1'] = {m: dict(auc=ci([c[m][0] for c in R['clean']]), f1=ci([c[m][1] for c in R['clean']]), loss=ci([c[m][2] for c in R['clean']])) for m in ['B0', 'B1', 'B2', 'B3']}
d21 = [c['B2'][0] - c['B1'][0] for c in R['clean']]; d20 = [c['B2'][0] - c['B0'][0] for c in R['clean']]
out['gain_B1'] = ci(d21); out['p_B1'] = float(wilcoxon(d21).pvalue); out['gain_B0'] = ci(d20); out['p_B0'] = float(wilcoxon(d20).pvalue)
# slopes
grid = [0.0, 0.05, 0.10, 0.20, 0.40]; slopes = {}
for t in types:
    slopes[t] = [-np.polyfit(grid, [curve[t][r][si][0] for r in grid], 1)[0] for si in range(n)]
out['slope'] = {t: ci(slopes[t]) for t in types}
def cmp(a, b):
    d = np.array(slopes[a]) - np.array(slopes[b]); return dict(diff=ci(d), p=float(wilcoxon(d).pvalue))
out['cmp'] = {'pol-lex': cmp('pol', 'lex'), 'mix-energy': cmp('mix', 'energy'), 'mix-lex': cmp('mix', 'lex'), 'mixu-mix': cmp('mixu', 'mix')}
# energy per node
en = {}
for e in R['energy']: en.setdefault(e['rho'], []).append(e['energy'])
out['energy'] = {str(r): float(np.mean(v)) for r, v in en.items()}
# blast
out['blast'] = {k: ci([np.mean(b[k]) for b in R['blast']]) for k in ['pol', 'lex']}
rd = [spearmanr(b['deg'], b['pol'])[0] for b in R['blast']]; re_ = [spearmanr(b['E'], b['pol'])[0] for b in R['blast']]
out['sp_deg'] = ci(rd); out['sp_E'] = ci(re_); p_h2 = float(wilcoxon(rd).pvalue)
# equity gap (H3)
gap = {k: {r: [] for r in rates} for k in ['mix', 'mixu']}
for t in R['terc']:
    for k in ['mix', 'mixu']:
        gap[k][t['rho']].append(t[k][0] - t[k][1])    # low-difficulty AUROC minus high-difficulty AUROC
out['gap'] = {k: {str(r): ci(v) for r, v in d.items()} for k, d in gap.items()}
ginc = [gap['mix'][0.4][i] - gap['mix'][0.0][i] for i in range(n)]
ginc_u = [gap['mixu'][0.4][i] - gap['mixu'][0.0][i] for i in range(n)]
out['gap_inc'] = ci(ginc); out['gap_inc_u'] = ci(ginc_u); p_h3 = float(wilcoxon(np.array(ginc)).pvalue)
terc = {k: {'low': {r: [] for r in rates}, 'high': {r: [] for r in rates}} for k in ['mix', 'mixu']}
for t in R['terc']:
    for k in ['mix', 'mixu']: terc[k]['low'][t['rho']].append(t[k][0]); terc[k]['high'][t['rho']].append(t[k][1])
out['terc_040'] = {k: {g: ci(terc[k][g][0.4]) for g in ['low', 'high']} for k in ['mix', 'mixu']}
out['terc_000'] = {k: {g: ci(terc[k][g][0.0]) for g in ['low', 'high']} for k in ['mix', 'mixu']}
# mitigation
mitd = {}
for r in R['mit']: mitd.setdefault((r['rho'], r['stage']), []).append(r['m'])
out['mit'] = {}; out['recovery'] = {}; out['mit_gain'] = {}; p_h4s = {}
for rho in [0.2, 0.4]:
    mit = {k[1]: v for k, v in mitd.items() if k[0] == rho}
    out['mit'][str(rho)] = {k: dict(auc=ci([x[0] for x in v]), loss=ci([x[2] for x in v])) for k, v in mit.items()}
    rec = {}
    for k in mit:
        # recovered share computed from means over seeds (more stable than a mean of ratios)
        base = np.array([R['clean'][si]['B2'][0] for si in range(n)]); nn = np.array([mit['none'][si][0] for si in range(n)]); mm = np.array([mit[k][si][0] for si in range(n)])
        bs = []
        for _ in range(4000):
            ix = rng.integers(0, n, n); den = (base[ix] - nn[ix]).mean(); bs.append((mm[ix] - nn[ix]).mean() / den if abs(den) > 1e-9 else np.nan)
        den = (base - nn).mean(); rec[k] = [float((mm - nn).mean() / den), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    out['recovery'][str(rho)] = rec
    dd = [mit['reasoning_mt'][i][0] - mit['none'][i][0] for i in range(n)]; out['mit_gain'][str(rho)] = ci(dd); p_h4s[rho] = float(wilcoxon(dd).pvalue)
    out['mit_deg'] = out.get('mit_deg', {}); out['mit_deg'][str(rho)] = ci(base - nn)
p_h4 = p_h4s[0.4]; out['p_h4'] = {str(k): v for k, v in p_h4s.items()}
# dloss curves
out['dloss'] = {t: {str(r): ci([curve[t][r][si][2] - curve[t][0.0][si][2] for si in range(n)]) for r in rates} for t in ['lex', 'pol', 'mix', 'mit']}
# concentration
conc = {}
for c in R['conc']: conc.setdefault(c['mode'], []).append(c)
out['conc'] = {m: {k: ci([x[k] for x in v]) for k in ['rel_auc', 'rel_loss', 'auc_drop', 'loss_inc']} for m, v in conc.items()}
gt = [a['rel_loss'] - a['rel_auc'] for a in conc['top_exposure']]; gr = [a['rel_loss'] - a['rel_auc'] for a in conc['random']]
out['gap_top'] = ci(gt); out['gap_rand'] = ci(gr); p_h5 = float(wilcoxon(gt).pvalue); out['p_gap_rand'] = float(wilcoxon(gr).pvalue)
dl = [t['loss_inc'] - r['loss_inc'] for t, r in zip(conc['top_exposure'], conc['random'])]; out['loss_diff'] = ci(dl); out['p_loss_diff'] = float(wilcoxon(dl).pvalue)
# Holm over H1..H5
ps = [out['cmp']['mix-energy']['p'], p_h2, p_h3, p_h4, p_h5]; order = np.argsort(ps); h = [0] * 5; prev = 0
for k, i in enumerate(order):
    h[i] = min(1, max(prev, ps[i] * (5 - k))); prev = h[i]
out['holm'] = dict(raw=ps, adj=h, names=['H1', 'H2', 'H3', 'H4', 'H5'])
# SGC
sg = {}
for r in R['sgc']: sg.setdefault(r['K'], []).append(r)
out['sgc'] = {K: dict(clean=ci([x['clean'] for x in v]), mix=ci([x['mix'] for x in v]), pol=ci([x['pol'] for x in v])) for K, v in sg.items()}
# break-even rates
def cross(t, base):
    xs = rates; ys = [mc[t][r][0] for r in xs]
    for i in range(1, len(xs)):
        if ys[i] <= base < ys[i - 1]: return xs[i - 1] + (ys[i - 1] - base) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1])
    return None
b = {m: out['t1'][m]['auc'][0] for m in ['B0', 'B1', 'B3']}
out['cross'] = {t: {m: cross(t, b[m]) for m in b} for t in ['lex', 'pol', 'mix', 'mit']}
json.dump(out, open('/home/claude/summary2.json', 'w'), indent=1)
# figure
lab = dict(lex='Lexical drift only', pol='Polarity reversal only', mix='Mixed (75/25)', mit='Mixed + reasoning MT', energy='Energy-matched Gaussian')
sty = dict(lex='-o', pol='-^', mix='-D', mit='-s', energy='--x')
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for t in ['lex', 'pol', 'mix', 'mit', 'energy']: ax[0].plot(rates, [mc[t][r][0] for r in rates], sty[t], label=lab[t], ms=4)
for m, l, ls in [('B1', 'B1: GCN without sentiment', ':'), ('B0', 'B0: logistic', '-.'), ('B3', 'B3: direct scoring', (0, (1, 3)))]:
    ax[0].axhline(b[m], color='grey', ls=ls, lw=1, label=l)
ax[0].set_xlabel('Mean translation-error rate $\\rho$'); ax[0].set_ylabel('Test AUROC'); ax[0].legend(fontsize=6, loc='lower left', ncol=1, framealpha=0.9); ax[0].set_title('Predictive degradation', fontsize=10)
for t in ['lex', 'pol', 'mix', 'mit']: ax[1].plot(rates, [out['dloss'][t][str(r)][0] for r in rates], sty[t], label=lab[t], ms=4)
ax[1].set_xlabel('Mean translation-error rate $\\rho$'); ax[1].set_ylabel('Increase in cost-weighted loss $\\Delta L$'); ax[1].set_title('Financial loss increase', fontsize=10)
plt.tight_layout(); plt.savefig('/home/claude/fig3.png', dpi=200)
