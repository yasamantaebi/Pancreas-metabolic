# -*- coding: utf-8 -*-
"""Stage 5 - diffusion pseudotime on the marker-verified beta compartment only.

Differences from the previous attempt, each addressing a specific failure:
  * input is beta cells confirmed by marker score, with no alpha-like or
    hormone-ambiguous population included;
  * low-complexity cells were already removed upstream, so the axis cannot
    simply track library depth;
  * hormone, exocrine and sex-linked transcripts are excluded from feature
    selection, so the axis cannot track ambient load or donor sex;
  * the root is the centroid-based beta cell of control donors, and two
    alternative roots are tested;
  * every disease comparison is made at the donor level.
"""
import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, SEED, excluded_features, STRESS, log, mem

sc.settings.verbosity = 1
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

log('loading ...')
A = sc.read_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'))
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
A.obs = obs.reindex(A.obs_names)

beta = A.obs.identity.astype(str).str.startswith('Beta').values
B = A[beta].copy()
del A
gc.collect()
log('beta compartment: %s | %s' % (str(B.shape), mem()))
log('populations: %s' % dict(B.obs.identity.value_counts()))

# ── re-embed the beta compartment ────────────────────────────────────────────
sc.pp.highly_variable_genes(B, flavor='seurat', min_mean=0.0125, max_mean=3,
                            min_disp=0.5, batch_key='donor_id')
excl = excluded_features(B.var_names.values)
cand = B.var.loc[~excl].sort_values(
    ['highly_variable_nbatches', 'dispersions_norm'], ascending=[False, False])
hvg = cand.index[:2000]
Bh = B[:, B.var_names.isin(hvg)].copy()
sc.pp.scale(Bh, zero_center=False, max_value=10)
sc.tl.pca(Bh, n_comps=30, svd_solver='arpack', zero_center=False, random_state=SEED)
sc.external.pp.harmony_integrate(Bh, key='donor_id', basis='X_pca',
                                 adjusted_basis='X_pca_harmony',
                                 max_iter_harmony=15, random_state=SEED)
sc.pp.neighbors(Bh, n_neighbors=15, n_pcs=20, use_rep='X_pca_harmony',
                random_state=SEED)
sc.tl.umap(Bh, random_state=SEED)
sc.tl.diffmap(Bh, n_comps=15)
log('diffusion map done | %s' % mem())

# ── root definitions ─────────────────────────────────────────────────────────
P = Bh.obsm['X_pca_harmony']
ins = np.asarray(B[:, 'INS'].X.todense()).ravel() if hasattr(B[:, 'INS'].X, 'todense') \
    else np.asarray(B[:, 'INS'].X).ravel()
ctrl = (Bh.obs.disease_state == 'Control').values
roots = {}
# Root B: centroid of the top-decile INS control beta cells
thr = np.quantile(ins[ctrl], 0.90)
sel = ctrl & (ins >= thr)
cen = P[sel].mean(axis=0)
roots['B'] = int(np.where(sel)[0][np.argmin(((P[sel] - cen) ** 2).sum(1))])
# Root A: single highest-INS control beta cell
roots['A'] = int(np.where(ctrl)[0][np.argmax(ins[ctrl])])
# Root C: random control beta cell
roots['C'] = int(rng.choice(np.where(ctrl)[0]))

pt = {}
for name, idx in roots.items():
    Bh.uns['iroot'] = idx
    sc.tl.dpt(Bh, n_dcs=15)
    pt[name] = Bh.obs['dpt_pseudotime'].values.copy()
    log('root %s: idx=%d INS=%.2f  pseudotime range %.3f-%.3f'
        % (name, idx, ins[idx], np.nanmin(pt[name]), np.nanmax(pt[name])))

R = pd.DataFrame(pt).corr(method='spearman')
R.to_csv(os.path.join(RES, 's05_root_robustness.csv'))
log('\nroot correlation matrix (Spearman):\n%s' % R.round(3).to_string())

Bh.obs['pt'] = pt['B']
Bh.obs['UMAP1'] = Bh.obsm['X_umap'][:, 0]
Bh.obs['UMAP2'] = Bh.obsm['X_umap'][:, 1]
Bh.obs['DC1'] = Bh.obsm['X_diffmap'][:, 1]
for k, v in pt.items():
    Bh.obs['pt_root' + k] = v

# ── what does the axis actually track? ───────────────────────────────────────
log('\n=== technical covariates of pseudotime ===')
tech = {}
for c in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'doublet_score']:
    r, p = spearmanr(Bh.obs.pt, Bh.obs[c])
    tech[c] = r
    log('  %-20s rho = %+.3f' % (c, r))
pd.Series(tech).to_csv(os.path.join(RES, 's05_pt_technical_corr.csv'))

# gene-pseudotime correlations across the HVG set
from scipy.stats import rankdata
Xh = Bh.X.toarray() if hasattr(Bh.X, 'toarray') else np.asarray(Bh.X)
pr = rankdata(Bh.obs.pt.values); pr = (pr - pr.mean()) / pr.std()
Rk = np.apply_along_axis(rankdata, 0, Xh).astype(np.float32)
Rk -= Rk.mean(axis=0, keepdims=True)
sd = Rk.std(axis=0)
ok = sd > 1e-9
rho = np.zeros(Xh.shape[1], dtype=np.float32)
rho[ok] = (Rk[:, ok] * pr[:, None]).mean(axis=0) / sd[ok]
del Rk, Xh
gc.collect()
gc = __import__('gc')
gr = pd.Series(rho, index=Bh.var_names).sort_values()
gr.to_csv(os.path.join(RES, 's05_gene_pseudotime_corr.csv'))
log('\nmost negative: %s' % dict(gr.head(12).round(2)))
log('most positive: %s' % dict(gr.tail(12).round(2)))
for g in ['INS', 'IAPP', 'MAFA', 'PDX1', 'NKX6-1', 'G6PC2', 'MEG3', 'ALDH1A3',
          'VIM', 'HSPA5', 'DDIT3', 'HERPUD1', 'TXNIP', 'XIST']:
    if g in gr.index:
        log('  %-9s rho = %+.3f' % (g, gr[g]))

# ── donor-level disease tests ────────────────────────────────────────────────
log('\n=== donor-level pseudotime tests ===')
rows = []
for p in sorted(Bh.obs.identity.unique()):
    s = Bh.obs[Bh.obs.identity == p]
    dm = s.groupby(['donor_id', 'disease_state'], observed=True).pt.median().reset_index()
    c = dm.loc[dm.disease_state == 'Control', 'pt'].values
    t = dm.loc[dm.disease_state == 'T2D', 'pt'].values
    if len(c) >= 3 and len(t) >= 3:
        st, pv = mannwhitneyu(t, c, alternative='two-sided')
    else:
        pv = np.nan
    rows.append(dict(population=p, n_cells=len(s), n_ctrl_donors=len(c),
                     n_t2d_donors=len(t), median_ctrl=np.median(c) if len(c) else np.nan,
                     median_t2d=np.median(t) if len(t) else np.nan,
                     delta=(np.median(t) - np.median(c)) if len(c) and len(t) else np.nan,
                     p_twosided=pv))
T = pd.DataFrame(rows).sort_values('median_ctrl')
T.to_csv(os.path.join(RES, 's05_pt_donor_tests.csv'), index=False)
print(T.round(4).to_string(index=False))

# terminal-quintile enrichment, cell level then donor level
q = pd.qcut(Bh.obs.pt, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
Bh.obs['pt_quintile'] = q
ct = pd.crosstab(q, Bh.obs.disease_state)
chi2, pchi, dof, _ = chi2_contingency(ct)
log('\ncell-level chi-square across quintiles: chi2=%.1f dof=%d p=%.3g' % (chi2, dof, pchi))
log('%% T2D per quintile:\n%s' % (100 * ct.div(ct.sum(1), axis=0)).round(1).to_string())

dq = (Bh.obs.assign(isQ5=(q == 'Q5'))
      .groupby(['donor_id', 'disease_state'], observed=True).isQ5.mean() * 100).reset_index()
c = dq.loc[dq.disease_state == 'Control', 'isQ5'].values
t = dq.loc[dq.disease_state == 'T2D', 'isQ5'].values
st, pq = mannwhitneyu(t, c, alternative='two-sided')
log('donor-level Q5: control median %.1f%% (n=%d) vs T2D %.1f%% (n=%d), p=%.3f'
    % (np.median(c), len(c), np.median(t), len(t), pq))
dq.to_csv(os.path.join(RES, 's05_donor_pct_q5.csv'), index=False)

keep = ['donor_id', 'disease_state', 'age', 'sex', 'identity', 'n_genes_by_counts',
        'total_counts', 'doublet_score', 'pt', 'pt_rootA', 'pt_rootB', 'pt_rootC',
        'UMAP1', 'UMAP2', 'DC1', 'pt_quintile', 'sig_Stress', 'sig_Beta']
Bh.obs[[c for c in keep if c in Bh.obs.columns]].to_csv(
    os.path.join(RES, 's05_trajectory_obs.csv'))
with open(os.path.join(RES, 's05_roots.txt'), 'w') as fh:
    fh.write(repr({k: (v, float(ins[v])) for k, v in roots.items()}))
log('STAGE 5 COMPLETE | %s' % mem())
