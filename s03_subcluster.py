# -*- coding: utf-8 -*-
"""Stage 3 - resolve subpopulations within the verified beta and alpha lineages,
then emit the final cell identity column used by all downstream analyses.
"""
import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, RES, SEED, SIGNATURES, REPORT_MARKERS, STRESS,
                    excluded_features, log, mem)

sc.settings.verbosity = 1
np.random.seed(SEED)
SUB_RES = {'Beta': 0.4, 'Alpha': 0.4}
MIN_SUB = 150            # merge subclusters smaller than this into the nearest

log('loading full matrix ...')
A = sc.read_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'))
obs = pd.read_csv(os.path.join(RES, 's02_obs_lineage.csv'), index_col=0)
A.obs = obs.reindex(A.obs_names)
log('loaded %s | %s' % (str(A.shape), mem()))

final = pd.Series('unassigned', index=A.obs_names, dtype=object)
sub_tables = []

for LIN, res in SUB_RES.items():
    m = A.obs.lineage.values == LIN
    log('\n===== sub-clustering %s (n = %d) =====' % (LIN, m.sum()))
    if m.sum() < 500:
        final[m] = LIN
        continue
    B = A[m].copy()
    sc.pp.highly_variable_genes(B, flavor='seurat', min_mean=0.0125,
                                max_mean=3, min_disp=0.5, batch_key='donor_id')
    excl = excluded_features(B.var_names.values)
    cand = B.var.loc[~excl].sort_values(
        ['highly_variable_nbatches', 'dispersions_norm'], ascending=[False, False])
    B.var['hvg_final'] = B.var_names.isin(cand.index[:2000])
    B = B[:, B.var.hvg_final.values].copy()
    sc.pp.scale(B, zero_center=False, max_value=10)
    sc.tl.pca(B, n_comps=30, svd_solver='arpack', zero_center=False,
              random_state=SEED)
    sc.external.pp.harmony_integrate(B, key='donor_id', basis='X_pca',
                                     adjusted_basis='X_pca_harmony',
                                     max_iter_harmony=15, random_state=SEED)
    sc.pp.neighbors(B, n_neighbors=15, n_pcs=20, use_rep='X_pca_harmony',
                    random_state=SEED)
    sc.tl.umap(B, random_state=SEED)
    sc.tl.leiden(B, resolution=res, key_added='sub', flavor='igraph',
                 n_iterations=2, directed=False, random_state=SEED)

    # merge tiny subclusters into the largest
    vc = B.obs['sub'].value_counts()
    small = vc[vc < MIN_SUB].index.tolist()
    if small:
        log('  merging %d subclusters < %d cells into the largest' % (len(small), MIN_SUB))
        B.obs['sub'] = B.obs['sub'].astype(str).replace(
            {s: vc.index[0] for s in small})

    # order subclusters by descending lineage-identity score so -1 is canonical
    idscore = 'sig_' + LIN
    order = (B.obs.groupby('sub', observed=True)[idscore].mean()
             .sort_values(ascending=False).index.tolist())
    ren = {c: '%s-%d' % (LIN, i + 1) for i, c in enumerate(order)}
    B.obs['identity'] = B.obs['sub'].map(ren)
    final[B.obs_names] = B.obs['identity'].values

    t = B.obs.groupby('identity', observed=True).agg(
        n_cells=('donor_id', 'size'),
        n_donors=('donor_id', 'nunique'),
        pct_T2D=('disease_state', lambda x: 100 * (x == 'T2D').mean()),
        median_genes=('n_genes_by_counts', 'median'),
        mean_doublet=('doublet_score', 'mean'),
        stress=('sig_Stress', 'mean'),
        id_score=(idscore, 'mean'))
    t['lineage'] = LIN
    sub_tables.append(t)
    print(t.round(3).to_string())

    np.save(os.path.join(DATA, 'sub_%s_umap.npy' % LIN),
            B.obsm['X_umap'].astype(np.float32))
    pd.DataFrame({'identity': B.obs['identity'].values,
                  'sub_UMAP1': B.obsm['X_umap'][:, 0],
                  'sub_UMAP2': B.obsm['X_umap'][:, 1]},
                 index=B.obs_names).to_csv(
        os.path.join(RES, 's03_sub_%s.csv' % LIN))
    del B
    gc.collect()

# non-subclustered lineages keep their lineage name
rest = final == 'unassigned'
final[rest] = A.obs.lineage.values[rest.values]
A.obs['identity'] = final.values
log('\n=== final identities ===')
print(A.obs.identity.value_counts().to_string())

if sub_tables:
    pd.concat(sub_tables).to_csv(os.path.join(RES, 's03_subpopulations.csv'))

# per-identity marker means and T2D enrichment
mk = [g for g in REPORT_MARKERS if g in A.var_names]
mdf = sc.get.obs_df(A, keys=mk + ['identity'])
mmean = mdf.groupby('identity', observed=True)[mk].mean()

summ = A.obs.groupby('identity', observed=True).agg(
    n_cells=('donor_id', 'size'),
    n_donors=('donor_id', 'nunique'),
    n_ctrl=('disease_state', lambda x: int((x == 'Control').sum())),
    n_t2d=('disease_state', lambda x: int((x == 'T2D').sum())),
    median_genes=('n_genes_by_counts', 'median'),
    mean_doublet=('doublet_score', 'mean'),
    stress=('sig_Stress', 'mean'))
summ['pct_T2D'] = 100 * summ.n_t2d / summ.n_cells
base = 100 * (A.obs.disease_state == 'T2D').mean()
summ['T2D_enrichment'] = summ.pct_T2D / base
summ = summ.join(mmean)
summ.sort_values('n_cells', ascending=False).to_csv(
    os.path.join(RES, 's03_identity_summary.csv'))
log('\nbaseline T2D cell fraction: %.1f%%' % base)
print(summ.sort_values('n_cells', ascending=False)[
    ['n_cells', 'n_ctrl', 'n_t2d', 'pct_T2D', 'T2D_enrichment', 'n_donors',
     'median_genes', 'stress']].round(2).to_string())

A.obs.to_csv(os.path.join(RES, 's03_obs_final.csv'))
log('STAGE 3 COMPLETE | %s' % mem())
