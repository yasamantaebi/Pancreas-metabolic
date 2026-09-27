# -*- coding: utf-8 -*-
"""Re-embed the retained atlas WITHOUT donor HPAP090 (reviewer request, Figure 2).

Identical pipeline to s01_embed.py - complexity filter, normalisation, ambient-
aware HVG selection, PCA, Harmony (donor), kNN, UMAP, Leiden 1.0 - run on the
61,859 retained cells minus the 4,024 HPAP090 cells. Leaner than s01: the
counts layer is not duplicated and nothing full-width is written to disk.

Outputs
  02_results/s24_obs_embedding_noHPAP090.csv   UMAP1/2 + leiden_1.0 per cell
  01_data/X_umap_noHPAP090.npy
  02_results/s24_log.txt
"""
import os, sys, gc
import numpy as np
import scipy.sparse as sp
import h5py
import anndata as ad
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (H5AD, DATA, RES, SEED, excluded_features, log, mem)

sc.settings.verbosity = 1
np.random.seed(SEED)
EXCLUDE_DONOR = 'HPAP090'
N_HVG, N_PCS, N_HARMONY_PCS, K = 3000, 50, 30, 15

# cells to keep: the final retained atlas minus the excluded donor
final = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
keep_ids = final.index[final.donor_id != EXCLUDE_DONOR]
log('target: %d retained cells minus %s -> %d cells'
    % (len(final), EXCLUDE_DONOR, len(keep_ids)))

f = h5py.File(H5AD, 'r')
var_names = np.array([v.decode() if isinstance(v, bytes) else v for v in f['var']['_index'][:]])
obs_names = np.array([v.decode() if isinstance(v, bytes) else v for v in f['obs']['_index'][:]])
pos = pd.Series(np.arange(len(obs_names)), index=obs_names)
rows = np.sort(pos.loc[keep_ids].values)

C = f['layers']['counts']
indptr = C['indptr'][:].astype(np.int64)
log('reading %d rows of CSR counts in blocks | %s' % (len(rows), mem()))
data_parts, idx_parts, new_indptr = [], [], [0]
sel = np.zeros(len(obs_names), bool); sel[rows] = True
CH = 4000
for a in range(0, len(obs_names), CH):
    b = min(a + CH, len(obs_names))
    if not sel[a:b].any():
        continue
    lo, hi = indptr[a], indptr[b]
    d = C['data'][lo:hi].astype(np.float32)
    ix = C['indices'][lo:hi].astype(np.int32)
    ip = indptr[a:b + 1] - lo
    blk = sp.csr_matrix((d, ix, ip), shape=(b - a, len(var_names)))[sel[a:b]]
    data_parts.append(blk)
X = sp.vstack(data_parts, format='csr')
del data_parts
f.close(); gc.collect()
log('counts %s nnz=%d | %s' % (str(X.shape), X.nnz, mem()))

A = ad.AnnData(X=X, obs=final.loc[obs_names[rows], ['donor_id', 'disease_state', 'identity']].copy(),
               var=pd.DataFrame(index=var_names))
del X; gc.collect()

sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
sc.pp.filter_genes(A, min_cells=20)
log('normalised; %d genes after min_cells=20 | %s' % (A.n_vars, mem()))

excl = excluded_features(A.var_names.values)
sc.pp.highly_variable_genes(A, flavor='seurat', n_top_genes=None,
                            min_mean=0.0125, max_mean=3, min_disp=0.5,
                            batch_key='donor_id')
hv = A.var.copy(); hv['excluded'] = excl
cand = hv.loc[~hv.excluded].sort_values(['highly_variable_nbatches', 'dispersions_norm'],
                                        ascending=[False, False])
sel_genes = cand.index[:N_HVG]
log('selected %d HVGs; top 10: %s' % (len(sel_genes), list(sel_genes[:10])))

B = A[:, A.var_names.isin(sel_genes)].copy()
del A; gc.collect()
log('HVG subset %s | %s' % (str(B.shape), mem()))

sc.pp.scale(B, zero_center=False, max_value=10)
sc.tl.pca(B, n_comps=N_PCS, svd_solver='arpack', zero_center=False, random_state=SEED)
log('PCA done | %s' % mem())
sc.external.pp.harmony_integrate(B, key='donor_id', basis='X_pca',
                                 adjusted_basis='X_pca_harmony',
                                 max_iter_harmony=20, random_state=SEED)
log('harmony done | %s' % mem())
sc.pp.neighbors(B, n_neighbors=K, n_pcs=N_HARMONY_PCS, use_rep='X_pca_harmony',
                random_state=SEED)
sc.tl.umap(B, random_state=SEED)
sc.tl.leiden(B, resolution=1.0, key_added='leiden_1.0', flavor='igraph',
             n_iterations=2, directed=False, random_state=SEED)
log('umap + leiden done: %d clusters | %s' % (B.obs['leiden_1.0'].nunique(), mem()))

out = B.obs.copy()
out['UMAP1'] = B.obsm['X_umap'][:, 0]
out['UMAP2'] = B.obsm['X_umap'][:, 1]
out.to_csv(os.path.join(RES, 's24_obs_embedding_noHPAP090.csv'))
np.save(os.path.join(DATA, 'X_umap_noHPAP090.npy'), B.obsm['X_umap'].astype(np.float32))

# where do the surviving Alpha-2 / Alpha-3 cells land?
for pop in ['Alpha-2', 'Alpha-3']:
    m = out.identity == pop
    vc = out.loc[m, 'leiden_1.0'].value_counts()
    log('%s: %d cells -> leiden clusters %s' % (pop, m.sum(), vc.head(4).to_dict()))
    top = vc.index[0]
    comp = out.loc[out['leiden_1.0'] == top, 'identity'].value_counts()
    log('   cluster %s composition (top 3): %s' % (top, comp.head(3).to_dict()))
log('DONE | %s' % mem())
