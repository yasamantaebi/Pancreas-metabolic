# -*- coding: utf-8 -*-
"""Stage 1 - rebuild the embedding on an ambient-aware feature set.

Reads raw counts directly from the HDF5 (never touching the 15 GB dense X),
normalises, selects highly variable genes with hormone/exocrine/sex-linked
transcripts excluded, then PCA -> Harmony -> kNN -> UMAP -> Leiden.

Memory-frugal by design: sparse throughout, float32, and the full-width matrix
is released as soon as the HVG subset exists.
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

N_HVG = 3000
N_PCS = 50
N_HARMONY_PCS = 30
K = 15

# Complexity filter. The distribution of genes detected per cell is bimodal in
# this dataset with a trough at ~1,000 genes; the lower mode is dominated by
# ambient hormone and exocrine transcripts and was the source of the
# hormone-ambiguous "mixed"/"stressed" populations in the previous analysis.
# Cells below the trough are excluded from lineage assignment and reported
# separately as a low-complexity fraction.
MIN_GENES = 1000

# ── read counts + obs straight from HDF5 ──────────────────────────────────────
log('opening %s' % H5AD)
f = h5py.File(H5AD, 'r')

var_names = np.array([v.decode() if isinstance(v, bytes) else v
                      for v in f['var']['_index'][:]])
obs_names = np.array([v.decode() if isinstance(v, bytes) else v
                      for v in f['obs']['_index'][:]])


def read_obs_col(name):
    g = f['obs'][name]
    if isinstance(g, h5py.Group):          # categorical
        cats = np.array([c.decode() if isinstance(c, bytes) else c
                         for c in g['categories'][:]])
        return pd.Categorical.from_codes(g['codes'][:], cats)
    arr = g[:]
    if arr.dtype.kind == 'S':
        return np.array([x.decode() for x in arr])
    return arr


obs = pd.DataFrame(index=obs_names)
for c in ['donor_id', 'disease_state', 'assay', 'sex', 'age', 'percent_mito',
          'n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'doublet_score',
          'cell_identity']:
    if c in f['obs']:
        obs[c] = read_obs_col(c)
obs.rename(columns={'cell_identity': 'identity_v1'}, inplace=True)
log('obs: %d cells x %d fields | %s' % (len(obs), obs.shape[1], mem()))

C = f['layers']['counts']
n_cells = len(obs_names)
n_genes = len(var_names)
log('loading counts CSR (%d nnz)' % C['data'].shape[0])
X = sp.csr_matrix((C['data'][:].astype(np.float32),
                   C['indices'][:].astype(np.int32),
                   C['indptr'][:].astype(np.int64)),
                  shape=(n_cells, n_genes))
f.close()
gc.collect()
log('counts loaded %s | %s' % (str(X.shape), mem()))

A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
del X
gc.collect()

# ── complexity filter ────────────────────────────────────────────────────────
keep = (A.obs.n_genes_by_counts.values >= MIN_GENES)
excluded_cells = A.obs.loc[~keep].copy()
excluded_cells.to_csv(os.path.join(RES, 's01_excluded_lowcomplexity_cells.csv'))
log('complexity filter >= %d genes: keeping %d / %d cells (%.1f%%)'
    % (MIN_GENES, keep.sum(), len(keep), 100 * keep.mean()))
log('  donors before %d, after %d; lost: %s'
    % (A.obs.donor_id.nunique(), A.obs.loc[keep].donor_id.nunique(),
       sorted(set(A.obs.donor_id.unique()) - set(A.obs.loc[keep].donor_id.unique()))))
A = A[keep].copy()
gc.collect()
A.layers['counts'] = A.X.copy()
log('working set %s | %s' % (str(A.shape), mem()))

# ── normalise ────────────────────────────────────────────────────────────────
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
log('normalised | %s' % mem())

# ── feature selection with ambient-dominated transcripts excluded ─────────────
sc.pp.filter_genes(A, min_cells=20)
log('genes after min_cells=20: %d' % A.n_vars)

excl = excluded_features(A.var_names.values)
log('excluded from feature selection: %d of %d genes' % (excl.sum(), A.n_vars))

sc.pp.highly_variable_genes(A, flavor='seurat', n_top_genes=None,
                            min_mean=0.0125, max_mean=3, min_disp=0.5,
                            batch_key='donor_id')
hv = A.var.copy()
hv['excluded'] = excl
# rank by normalised dispersion among non-excluded genes
cand = hv.loc[~hv.excluded].copy()
cand = cand.sort_values(['highly_variable_nbatches', 'dispersions_norm'],
                        ascending=[False, False])
sel = cand.index[:N_HVG]
A.var['hvg_final'] = A.var_names.isin(sel)
log('selected %d HVGs; top 15: %s' % (A.var.hvg_final.sum(), list(sel[:15])))
hv.to_csv(os.path.join(RES, 's01_hvg_table.csv'))

# keep the full log-normalised matrix on disk for later stages, then subset
log('writing full log-normalised matrix to disk ...')
A.write_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'), compression='lzf')
log('written | %s' % mem())

B = A[:, A.var.hvg_final.values].copy()
del A
gc.collect()
log('HVG subset %s | %s' % (str(B.shape), mem()))

# ── scale (no zero-centering: keeps the matrix sparse) + PCA ──────────────────
sc.pp.scale(B, zero_center=False, max_value=10)
sc.tl.pca(B, n_comps=N_PCS, svd_solver='arpack', zero_center=False,
          random_state=SEED)
log('PCA done | %s' % mem())

# ── Harmony on donor ─────────────────────────────────────────────────────────
sc.external.pp.harmony_integrate(B, key='donor_id', basis='X_pca',
                                 adjusted_basis='X_pca_harmony',
                                 max_iter_harmony=20, random_state=SEED)
log('harmony done | %s' % mem())

# ── graph, UMAP, Leiden at several resolutions ───────────────────────────────
sc.pp.neighbors(B, n_neighbors=K, n_pcs=N_HARMONY_PCS,
                use_rep='X_pca_harmony', random_state=SEED)
sc.tl.umap(B, random_state=SEED)
log('umap done | %s' % mem())

for r in (0.6, 1.0, 1.5, 2.0):
    key = 'leiden_%.1f' % r
    sc.tl.leiden(B, resolution=r, key_added=key, flavor='igraph',
                 n_iterations=2, directed=False, random_state=SEED)
    log('%s -> %d clusters' % (key, B.obs[key].nunique()))

out = B.obs.copy()
out['UMAP1'] = B.obsm['X_umap'][:, 0]
out['UMAP2'] = B.obsm['X_umap'][:, 1]
out.to_csv(os.path.join(RES, 's01_obs_embedding.csv'))
np.save(os.path.join(DATA, 'X_pca_harmony.npy'),
        B.obsm['X_pca_harmony'].astype(np.float32))
np.save(os.path.join(DATA, 'X_umap.npy'), B.obsm['X_umap'].astype(np.float32))
B.write_h5ad(os.path.join(DATA, 'hvg_embedded.h5ad'), compression='lzf')
log('STAGE 1 COMPLETE | %s' % mem())
