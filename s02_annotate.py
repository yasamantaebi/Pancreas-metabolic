# -*- coding: utf-8 -*-
"""Stage 2 - marker-driven lineage annotation.

Every Leiden cluster is assigned to the lineage with the highest mean signature
score. Signatures are weighted toward transcription factors and low-abundance
identity genes rather than secreted hormones, because the hormones are the
ambient-contaminated transcripts that drove the previous mis-annotation.
"""
import os, sys, gc
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, SEED, SIGNATURES, REPORT_MARKERS, STRESS, log, mem

sc.settings.verbosity = 1
np.random.seed(SEED)
LEIDEN = 'leiden_1.0'
AMBIG = 0.20          # min margin between best and second-best lineage score

log('loading full log-normalised matrix ...')
A = sc.read_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'))
log('loaded %s | %s' % (str(A.shape), mem()))

emb = pd.read_csv(os.path.join(RES, 's01_obs_embedding.csv'), index_col=0)
for c in emb.columns:
    if c not in A.obs.columns or c.startswith('leiden') or c.startswith('UMAP'):
        A.obs[c] = emb[c].reindex(A.obs_names).values
log('clusters at %s: %d' % (LEIDEN, A.obs[LEIDEN].nunique()))

# ── per-cell signature scores ────────────────────────────────────────────────
present = {k: [g for g in v if g in A.var_names] for k, v in SIGNATURES.items()}
for k, v in present.items():
    log('  %-12s %2d/%2d markers present' % (k, len(v), len(SIGNATURES[k])))
    sc.tl.score_genes(A, v, score_name='sig_' + k, random_state=SEED)
sc.tl.score_genes(A, [g for g in STRESS if g in A.var_names],
                  score_name='sig_Stress', random_state=SEED)
log('signature scoring done | %s' % mem())

SIGCOLS = ['sig_' + k for k in SIGNATURES]

# ── per-cluster assignment ───────────────────────────────────────────────────
cl = A.obs.groupby(LEIDEN, observed=True)[SIGCOLS + ['sig_Stress']].mean()
cl['n_cells'] = A.obs.groupby(LEIDEN, observed=True).size()
cl['pct_T2D'] = (A.obs.assign(t=(A.obs.disease_state == 'T2D'))
                 .groupby(LEIDEN, observed=True).t.mean() * 100)
cl['n_donors'] = A.obs.groupby(LEIDEN, observed=True).donor_id.nunique()
cl['median_genes'] = A.obs.groupby(LEIDEN, observed=True).n_genes_by_counts.median()
cl['mean_doublet'] = A.obs.groupby(LEIDEN, observed=True).doublet_score.mean()

S = cl[SIGCOLS]
best = S.idxmax(axis=1).str.replace('sig_', '', regex=False)
srt = np.sort(S.values, axis=1)
margin = srt[:, -1] - srt[:, -2]
cl['lineage'] = best.values
cl['margin'] = margin
cl.loc[cl.margin < AMBIG, 'lineage'] = cl.loc[cl.margin < AMBIG, 'lineage'] + '?'
cl['top_score'] = S.max(axis=1).values

# marker means per cluster for the manuscript table
mk = [g for g in REPORT_MARKERS if g in A.var_names]
mdf = sc.get.obs_df(A, keys=mk + [LEIDEN])
mmean = mdf.groupby(LEIDEN, observed=True)[mk].mean()
mmean.to_csv(os.path.join(RES, 's02_cluster_marker_means.csv'))

cl = cl.join(mmean, how='left')
cl.sort_values('n_cells', ascending=False).to_csv(
    os.path.join(RES, 's02_cluster_assignment.csv'))

log('\n=== cluster -> lineage ===')
show = ['n_cells', 'pct_T2D', 'n_donors', 'lineage', 'margin', 'top_score',
        'median_genes', 'mean_doublet']
print(cl.sort_values('n_cells', ascending=False)[show].round(2).to_string())

A.obs['lineage'] = A.obs[LEIDEN].map(cl['lineage']).astype(str)
log('\n=== cells per lineage ===')
print(A.obs.lineage.value_counts().to_string())

keep = ['donor_id', 'disease_state', 'assay', 'sex', 'age', 'n_genes_by_counts',
        'total_counts', 'pct_counts_mt', 'doublet_score', 'identity_v1',
        LEIDEN, 'lineage', 'UMAP1', 'UMAP2'] + SIGCOLS + ['sig_Stress']
A.obs[[c for c in keep if c in A.obs.columns]].to_csv(
    os.path.join(RES, 's02_obs_lineage.csv'))
log('STAGE 2 COMPLETE | %s' % mem())
