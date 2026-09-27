# -*- coding: utf-8 -*-
"""Stage 4 - donor-level (pseudobulk) differential expression on the verified
populations, with age and sex as covariates and per-donor cell numbers equalised.
"""
import os, sys, gc, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, SEED, log, mem

warnings.filterwarnings('ignore')
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

CAP = 500            # max cells per donor per population before aggregation
MIN_DONORS = 5       # per condition
MIN_COUNTS = 10      # gene filter across pseudobulk samples
PADJ = 0.05
LFC = 0.5

log('loading counts + final identities ...')
A = sc.read_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'))
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
A.obs = obs.reindex(A.obs_names)
C = A.layers['counts']
if not sp.isspmatrix_csr(C):
    C = sp.csr_matrix(C)
genes = np.array(A.var_names)
log('loaded %s | %s' % (str(A.shape), mem()))

donor_meta = (A.obs.groupby('donor_id', observed=True)
              .agg(disease_state=('disease_state', 'first'),
                   age=('age', 'first'), sex=('sex', 'first')))

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

PASS = os.environ.get('DE_PASS', 'all')
if PASS == 'noHPAP090':
    keepmask = (A.obs.donor_id.values != 'HPAP090')
    log('SENSITIVITY PASS: excluding HPAP090 (%d cells)' % (~keepmask).sum())
else:
    keepmask = np.ones(A.n_obs, bool)
SUFFIX = '' if PASS == 'all' else '_' + PASS

summary, all_res = [], {}
vc = A.obs.identity[keepmask].value_counts()
identities = vc[vc >= 200].index.tolist()
log('pass=%s: testing %d populations with >= 200 cells' % (PASS, len(identities)))

for ident in identities:
    cells = np.where((A.obs.identity.values == ident) & keepmask)[0]
    sub_obs = A.obs.iloc[cells]
    per_donor = sub_obs.groupby('donor_id', observed=True).size()
    dm = donor_meta.loc[per_donor.index]
    n_ctrl = int((dm.disease_state == 'Control').sum())
    n_t2d = int((dm.disease_state == 'T2D').sum())
    if min(n_ctrl, n_t2d) < MIN_DONORS:
        log('  skip %-22s (donors %d/%d)' % (ident, n_ctrl, n_t2d))
        summary.append(dict(identity=ident, n_cells=len(cells), n_ctrl_donors=n_ctrl,
                            n_t2d_donors=n_t2d, tested=False, n_deg=0, n_up=0,
                            n_down=0))
        continue

    # cap cells per donor, then aggregate raw counts
    rows, keep_donors = [], []
    for d, idx in sub_obs.groupby('donor_id', observed=True).indices.items():
        gi = cells[idx]
        if len(gi) > CAP:
            gi = rng.choice(gi, CAP, replace=False)
        rows.append(np.asarray(C[gi].sum(axis=0)).ravel())
        keep_donors.append(d)
    counts = pd.DataFrame(np.vstack(rows).astype(np.int64),
                          index=keep_donors, columns=genes)
    meta = donor_meta.loc[keep_donors].copy()
    meta['disease_state'] = pd.Categorical(meta.disease_state,
                                           categories=['Control', 'T2D'])
    meta['sex'] = meta.sex.astype(str)
    meta['age'] = meta.age.astype(float)

    counts = counts.loc[:, counts.sum(axis=0) >= MIN_COUNTS]
    log('  %-22s cells=%5d donors=%2d/%2d genes=%d'
        % (ident, len(cells), n_ctrl, n_t2d, counts.shape[1]))

    try:
        dds = DeseqDataSet(counts=counts, metadata=meta,
                           design='~ age + sex + disease_state',
                           refit_cooks=True, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=['disease_state', 'T2D', 'Control'],
                        quiet=True)
        st.summary()
        r = st.results_df.copy()
    except Exception as e:
        log('    FAILED: %s' % e)
        summary.append(dict(identity=ident, n_cells=len(cells), n_ctrl_donors=n_ctrl,
                            n_t2d_donors=n_t2d, tested=False, n_deg=0, n_up=0,
                            n_down=0))
        continue

    r['identity'] = ident
    sig = r[(r.padj < PADJ) & (r.log2FoldChange.abs() > LFC)]
    up = sig[sig.log2FoldChange > 0].sort_values('log2FoldChange', ascending=False)
    dn = sig[sig.log2FoldChange < 0].sort_values('log2FoldChange')
    all_res[ident] = r
    r.to_csv(os.path.join(RES, 's04_de%s_%s.csv' % (SUFFIX, ident.replace('/', '_'))))
    summary.append(dict(identity=ident, n_cells=len(cells), n_ctrl_donors=n_ctrl,
                        n_t2d_donors=n_t2d, tested=True, n_genes=int(counts.shape[1]),
                        n_deg=len(sig), n_up=len(up), n_down=len(dn),
                        top_up='; '.join(up.index[:5]),
                        top_down='; '.join(dn.index[:5])))
    log('    -> %d DEGs (%d up, %d down)  up: %s'
        % (len(sig), len(up), len(dn), ', '.join(up.index[:6])))

S = pd.DataFrame(summary).sort_values('n_deg', ascending=False)
S.to_csv(os.path.join(RES, 's04_de_summary%s.csv' % SUFFIX), index=False)
log('\n=== DE summary ===')
print(S[['identity', 'n_cells', 'n_ctrl_donors', 'n_t2d_donors', 'tested',
         'n_deg', 'n_up', 'n_down']].to_string(index=False))
log('STAGE 4 COMPLETE | %s' % mem())
