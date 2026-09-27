# -*- coding: utf-8 -*-
"""Stage 16 - donor-level differential expression on the beta compartment as a
whole, with subpopulation composition as a covariate.

Rationale. Testing each beta subpopulation separately (Stage 4) splits one
shared signal across six gene lists and pays a six-fold multiple-testing cost.
Pooling every beta cell of a donor into a single pseudobulk profile tests the
compartment once. Because the beta subpopulation mix differs between conditions,
the donor fractions of the two subpopulations that differ significantly are
included as covariates, so a shift in composition cannot by itself produce
apparent differential expression.

INPUT   part 1/01_input/hpap_processed.h5ad   raw counts (layers/counts, CSR)
        02_results/s03_obs_final.csv          final identities + donor metadata
OUTPUT  02_results/s16_de_BetaCompartment_compAdjusted.csv       primary
        02_results/s16_de_BetaCompartment_compAdjusted_all.csv   all donors
        02_results/s16_de_BetaCompartment_noHPAP090.csv          no covariates
        02_results/s16_de_BetaCompartment_summary.csv            counts per pass
        02_results/s16_de_BetaCompartment_composition.csv        composition test
        02_results/s16_de_BetaCompartment_permutation.csv        type-I error
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, H5AD, SEED, log

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

CAP = 500          # max beta cells per donor, applied ONCE to the compartment
MIN_COUNTS = 10    # gene filter across pseudobulk samples
PADJ, LFC = 0.05, 0.5
NPERM = 15         # label shuffles for the type-I error check

# ── 1. pull raw counts for the beta compartment out of the 15 GB h5ad ────────
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
beta = obs[obs.identity.str.startswith('Beta')]
log('beta compartment: %d cells, %d donors' % (len(beta), beta.donor_id.nunique()))

with h5py.File(H5AD, 'r') as f:
    names = np.array([x.decode() if isinstance(x, bytes) else x
                      for x in f['obs/_index'][:]])
    genes = np.array([x.decode() if isinstance(x, bytes) else x
                      for x in f['var/_index'][:]])
    pos = pd.Series(np.arange(len(names)), index=names)
    rows = np.sort(pos.loc[beta.index].values)
    sel = np.zeros(len(names), bool)
    sel[rows] = True
    G = f['layers/counts']
    indptr = G['indptr'][:]
    blocks, CH = [], 4000
    for a in range(0, len(names), CH):
        b = min(a + CH, len(names))
        if not sel[a:b].any():
            continue
        lo, hi = indptr[a], indptr[b]
        M = sp.csr_matrix((G['data'][lo:hi], G['indices'][lo:hi],
                           indptr[a:b + 1] - lo), shape=(b - a, len(genes)))
        blocks.append(M[sel[a:b]])
X = sp.vstack(blocks).tocsr().astype(np.int64)
cells = names[rows]
obs_b = obs.loc[cells]
idx_of = {b: i for i, b in enumerate(cells)}
log('counts extracted %s' % str(X.shape))

donor_meta = obs_b.groupby('donor_id', observed=True).agg(
    disease_state=('disease_state', 'first'), age=('age', 'first'), sex=('sex', 'first'))
frac = obs_b.groupby(['donor_id', 'identity'], observed=True).size().unstack(fill_value=0)
frac = frac.div(frac.sum(axis=1), axis=0)

# ── 2. does beta subpopulation composition differ by condition? ──────────────
comp_rows = []
for p in frac.columns:
    st = donor_meta.loc[frac.index].disease_state
    c, t = frac.loc[st == 'Control', p], frac.loc[st == 'T2D', p]
    u, pv = mannwhitneyu(t, c, alternative='two-sided')
    comp_rows.append(dict(subpopulation=p, ctrl_median=c.median(),
                          t2d_median=t.median(), p=pv))
COMP = pd.DataFrame(comp_rows).sort_values('p')
COMP.to_csv(os.path.join(RES, 's16_de_BetaCompartment_composition.csv'), index=False)
log('\ncomposition of the beta compartment by condition (donor medians):')
print(COMP.to_string(index=False))
ADJ = COMP.loc[COMP.p < 0.05, 'subpopulation'].tolist()
log('subpopulations included as covariates (p < 0.05): %s' % (ADJ or 'none'))

# ── 3. pseudobulk + DESeq2 ───────────────────────────────────────────────────
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def pseudobulk(exclude=None):
    sub = obs_b if exclude is None else obs_b[obs_b.donor_id != exclude]
    rows_, donors = [], []
    for d, g in sub.groupby('donor_id', observed=True).groups.items():
        gi = np.array([idx_of[b] for b in g])
        if len(gi) > CAP:
            gi = rng.choice(gi, CAP, replace=False)
        rows_.append(np.asarray(X[gi].sum(axis=0)).ravel())
        donors.append(d)
    C = pd.DataFrame(np.vstack(rows_).astype(np.int64), index=donors, columns=genes)
    return C.loc[:, C.sum(axis=0) >= MIN_COUNTS]


def metadata(donors, labels=None):
    m = donor_meta.loc[donors].copy()
    lab = m.disease_state.values if labels is None else labels
    m['disease_state'] = pd.Categorical(lab, categories=['Control', 'T2D'])
    m['sex'] = m.sex.astype(str)
    m['age'] = m.age.astype(float)
    for p in ADJ:
        m['f_' + p.replace('-', '')] = frac.loc[donors, p].values.astype(float)
    return m


def run(counts, meta, adjust, label):
    terms = ['age', 'sex'] + (['f_' + p.replace('-', '') for p in ADJ] if adjust else [])
    design = '~ ' + ' + '.join(terms + ['disease_state'])
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design,
                       refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=['disease_state', 'T2D', 'Control'], quiet=True)
    st.summary()
    r = st.results_df.copy()
    s = r[(r.padj < PADJ) & (r.log2FoldChange.abs() > LFC)]
    log('  %-46s %3d DEGs (%2d up, %2d down)  [%s]'
        % (label, len(s), (s.log2FoldChange > 0).sum(), (s.log2FoldChange < 0).sum(), design))
    return r, s


log('\nprimary and sensitivity passes:')
C90 = pseudobulk(exclude='HPAP090')
M90 = metadata(C90.index)
CALL = pseudobulk()
MALL = metadata(CALL.index)

r_pri, s_pri = run(C90, M90, True, 'PRIMARY  composition-adjusted, no HPAP090')
r_all, s_all = run(CALL, MALL, True, 'sensitivity  composition-adjusted, all donors')
r_unadj, s_unadj = run(C90, M90, False, 'sensitivity  unadjusted, no HPAP090')

r_pri.to_csv(os.path.join(RES, 's16_de_BetaCompartment_compAdjusted.csv'))
r_all.to_csv(os.path.join(RES, 's16_de_BetaCompartment_compAdjusted_all.csv'))
r_unadj.to_csv(os.path.join(RES, 's16_de_BetaCompartment_noHPAP090.csv'))

# ── 4. type-I error: shuffle the donor disease labels ────────────────────────
log('\npermutation calibration (%d shuffles of the donor label):' % NPERM)
labels = donor_meta.loc[C90.index].disease_state.values
nulls = []
for i in range(NPERM):
    m = metadata(C90.index, labels=rng.permutation(labels))
    _, s = run(C90, m, True, '  permutation %2d' % (i + 1))
    nulls.append(len(s))
PERM = pd.DataFrame(dict(permutation=range(1, NPERM + 1), n_deg=nulls))
PERM.to_csv(os.path.join(RES, 's16_de_BetaCompartment_permutation.csv'), index=False)
log('observed %d | permuted median %.0f, max %d, >= observed %d/%d'
    % (len(s_pri), np.median(nulls), max(nulls), sum(n >= len(s_pri) for n in nulls), NPERM))

# ── 5. summary + comparison with the per-subpopulation analysis ──────────────
def sig_of(fn):
    d = pd.read_csv(os.path.join(RES, fn), index_col=0)
    return set(d[(d.padj < PADJ) & (d.log2FoldChange.abs() > LFC)].index)


subs = {p: sig_of('s04_de_noHPAP090_%s.csv' % p)
        for p in sorted(obs_b.identity.unique())
        if os.path.exists(os.path.join(RES, 's04_de_noHPAP090_%s.csv' % p))}
union = set().union(*subs.values()) if subs else set()
S = pd.DataFrame([
    dict(pass_='primary (adjusted, no HPAP090)', n_deg=len(s_pri),
         n_up=int((s_pri.log2FoldChange > 0).sum()),
         n_down=int((s_pri.log2FoldChange < 0).sum()), n_genes=C90.shape[1]),
    dict(pass_='adjusted, all donors', n_deg=len(s_all),
         n_up=int((s_all.log2FoldChange > 0).sum()),
         n_down=int((s_all.log2FoldChange < 0).sum()), n_genes=CALL.shape[1]),
    dict(pass_='unadjusted, no HPAP090', n_deg=len(s_unadj),
         n_up=int((s_unadj.log2FoldChange > 0).sum()),
         n_down=int((s_unadj.log2FoldChange < 0).sum()), n_genes=C90.shape[1]),
    dict(pass_='union of six per-subpopulation tests', n_deg=len(union),
         n_up=np.nan, n_down=np.nan, n_genes=np.nan)])
S.to_csv(os.path.join(RES, 's16_de_BetaCompartment_summary.csv'), index=False)
log('\n=== summary ===')
print(S.to_string(index=False))
log('genes recovered only by the compartment test: %d' % len(set(s_pri.index) - union))
log('STAGE 16 COMPLETE')
