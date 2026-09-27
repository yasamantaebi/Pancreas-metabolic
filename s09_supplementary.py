# -*- coding: utf-8 -*-
"""Stage 9 - supplementary analyses: QC/doublets, curated energy pathways,
pathway over-representation, hormone fraction, per-donor DE support."""
import os, sys, io, gc, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, PART1, SEED, HORMONE, EXOCRINE, log, mem

warnings.filterwarnings('ignore')
BETA = ['Beta-%d' % i for i in range(1, 7)]
ALPHA = ['Alpha-%d' % i for i in range(1, 9)]
ORDER = BETA + ALPHA + ['Delta', 'PP', 'Acinar']

obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
full = pd.read_csv(os.path.join(PART1, '04_results', 'reviewer_diagnostics',
                                'full_obs_with_umap.csv'), low_memory=False)

# ── 1. QC and doublet summaries ──────────────────────────────────────────────
log('=== QC / doublet summaries ===')
qc = obs.groupby('identity').agg(
    n_cells=('donor_id', 'size'), n_donors=('donor_id', 'nunique'),
    median_genes=('n_genes_by_counts', 'median'),
    median_umi=('total_counts', 'median'),
    median_mito=('pct_counts_mt', 'median'),
    mean_doublet=('doublet_score', 'mean'),
    median_doublet=('doublet_score', 'median')).reindex(ORDER)
# each population vs all others
pv = {}
for p in ORDER:
    a = obs.loc[obs.identity == p, 'doublet_score'].values
    b = obs.loc[obs.identity != p, 'doublet_score'].values
    pv[p] = mannwhitneyu(a, b, alternative='greater').pvalue
qc['p_doublet_greater'] = pd.Series(pv)
qc.to_csv(os.path.join(RES, 's09_qc_by_population.csv'))
print(qc.round(4).to_string())

qcd = obs.groupby('disease_state').agg(
    n_cells=('donor_id', 'size'),
    median_genes=('n_genes_by_counts', 'median'),
    median_umi=('total_counts', 'median'),
    median_mito=('pct_counts_mt', 'median'),
    mean_doublet=('doublet_score', 'mean'))
qcd.to_csv(os.path.join(RES, 's09_qc_by_condition.csv'))
log('QC by condition:\n%s' % qcd.round(3).to_string())

# per-donor QC for the supplementary table
don = obs.groupby('donor_id').agg(
    disease_state=('disease_state', 'first'), age=('age', 'first'),
    sex=('sex', 'first'), assay=('assay', 'first'),
    n_cells=('age', 'size'), median_genes=('n_genes_by_counts', 'median'),
    median_umi=('total_counts', 'median'), median_mito=('pct_counts_mt', 'median'),
    mean_doublet=('doublet_score', 'mean'))
pre = full.groupby('donor_id').size().rename('cells_before_filter')
don = don.join(pre)
don['retention_pct'] = 100 * don.n_cells / don.cells_before_filter
don.sort_values(['disease_state', 'donor_id']).to_csv(
    os.path.join(RES, 's09_per_donor_qc.csv'))
log('per-donor QC written (%d donors)' % len(don))

# chemistry mixing
chem = obs.assign(chem=np.where(obs.assay.astype(str).str.lower().str.contains('v2'),
                                "v2", "v3"))
ct = pd.crosstab(chem.identity, chem.chem, normalize='index') * 100
ct.reindex(ORDER).to_csv(os.path.join(RES, 's09_chemistry_by_population.csv'))
log('chemistry: v2 = %d cells (%d donors), v3 = %d cells (%d donors)'
    % ((chem.chem == 'v2').sum(), chem.loc[chem.chem == 'v2', 'donor_id'].nunique(),
       (chem.chem == 'v3').sum(), chem.loc[chem.chem == 'v3', 'donor_id'].nunique()))

# ── 2. hormone fraction (justifies hormone-aware normalisation) ──────────────
log('\n=== hormone fraction of library ===')
import h5py as _h5
from common import H5AD as _H5AD
_f = _h5.File(_H5AD, 'r')
genes = np.array([v.decode() if isinstance(v, bytes) else v
                  for v in _f['var']['_index'][:]])
_on = np.array([v.decode() if isinstance(v, bytes) else v
                for v in _f['obs']['_index'][:]])
_L = _f['layers']['counts']
Cfull = sp.csr_matrix((_L['data'][:].astype(np.float32),
                       _L['indices'][:].astype(np.int32),
                       _L['indptr'][:].astype(np.int64)),
                      shape=(len(_on), len(genes)))
_f.close()
_keep = pd.Index(_on).get_indexer(obs.index)
C = Cfull[_keep]
del Cfull
gc.collect()
import anndata as _ad
A = _ad.AnnData(X=sp.csr_matrix((len(obs), 1), dtype=np.float32),
                obs=obs.copy())
log('counts for hormone analysis: %s | %s' % (str(C.shape), mem()))
horm = np.isin(genes, HORMONE + EXOCRINE)
tot = np.asarray(C.sum(axis=1)).ravel()
hsum = np.asarray(C[:, horm].sum(axis=1)).ravel()
A.obs['hormone_frac'] = 100 * hsum / np.maximum(tot, 1)
ins_i = int(np.where(genes == 'INS')[0][0])
A.obs['ins_frac'] = 100 * np.asarray(C[:, ins_i].todense()).ravel() / np.maximum(tot, 1)
hf = A.obs.groupby(['identity', 'disease_state']).agg(
    hormone_pct=('hormone_frac', 'mean'), ins_pct=('ins_frac', 'mean')).unstack()
hf.reindex(ORDER).to_csv(os.path.join(RES, 's09_hormone_fraction.csv'))
print(hf.reindex(ORDER).round(1).to_string())

# detection rate of INS / GCG in every population (ambient evidence)
for g in ['INS', 'GCG']:
    gi = int(np.where(genes == g)[0][0])
    v = np.asarray(C[:, gi].todense()).ravel()
    A.obs['det_' + g] = v > 0
det = A.obs.groupby('identity')[['det_INS', 'det_GCG']].mean().reindex(ORDER) * 100
det.to_csv(os.path.join(RES, 's09_hormone_detection_rate.csv'))
log('\nINS/GCG detection rate per population (%%):\n%s' % det.round(1).to_string())
del A, C
gc.collect()

# ── 3. curated energy-metabolism pathways ────────────────────────────────────
log('\n=== curated energy pathways ===')
S = pd.read_csv(os.path.join(RES, 's06_reaction_scores.csv'), index_col=0)
scan = pd.read_csv(os.path.join(RES, 's06_subsystem_scan.csv'))
from cobra.io import read_sbml_model
model = read_sbml_model(os.path.join(PART1, '01_input', 'Human-GEM-v1.19.xml'))
subsys = {r.id: (r.subsystem or 'unassigned') for r in model.reactions}
rules = {r.id: r.gene_reaction_rule.strip() for r in model.reactions}
has_gpr = [rid for rid, r in rules.items() if r]
ref = S.loc[has_gpr].mean()
CTS = [c for c in scan.columns if c in ORDER]


def rxns_where(pred):
    return [rid for rid in has_gpr if pred(subsys[rid])]


CUR = {
 'Oxidative phosphorylation': rxns_where(lambda s: s == 'Oxidative phosphorylation'),
 'Glycolysis / gluconeogenesis': rxns_where(lambda s: s.startswith('Glycolysis')),
 'TCA cycle': rxns_where(lambda s: s.startswith('Tricarboxylic acid')),
 'Pentose phosphate pathway': rxns_where(lambda s: s == 'Pentose phosphate pathway'),
 'Pyruvate metabolism': rxns_where(lambda s: s == 'Pyruvate metabolism'),
 'Fatty acid β-oxidation': rxns_where(
     lambda s: s.startswith('Beta oxidation') or s == 'Fatty acid oxidation'),
 'Fatty acid synthesis': rxns_where(
     lambda s: s.startswith('Fatty acid biosynthesis') or s.startswith('Fatty acid elongation')),
}


def rel(ids):
    o = {}
    for ct in CTS:
        c = [x for x in S.columns if x.startswith(ct + '_Control')]
        t = [x for x in S.columns if x.startswith(ct + '_T2D')]
        ids2 = [i for i in ids if i in S.index]
        o[ct] = ((S.loc[ids2, t].mean().mean() - ref[t].mean())
                 - (S.loc[ids2, c].mean().mean() - ref[c].mean()))
    return pd.Series(o)


rows = []
for name, ids in CUR.items():
    r = rel(ids)
    rows.append(dict(pathway=name, n_rxns=len(ids), **r.to_dict(), mean=r.mean(),
                     direction_all=('down' if (r < 0).all() else
                                    'up' if (r > 0).all() else 'mixed')))
CU = pd.DataFrame(rows)
CU.to_csv(os.path.join(RES, 's09_curated_pathways.csv'), index=False)
print(CU.round(3).to_string(index=False))

# global shift, reported in the methods
shift = {}
for ct in CTS:
    c = [x for x in S.columns if x.startswith(ct + '_Control')]
    t = [x for x in S.columns if x.startswith(ct + '_T2D')]
    shift[ct] = ref[t].mean() - ref[c].mean()
pd.Series(shift).to_csv(os.path.join(RES, 's09_global_shift.csv'))
log('global reaction-score shift: %s' % {k: round(v, 3) for k, v in shift.items()})

# ── 4. pathway over-representation of the beta DEG sets ──────────────────────
log('\n=== over-representation analysis ===')
import gseapy as gp
LIBS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human']
ora_rows = []
for pop in ['Beta-3', 'Beta-1']:
    f = os.path.join(RES, 's04_de_noHPAP090_%s.csv' % pop)
    r = pd.read_csv(f, index_col=0).dropna(subset=['padj'])
    sig = r[(r.padj < 0.05) & (r.log2FoldChange.abs() > 0.5)]
    for direction, sub in [('up', sig[sig.log2FoldChange > 0]),
                           ('down', sig[sig.log2FoldChange < 0])]:
        genes_l = list(sub.index)
        if len(genes_l) < 3:
            continue
        try:
            e = gp.enrichr(gene_list=genes_l, gene_sets=LIBS,
                           background=list(r.index), outdir=None, no_plot=True)
            d = e.results.copy()
            d['population'] = pop
            d['direction'] = direction
            d['n_genes_in'] = len(genes_l)
            ora_rows.append(d)
            top = d.sort_values('Adjusted P-value').head(3)
            log('  %s %-5s (%d genes): %s' % (
                pop, direction, len(genes_l),
                '; '.join('%s (padj=%.3f)' % (t.Term[:44], t['Adjusted P-value'])
                          for _, t in top.iterrows())))
        except Exception as ex:
            log('  %s %s ORA failed: %s' % (pop, direction, ex))
if ora_rows:
    O = pd.concat(ora_rows)
    O.to_csv(os.path.join(RES, 's09_ora_results.csv'), index=False)
    sigo = O[O['Adjusted P-value'] < 0.1]
    log('terms reaching adjusted p < 0.1: %d' % len(sigo))
    if len(sigo):
        print(sigo[['population', 'direction', 'Gene_set', 'Term',
                    'Adjusted P-value', 'Genes']].to_string(index=False))

# ── 5. per-donor support for the top DE genes ────────────────────────────────
log('\n=== per-donor detection of the top DE genes ===')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py
from common import H5AD
KEY = ['BARX1', 'TBX2', 'TBX2-AS1', 'FAIM2', 'TSHR', 'PPP1R1A', 'GOLT1A', 'CD82']
f = h5py.File(H5AD, 'r')
vn = np.array([v.decode() if isinstance(v, bytes) else v
               for v in f['raw']['var']['_index'][:]])
on = np.array([v.decode() if isinstance(v, bytes) else v
               for v in f['obs']['_index'][:]])
R = f['raw']['X']
indptr = R['indptr'][:]
gidx = {g: int(np.where(vn == g)[0][0]) for g in KEY if (vn == g).any()}
out = np.zeros((len(on), len(gidx)), dtype=np.float32)
cols = {v: i for i, v in enumerate(gidx.values())}
CH = 20_000_000
nnz = R['data'].shape[0]
for st in range(0, nnz, CH):
    sp_ = min(st + CH, nnz)
    ind = R['indices'][st:sp_]
    hit = np.isin(ind, list(gidx.values()))
    if hit.any():
        pos = np.nonzero(hit)[0]
        dat = R['data'][st:sp_][pos]
        rws = np.searchsorted(indptr, st + pos, side='right') - 1
        for g in np.unique(ind[pos]):
            m = ind[pos] == g
            out[rws[m], cols[int(g)]] = dat[m]
f.close()
E = pd.DataFrame(out, columns=list(gidx), index=on).reindex(obs.index)
E['donor_id'] = obs.donor_id.values
E['disease_state'] = obs.disease_state.values
E['identity'] = obs.identity.values
beta_e = E[E.identity.isin(['Beta-1', 'Beta-3'])]
rows = []
for g in gidx:
    d = beta_e.groupby(['donor_id', 'disease_state'])[g].apply(lambda v: (v > 0).mean() * 100)
    d = d.reset_index()
    for ds in ['Control', 'T2D']:
        sub = d[d.disease_state == ds]
        rows.append(dict(gene=g, condition=ds, n_donors=len(sub),
                         donors_with_detection=int((sub[g] > 0).sum()),
                         median_pct_cells=float(sub[g].median()),
                         mean_pct_cells=float(sub[g].mean())))
PD = pd.DataFrame(rows)
PD.to_csv(os.path.join(RES, 's09_per_donor_detection.csv'), index=False)
print(PD.round(2).to_string(index=False))
beta_e.groupby(['donor_id', 'disease_state'])[list(gidx)].apply(
    lambda v: (v > 0).mean() * 100).to_csv(
    os.path.join(RES, 's09_per_donor_detection_matrix.csv'))
log('STAGE 9 COMPLETE | %s' % mem())
