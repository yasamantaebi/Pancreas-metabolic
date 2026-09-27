# -*- coding: utf-8 -*-
"""Stage 23 - metabolic-gene expression matrices for the metabolic modelling work.

Builds three deliverables from the raw count layer:

  FILE 1  metabolic genes x 224 pseudobulk samples, RAW SUMMED COUNTS
          Columns are exactly the 224 `Population|Donor` samples that were sent
          for the ftINIT run (same populations, same donors, same 500-cell cap,
          same random seed), so this file is the count-level source of
          `s12_per_donor_cpm.csv`.

  FILE 2  the same matrix, CPM-normalised (counts / full library total * 1e6).

  FILE 3  metabolic genes x cell type, raw counts and CPM, using EVERY cell of
          each annotated identity (no cap, no subsampling).

INPUT   part 1/01_input/hpap_processed.h5ad          layers/counts (raw UMIs)
        part 1/01_input/genes.tsv                    Human-GEM gene table
        02_results/s03_obs_final.csv                 per-cell identity + donor
        02_results/s12_sample_metadata.csv           the 224 sample definitions
        02_results/s12_per_donor_cpm.csv             for the reproduction check
OUTPUT  08_metabolic_matrices/*.csv

The counts layer is streamed in row blocks: the machine has ~1.4 GB free and the
full CSR is ~1.5 GB.
"""
import os, sys, io, gc, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PART1, BASE, H5AD, SEED, HORMONE, EXOCRINE, log

warnings.filterwarnings('ignore')

OUT = os.path.join(BASE, '08_metabolic_matrices')
os.makedirs(OUT, exist_ok=True)

CAP = 500          # cells per donor per population, as in stage 12
MIN_CELLS = 20     # a donor must contribute this many cells to be profiled
POPS = ['Beta-1', 'Beta-2', 'Beta-3', 'Alpha-4', 'Alpha-6', 'Delta']
NORM_EXCLUDE = set(HORMONE + EXOCRINE)
BLOCK = 4000       # rows per streaming block

t0 = time.time()
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
log('obs %d cells, %d identities' % (len(obs), obs.identity.nunique()))

# ── gene / cell indices from the h5ad, without reading the matrix ─────────────
f = h5py.File(H5AD, 'r')
genes = np.array([v.decode() if isinstance(v, bytes) else v
                  for v in f['var']['_index'][:]])
allobs = np.array([v.decode() if isinstance(v, bytes) else v
                   for v in f['obs']['_index'][:]])
L = f['layers']['counts']
indptr = L['indptr'][:].astype(np.int64)
n_rows_h5, n_genes = len(allobs), len(genes)
log('h5ad %d cells x %d genes, %d non-zeros' % (n_rows_h5, n_genes, indptr[-1]))

pos = pd.Index(allobs).get_indexer(obs.index)      # h5ad row for each kept cell
assert (pos >= 0).all(), 'some cells in s03_obs_final are absent from the h5ad'

# ── FILE 1/2 sample definition: replicate stage 12 exactly ───────────────────
# Same seed, same loop order, same cap, so the resulting profiles correspond
# cell-for-cell to the matrix the collaborator already has.
rng = np.random.default_rng(SEED)
samples, sample_cells = [], []
for pop in POPS:
    m = obs.identity.values == pop
    for donor in pd.unique(obs.donor_id.values[m]):
        idx = np.where(m & (obs.donor_id.values == donor))[0]
        if len(idx) < MIN_CELLS:
            continue
        if len(idx) > CAP:
            idx = rng.choice(idx, CAP, replace=False)
        samples.append('%s|%s' % (pop, donor))
        sample_cells.append(idx)
log('defined %d pseudobulk samples' % len(samples))

ref_meta = pd.read_csv(os.path.join(RES, 's12_sample_metadata.csv'), index_col=0)
assert list(ref_meta.index) == samples, 'sample list differs from stage 12'
log('sample list matches s12_sample_metadata.csv exactly')

# map each h5ad row to a sample (-1 = not used) and to a cell type
samp_of_row = np.full(n_rows_h5, -1, np.int32)
for j, idx in enumerate(sample_cells):
    samp_of_row[pos[idx]] = j

celltypes = sorted(obs.identity.unique())
ct_index = {c: i for i, c in enumerate(celltypes)}
ct_of_row = np.full(n_rows_h5, -1, np.int32)
ct_of_row[pos] = obs.identity.map(ct_index).values.astype(np.int32)

lineages = sorted(obs.lineage.unique())
ln_index = {c: i for i, c in enumerate(lineages)}
ln_of_row = np.full(n_rows_h5, -1, np.int32)
ln_of_row[pos] = obs.lineage.map(ln_index).values.astype(np.int32)

# ── stream the count layer, accumulating group sums ──────────────────────────
A_samp = np.zeros((len(samples), n_genes), np.float64)
A_ct = np.zeros((len(celltypes), n_genes), np.float64)
A_ln = np.zeros((len(lineages), n_genes), np.float64)


def accumulate(acc, group_of_row, chunk, r0, r1):
    g = group_of_row[r0:r1]
    keep = np.where(g >= 0)[0]
    if not len(keep):
        return
    G = sp.csr_matrix((np.ones(len(keep)), (g[keep], keep)),
                      shape=(acc.shape[0], r1 - r0))
    acc += np.asarray((G @ chunk).todense())


log('streaming counts in blocks of %d rows ...' % BLOCK)
for r0 in range(0, n_rows_h5, BLOCK):
    r1 = min(r0 + BLOCK, n_rows_h5)
    a, b = indptr[r0], indptr[r1]
    if b > a:
        chunk = sp.csr_matrix(
            (L['data'][a:b].astype(np.float64), L['indices'][a:b].astype(np.int32),
             indptr[r0:r1 + 1] - a), shape=(r1 - r0, n_genes))
        accumulate(A_samp, samp_of_row, chunk, r0, r1)
        accumulate(A_ct, ct_of_row, chunk, r0, r1)
        accumulate(A_ln, ln_of_row, chunk, r0, r1)
        del chunk
    if (r0 // BLOCK) % 5 == 0:
        log('  rows %d / %d' % (r1, n_rows_h5))
    gc.collect()
f.close()
log('streaming done in %.0fs' % (time.time() - t0))

RAW = pd.DataFrame(A_samp.T, index=genes, columns=samples)
CT_RAW = pd.DataFrame(A_ct.T, index=genes, columns=celltypes)
LN_RAW = pd.DataFrame(A_ln.T, index=genes, columns=lineages)
assert (RAW.values == np.round(RAW.values)).all(), 'counts are not integers'

# ── reproduction check against the matrix the collaborator received ──────────
# Stage 12 normalised by the library total EXCLUDING hormone and exocrine genes
# (the ambient transcripts). Rebuilding that here proves the raw counts below
# are the same cells in the same order.
norm_mask = np.array([g not in NORM_EXCLUDE for g in genes])
repro = RAW.values / RAW.values[norm_mask].sum(axis=0) * 1e6
ref = pd.read_csv(os.path.join(RES, 's12_per_donor_cpm.csv'), index_col=0)
ref = ref.loc[genes, samples]
dev = np.abs(repro - ref.values).max()
rel = np.abs(repro - ref.values).max() / max(ref.values.max(), 1.0)
log('reproduction check vs s12_per_donor_cpm.csv: max abs dev %.3e (rel %.2e)'
    % (dev, rel))
assert rel < 1e-6, 'raw counts do not reproduce the stage-12 CPM matrix'
del repro, ref
gc.collect()

# ── CPM: full library total, then subset to metabolic genes ──────────────────
# Normalising before subsetting keeps the values comparable to any other CPM
# table from this dataset; normalising after would make "per million" mean
# "per million metabolic transcripts", which is not what CPM means.
lib = RAW.values.sum(axis=0)
CPM = pd.DataFrame(RAW.values / lib * 1e6, index=genes, columns=samples)
CT_CPM = pd.DataFrame(CT_RAW.values / CT_RAW.values.sum(axis=0) * 1e6,
                      index=genes, columns=celltypes)
LN_CPM = pd.DataFrame(LN_RAW.values / LN_RAW.values.sum(axis=0) * 1e6,
                      index=genes, columns=lineages)

# Ambient-aware variant: the same CPM with hormone and exocrine transcripts
# removed from the DENOMINATOR only. They are 20% of the library in alpha cells
# and up to 71% in beta, so plain CPM deflates every metabolic gene in beta by
# roughly two-fold for a reason that is technical, not biological. This is the
# normalisation stage 12 used, i.e. the one behind the matrix already sent out.
CPM_AA = pd.DataFrame(RAW.values / RAW.values[norm_mask].sum(axis=0) * 1e6,
                      index=genes, columns=samples)
CT_CPM_AA = pd.DataFrame(
    CT_RAW.values / CT_RAW.values[norm_mask].sum(axis=0) * 1e6,
    index=genes, columns=celltypes)

# ── metabolic gene set = Human-GEM genes present in the data ─────────────────
gt = pd.read_csv(os.path.join(PART1, '01_input', 'genes.tsv'), sep='\t')
gem = gt[['genes', 'geneSymbols']].dropna().drop_duplicates('geneSymbols')
present = gem[gem.geneSymbols.isin(set(genes))]
met = [g for g in genes if g in set(present.geneSymbols)]      # dataset order
log('Human-GEM genes %d | detected in this dataset %d | missing %d'
    % (len(gem), len(met), len(gem) - len(met)))

ann = present.set_index('geneSymbols').loc[met, 'genes'].rename('ensembl_id')
detected = (RAW.loc[met] > 0).any(axis=1)
log('metabolic genes with >0 counts anywhere: %d of %d' % (detected.sum(), len(met)))

# ── write ────────────────────────────────────────────────────────────────────
def write(df, name, note):
    out = df.copy()
    out.index.name = 'gene_symbol'
    out.insert(0, 'ensembl_id', ann.reindex(out.index).values)
    p = os.path.join(OUT, name)
    out.to_csv(p, float_format='%.6g')
    log('wrote %-52s %d genes x %d columns  (%s)'
        % (name, len(out), out.shape[1] - 1, note))


write(RAW.loc[met].astype(np.int64), 'F1_metabolic_genes_RAW_counts_224samples.csv',
      'raw summed UMIs')
write(CPM.loc[met], 'F2_metabolic_genes_CPM_224samples.csv', 'CPM')
write(CPM_AA.loc[met], 'F2b_metabolic_genes_CPM_ambientAware_224samples.csv',
      'CPM, ambient excluded from denominator')
write(CT_CPM.loc[met], 'F3_metabolic_genes_CPM_by_celltype.csv', 'CPM')
write(CT_CPM_AA.loc[met], 'F3d_metabolic_genes_CPM_ambientAware_by_celltype.csv',
      'CPM, ambient excluded from denominator')
write(CT_RAW.loc[met].astype(np.int64), 'F3b_metabolic_genes_RAW_counts_by_celltype.csv',
      'raw summed UMIs')
write(LN_CPM.loc[met], 'F3c_metabolic_genes_CPM_by_lineage.csv', 'CPM')

# ── companion metadata ───────────────────────────────────────────────────────
meta = ref_meta.copy()
meta['library_total_counts'] = lib.astype(np.int64)
meta['metabolic_counts'] = RAW.loc[met].values.sum(axis=0).astype(np.int64)
meta['pct_counts_metabolic'] = 100.0 * meta.metabolic_counts / meta.library_total_counts
meta['pct_counts_hormone_exocrine'] = 100.0 * RAW.values[~norm_mask].sum(axis=0) / lib
meta.to_csv(os.path.join(OUT, 'sample_metadata_224.csv'))

ctm = pd.DataFrame(dict(
    n_cells=obs.identity.value_counts().reindex(celltypes),
    n_donors=obs.groupby('identity').donor_id.nunique().reindex(celltypes),
    library_total_counts=CT_RAW.values.sum(axis=0).astype(np.int64),
    pct_counts_metabolic=100.0 * CT_RAW.loc[met].values.sum(axis=0)
    / CT_RAW.values.sum(axis=0),
    pct_counts_hormone_exocrine=100.0 * CT_RAW.values[~norm_mask].sum(axis=0)
    / CT_RAW.values.sum(axis=0)), index=celltypes)
ctm.index.name = 'cell_type'
ctm.to_csv(os.path.join(OUT, 'celltype_metadata.csv'))

print('\n' + '=' * 74)
print('AMBIENT LOAD - why the hormone/exocrine fraction matters for CPM')
print('=' * 74)
print(ctm.round(2).to_string())
print('\nacross the 224 samples, hormone+exocrine share of the library:')
print('  median %.1f%%   range %.1f-%.1f%%'
      % (meta.pct_counts_hormone_exocrine.median(),
         meta.pct_counts_hormone_exocrine.min(),
         meta.pct_counts_hormone_exocrine.max()))
print('  by population:')
print(meta.groupby('population').pct_counts_hormone_exocrine
      .agg(['median', 'min', 'max']).round(1).to_string())
log('STAGE 23 COMPLETE in %.0fs' % (time.time() - t0))
