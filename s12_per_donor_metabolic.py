# -*- coding: utf-8 -*-
"""Stage 12 - per-donor metabolic reaction scoring with donor-level statistics.

Reaction activity scores are computed from expression before any optimisation, so
one profile per donor per population can be built without a mixed-integer solver.
This replaces the resampled-replicate design with genuine biological replicates
and permits formal inference.
"""
import os, sys, io, gc, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, PART1, H5AD, SEED, HORMONE, EXOCRINE, log, mem

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

CAP = 500            # cells per donor per population before aggregation
MIN_CELLS = 20       # a donor must contribute this many cells to be profiled
MIN_DONORS = 5       # per condition, per population
THRESH_PCT = 25.0
POPS = ['Beta-1', 'Beta-2', 'Beta-3', 'Alpha-4', 'Alpha-6', 'Delta']
NORM_EXCLUDE = set(HORMONE + EXOCRINE)

obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)

# ── counts ───────────────────────────────────────────────────────────────────
log('loading counts ...')
f = h5py.File(H5AD, 'r')
genes = np.array([v.decode() if isinstance(v, bytes) else v
                  for v in f['var']['_index'][:]])
allobs = np.array([v.decode() if isinstance(v, bytes) else v
                   for v in f['obs']['_index'][:]])
L = f['layers']['counts']
Cfull = sp.csr_matrix((L['data'][:].astype(np.float32),
                       L['indices'][:].astype(np.int32),
                       L['indptr'][:].astype(np.int64)),
                      shape=(len(allobs), len(genes)))
f.close()
C = Cfull[pd.Index(allobs).get_indexer(obs.index)]
del Cfull
gc.collect()
log('counts %s | %s' % (str(C.shape), mem()))

norm_mask = np.array([g not in NORM_EXCLUDE for g in genes])

# ── one pseudobulk profile per donor per population ──────────────────────────
profiles, meta = {}, []
for pop in POPS:
    m = obs.identity.values == pop
    for donor in pd.unique(obs.donor_id.values[m]):
        idx = np.where(m & (obs.donor_id.values == donor))[0]
        if len(idx) < MIN_CELLS:
            continue
        if len(idx) > CAP:
            idx = rng.choice(idx, CAP, replace=False)
        tot = np.asarray(C[idx].sum(axis=0)).ravel()
        profiles['%s|%s' % (pop, donor)] = tot / tot[norm_mask].sum() * 1e6
        meta.append(dict(sample='%s|%s' % (pop, donor), population=pop, donor=donor,
                         n_cells=len(idx),
                         disease_state=obs.disease_state.values[
                             obs.donor_id.values == donor][0]))
M = pd.DataFrame(meta).set_index('sample')
E = pd.DataFrame(profiles, index=genes)
log('built %d per-donor profiles across %d populations' % (E.shape[1], len(POPS)))
print(M.groupby(['population', 'disease_state']).size().unstack().to_string())
E.to_csv(os.path.join(RES, 's12_per_donor_cpm.csv'))
M.to_csv(os.path.join(RES, 's12_sample_metadata.csv'))
del C
gc.collect()

# ── Human-GEM and GPR scoring (identical to the group-level analysis) ────────
from cobra.io import read_sbml_model
log('loading Human-GEM ...')
model = read_sbml_model(os.path.join(PART1, '01_input', 'Human-GEM-v1.19.xml'))
gt = pd.read_csv(os.path.join(PART1, '01_input', 'genes.tsv'), sep='\t')
sym2ens = dict(zip(gt['geneSymbols'], gt['genes']))
subsys = {r.id: (r.subsystem or 'unassigned') for r in model.reactions}
rules = {r.id: r.gene_reaction_rule.strip() for r in model.reactions}
TOK = {rid: r.replace('(', ' ( ').replace(')', ' ) ').split()
       for rid, r in rules.items() if r}


def pexpr(t, p, gs):
    v, p = pterm(t, p, gs)
    while p < len(t) and t[p].lower() == 'or':
        p += 1
        r, p = pterm(t, p, gs)
        v = max(v, r)
    return v, p


def pterm(t, p, gs):
    v, p = pfac(t, p, gs)
    while p < len(t) and t[p].lower() == 'and':
        p += 1
        r, p = pfac(t, p, gs)
        v = min(v, r)
    return v, p


def pfac(t, p, gs):
    if p >= len(t):
        return 0.0, p
    if t[p] == '(':
        p += 1
        v, p = pexpr(t, p, gs)
        if p < len(t) and t[p] == ')':
            p += 1
        return v, p
    return gs.get(t[p], 0.0), p + 1


log('scoring %d profiles ...' % E.shape[1])
scores = {}
for k, samp in enumerate(E.columns):
    gs = {}
    for sym, v in E[samp].items():
        e = sym2ens.get(sym)
        if e:
            gs[e] = float(np.log2(v + 1.0))
    exprsd = [v for v in gs.values() if v > 0]
    thr = float(np.percentile(exprsd, THRESH_PCT)) if exprsd else 0.0
    scores[samp] = {rid: (0.5 if not rules[rid] else pexpr(TOK[rid], 0, gs)[0] - thr)
                    for rid in rules}
    if (k + 1) % 40 == 0:
        log('  %d / %d' % (k + 1, E.shape[1]))
S = pd.DataFrame(scores)
S.to_csv(os.path.join(RES, 's12_per_donor_reaction_scores.csv'))
log('score matrix %s | %s' % (str(S.shape), mem()))

# ── per-donor subsystem scores, corrected for each sample's global level ─────
has_gpr = [rid for rid, r in rules.items() if r]
ref = S.loc[has_gpr].mean()                       # per-sample global mean
sub2rx = {}
for rid in has_gpr:
    sub2rx.setdefault(subsys[rid], []).append(rid)
sub2rx = {k: v for k, v in sub2rx.items() if len(v) >= 15}
log('%d subsystems with >= 15 GPR reactions' % len(sub2rx))

REL = pd.DataFrame({name: S.loc[ids].mean() - ref for name, ids in sub2rx.items()})
REL = REL.join(M)
REL.to_csv(os.path.join(RES, 's12_per_donor_subsystem_scores.csv'))

# ── (a) per-population donor-level tests ─────────────────────────────────────
rows = []
for pop in POPS:
    sub = REL[REL.population == pop]
    c = sub[sub.disease_state == 'Control']
    t = sub[sub.disease_state == 'T2D']
    if len(c) < MIN_DONORS or len(t) < MIN_DONORS:
        continue
    for name in sub2rx:
        u, p = mannwhitneyu(t[name], c[name], alternative='two-sided')
        rows.append(dict(population=pop, subsystem=name, n_ctrl=len(c), n_t2d=len(t),
                         median_ctrl=c[name].median(), median_t2d=t[name].median(),
                         delta=t[name].median() - c[name].median(), p=p))
PP = pd.DataFrame(rows)
PP['padj'] = np.nan
for pop in PP.population.unique():
    m = PP.population == pop
    PP.loc[m, 'padj'] = multipletests(PP.loc[m, 'p'], method='fdr_bh')[1]
PP.to_csv(os.path.join(RES, 's12_subsystem_tests_per_population.csv'), index=False)

# ── (b) pooled donor-level test (one value per donor, averaged over
#        the populations in which that donor is represented) ─────────────────
pool = REL.groupby(['donor', 'disease_state'])[list(sub2rx)].mean().reset_index()
n_c = int((pool.disease_state == 'Control').sum())
n_t = int((pool.disease_state == 'T2D').sum())
log('\npooled donor-level test: %d control vs %d T2D donors' % (n_c, n_t))
rows = []
for name in sub2rx:
    c = pool.loc[pool.disease_state == 'Control', name]
    t = pool.loc[pool.disease_state == 'T2D', name]
    u, p = mannwhitneyu(t, c, alternative='two-sided')
    # rank-biserial correlation as effect size
    rb = 2 * u / (len(c) * len(t)) - 1
    rows.append(dict(subsystem=name, n_rxns=len(sub2rx[name]),
                     median_ctrl=c.median(), median_t2d=t.median(),
                     delta=t.median() - c.median(), effect_r=rb, p=p))
PO = pd.DataFrame(rows)
PO['padj'] = multipletests(PO.p, method='fdr_bh')[1]
PO['n_pops_sig'] = [int(((PP.subsystem == s) & (PP.padj < 0.05)).sum()) for s in PO.subsystem]
PO = PO.sort_values('delta')
PO.to_csv(os.path.join(RES, 's12_subsystem_tests_pooled.csv'), index=False)

sig = PO[PO.padj < 0.05]
log('\n=== %d of %d subsystems significant at FDR 5%% (pooled donor-level) ==='
    % (len(sig), len(PO)))
print(sig[['subsystem', 'n_rxns', 'delta', 'effect_r', 'p', 'padj', 'n_pops_sig']]
      .round(4).to_string(index=False))

# ── gene-level, same donor-level framework ───────────────────────────────────
gene2rx = {}
for rid in has_gpr:
    for g in TOK[rid]:
        if g.startswith('ENSG'):
            gene2rx.setdefault(g, []).append(rid)
ens2sym = {v: k for k, v in sym2ens.items()}
KEY = ['G6PD', 'PGD', 'TALDO1', 'TKT', 'SHPK', 'TPK1',
       'GCH1', 'PTS', 'SPR', 'PCBD1', 'QDPR', 'DHFR']
key_ens = {ens2sym.get(g, g): g for g in gene2rx if ens2sym.get(g, g) in KEY}
GR = pd.DataFrame({sym: S.loc[gene2rx[e]].mean() - ref for sym, e in key_ens.items()})
GR = GR.join(M)
gpool = GR.groupby(['donor', 'disease_state'])[list(key_ens)].mean().reset_index()
rows = []
for sym in key_ens:
    c = gpool.loc[gpool.disease_state == 'Control', sym]
    t = gpool.loc[gpool.disease_state == 'T2D', sym]
    u, p = mannwhitneyu(t, c, alternative='two-sided')
    rows.append(dict(gene=sym, n_rxns=len(gene2rx[key_ens[sym]]),
                     median_ctrl=c.median(), median_t2d=t.median(),
                     delta=t.median() - c.median(),
                     effect_r=2 * u / (len(c) * len(t)) - 1, p=p))
G = pd.DataFrame(rows)
G['padj'] = multipletests(G.p, method='fdr_bh')[1]
G = G.sort_values('delta')
G.to_csv(os.path.join(RES, 's12_gene_tests_pooled.csv'), index=False)
GR.to_csv(os.path.join(RES, 's12_per_donor_gene_scores.csv'))
log('\n=== NADPH / BH4 axis, pooled donor-level ===')
print(G[['gene', 'n_rxns', 'delta', 'effect_r', 'p', 'padj']].round(4).to_string(index=False))
log('STAGE 12 COMPLETE | %s' % mem())
