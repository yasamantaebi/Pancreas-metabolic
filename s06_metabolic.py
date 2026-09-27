# -*- coding: utf-8 -*-
"""Stage 6 - GPR-resolved reaction activity scores on the re-annotated populations.

Reproduces the scoring used previously (hormone-aware CPM, log2(CPM+1), OR=max /
AND=min GPR evaluation, threshold = 25th percentile of expressed genes) so the
results are directly comparable, but computes it on the marker-verified
populations. No mixed-integer solver is required: these scores are computed from
expression before any optimisation.
"""
import os, sys, gc, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, PART1, SEED, HORMONE, EXOCRINE, log, mem

warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

CAP = 500
N_REP = 3
FRAC = 0.80
THRESH_PCT = 25.0
POPS = ['Beta-1', 'Beta-2', 'Beta-3', 'Alpha-4', 'Alpha-6', 'Delta']
NORM_EXCLUDE = set(HORMONE + EXOCRINE)

# ── pseudobulk profiles ──────────────────────────────────────────────────────
log('loading counts ...')
A = sc.read_h5ad(os.path.join(DATA, 'lognorm_full.h5ad'))
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
A.obs = obs.reindex(A.obs_names)
C = A.layers['counts']
C = C if sp.isspmatrix_csr(C) else sp.csr_matrix(C)
genes = np.array(A.var_names)
log('loaded %s | %s' % (str(A.shape), mem()))

norm_mask = np.array([g not in NORM_EXCLUDE for g in genes])
log('normalisation denominator excludes %d hormone/exocrine transcripts'
    % (~norm_mask).sum())

profiles = {}
for pop in POPS:
    for cond in ['Control', 'T2D']:
        m = ((A.obs.identity.values == pop) &
             (A.obs.disease_state.values == cond))
        donors = pd.unique(A.obs.donor_id.values[m])
        if len(donors) < 5:
            log('  skip %s %s (only %d donors)' % (pop, cond, len(donors)))
            continue
        for rep in range(N_REP):
            keep = rng.choice(donors, max(3, int(round(FRAC * len(donors)))),
                              replace=False)
            idx = np.where(m & np.isin(A.obs.donor_id.values, keep))[0]
            # equalise donor contribution before aggregation
            sel = []
            for d in keep:
                gi = idx[A.obs.donor_id.values[idx] == d]
                sel.append(rng.choice(gi, CAP, replace=False) if len(gi) > CAP else gi)
            sel = np.concatenate(sel)
            tot = np.asarray(C[sel].sum(axis=0)).ravel()
            denom = tot[norm_mask].sum()
            cpm = tot / denom * 1e6
            profiles['%s_%s_rep%d' % (pop, cond, rep + 1)] = cpm
        log('  %-9s %-7s donors=%2d  reps=%d' % (pop, cond, len(donors), N_REP))

E = pd.DataFrame(profiles, index=genes)
E.to_csv(os.path.join(RES, 's06_pseudobulk_cpm.csv'))
log('pseudobulk matrix %s' % str(E.shape))
del A, C
gc.collect()

# ── Human-GEM ────────────────────────────────────────────────────────────────
from cobra.io import read_sbml_model
log('loading Human-GEM v1.19 ...')
model = read_sbml_model(os.path.join(PART1, '01_input', 'Human-GEM-v1.19.xml'))
gt = pd.read_csv(os.path.join(PART1, '01_input', 'genes.tsv'), sep='\t')
sym2ens = dict(zip(gt['geneSymbols'], gt['genes']))
subsys = {r.id: (r.subsystem or 'unassigned') for r in model.reactions}
rules = {r.id: r.gene_reaction_rule.strip() for r in model.reactions}
log('model: %d reactions, %d with GPR' % (len(rules), sum(1 for v in rules.values() if v)))


def parse_expr(tok, pos, gs):
    val, pos = parse_term(tok, pos, gs)
    while pos < len(tok) and tok[pos].lower() == 'or':
        pos += 1
        r, pos = parse_term(tok, pos, gs)
        val = max(val, r)
    return val, pos


def parse_term(tok, pos, gs):
    val, pos = parse_factor(tok, pos, gs)
    while pos < len(tok) and tok[pos].lower() == 'and':
        pos += 1
        r, pos = parse_factor(tok, pos, gs)
        val = min(val, r)
    return val, pos


def parse_factor(tok, pos, gs):
    if pos >= len(tok):
        return 0.0, pos
    t = tok[pos]
    if t == '(':
        pos += 1
        val, pos = parse_expr(tok, pos, gs)
        if pos < len(tok) and tok[pos] == ')':
            pos += 1
        return val, pos
    return gs.get(t, 0.0), pos + 1


TOKENS = {rid: r.replace('(', ' ( ').replace(')', ' ) ').split()
          for rid, r in rules.items() if r}

scores = {}
for samp in E.columns:
    cpm = E[samp]
    gs = {}
    for sym, v in cpm.items():
        e = sym2ens.get(sym)
        if e:
            gs[e] = float(np.log2(v + 1.0))
    expressed = [v for v in gs.values() if v > 0]
    thr = float(np.percentile(expressed, THRESH_PCT)) if expressed else 0.0
    col = {}
    for rid, rule in rules.items():
        if not rule:
            col[rid] = 0.5
        else:
            col[rid] = parse_expr(TOKENS[rid], 0, gs)[0] - thr
    scores[samp] = col
    log('  scored %-22s threshold=%.3f' % (samp, thr))

S = pd.DataFrame(scores)
S.to_csv(os.path.join(RES, 's06_reaction_scores.csv'))
log('reaction score matrix %s' % str(S.shape))

# ── comparative analysis, corrected for the global shift ─────────────────────
has_gpr = [rid for rid, r in rules.items() if r]
ref = S.loc[has_gpr].mean()
log('\nglobal shift per population (T2D - Control, all %d GPR reactions):' % len(has_gpr))
CTS = sorted({c.rsplit('_', 2)[0] for c in S.columns})
shift = {}
for ct in CTS:
    c = [x for x in S.columns if x.startswith(ct + '_Control')]
    t = [x for x in S.columns if x.startswith(ct + '_T2D')]
    if c and t:
        shift[ct] = ref[t].mean() - ref[c].mean()
        log('  %-9s %+.3f' % (ct, shift[ct]))
CTS = [c for c in CTS if c in shift]


def rel(ids):
    o = {}
    for ct in CTS:
        c = [x for x in S.columns if x.startswith(ct + '_Control')]
        t = [x for x in S.columns if x.startswith(ct + '_T2D')]
        ids2 = [i for i in ids if i in S.index]
        if not ids2:
            continue
        o[ct] = ((S.loc[ids2, t].mean().mean() - ref[t].mean())
                 - (S.loc[ids2, c].mean().mean() - ref[c].mean()))
    return pd.Series(o)


def pairs_consistent(ids):
    """Direction agreement across all Control-vs-T2D replicate pairs."""
    out = {}
    for ct in CTS:
        c = [x for x in S.columns if x.startswith(ct + '_Control')]
        t = [x for x in S.columns if x.startswith(ct + '_T2D')]
        ids2 = [i for i in ids if i in S.index]
        d = []
        for ci in c:
            for ti in t:
                d.append((S.loc[ids2, ti].mean() - ref[ti])
                         - (S.loc[ids2, ci].mean() - ref[ci]))
        d = np.array(d)
        out[ct] = (d < 0).sum() if np.mean(d) < 0 else (d > 0).sum()
        out[ct + '_n'] = len(d)
    return out


# unbiased subsystem survey
sub2rx = {}
for rid in has_gpr:
    sub2rx.setdefault(subsys[rid], []).append(rid)
rows = []
for sname, ids in sub2rx.items():
    if len(ids) < 15:
        continue
    r = rel(ids)
    if len(r) < len(CTS):
        continue
    pc = pairs_consistent(ids)
    allc = (np.sign(r).nunique() == 1) and all(
        pc[ct] == pc[ct + '_n'] for ct in CTS)
    rows.append(dict(subsystem=sname, n_rxns=len(ids), **r.to_dict(),
                     mean=r.mean(), all_consistent=bool(allc)))
SUB = pd.DataFrame(rows).sort_values('mean')
SUB.to_csv(os.path.join(RES, 's06_subsystem_scan.csv'), index=False)
cons = SUB[SUB.all_consistent]
log('\n=== %d of %d subsystems changed concordantly in all %d populations ==='
    % (len(cons), len(SUB), len(CTS)))
print(cons[['subsystem', 'n_rxns'] + CTS + ['mean']].round(3).to_string(index=False))

# gene-level attribution
gene2rx = {}
for rid in has_gpr:
    for g in TOKENS[rid]:
        if g.startswith('ENSG'):
            gene2rx.setdefault(g, []).append(rid)
ens2sym = {v: k for k, v in sym2ens.items()}
grows = []
for g, ids in gene2rx.items():
    if len(ids) < 1:
        continue
    r = rel(ids)
    if len(r) < len(CTS):
        continue
    grows.append(dict(gene=ens2sym.get(g, g), n_rxns=len(ids), **r.to_dict(),
                      mean=r.mean()))
G = pd.DataFrame(grows).sort_values('mean')
G.to_csv(os.path.join(RES, 's06_gene_level.csv'), index=False)

KEY = ['G6PD', 'PGD', 'TALDO1', 'TKT', 'SHPK', 'PGM2',
       'GCH1', 'PTS', 'SPR', 'PCBD1', 'QDPR', 'DHFR', 'TPK1', 'CD36', 'SLC27A6']
log('\n=== NADPH / BH4 axis, gene level ===')
print(G[G.gene.isin(KEY)].set_index('gene').reindex(KEY)[
    ['n_rxns'] + CTS + ['mean']].round(3).to_string())
log('STAGE 6 COMPLETE | %s' % mem())
