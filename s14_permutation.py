# -*- coding: utf-8 -*-
"""Stage 14 - is the metabolic pattern real even though no single subsystem
survives FDR?

Three questions, each with a proper test:
  Q1  Is the number of subsystems concordant in direction across all six
      populations greater than expected by chance?  -> donor-label permutation
  Q2  Is biopterin metabolism's rank among the 98 subsystems unusual?
  Q3  Treating the NADPH-BH4 axis as a single composite, is it reduced?
      (reported with the caveat that the composite was suggested by these data)
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, log

N_PERM = 2000
rng = np.random.default_rng(42)
POPS = ['Beta-1', 'Beta-2', 'Beta-3', 'Alpha-4', 'Alpha-6', 'Delta']

REL = pd.read_csv(os.path.join(RES, 's12_per_donor_subsystem_scores.csv'), index_col=0)
SUBS = [c for c in REL.columns
        if c not in ('population', 'donor', 'n_cells', 'disease_state')]
donors = REL[['donor', 'disease_state']].drop_duplicates().set_index('donor')['disease_state']
log('%d subsystems, %d profiles, %d donors (%d control / %d T2D)'
    % (len(SUBS), len(REL), len(donors), (donors == 'Control').sum(),
       (donors == 'T2D').sum()))


def concordance(labels):
    """Per-population median difference, then count subsystems whose direction
    agrees across all six populations. Returns (count, per-subsystem mean delta)."""
    lab = REL.donor.map(labels)
    deltas = np.full((len(POPS), len(SUBS)), np.nan)
    for i, pop in enumerate(POPS):
        m = (REL.population == pop).values
        c = REL.loc[m & (lab == 'Control').values, SUBS].median()
        t = REL.loc[m & (lab == 'T2D').values, SUBS].median()
        deltas[i] = (t - c).values
    sign = np.sign(deltas)
    n_conc = int(np.sum(np.all(sign == sign[0], axis=0) & np.all(sign != 0, axis=0)))
    return n_conc, np.nanmean(deltas, axis=0)


obs_n, obs_delta = concordance(donors)
bio = SUBS.index([s for s in SUBS if 'Biopterin' in s][0])
obs_rank = int(np.argsort(obs_delta).tolist().index(bio)) + 1
log('\nOBSERVED: %d of %d subsystems concordant across all six populations'
    % (obs_n, len(SUBS)))
log('OBSERVED: biopterin metabolism ranks %d of %d by mean delta (1 = most reduced)'
    % (obs_rank, len(SUBS)))

# ── Q1/Q2: permute disease labels at the donor level ─────────────────────────
lab_vals = donors.values.copy()
null_n, null_rank = np.zeros(N_PERM, int), np.zeros(N_PERM, int)
for k in range(N_PERM):
    perm = pd.Series(rng.permutation(lab_vals), index=donors.index)
    n, d = concordance(perm)
    null_n[k] = n
    null_rank[k] = int(np.argsort(d).tolist().index(bio)) + 1
    if (k + 1) % 500 == 0:
        log('  %d / %d permutations' % (k + 1, N_PERM))

p_conc = (np.sum(null_n >= obs_n) + 1) / (N_PERM + 1)
p_rank = (np.sum(null_rank <= obs_rank) + 1) / (N_PERM + 1)
log('\nQ1  concordant subsystems: observed %d, null median %d (95th pct %d)'
    % (obs_n, int(np.median(null_n)), int(np.percentile(null_n, 95))))
log('    permutation p = %.4f' % p_conc)
log('Q2  biopterin rank: observed %d, null median %d (5th pct %d)'
    % (obs_rank, int(np.median(null_rank)), int(np.percentile(null_rank, 5))))
log('    permutation p = %.4f' % p_rank)

# ── Q3: the NADPH-BH4 axis as one composite ──────────────────────────────────
GR = pd.read_csv(os.path.join(RES, 's12_per_donor_gene_scores.csv'), index_col=0)
AXIS = [g for g in ['G6PD', 'TALDO1', 'TKT', 'SHPK', 'GCH1', 'PTS', 'SPR', 'PCBD1']
        if g in GR.columns]
gp = GR.groupby(['donor', 'disease_state'])[AXIS].mean()
composite = gp.mean(axis=1).reset_index()
composite.columns = ['donor', 'disease_state', 'axis_score']
c = composite.loc[composite.disease_state == 'Control', 'axis_score']
t = composite.loc[composite.disease_state == 'T2D', 'axis_score']
u, p_two = mannwhitneyu(t, c, alternative='two-sided')
_, p_one = mannwhitneyu(t, c, alternative='less')
rb = 2 * u / (len(c) * len(t)) - 1
boot = [np.median(rng.choice(t, len(t))) - np.median(rng.choice(c, len(c)))
        for _ in range(5000)]
log('\nQ3  NADPH-BH4 composite (%d genes: %s)' % (len(AXIS), ', '.join(AXIS)))
log('    control median %.3f (n=%d), T2D median %.3f (n=%d)'
    % (c.median(), len(c), t.median(), len(t)))
log('    delta %.3f, 95%% CI [%.3f, %.3f]'
    % (t.median() - c.median(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
log('    rank-biserial r = %.3f | p(two-sided) = %.4f | p(one-sided, reduced) = %.4f'
    % (rb, p_two, p_one))
composite.to_csv(os.path.join(RES, 's14_axis_composite.csv'), index=False)

# ── how many donors would be needed? ─────────────────────────────────────────
from scipy.stats import norm
d_eff = (c.mean() - t.mean()) / np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2)
for power in (0.8, 0.9):
    z = norm.ppf(1 - 0.05 / 2) + norm.ppf(power)
    ratio = len(c) / len(t)
    n_t = (z / d_eff) ** 2 * (1 + 1 / ratio)
    log('    to reach %.0f%% power at alpha 0.05: ~%d T2D and ~%d control donors'
        % (power * 100, np.ceil(n_t), np.ceil(n_t * ratio)))
log('    observed Cohen d = %.3f' % d_eff)

pd.DataFrame(dict(metric=['n_concordant', 'biopterin_rank'],
                  observed=[obs_n, obs_rank],
                  null_median=[np.median(null_n), np.median(null_rank)],
                  perm_p=[p_conc, p_rank])).to_csv(
    os.path.join(RES, 's14_permutation_results.csv'), index=False)
np.save(os.path.join(RES, 's14_null_nconc.npy'), null_n)
np.save(os.path.join(RES, 's14_null_biopterin_rank.npy'), null_rank)
log('\nSTAGE 14 COMPLETE')
