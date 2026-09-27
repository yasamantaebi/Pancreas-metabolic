# -*- coding: utf-8 -*-
"""Sensitivity analysis for the pooled donor-level subsystem test (Section 3.4).

Repeats the s12 pooled test (one value per donor = mean over the populations in
which the donor is represented; two-sided Mann-Whitney; BH across the 98
subsystems) after (i) requiring >= 50 / 100 / 200 cells per donor-population
profile and (ii) restricting to the three beta populations.

Outputs  02_results/s29_activity_sensitivity.csv
         03_figures/supplementary/FigureS7_activity_sensitivity.png
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, log
import figstyle as F

REL = pd.read_csv(os.path.join(RES, 's12_per_donor_subsystem_scores.csv'), index_col=0)
SUBS = [c for c in REL.columns if c not in ('population', 'donor', 'n_cells', 'disease_state')]
REF = pd.read_csv(os.path.join(RES, 's12_subsystem_tests_pooled.csv')).set_index('subsystem')
BETA = ['Beta-1', 'Beta-2', 'Beta-3']


def pooled_test(df, label):
    per_donor = df.groupby('donor')[SUBS].mean()
    lab = df.groupby('donor').disease_state.first().loc[per_donor.index]
    c, t = per_donor[lab == 'Control'], per_donor[lab == 'T2D']
    rows = []
    for s in SUBS:
        u, p = mannwhitneyu(t[s], c[s], alternative='two-sided')
        rows.append(dict(variant=label, subsystem=s, n_ctrl=len(c), n_t2d=len(t),
                         n_profiles=len(df), delta=t[s].median() - c[s].median(), p=p))
    out = pd.DataFrame(rows)
    out['padj'] = multipletests(out.p, method='fdr_bh')[1]
    return out

variants = [('all profiles (>=20 cells)', REL)]
for k in (50, 100, 200):
    variants.append(('>=%d cells per profile' % k, REL[REL.n_cells >= k]))
variants.append(('beta populations only', REL[REL.population.isin(BETA)]))

ALL = pd.concat([pooled_test(df, lab) for lab, df in variants], ignore_index=True)
ALL.to_csv(os.path.join(RES, 's29_activity_sensitivity.csv'), index=False)

base = ALL[ALL.variant == variants[0][0]].set_index('subsystem')
log('variant                        profiles  donors(C/T)  min padj  n padj<0.05  sign-agreement with primary (98 subsystems)')
summary = []
for lab, _ in variants:
    v = ALL[ALL.variant == lab].set_index('subsystem')
    agree = (np.sign(v.delta) == np.sign(base.delta)).mean()
    log('%-30s %8d  %3d/%-3d      %.3f    %d           %.1f%%'
        % (lab, v.n_profiles.iloc[0], v.n_ctrl.iloc[0], v.n_t2d.iloc[0], v.padj.min(),
           (v.padj < 0.05).sum(), 100 * agree))
    summary.append(dict(variant=lab, n_profiles=v.n_profiles.iloc[0], n_ctrl=v.n_ctrl.iloc[0],
                        n_t2d=v.n_t2d.iloc[0], min_padj=v.padj.min(), n_sig=(v.padj < 0.05).sum(),
                        sign_agreement=agree))
SUM = pd.DataFrame(summary)

# the 18 concordant subsystems of Table S4 / Figure 4A, in Table S4 order
T4 = ['Biopterin metabolism', 'Beta oxidation of even-chain fatty acids (mitochondrial)',
      'Beta oxidation of unsaturated fatty acids (n-9) (mitochondrial)',
      'Beta oxidation of unsaturated fatty acids (n-7) (mitochondrial)',
      'Beta oxidation of di-unsaturated fatty acids (n-6) (mitochondrial)',
      'Beta oxidation of odd-chain fatty acids (mitochondrial)', 'Porphyrin metabolism',
      'Folate metabolism', 'Glycine, serine and threonine metabolism', 'Pentose phosphate pathway',
      'Pyruvate metabolism', 'Leukotriene metabolism', 'Alanine, aspartate and glutamate metabolism',
      'Bile acid biosynthesis', 'Fatty acid oxidation', 'Glycerolipid metabolism',
      'Heparan sulfate degradation', 'Protein degradation']
T4 = [s for s in T4 if s in SUBS]
short = {s: s.replace(' (mitochondrial)', '').replace('Beta oxidation of ', 'β-ox. ')
         .replace(' fatty acids', ' FA') for s in T4}

M = np.array([[ALL[(ALL.variant == lab) & (ALL.subsystem == s)].delta.iloc[0] for lab, _ in variants]
              for s in T4])
P = np.array([[ALL[(ALL.variant == lab) & (ALL.subsystem == s)].padj.iloc[0] for lab, _ in variants]
              for s in T4])

fig, axes = plt.subplots(1, 2, figsize=(7.09, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.08),
                         constrained_layout=False)
fig.subplots_adjust(left=0.36, right=0.98, top=0.86, bottom=0.24)
v = np.nanmax(np.abs(M))
ax = axes[0]
im = ax.imshow(M, cmap=F.DIV, vmin=-v, vmax=v, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, '%+.2f' % M[i, j], ha='center', va='center', fontsize=5.2,
                color=F.INK if abs(M[i, j]) < v * 0.6 else 'white')
ax.set_yticks(range(len(T4))); ax.set_yticklabels([short[s] for s in T4], fontsize=6)
ax.set_xticks(range(len(variants)))
XT = ['all profiles\n(≥20 cells)', '≥50 cells\nper profile', '≥100 cells\nper profile',
      '≥200 cells\nper profile', 'beta\npopulations\nonly']
ax.set_xticklabels(XT, fontsize=5.4, rotation=0)
ax.set_title('pooled donor-level Δ (T2D − control)', fontsize=7, pad=4)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
F.panel(ax, 'A', dx=-0.95, dy=1.06)

ax = axes[1]
im2 = ax.imshow(P, cmap=F.SEQ.reversed(), vmin=0, vmax=1, aspect='auto')
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        ax.text(j, i, '%.2f' % P[i, j], ha='center', va='center', fontsize=5.2,
                color=F.INK if P[i, j] > 0.35 else 'white')
ax.set_yticks([]); ax.set_xticks(range(len(variants)))
ax.set_xticklabels(XT, fontsize=5.4)
ax.set_title('Benjamini–Hochberg adjusted p (98 subsystems)', fontsize=7, pad=4)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
F.panel(ax, 'B', dx=-0.08, dy=1.06)
fig.text(0.36, 0.955, 'Sensitivity of the pooled subsystem test to profile-size threshold and lineage restriction '
         '(18 concordant subsystems of Table S4)', fontsize=7, color=F.MUTED)
fig.text(0.36, 0.01, '\n'.join('%s: %d profiles, %d control / %d T2D donors, smallest adjusted p = %.2f, %d subsystems below 0.05'
                               % (r.variant.replace('>=', '≥'), r.n_profiles, r.n_ctrl, r.n_t2d, r.min_padj, r.n_sig)
                               for r in SUM.itertuples()),
         fontsize=5.0, color=F.MUTED, va='bottom')
out = os.path.join(FIG, 'supplementary', 'FigureS7_activity_sensitivity.png')
fig.savefig(out, dpi=400, facecolor='white')
log('wrote %s' % out)
