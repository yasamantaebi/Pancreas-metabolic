# -*- coding: utf-8 -*-
"""Rebuild Figure 4 around donor-level metabolic statistics."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, log
from figstyle import INK, MUTED, DIV, DISEASE, panel, clean, heat, cbar

PO = pd.read_csv(os.path.join(RES, 's12_subsystem_tests_pooled.csv'))
GT = pd.read_csv(os.path.join(RES, 's12_gene_tests_pooled.csv'))
REL = pd.read_csv(os.path.join(RES, 's12_per_donor_subsystem_scores.csv'), index_col=0)
GR = pd.read_csv(os.path.join(RES, 's12_per_donor_gene_scores.csv'), index_col=0)
CONS = pd.read_csv(os.path.join(RES, 's06_subsystem_scan.csv'))
POPS = ['Beta-1', 'Beta-2', 'Beta-3', 'Alpha-4', 'Alpha-6', 'Delta']

SHORT = {'Beta oxidation of di-unsaturated fatty acids (n-6) (mitochondrial)': 'β-oxidation, di-unsat FA (n-6)',
         'Beta oxidation of unsaturated fatty acids (n-7) (mitochondrial)': 'β-oxidation, unsat FA (n-7)',
         'Beta oxidation of unsaturated fatty acids (n-9) (mitochondrial)': 'β-oxidation, unsat FA (n-9)',
         'Beta oxidation of odd-chain fatty acids (mitochondrial)': 'β-oxidation, odd-chain FA',
         'Beta oxidation of even-chain fatty acids (mitochondrial)': 'β-oxidation, even-chain FA',
         'Beta oxidation of poly-unsaturated fatty acids (mitochondrial)': 'β-oxidation, poly-unsat FA',
         'Valine, leucine, and isoleucine metabolism': 'Branched-chain amino acid metab.'}

cons = CONS[CONS.all_consistent].sort_values('mean')
tbl = cons.merge(PO[['subsystem', 'delta', 'effect_r', 'p', 'padj']], on='subsystem',
                 how='left')
tbl['label'] = [SHORT.get(s, s) + '  (%d)' % n for s, n in zip(tbl.subsystem, tbl.n_rxns)]

fig = plt.figure(figsize=(7.09, 8.6))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.30, 0.72, 0.78], hspace=0.42,
              wspace=0.30, left=0.315, right=0.90, top=0.955, bottom=0.075)

# A: effect sizes across populations, with donor-level statistics alongside
ax = fig.add_subplot(gs[0, :])
im = heat(ax, tbl.set_index('label')[POPS], vmax=0.62, fontsize=5.3)
ax.axhline(len(tbl) - 3.5, color=INK, lw=1.0)
panel(ax, 'A', dx=-0.455, dy=1.02)
cbar(fig, im, ax, 'Δ reaction activity score\n(T2D − Control, vs global level)',
     shrink=0.58, fraction=0.026, pad=0.13)
ax.set_ylabel('Human-GEM subsystem (n reactions)', fontsize=6.8, labelpad=2)
for i, (_, r) in enumerate(tbl.iterrows()):
    ax.text(len(POPS) - 0.30, i, '%.3f' % r.p, transform=ax.transData, ha='left',
            va='center', fontsize=5.2,
            color=INK if r.p < 0.05 else MUTED)
ax.text(len(POPS) - 0.30, -1.05, 'donor-level\n$p$', ha='left', va='bottom',
        fontsize=5.6, color=INK, fontweight='bold')
ax.text(0.5, 1.10, 'no subsystem reaches FDR 5% at donor level (all $p_{adj}$ > 0.21)',
        transform=ax.transAxes, ha='center', fontsize=6.2, color='#b3200f',
        fontweight='bold')

# B: per-donor distributions for the pterin/PPP subsystems
ax = fig.add_subplot(gs[1, :])
SHOW = ['Biopterin metabolism', 'Folate metabolism', 'Pentose phosphate pathway',
        'Beta oxidation of even-chain fatty acids (mitochondrial)',
        'Oxidative phosphorylation']
SHOW = [s for s in SHOW if s in REL.columns]
pool = REL.groupby(['donor', 'disease_state'])[SHOW].mean().reset_index()
rng = np.random.default_rng(0)
for i, s in enumerate(SHOW):
    for j, ds in enumerate(['Control', 'T2D']):
        v = pool.loc[pool.disease_state == ds, s].values
        pos = i + (j - 0.5) * 0.36
        ax.boxplot([v], positions=[pos], widths=0.30, patch_artist=True,
                   showfliers=False, medianprops=dict(color=INK, lw=1.0),
                   whiskerprops=dict(color='#9a9a9a', lw=0.6),
                   capprops=dict(color='#9a9a9a', lw=0.6),
                   boxprops=dict(facecolor=DISEASE[ds], alpha=0.30, edgecolor='none'))
        ax.scatter(pos + (rng.random(len(v)) - .5) * .17, v, s=5.5,
                   color=DISEASE[ds], linewidths=0.2, edgecolors='white', zorder=4)
    row = PO[PO.subsystem == s]
    if len(row):
        ax.text(i, 1.03, '$p$ = %.3f' % row.p.iloc[0], transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5.8, color=MUTED)
ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(len(SHOW)))
ax.set_xticklabels([SHORT.get(s, s) for s in SHOW], rotation=18, ha='right', fontsize=5.8)
ax.set_ylabel('per-donor subsystem score\n(vs global level)', fontsize=6.6)
ax.set_xlim(-0.55, len(SHOW) - 0.45)
clean(ax); panel(ax, 'B', dx=-0.455, dy=1.10)

# C: gene-level effect sizes with donor-level p
ax = fig.add_subplot(gs[2, 0])
G = GT.sort_values('delta')
ys = np.arange(len(G))
ax.barh(ys, G.delta.values, color=[DIV(0.5 + v / 1.6) for v in G.delta.values],
        linewidth=0, height=0.66)
ax.set_yticks(ys); ax.set_yticklabels(G.gene.values, style='italic')
ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel('Δ per-donor gene score\n(T2D − Control)', fontsize=6.6)
for i, (_, r) in enumerate(G.iterrows()):
    ax.text(r.delta - 0.03, i, '%.3f' % r.p, va='center', ha='right', fontsize=5.2,
            color=INK if r.p < 0.05 else MUTED)
ax.set_xlim(-0.95, 0.22)
clean(ax); panel(ax, 'C', dx=-0.46, dy=1.05)

ax = fig.add_subplot(gs[2, 1])
KEY = ['PCBD1', 'TPK1', 'SPR', 'G6PD']
KEY = [k for k in KEY if k in GR.columns]
gp = GR.groupby(['donor', 'disease_state'])[KEY].mean().reset_index()
for i, g in enumerate(KEY):
    for j, ds in enumerate(['Control', 'T2D']):
        v = gp.loc[gp.disease_state == ds, g].values
        pos = i + (j - 0.5) * 0.36
        ax.boxplot([v], positions=[pos], widths=0.30, patch_artist=True,
                   showfliers=False, medianprops=dict(color=INK, lw=1.0),
                   whiskerprops=dict(color='#9a9a9a', lw=0.6),
                   capprops=dict(color='#9a9a9a', lw=0.6),
                   boxprops=dict(facecolor=DISEASE[ds], alpha=0.30, edgecolor='none'))
        ax.scatter(pos + (rng.random(len(v)) - .5) * .17, v, s=5.5, color=DISEASE[ds],
                   linewidths=0.2, edgecolors='white', zorder=4)
    r = GT[GT.gene == g]
    ax.text(i, 1.03, '%.3f' % r.p.iloc[0], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=5.8, color=MUTED)
ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(len(KEY)))
ax.set_xticklabels(KEY, style='italic', fontsize=6.4)
ax.set_ylabel('per-donor gene score', fontsize=6.6)
ax.set_xlim(-0.55, len(KEY) - 0.45)
clean(ax); panel(ax, 'D', dx=-0.30, dy=1.05)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='none', ms=4,
                          markerfacecolor=DISEASE[d], markeredgecolor='none', label=d)
                   for d in ['Control', 'T2D']],
          loc='lower right', frameon=False, fontsize=6)

fig.savefig(os.path.join(FIG, 'Figure4_metabolic.png'))
plt.close(fig)
log('Figure 4 rebuilt with donor-level statistics')
