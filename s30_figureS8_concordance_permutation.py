# -*- coding: utf-8 -*-
"""Supplementary Figure S8: donor-label permutation null for the multi-population
directional concordance and for the rank of biopterin metabolism (s14)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, log
import figstyle as F

null_n = np.load(os.path.join(RES, 's14_null_nconc.npy'))
null_r = np.load(os.path.join(RES, 's14_null_biopterin_rank.npy'))
res = pd.read_csv(os.path.join(RES, 's14_permutation_results.csv')).set_index('metric')
obs_n, obs_r = int(res.loc['n_concordant', 'observed']), int(res.loc['biopterin_rank', 'observed'])
p_n, p_r = res.loc['n_concordant', 'perm_p'], res.loc['biopterin_rank', 'perm_p']
N = len(null_n)

fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.9))
fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.26, wspace=0.32)

ax = axes[0]
bins = np.arange(-0.5, max(null_n.max(), obs_n) + 1.5, 1)
ax.hist(null_n, bins=bins, color=F.FAINT, edgecolor=F.INK, linewidth=0.4)
ax.axvline(obs_n, color=F.HUE['beta'], lw=1.6)
ax.text(obs_n - 0.6, ax.get_ylim()[1] * 0.92, 'observed %d' % obs_n, color=F.HUE['beta'],
        fontsize=6.4, ha='right', va='top')
ax.text(0.98, 0.70, 'null median %d\n95th percentile %d\npermutation p = %.3f'
        % (int(np.median(null_n)), int(np.percentile(null_n, 95)), p_n),
        transform=ax.transAxes, ha='right', va='top', fontsize=6.0, color=F.INK)
ax.set_xlabel('subsystems concordant across all six populations')
ax.set_ylabel('permutations (of %d)' % N)
ax.set_title('directional concordance', fontsize=7.5, pad=3)
F.clean(ax); F.panel(ax, 'A')

ax = axes[1]
bins = np.arange(0.5, 99.5, 2)
ax.hist(null_r, bins=bins, color=F.FAINT, edgecolor=F.INK, linewidth=0.4)
ax.axvline(obs_r, color=F.HUE['beta'], lw=1.6)
ax.text(obs_r + 1.5, ax.get_ylim()[1] * 0.92, 'observed rank %d' % obs_r, color=F.HUE['beta'],
        fontsize=6.4, ha='left', va='top')
ax.text(0.98, 0.70, 'null median rank %d\n5th percentile %d\npermutation p = %.3f'
        % (int(np.median(null_r)), int(np.percentile(null_r, 5)), p_r),
        transform=ax.transAxes, ha='right', va='top', fontsize=6.0, color=F.INK)
ax.set_xlabel('biopterin metabolism rank by mean effect' + chr(10) + '(1 = most reduced)')  # two lines: was clipped at the right edge
ax.set_ylabel('permutations (of %d)' % N)
ax.set_title('rank of biopterin metabolism among 98 subsystems', fontsize=7.5, pad=3)
F.clean(ax); F.panel(ax, 'B')

out = os.path.join(FIG, 'supplementary', 'FigureS8_concordance_permutation.png')
fig.savefig(out, dpi=400, facecolor='white')
log('wrote %s | obs_n=%d p=%.4f | obs_rank=%d p=%.4f' % (out, obs_n, p_n, obs_r, p_r))
