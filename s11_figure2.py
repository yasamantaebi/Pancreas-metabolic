# -*- coding: utf-8 -*-
"""Rebuild Figure 2 as a self-contained analysis of donor composition."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, log
from figstyle import INK, MUTED, HUE, DISEASE, shades, panel, clean, legend_swatches

obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
summ = pd.read_csv(os.path.join(RES, 's03_identity_summary.csv'), index_col=0)
BETA = ['Beta-%d' % i for i in range(1, 7)]
ALPHA = ['Alpha-%d' % i for i in range(1, 9)]
ORDER = BETA + ALPHA + ['Delta', 'PP', 'Acinar']
COL = {}
for k, c in zip(BETA, shades(HUE['beta'], len(BETA), .25, .85)):
    COL[k] = c
for k, c in zip(ALPHA, shades(HUE['alpha'], len(ALPHA), .20, .88)):
    COL[k] = c
COL['Delta'] = HUE['delta']; COL['PP'] = HUE['pp']; COL['Acinar'] = HUE['mixed']
RED = '#b3200f'

base = 100 * (obs.disease_state == 'T2D').mean()
n_t2d_donors = obs.loc[obs.disease_state == 'T2D', 'donor_id'].nunique()

stats = []
for cid in ORDER:
    s = obs[obs.identity == cid]
    vc = s.groupby('donor_id').size().sort_values(ascending=False)
    frac = vc / len(s)
    t2d_don = s[s.disease_state == 'T2D'].groupby('donor_id').size()
    stats.append(dict(identity=cid, n=len(s), pct_T2D=100 * (s.disease_state == 'T2D').mean(),
                      top_donor=vc.index[0], top_frac=100 * frac.iloc[0],
                      n_donors_1pct=int((frac >= 0.01).sum()),
                      n_t2d_donors_1pct=int((t2d_don / len(s) >= 0.01).sum())))
ST = pd.DataFrame(stats).set_index('identity')
ST.to_csv(os.path.join(RES, 's11_donor_composition.csv'))
log('single-donor populations (>50%% from one donor):\n%s'
    % ST[ST.top_frac > 50][['n', 'pct_T2D', 'top_donor', 'top_frac']].round(1).to_string())

fig = plt.figure(figsize=(7.09, 6.6))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15], hspace=0.70,
              wspace=0.28, left=0.115, right=0.885, top=0.955, bottom=0.115)
xs = np.arange(len(ORDER))
flag = ST.top_frac.values > 50

ax = fig.add_subplot(gs[0, :])
ax.bar(xs, ST.pct_T2D.values, color=[RED if f else COL[c] for f, c in zip(flag, ORDER)],
       linewidth=0, width=0.74)
ax.axhline(base, color=INK, ls=(0, (3, 2)), lw=0.9)
ax.text(len(ORDER) - 0.4, base + 2.5, 'overall retained-cell fraction (%.1f%%)' % base,
        ha='right', fontsize=6, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([]); ax.set_ylabel('% T2D cells')
ax.set_ylim(0, 112)
clean(ax); panel(ax, 'A', dx=-0.085, dy=1.03)

ax = fig.add_subplot(gs[1, :])
ax.bar(xs, ST.top_frac.values, color=[RED if f else '#b9b9b9' for f in flag],
       linewidth=0, width=0.74)
ax.axhline(50, color=RED, ls=(0, (3, 2)), lw=0.9)
for i, f in enumerate(flag):
    if f:
        ax.text(i, ST.top_frac.values[i] + 2.5, ST.top_donor.values[i], ha='center',
                fontsize=5.4, color=RED, fontweight='bold', rotation=90)
ax.set_xticks(xs); ax.set_xticklabels(ORDER, rotation=60, ha='right')
ax.set_ylabel('% of population from\nits single largest donor')
ax.set_ylim(0, 118)
clean(ax); panel(ax, 'B', dx=-0.085, dy=1.03)

hi = ST[ST.pct_T2D > 80].index.tolist()
for k, cid in enumerate(hi[:2]):
    ax = fig.add_subplot(gs[2, k])
    s = obs[obs.identity == cid]
    vc = s.groupby('donor_id').size().sort_values(ascending=False).head(8)
    ax.bar(range(len(vc)), vc.values,
           color=[RED if i == 0 else DISEASE['T2D'] for i in range(len(vc))],
           linewidth=0, width=0.72)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=60, ha='right', fontsize=5.4)
    ax.set_ylabel('cells contributed')
    ax.set_title('%s  (n = %d, %.0f%% T2D)' % (cid, len(s), ST.loc[cid, 'pct_T2D']),
                 fontsize=7, pad=3)
    ax.text(0.96, 0.86, '%s = %.1f%%' % (ST.loc[cid, 'top_donor'], ST.loc[cid, 'top_frac']),
            transform=ax.transAxes, ha='right', fontsize=6.4, color=RED,
            fontweight='bold')
    ax.text(0.96, 0.70, '%d of %d T2D donors\ncontribute ≥1%%'
            % (ST.loc[cid, 'n_t2d_donors_1pct'], n_t2d_donors),
            transform=ax.transAxes, ha='right', fontsize=5.8, color=MUTED)
    clean(ax); panel(ax, 'CD'[k], dx=-0.24, dy=1.14)

fig.savefig(os.path.join(FIG, 'Figure2_donor_composition.png'))
plt.close(fig)
log('Figure 2 rebuilt')
