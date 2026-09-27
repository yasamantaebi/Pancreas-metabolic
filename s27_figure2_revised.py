# -*- coding: utf-8 -*-
"""Figure 2, revised after review.

Panels A-D unchanged from s11_figure2.py. New panels:
  E  original embedding (61,859 cells) coloured by population, HPAP090 cells
     over-plotted in red
  F  independent re-embedding of the 57,835 cells remaining after HPAP090
     exclusion (s24), coloured by the original population label
  G  cells per population before and after HPAP090 exclusion
Also writes 02_results/s27_population_counts_HPAP090_exclusion.csv.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, log
from figstyle import (INK, MUTED, HUE, DISEASE, shades, panel, clean, bare,
                      legend_swatches, label_pos)

EXCL = 'HPAP090'
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
new = pd.read_csv(os.path.join(RES, 's24_obs_embedding_noHPAP090.csv'), index_col=0)
assert (new.donor_id != EXCL).all() and len(new) == (obs.donor_id != EXCL).sum()

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

# ── donor-composition statistics (as s11) ────────────────────────────────────
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

# ── before / after HPAP090 exclusion ─────────────────────────────────────────
x = obs[obs.donor_id != EXCL]
CT = pd.DataFrame({'cells_before': obs.identity.value_counts(),
                   'cells_after': x.identity.value_counts()}).reindex(ORDER)
CT['cells_removed'] = CT.cells_before - CT.cells_after
CT['pct_removed'] = 100 * CT.cells_removed / CT.cells_before
CT['ctrl_after'] = x[x.disease_state == 'Control'].identity.value_counts().reindex(ORDER).fillna(0).astype(int)
CT['t2d_after'] = x[x.disease_state == 'T2D'].identity.value_counts().reindex(ORDER).fillna(0).astype(int)
CT['pct_T2D_before'] = ST.pct_T2D
CT['pct_T2D_after'] = 100 * CT.t2d_after / CT.cells_after
CT['donors_before'] = obs.groupby('identity').donor_id.nunique().reindex(ORDER)
CT['donors_after'] = x.groupby('identity').donor_id.nunique().reindex(ORDER)
CT.to_csv(os.path.join(RES, 's27_population_counts_HPAP090_exclusion.csv'))

def totals(df):
    d = df.drop_duplicates('donor_id')
    return dict(cells=len(df), ctrl=int((df.disease_state == 'Control').sum()),
                t2d=int((df.disease_state == 'T2D').sum()), donors=df.donor_id.nunique(),
                d_ctrl=int((d.disease_state == 'Control').sum()),
                d_t2d=int((d.disease_state == 'T2D').sum()),
                pct=100 * (df.disease_state == 'T2D').mean())
TB, TA = totals(obs), totals(x)
log('before: %s' % TB); log('after : %s' % TA)

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.09, 11.4))
gs = GridSpec(5, 2, figure=fig, height_ratios=[0.78, 0.78, 0.95, 1.10, 0.95],
              hspace=0.62, wspace=0.28, left=0.115, right=0.885, top=0.972, bottom=0.06)
xs = np.arange(len(ORDER))
flag = ST.top_frac.values > 50

# A
ax = fig.add_subplot(gs[0, :])
ax.bar(xs, ST.pct_T2D.values, color=[RED if f else COL[c] for f, c in zip(flag, ORDER)],
       linewidth=0, width=0.74)
ax.axhline(base, color=INK, ls=(0, (3, 2)), lw=0.9)
ax.text(len(ORDER) - 0.4, base + 2.5, 'overall retained-cell fraction (%.1f%%)' % base,
        ha='right', fontsize=6, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([]); ax.set_ylabel('% T2D cells')
ax.set_ylim(0, 112)
clean(ax); panel(ax, 'A', dx=-0.085, dy=1.03)

# B
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

# C, D
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
            transform=ax.transAxes, ha='right', fontsize=6.4, color=RED, fontweight='bold')
    ax.text(0.96, 0.70, '%d of %d T2D donors\ncontribute ≥1%%'
            % (ST.loc[cid, 'n_t2d_donors_1pct'], n_t2d_donors),
            transform=ax.transAxes, ha='right', fontsize=5.8, color=MUTED)
    clean(ax); panel(ax, 'CD'[k], dx=-0.24, dy=1.14)

# E: original embedding, HPAP090 highlighted
rng = np.random.default_rng(0)
def umap_panel(ax, df, title, letter, highlight=None, label_a23=True):
    order = rng.permutation(len(df))
    d = df.iloc[order]
    ax.scatter(d.UMAP1, d.UMAP2, c=[COL[i] for i in d.identity], s=0.35, lw=0,
               alpha=0.5, rasterized=True)
    if highlight is not None:
        h = df[highlight]
        ax.scatter(h.UMAP1, h.UMAP2, c=RED, s=0.9, lw=0, alpha=0.9, rasterized=True)
    for cid in (['Alpha-2', 'Alpha-3'] if label_a23 else []):
        m = (df.identity == cid).values
        if m.sum() >= 20:
            px, py = label_pos(df.UMAP1.values, df.UMAP2.values, m)
            ax.annotate(cid, (px, py), fontsize=6, color=RED if highlight is not None else INK,
                        fontweight='bold', ha='center', va='center',
                        xytext=((34, -4) if cid == 'Alpha-2' else (0, 14)), textcoords='offset points',
                        arrowprops=dict(arrowstyle='-', lw=0.5, color=INK))
    bare(ax)
    ax.set_xlabel('UMAP 1', fontsize=6.5, color=MUTED, labelpad=1)
    ax.set_ylabel('UMAP 2', fontsize=6.5, color=MUTED, labelpad=1)
    ax.set_title(title, fontsize=7, pad=3)
    panel(ax, letter, dx=-0.10, dy=1.06)

ax = fig.add_subplot(gs[3, 0])
umap_panel(ax, obs, 'original embedding: %s cells, %d donors' % (format(TB['cells'], ','), TB['donors']),
           'E', highlight=(obs.donor_id == EXCL).values)
ax.text(0.02, 0.02, '%s cells (%s) in red' % (format((obs.donor_id == EXCL).sum(), ','), EXCL),
        transform=ax.transAxes, fontsize=5.8, color=RED)

ax = fig.add_subplot(gs[3, 1])
umap_panel(ax, new, 're-embedded without %s: %s cells, %d donors'
           % (EXCL, format(TA['cells'], ','), TA['donors']), 'F', label_a23=False)
surv = new[new.identity.isin(['Alpha-2', 'Alpha-3'])]
ax.scatter(surv.UMAP1, surv.UMAP2, s=7, facecolors='none', edgecolors=RED,
           linewidths=0.45, zorder=5)
ax.text(0.02, 0.02, 'red circles: the %d remaining Alpha-2 (n = %d) and Alpha-3 (n = %d)\ncells, now dispersed among the alpha populations'
        % (len(surv), CT.loc['Alpha-2', 'cells_after'], CT.loc['Alpha-3', 'cells_after']),
        transform=ax.transAxes, fontsize=5.4, color=INK)
legend_swatches(ax, ORDER, [COL[c] for c in ORDER], ncol=1, loc='center left',
                bbox=(1.01, 0.5), ms=3.6)
ax.get_legend().prop.set_size(5.4) if False else None
for t in ax.get_legend().get_texts():
    t.set_fontsize(5.4)

# G: before / after counts
ax = fig.add_subplot(gs[4, :])
w = 0.38
ax.bar(xs - w / 2, CT.cells_before, width=w, color='#c9c9c9', linewidth=0, label='before exclusion')
ax.bar(xs + w / 2, CT.cells_after, width=w,
       color=[RED if f else COL[c] for f, c in zip(flag, ORDER)], linewidth=0,
       label='after excluding %s' % EXCL)
ax.set_yscale('log')
ax.set_ylim(10, 60000)
for i, cid in enumerate(ORDER):
    ax.text(i + w / 2, CT.loc[cid, 'cells_after'] * 1.18, format(int(CT.loc[cid, 'cells_after']), ','),
            ha='center', fontsize=4.9, color=INK, rotation=90, va='bottom')
    if flag[i]:
        ax.text(i - w / 2, CT.loc[cid, 'cells_before'] * 1.18, format(int(CT.loc[cid, 'cells_before']), ','),
                ha='center', fontsize=4.9, color=MUTED, rotation=90, va='bottom')
        ax.annotate('−%.1f%%' % CT.loc[cid, 'pct_removed'], (i, CT.loc[cid, 'cells_before'] * 4.0), ha='center', va='bottom',
                    fontsize=5.6, color=RED, fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(ORDER, rotation=60, ha='right')
ax.set_ylabel('cells (log scale)')
ax.legend(frameon=False, fontsize=6, loc='upper right', ncol=2, handlelength=1.0)
ax.text(0.5, 1.02,
        'all populations: %s → %s cells  |  control %s → %s, T2D %s → %s  |  '
        'donors %d → %d (control %d → %d, T2D %d → %d)  |  T2D fraction %.1f%% → %.1f%%'
        % (format(TB['cells'], ','), format(TA['cells'], ','), format(TB['ctrl'], ','), format(TA['ctrl'], ','),
           format(TB['t2d'], ','), format(TA['t2d'], ','), TB['donors'], TA['donors'],
           TB['d_ctrl'], TA['d_ctrl'], TB['d_t2d'], TA['d_t2d'], TB['pct'], TA['pct']),
        transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=INK)
clean(ax); panel(ax, 'G', dx=-0.085, dy=1.10)

out = os.path.join(FIG, 'Figure2_donor_composition.png')
fig.savefig(out, dpi=400, facecolor='white')
plt.close(fig)
log('Figure 2 (revised) written: %s' % out)
print(CT[['cells_before', 'cells_after', 'cells_removed', 'pct_removed', 'ctrl_after',
          't2d_after', 'pct_T2D_before', 'pct_T2D_after', 'donors_before', 'donors_after']]
      .round(1).to_string())
