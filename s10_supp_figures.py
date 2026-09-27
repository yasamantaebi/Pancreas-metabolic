# -*- coding: utf-8 -*-
"""Stage 10 - supplementary figures."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, PART1, log
from figstyle import (INK, MUTED, FAINT, SURFACE, SEQ, DIV, HUE, DISEASE, shades,
                      panel, clean, bare, heat, cbar, scatter_umap,
                      legend_swatches, label_pos)

SD = os.path.join(FIG, 'supplementary')
os.makedirs(SD, exist_ok=True)
W = 7.09

obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
full = pd.read_csv(os.path.join(PART1, '04_results', 'reviewer_diagnostics',
                                'full_obs_with_umap.csv'), low_memory=False)
BETA = ['Beta-%d' % i for i in range(1, 7)]
ALPHA = ['Alpha-%d' % i for i in range(1, 9)]
ORDER = BETA + ALPHA + ['Delta', 'PP', 'Acinar']
COL = {}
for k, c in zip(BETA, shades(HUE['beta'], len(BETA), .25, .85)):
    COL[k] = c
for k, c in zip(ALPHA, shades(HUE['alpha'], len(ALPHA), .20, .88)):
    COL[k] = c
COL['Delta'] = HUE['delta']; COL['PP'] = HUE['pp']; COL['Acinar'] = HUE['mixed']
X, Y = obs.UMAP1.values, obs.UMAP2.values
rng = np.random.default_rng(0)
sh = rng.permutation(len(obs))

# ── S1: quality control ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(W, 6.4))
gs = GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.30, left=0.11, right=0.90,
              top=0.95, bottom=0.09)

ax = fig.add_subplot(gs[0, :])
g = np.log10(full.n_genes_by_counts.values)
ax.hist(g, bins=80, color='#9ab8d4', edgecolor='none')
ax.axvline(np.log10(1000), color='#b3200f', lw=1.4)
ax.text(np.log10(1000), ax.get_ylim()[1] * 0.92, '  threshold: 1,000 genes',
        color='#b3200f', fontsize=6.6, fontweight='bold')
ax.set_xticks([np.log10(v) for v in (300, 500, 1000, 2000, 4000, 8000)])
ax.set_xticklabels(['300', '500', '1,000', '2,000', '4,000', '8,000'])
ax.set_xlabel('genes detected per cell (log scale)')
ax.set_ylabel('cells')
ax.text(0.02, 0.9, 'ambient-dominated mode', transform=ax.transAxes, fontsize=6.2,
        color=MUTED)
ax.text(0.55, 0.9, 'main mode', transform=ax.transAxes, fontsize=6.2, color=MUTED)
clean(ax); panel(ax, 'A', dx=-0.075, dy=1.02)

don = pd.read_csv(os.path.join(RES, 's09_per_donor_qc.csv'), index_col=0)
don = don.sort_values(['disease_state', 'median_genes'])
for k, (col, lab) in enumerate([('median_genes', 'median genes per cell'),
                                ('retention_pct', '% cells retained')]):
    ax = fig.add_subplot(gs[1, k])
    cols = [DISEASE[d] for d in don.disease_state]
    ax.bar(range(len(don)), don[col].values, color=cols, linewidth=0, width=0.8)
    ax.set_xticks([]); ax.set_xlabel('donors (n = %d)' % len(don))
    ax.set_ylabel(lab)
    clean(ax); panel(ax, 'BC'[k], dx=-0.22, dy=1.04)
    if k == 1:
        legend_swatches(ax, ['Control', 'T2D'],
                        [DISEASE['Control'], DISEASE['T2D']], bbox=(1.0, 0.5), ms=4)

qc = pd.read_csv(os.path.join(RES, 's09_qc_by_population.csv'), index_col=0).reindex(ORDER)
for k, (col, lab) in enumerate([('median_genes', 'median genes per cell'),
                                ('median_mito', 'median mitochondrial %')]):
    ax = fig.add_subplot(gs[2, k])
    ax.bar(range(len(ORDER)), qc[col].values, color=[COL[c] for c in ORDER],
           linewidth=0, width=0.74)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=60, ha='right', fontsize=5.2)
    ax.set_ylabel(lab)
    clean(ax); panel(ax, 'DE'[k], dx=-0.22, dy=1.04)
fig.savefig(os.path.join(SD, 'FigureS1_quality_control.png'))
plt.close(fig); log('S1 done')

# ── S2: doublet scores ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(W, 4.6))
gs = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.28, left=0.11, right=0.90,
              top=0.94, bottom=0.16)
ax = fig.add_subplot(gs[0, :])
data = [obs.loc[obs.identity == p, 'doublet_score'].values for p in ORDER]
vp = ax.violinplot(data, positions=range(len(ORDER)), widths=0.78,
                   showextrema=False)
for b, p in zip(vp['bodies'], ORDER):
    b.set_facecolor(COL[p]); b.set_alpha(0.85); b.set_edgecolor('white'); b.set_linewidth(0.5)
for i, d in enumerate(data):
    ax.plot([i - .2, i + .2], [np.median(d)] * 2, color=INK, lw=1.0, zorder=5)
ax.axhline(0.25, color='#b3200f', ls=(0, (3, 2)), lw=0.9)
ax.text(len(ORDER) - .4, 0.26, 'Scrublet call threshold (0.25)', ha='right',
        fontsize=5.8, color='#b3200f')
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=60, ha='right', fontsize=5.4)
ax.set_ylabel('Scrublet doublet score')
clean(ax); panel(ax, 'A', dx=-0.075, dy=1.03)

ax = fig.add_subplot(gs[1, 0])
sc = ax.scatter(X[sh], Y[sh], c=obs.doublet_score.values[sh], cmap=SEQ,
                norm=Normalize(0, 0.3), s=0.6, linewidths=0, alpha=0.7,
                rasterized=True)
bare(ax); panel(ax, 'B', dx=-0.10, dy=1.02)
cbar(fig, sc, ax, 'doublet score', shrink=0.75, fraction=0.04)

ax = fig.add_subplot(gs[1, 1])
dd = obs.groupby(['donor_id', 'disease_state']).doublet_score.mean().reset_index()
for j, ds in enumerate(['Control', 'T2D']):
    v = dd.loc[dd.disease_state == ds, 'doublet_score'].values
    ax.scatter(j + (np.random.default_rng(j).random(len(v)) - .5) * .3, v, s=8,
               color=DISEASE[ds], linewidths=0.25, edgecolors='white')
    ax.plot([j - .2, j + .2], [np.median(v)] * 2, color=INK, lw=1.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'T2D'])
ax.set_ylabel('per-donor mean doublet score')
ax.set_xlim(-.6, 1.6)
clean(ax); panel(ax, 'C', dx=-0.28, dy=1.02)
fig.savefig(os.path.join(SD, 'FigureS2_doublets.png'))
plt.close(fig); log('S2 done')

# ── S3: batch structure ──────────────────────────────────────────────────────
chem = np.where(obs.assay.astype(str).str.lower().str.contains('v2'), 'v2', 'v3')
fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.12, wspace=0.10)
m2 = chem == 'v2'
axes[0].scatter(X[~m2], Y[~m2], c='#009E73', s=0.5, linewidths=0, alpha=0.4,
                rasterized=True)
axes[0].scatter(X[m2], Y[m2], c='#7B5FB0', s=1.4, linewidths=0, alpha=0.85,
                rasterized=True)
bare(axes[0]); panel(axes[0], 'A', dx=-0.02, dy=1.0)
axes[0].set_title("10x chemistry (v2 over v3)", fontsize=6.8, pad=2)
dn = obs.donor_id.astype('category')
axes[1].scatter(X[sh], Y[sh], c=dn.cat.codes.values[sh], cmap='tab20', s=0.5,
                linewidths=0, alpha=0.5, rasterized=True)
bare(axes[1]); panel(axes[1], 'B', dx=-0.02, dy=1.0)
axes[1].set_title('donor identity (%d donors)' % dn.nunique(), fontsize=6.8, pad=2)
sc = axes[2].scatter(X[sh], Y[sh], c=obs.age.values[sh], cmap=SEQ, s=0.5,
                     linewidths=0, alpha=0.6, rasterized=True)
bare(axes[2]); panel(axes[2], 'C', dx=-0.02, dy=1.0)
axes[2].set_title('donor age', fontsize=6.8, pad=2)
cbar(fig, sc, axes[2], 'age (years)', shrink=0.7, fraction=0.04)
fig.savefig(os.path.join(SD, 'FigureS3_batch.png'))
plt.close(fig); log('S3 done')

# ── S4: ambient hormone evidence ─────────────────────────────────────────────
det = pd.read_csv(os.path.join(RES, 's09_hormone_detection_rate.csv'), index_col=0).reindex(ORDER)
hf = pd.read_csv(os.path.join(RES, 's09_hormone_fraction.csv'), index_col=0, header=[0, 1]).reindex(ORDER)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.30, wspace=0.28)
xs = np.arange(len(ORDER))
axes[0].bar(xs - 0.19, det.det_INS.values, width=0.36, color=HUE['beta'],
            linewidth=0, label='INS')
axes[0].bar(xs + 0.19, det.det_GCG.values, width=0.36, color=HUE['alpha'],
            linewidth=0, label='GCG')
axes[0].set_ylim(0, 108); axes[0].set_ylabel('% of cells with non-zero counts')
axes[0].set_xticks(xs); axes[0].set_xticklabels(ORDER, rotation=60, ha='right', fontsize=5.2)
axes[0].legend(frameon=False, fontsize=6, loc='lower right')
clean(axes[0]); panel(axes[0], 'A', dx=-0.16, dy=1.03)
hp = hf['hormone_pct']
axes[1].bar(xs - 0.19, hp['Control'].values, width=0.36, color=DISEASE['Control'],
            linewidth=0, label='Control')
axes[1].bar(xs + 0.19, hp['T2D'].values, width=0.36, color=DISEASE['T2D'],
            linewidth=0, label='T2D')
axes[1].set_ylabel('hormone + exocrine\n% of library')
axes[1].set_xticks(xs); axes[1].set_xticklabels(ORDER, rotation=60, ha='right', fontsize=5.2)
axes[1].legend(frameon=False, fontsize=6)
clean(axes[1]); panel(axes[1], 'B', dx=-0.18, dy=1.03)
fig.savefig(os.path.join(SD, 'FigureS4_ambient.png'))
plt.close(fig); log('S4 done')

# ── S5: signature scores per population ──────────────────────────────────────
SIG = [c for c in obs.columns if c.startswith('sig_')]
M = obs.groupby('identity')[SIG].mean().reindex(ORDER)
M.columns = [c.replace('sig_', '') for c in M.columns]
fig, ax = plt.subplots(figsize=(W, 4.0))
fig.subplots_adjust(left=0.20, right=0.86, top=0.93, bottom=0.22)
im = heat(ax, M, vmax=float(np.nanmax(np.abs(M.values))), fontsize=5.2, fmt='{:+.1f}')
cbar(fig, im, ax, 'mean signature score', shrink=0.85, fraction=0.026, pad=0.02)
panel(ax, '', dx=0, dy=1.0)
fig.savefig(os.path.join(SD, 'FigureS5_signatures.png'))
plt.close(fig); log('S5 done')

# ── S6: curated energy pathways ──────────────────────────────────────────────
CU = pd.read_csv(os.path.join(RES, 's09_curated_pathways.csv'))
CTS = [c for c in CU.columns if c in ORDER]
fig, ax = plt.subplots(figsize=(W, 2.9))
fig.subplots_adjust(left=0.31, right=0.87, top=0.93, bottom=0.20)
M = CU.set_index(CU.pathway + '  (' + CU.n_rxns.astype(str) + ')')[CTS]
im = heat(ax, M, vmax=0.24, fontsize=5.8)
cbar(fig, im, ax, 'Δ reaction activity score\n(T2D − Control, vs global shift)',
     shrink=0.95, fraction=0.028, pad=0.025)
fig.savefig(os.path.join(SD, 'FigureS6_curated_pathways.png'))
plt.close(fig); log('S6 done')

# ── S7: per-donor detection of the top DE genes ──────────────────────────────
PD = pd.read_csv(os.path.join(RES, 's09_per_donor_detection.csv'))
mat = pd.read_csv(os.path.join(RES, 's09_per_donor_detection_matrix.csv'))
gcols = [c for c in mat.columns if c not in ('donor_id', 'disease_state')]
fig, axes = plt.subplots(2, 4, figsize=(W, 4.0))
fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.12, hspace=0.62,
                    wspace=0.36)
for ax, g in zip(axes.ravel(), gcols):
    for j, ds in enumerate(['Control', 'T2D']):
        v = mat.loc[mat.disease_state == ds, g].values
        ax.scatter(j + (np.random.default_rng(j).random(len(v)) - .5) * .3, v, s=7,
                   color=DISEASE[ds], linewidths=0.25, edgecolors='white')
        ax.plot([j - .22, j + .22], [np.median(v)] * 2, color=INK, lw=1.2, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Ctrl', 'T2D'], fontsize=6)
    ax.set_title(g, fontsize=7, style='italic', pad=2)
    ax.set_xlim(-.6, 1.6)
    if ax in axes[:, 0]:
        ax.set_ylabel('% of donor\'s beta cells\nwith non-zero counts', fontsize=6)
    clean(ax)
for ax in axes.ravel()[len(gcols):]:
    ax.set_visible(False)
fig.savefig(os.path.join(SD, 'FigureS7_per_donor_detection.png'))
plt.close(fig); log('S7 done')

# ── S8: root robustness and pseudotime technical correlations ────────────────
T = pd.read_csv(os.path.join(RES, 's05_trajectory_obs.csv'), index_col=0)
RB = pd.read_csv(os.path.join(RES, 's05_root_robustness.csv'), index_col=0)
fig = plt.figure(figsize=(W, 4.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38, left=0.10, right=0.93,
              top=0.92, bottom=0.13)
ax = fig.add_subplot(gs[0, 0])
im = heat(ax, RB, vmax=1.0, fontsize=6.4)
ax.set_title('root concordance (Spearman)', fontsize=6.8, pad=3)
panel(ax, 'A', dx=-0.30, dy=1.10)
for k, (a, b) in enumerate([('pt_rootA', 'pt_rootB'), ('pt_rootC', 'pt_rootB')]):
    ax = fig.add_subplot(gs[0, k + 1])
    ax.scatter(T[a], T[b], s=0.5, c='#9ab8d4', linewidths=0, alpha=0.4, rasterized=True)
    ax.set_xlabel(a.replace('pt_root', 'Root '))
    ax.set_ylabel(b.replace('pt_root', 'Root '))
    ax.text(0.05, 0.92, 'ρ = %.3f' % RB.loc[a[-1], b[-1]], transform=ax.transAxes,
            fontsize=6.4, color=INK)
    clean(ax); panel(ax, 'BC'[k], dx=-0.32, dy=1.06)
for k, (c, lab) in enumerate([('n_genes_by_counts', 'genes detected'),
                              ('total_counts', 'total UMI'),
                              ('doublet_score', 'doublet score')]):
    ax = fig.add_subplot(gs[1, k])
    ax.scatter(T[c], T.pt, s=0.5, c='#9ab8d4', linewidths=0, alpha=0.4, rasterized=True)
    from scipy.stats import spearmanr
    r, _ = spearmanr(T[c], T.pt)
    ax.text(0.95, 0.92, 'ρ = %+.3f' % r, transform=ax.transAxes, ha='right',
            fontsize=6.4, color=INK)
    ax.set_xlabel(lab); ax.set_ylabel('pseudotime' if k == 0 else '')
    clean(ax); panel(ax, 'DEF'[k], dx=-0.30, dy=1.06)
fig.savefig(os.path.join(SD, 'FigureS8_trajectory_qc.png'))
plt.close(fig); log('S8 done')
log('SUPPLEMENTARY FIGURES WRITTEN TO %s' % SD)
