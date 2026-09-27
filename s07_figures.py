# -*- coding: utf-8 -*-
"""Stage 7 - manuscript figures for the re-analysis."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, PART1, log
from figstyle import (INK, MUTED, FAINT, SURFACE, SEQ, DIV, HUE, DISEASE,
                      shades, panel, clean, bare, heat, cbar, scatter_umap,
                      legend_swatches, label_pos)

obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
summ = pd.read_csv(os.path.join(RES, 's03_identity_summary.csv'), index_col=0)

BETA = ['Beta-1', 'Beta-2', 'Beta-3', 'Beta-4', 'Beta-5', 'Beta-6']
ALPHA = ['Alpha-1', 'Alpha-2', 'Alpha-3', 'Alpha-4', 'Alpha-5', 'Alpha-6',
         'Alpha-7', 'Alpha-8']
OTHER = ['Delta', 'PP', 'Acinar']
ORDER = BETA + ALPHA + OTHER
COL = {}
for k, c in zip(BETA, shades(HUE['beta'], len(BETA), 0.25, 0.85)):
    COL[k] = c
for k, c in zip(ALPHA, shades(HUE['alpha'], len(ALPHA), 0.20, 0.88)):
    COL[k] = c
COL['Delta'] = HUE['delta']; COL['PP'] = HUE['pp']; COL['Acinar'] = HUE['mixed']
LIN = {**{k: 'Beta' for k in BETA}, **{k: 'Alpha' for k in ALPHA},
       'Delta': 'Delta', 'PP': 'PP', 'Acinar': 'Acinar'}
LINCOL = {'Beta': HUE['beta'], 'Alpha': HUE['alpha'], 'Delta': HUE['delta'],
          'PP': HUE['pp'], 'Acinar': HUE['mixed']}

X, Y = obs.UMAP1.values, obs.UMAP2.values
rng = np.random.default_rng(0)
sh = rng.permutation(len(obs))
W = 7.09

# ═══════════════════════ FIGURE 1 - atlas ════════════════════════════════════
fig = plt.figure(figsize=(W, 8.4))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.35, 1.15], hspace=0.34,
              wspace=0.20, left=0.115, right=0.855, top=0.965, bottom=0.075)

ax = fig.add_subplot(gs[0, 0])
scatter_umap(ax, X[sh], Y[sh], [LINCOL[LIN[v]] for v in obs.identity.values[sh]],
             s=0.7, alpha=0.55)
ax.set_xlabel(''); ax.set_ylabel('')
panel(ax, 'A', dx=-0.05, dy=1.01)
legend_swatches(ax, list(LINCOL), list(LINCOL.values()), ncol=3,
                loc='upper center', bbox=(0.5, -0.02), ms=3.6)

ax = fig.add_subplot(gs[0, 1])
scatter_umap(ax, X[sh], Y[sh], [DISEASE[v] for v in obs.disease_state.values[sh]],
             s=0.7, alpha=0.5)
ax.set_xlabel(''); ax.set_ylabel('')
panel(ax, 'B', dx=-0.05, dy=1.01)
nc = int((obs.disease_state == 'Control').sum()); nt = len(obs) - nc
legend_swatches(ax, ['Control (%s)' % f'{nc:,}', 'T2D (%s)' % f'{nt:,}'],
                [DISEASE['Control'], DISEASE['T2D']], ncol=2, loc='upper center',
                bbox=(0.5, -0.02), ms=3.6)

ax = fig.add_subplot(gs[1, :])
scatter_umap(ax, X[sh], Y[sh], [COL[v] for v in obs.identity.values[sh]],
             s=1.0, alpha=0.6)
for cid in ORDER:
    m = obs.identity.values == cid
    if m.sum() > 100:
        lx, ly = label_pos(X, Y, m)
        ax.text(lx, ly, cid, fontsize=5.4, ha='center', va='center', color=INK,
                path_effects=[pe.withStroke(linewidth=1.9, foreground='white')])
panel(ax, 'C', dx=-0.035, dy=1.005)
legend_swatches(ax, ORDER, [COL[i] for i in ORDER], ncol=1, bbox=(1.005, 0.5), ms=4)

ax = fig.add_subplot(gs[2, :])
MK = ['INS', 'IAPP', 'MAFA', 'NKX6-1', 'G6PC2', 'GCG', 'ARX', 'IRX2', 'PCSK2',
      'TTR', 'SST', 'HHEX', 'LEPR', 'PPY', 'PRSS1', 'CPA1', 'KRT19']
M = summ.reindex(ORDER)[MK]
Z = (M - M.mean()) / M.std()
im = heat(ax, Z, vmax=2.4, fontsize=4.6, fmt='{:+.1f}')
for x in (5.5, 10.5, 12.5, 13.5):
    ax.axvline(x, color=INK, lw=0.9)
for lab, x0, x1 in [('beta', 0, 5), ('alpha', 5, 10), ('delta', 10, 12),
                    ('PP', 12, 13), ('exo/duct', 14, 17)]:
    ax.text((x0 + x1) / 2 - 0.5 + 0.5, -1.1, lab, ha='center', fontsize=6,
            fontweight='bold', color=INK)
for t in ax.get_xticklabels():
    t.set_style('italic')
panel(ax, 'D', dx=-0.105, dy=1.10)
cbar(fig, im, ax, 'Mean expression\n(z across identities)', shrink=0.85,
     fraction=0.022, pad=0.02)
fig.savefig(os.path.join(FIG, 'Figure1_atlas.png'))
plt.close(fig); log('Figure 1 done')

# ═════════════ FIGURE 2 - the single-donor artefact ══════════════════════════
old = pd.read_csv(os.path.join(PART1, '04_results', 'reviewer_diagnostics',
                               'full_obs_with_umap.csv'), low_memory=False)
fig = plt.figure(figsize=(W, 5.4))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.62, wspace=0.30,
              left=0.10, right=0.87, top=0.93, bottom=0.13)

for k, pop in enumerate(['Alpha-ARX+', 'Beta-3']):
    ax = fig.add_subplot(gs[0, k])
    s = old[old.cell_identity == pop]
    vc = s.groupby('donor_id').size().sort_values(ascending=False)
    top = vc.head(8)
    cols = ['#b3200f' if d == 'HPAP090' else DISEASE['T2D'] for d in top.index]
    ax.bar(range(len(top)), top.values, color=cols, linewidth=0, width=0.72)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=55, ha='right', fontsize=5.6)
    ax.set_ylabel('cells contributed')
    ax.set_title('prior annotation: %s\n(n = %d, %d donors)' % (pop, len(s), vc.size),
                 fontsize=7, color=INK, pad=3)
    ax.text(0.97, 0.88, 'HPAP090 = %.1f%%' % (100 * vc.iloc[0] / len(s)),
            transform=ax.transAxes, ha='right', fontsize=6.6, color='#b3200f',
            fontweight='bold')
    clean(ax)
    panel(ax, 'AB'[k], dx=-0.20, dy=1.16)

ax = fig.add_subplot(gs[1, :])
newT = summ.reindex(ORDER)
base = 100 * (obs.disease_state == 'T2D').mean()
donor_frac = []
for cid in ORDER:
    s = obs[obs.identity == cid]
    vc = s.groupby('donor_id').size().sort_values(ascending=False)
    donor_frac.append(100 * vc.iloc[0] / len(s))
xs = np.arange(len(ORDER))
cols = ['#b3200f' if f > 50 else COL[c] for f, c in zip(donor_frac, ORDER)]
ax.bar(xs, newT.pct_T2D.values, color=cols, linewidth=0, width=0.72)
ax.axhline(base, color=INK, ls=(0, (3, 2)), lw=0.8)
ax.text(len(ORDER) - 0.4, base + 1.5, 'expected (%.1f%% T2D)' % base, ha='right',
        fontsize=5.8, color=INK)
for i, f in enumerate(donor_frac):
    if f > 50:
        ax.text(i, newT.pct_T2D.values[i] + 2, 'single\ndonor', ha='center',
                fontsize=5.2, color='#b3200f', fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(ORDER, rotation=55, ha='right')
ax.set_ylabel('% T2D cells in population')
ax.set_ylim(0, 112)
clean(ax)
panel(ax, 'C', dx=-0.085, dy=1.03)
ax.text(0, -0.62, 'Red bars: >50% of the population comes from one donor.',
        transform=ax.transAxes, fontsize=6, color='#b3200f')
fig.savefig(os.path.join(FIG, 'Figure2_donor_artefact.png'))
plt.close(fig); log('Figure 2 done')

# ═══════════════════════ FIGURE 3 - differential expression ══════════════════
de_sum = pd.read_csv(os.path.join(RES, 's04_de_summary_noHPAP090.csv'))
d3 = pd.read_csv(os.path.join(RES, 's04_de_noHPAP090_Beta-3.csv'), index_col=0)
d3 = d3.dropna(subset=['padj'])
sig = d3[(d3.padj < 0.05) & (d3.log2FoldChange.abs() > 0.5)]
up = sig[sig.log2FoldChange > 0].sort_values('log2FoldChange', ascending=False)
dn = sig[sig.log2FoldChange < 0].sort_values('log2FoldChange')

fig = plt.figure(figsize=(W, 7.2))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.95], hspace=0.42,
              wspace=0.34, left=0.11, right=0.86, top=0.955, bottom=0.09)

ax = fig.add_subplot(gs[0, 0])
ns = d3.drop(sig.index)
ax.scatter(ns.log2FoldChange, -np.log10(ns.padj), s=2.6, c='#c9c9c9',
           linewidths=0, rasterized=True)
ax.scatter(up.log2FoldChange, -np.log10(up.padj), s=12, c=DISEASE['T2D'],
           linewidths=0.3, edgecolors='white', label='up (n=%d)' % len(up))
ax.scatter(dn.log2FoldChange, -np.log10(dn.padj), s=12, c=DISEASE['Control'],
           linewidths=0.3, edgecolors='white', label='down (n=%d)' % len(dn))
ax.axhline(-np.log10(0.05), color=MUTED, ls=(0, (3, 2)), lw=0.6)
for g in list(up.index[:7]) + list(dn.index[:4]):
    r = d3.loc[g]
    ax.annotate(g, (r.log2FoldChange, -np.log10(r.padj)), fontsize=5.6,
                style='italic', xytext=(2.5, 2.5), textcoords='offset points')
ax.set_xlabel('log$_2$ fold change (T2D vs Control)')
ax.set_ylabel('$-$log$_{10}$ adjusted $p$')
ax.set_title('Beta-3 (largest beta population)', fontsize=7, pad=3)
clean(ax); ax.legend(frameon=False, loc='upper left', fontsize=6)
panel(ax, 'A', dx=-0.22, dy=1.06)

ax = fig.add_subplot(gs[0, 1])
d = de_sum.set_index('identity').reindex(ORDER).fillna(0)
ys = np.arange(len(ORDER))
ax.barh(ys, -d.n_down.values, color=DISEASE['Control'], linewidth=0, height=0.68)
ax.barh(ys, d.n_up.values, color=DISEASE['T2D'], linewidth=0, height=0.68)
ax.set_yticks(ys); ax.set_yticklabels(ORDER); ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel('significant genes (down | up)')
for i, c in enumerate(ORDER):
    n = int(d.n_deg.values[i])
    if n:
        ax.text(d.n_up.values[i] + 0.6, i, str(n), va='center', fontsize=5.8,
                color=INK)
ax.axhspan(-0.5, len(BETA) - 0.5, color=HUE['beta'], alpha=0.07, zorder=0)
ax.text(0.97, 0.02, 'signal is confined\nto beta populations', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=6, color=INK)
clean(ax)
panel(ax, 'B', dx=-0.30, dy=1.06)

ax = fig.add_subplot(gs[1, :])
CORE = ['BARX1', 'TBX2', 'TBX2-AS1', 'FAIM2', 'GAL', 'DKK3', 'PCOLCE2', 'PAX5',
        'SLC4A4', 'TSHR', 'ELFN1', 'SLC26A4', 'TNFRSF11B']
mat, ann = [], []
for pop in ORDER:
    f = os.path.join(RES, 's04_de_noHPAP090_%s.csv' % pop)
    if not os.path.exists(f):
        mat.append([np.nan] * len(CORE)); ann.append([''] * len(CORE)); continue
    r = pd.read_csv(f, index_col=0)
    row, arow = [], []
    for g in CORE:
        if g in r.index and np.isfinite(r.loc[g, 'log2FoldChange']):
            row.append(r.loc[g, 'log2FoldChange'])
            p = r.loc[g, 'padj']
            arow.append('*' if (np.isfinite(p) and p < 0.05) else '')
        else:
            row.append(np.nan); arow.append('')
    mat.append(row); ann.append(arow)
Mx = pd.DataFrame(mat, index=ORDER, columns=CORE)
im = ax.imshow(Mx.values, cmap=DIV, norm=TwoSlopeNorm(0, -4, 4), aspect='auto')
ax.set_xticks(range(len(CORE)))
ax.set_xticklabels(CORE, rotation=55, ha='right', style='italic')
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
ax.set_xticks(np.arange(-.5, len(CORE), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
ax.grid(which='minor', color=SURFACE, linewidth=1.3)
ax.tick_params(which='minor', length=0); ax.tick_params(length=1.5)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(ORDER)):
    for j in range(len(CORE)):
        if ann[i][j]:
            ax.text(j, i, '*', ha='center', va='center', fontsize=8,
                    color='white' if abs(Mx.values[i, j]) > 2.4 else INK)
ax.axhline(len(BETA) - 0.5, color=INK, lw=1.0)
panel(ax, 'C', dx=-0.105, dy=1.04)
cbar(fig, im, ax, 'log$_2$FC (T2D vs Control)', shrink=0.8, fraction=0.022, pad=0.02)
ax.text(1.005, -0.02, '* adjusted $p$ < 0.05', transform=ax.transAxes, fontsize=6,
        color=INK, va='bottom')
fig.savefig(os.path.join(FIG, 'Figure3_de.png'))
plt.close(fig); log('Figure 3 done')

# ═══════════════════════ FIGURE 4 - metabolic ════════════════════════════════
SUB = pd.read_csv(os.path.join(RES, 's06_subsystem_scan.csv'))
G = pd.read_csv(os.path.join(RES, 's06_gene_level.csv'))
CTS = [c for c in SUB.columns if c in ORDER]
cons = SUB[SUB.all_consistent].sort_values('mean')
SHORT = {'Beta oxidation of di-unsaturated fatty acids (n-6) (mitochondrial)': 'β-ox, di-unsat FA (n-6)',
         'Beta oxidation of unsaturated fatty acids (n-7) (mitochondrial)': 'β-ox, unsat FA (n-7)',
         'Beta oxidation of unsaturated fatty acids (n-9) (mitochondrial)': 'β-ox, unsat FA (n-9)',
         'Beta oxidation of odd-chain fatty acids (mitochondrial)': 'β-ox, odd-chain FA',
         'Beta oxidation of even-chain fatty acids (mitochondrial)': 'β-ox, even-chain FA'}
cons = cons.copy()
cons['label'] = [SHORT.get(s, s) + '  (%d)' % n
                 for s, n in zip(cons.subsystem, cons.n_rxns)]

fig = plt.figure(figsize=(W, 8.2))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.72], hspace=0.30,
              left=0.335, right=0.895, top=0.955, bottom=0.075)

ax = fig.add_subplot(gs[0])
im = heat(ax, cons.set_index('label')[CTS], vmax=0.62, fontsize=5.3)
ax.axhline(len(cons) - 3.5, color=INK, lw=1.0)
panel(ax, 'A', dx=-0.48, dy=1.02)
cbar(fig, im, ax, 'Δ reaction activity score\n(T2D − Control, vs global shift)',
     shrink=0.6, fraction=0.026, pad=0.02)
ax.set_ylabel('Human-GEM subsystem (n reactions)', fontsize=6.8, labelpad=2)

ax = fig.add_subplot(gs[1])
PPP = ['G6PD', 'PGD', 'TALDO1', 'TKT', 'SHPK', 'TPK1']
BH4 = ['GCH1', 'PTS', 'SPR', 'PCBD1', 'QDPR', 'DHFR']
Gi = G.set_index('gene')
M = Gi.reindex(PPP + BH4)[CTS]
im = heat(ax, M, vmax=1.15, fontsize=5.4)
ax.axhline(len(PPP) - 0.5, color=INK, lw=1.2)
ax.text(-0.285, len(PPP) / 2 - 0.5, 'Pentose phosphate pathway\n+ thiamine cofactor',
        transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center',
        fontsize=6.1, color=INK, fontweight='bold')
ax.text(-0.285, len(PPP) + len(BH4) / 2 - 0.5, 'Tetrahydrobiopterin (BH$_4$)\nsynthesis and salvage',
        transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center',
        fontsize=6.1, color=INK, fontweight='bold')
for t in ax.get_yticklabels():
    t.set_style('italic')
panel(ax, 'B', dx=-0.48, dy=1.03)
cbar(fig, im, ax, 'Δ reaction activity score', shrink=0.85, fraction=0.026, pad=0.02)
fig.savefig(os.path.join(FIG, 'Figure4_metabolic.png'))
plt.close(fig); log('Figure 4 done')

# ═══════════════════════ FIGURE 5 - trajectory ═══════════════════════════════
T = pd.read_csv(os.path.join(RES, 's05_trajectory_obs.csv'), index_col=0)
DT = pd.read_csv(os.path.join(RES, 's05_pt_donor_tests.csv'))
RB = pd.read_csv(os.path.join(RES, 's05_root_robustness.csv'), index_col=0)
GR = pd.read_csv(os.path.join(RES, 's05_gene_pseudotime_corr.csv'), index_col=0)
DQ = pd.read_csv(os.path.join(RES, 's05_donor_pct_q5.csv'))
tc = pd.read_csv(os.path.join(RES, 's05_pt_technical_corr.csv'), index_col=0)

fig = plt.figure(figsize=(W, 7.4))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.95, 0.95], hspace=0.52,
              wspace=0.34, left=0.115, right=0.86, top=0.955, bottom=0.085)
sh2 = np.random.default_rng(1).permutation(len(T))

ax = fig.add_subplot(gs[0, 0])
scatter_umap(ax, T.UMAP1.values[sh2], T.UMAP2.values[sh2],
             [COL[v] for v in T.identity.values[sh2]], s=0.9, alpha=0.6)
ax.set_xlabel(''); ax.set_ylabel('')
panel(ax, 'A', dx=-0.09, dy=1.01)
legend_swatches(ax, BETA, [COL[b] for b in BETA], ncol=3, loc='upper center',
                bbox=(0.5, -0.02), ms=3.4)

ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(T.UMAP1.values[sh2], T.UMAP2.values[sh2], c=T.pt.values[sh2],
                cmap='viridis', s=0.9, linewidths=0, alpha=0.7, rasterized=True)
bare(ax)
panel(ax, 'B', dx=-0.09, dy=1.01)
cbar(fig, sc, ax, 'Diffusion pseudotime', shrink=0.72, fraction=0.04)

ax = fig.add_subplot(gs[1, 0])
pops = DT.population.tolist()
for i, p in enumerate(pops):
    for j, ds in enumerate(['Control', 'T2D']):
        v = T[(T.identity == p) & (T.disease_state == ds)]
        pd_ = v.groupby('donor_id').pt.median().values
        pos = i + (j - 0.5) * 0.34
        if len(pd_):
            ax.scatter(pos + (np.random.default_rng(i * 10 + j).random(len(pd_)) - .5) * .18,
                       pd_, s=6, color=DISEASE[ds], linewidths=0.25,
                       edgecolors='white', zorder=4)
            ax.plot([pos - .13, pos + .13], [np.median(pd_)] * 2, color=INK, lw=1.1,
                    zorder=5)
    pv = DT.loc[DT.population == p, 'p_twosided'].iloc[0]
    ax.text(i, 1.02, '%.2f' % pv, transform=ax.get_xaxis_transform(), ha='center',
            va='bottom', fontsize=5.8, color=MUTED)
ax.set_xticks(range(len(pops)))
ax.set_xticklabels(pops, rotation=35, ha='right')
ax.set_ylabel('donor median pseudotime')
ax.set_xlim(-0.55, len(pops) - 0.45)
clean(ax)
ax.text(0.5, 1.13, 'two-sided Mann–Whitney $p$ (donor level)', transform=ax.transAxes,
        ha='center', fontsize=5.8, color=MUTED)
panel(ax, 'C', dx=-0.24, dy=1.18)
legend_swatches(ax, ['Control', 'T2D'], [DISEASE['Control'], DISEASE['T2D']],
                bbox=(1.0, 0.5), ms=4)

ax = fig.add_subplot(gs[1, 1])
key = ['MAFA', 'PDX1', 'NKX6-1', 'G6PC2', 'INS', 'IAPP', 'MEG3', 'VIM', 'HSPA5',
       'DDIT3', 'ALDH1A3']
vals = [GR.iloc[:, 0].get(g, np.nan) for g in key]
ok = [(g, v) for g, v in zip(key, vals) if np.isfinite(v)]
ax.barh(range(len(ok)), [v for _, v in ok],
        color=[DIV(0.5 + v / 1.2) for _, v in ok], linewidth=0, height=0.68)
ax.set_yticks(range(len(ok)))
ax.set_yticklabels([g for g, _ in ok], style='italic')
ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel('Spearman ρ with pseudotime')
ax.set_xlim(-0.45, 0.45)
gmax = float(np.nanmax(np.abs(GR.iloc[:, 0].values)))
ax.text(0.97, 0.03, 'strongest |ρ| in\nany gene: %.2f' % gmax,
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=INK)
clean(ax)
panel(ax, 'D', dx=-0.30, dy=1.06)

ax = fig.add_subplot(gs[2, 0])
labs = {'n_genes_by_counts': 'genes detected', 'total_counts': 'total UMI',
        'pct_counts_mt': 'mitochondrial %', 'doublet_score': 'doublet score'}
v = tc.iloc[:, 0]
ax.barh(range(len(v)), v.values, color=[DIV(0.5 + x / 1.2) for x in v.values],
        linewidth=0, height=0.62)
ax.set_yticks(range(len(v)))
ax.set_yticklabels([labs.get(i, i) for i in v.index])
ax.invert_yaxis(); ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel('Spearman ρ with pseudotime')
ax.set_xlim(-0.8, 0.8)
for x in (-0.5, 0.5):
    ax.axvline(x, color=MUTED, ls=(0, (2, 2)), lw=0.6)
clean(ax)
panel(ax, 'E', dx=-0.30, dy=1.06)

ax = fig.add_subplot(gs[2, 1])
for j, ds in enumerate(['Control', 'T2D']):
    v = DQ.loc[DQ.disease_state == ds, 'isQ5'].values
    ax.boxplot([v], positions=[j], widths=0.42, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=1.1),
               whiskerprops=dict(color='#9a9a9a', lw=0.7),
               capprops=dict(color='#9a9a9a', lw=0.7),
               boxprops=dict(facecolor=DISEASE[ds], alpha=0.3, edgecolor='none'))
    ax.scatter(j + (np.random.default_rng(j).random(len(v)) - .5) * .26, v, s=7,
               color=DISEASE[ds], linewidths=0.25, edgecolors='white', zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'T2D'])
ax.set_ylabel("% of donor's cells in Q5")
ax.set_xlim(-0.6, 1.6)
yl = ax.get_ylim()
ax.text(0.5, yl[1] * 0.97, 'n.s. ($p$ = 0.14)', ha='center', fontsize=6.4, color=INK)
clean(ax)
panel(ax, 'F', dx=-0.30, dy=1.06)
fig.savefig(os.path.join(FIG, 'Figure5_trajectory.png'))
plt.close(fig); log('Figure 5 done')
log('ALL FIGURES WRITTEN TO %s' % FIG)
