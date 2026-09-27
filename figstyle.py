# -*- coding: utf-8 -*-
"""Shared style for all regenerated manuscript figures.

Categorical palette: Okabe-Ito, ordered so the worst adjacent CVD separation is
maximised (validated: worst adjacent DeltaE 12.0 protan, normal-vision 16.4 -> all
checks PASS). Identity is never colour-alone: UMAPs carry centroid labels, every
multi-series panel carries a legend, and heatmaps carry printed values.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7.5,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

INK = '#1c1c1c'
MUTED = '#6b6b6b'
FAINT = '#d8d8d8'
SURFACE = '#ffffff'

# validated categorical order
CAT = ['#009E73', '#0072B2', '#E69F00', '#CC79A7', '#7B5FB0', '#D55E00']

# lineage hues (hue = lineage, lightness = subpopulation -> composite encoding
# instead of a 15-hue cycle)
HUE = {'alpha': '#0072B2', 'beta': '#D55E00', 'delta': '#009E73',
       'pp': '#7B5FB0', 'mixed': '#CC79A7'}

DISEASE = {'Control': '#0072B2', 'T2D': '#D55E00'}
CHEM = {"10x 3' v2": '#7B5FB0', "10x 3' v3": '#009E73'}


def shades(base, n, lo=0.30, hi=0.92):
    """n lightness steps of one hue, dark -> light, keeping chroma."""
    r, g, b = to_rgb(base)
    out = []
    for f in np.linspace(lo, hi, n):
        # f=0 -> toward black, f=1 -> toward white; centre near the base hue
        if f <= 0.5:
            k = f / 0.5
            out.append((r * (0.45 + 0.55 * k), g * (0.45 + 0.55 * k), b * (0.45 + 0.55 * k)))
        else:
            k = (f - 0.5) / 0.5
            out.append((r + (1 - r) * 0.62 * k, g + (1 - g) * 0.62 * k, b + (1 - b) * 0.62 * k))
    return out


ALPHA_IDS = ['Alpha-1', 'Alpha-2', 'Alpha-3', 'Alpha-4', 'Alpha-5', 'Alpha-ARX+']
BETA_IDS = ['Beta-1', 'Beta-2', 'Beta-3']
DELTA_IDS = ['Delta', 'Delta-like (unresolved)']
PP_IDS = ['PP cell']
MIX_IDS = ['Alpha/Beta-stressed', 'Alpha/Beta-transitional', 'Alpha/Beta-mixed']

IDENT_ORDER = ALPHA_IDS + BETA_IDS + DELTA_IDS + PP_IDS + MIX_IDS

_a = shades(HUE['alpha'], 6, 0.22, 0.88)
_b = shades(HUE['beta'], 3, 0.28, 0.80)
_d = shades(HUE['delta'], 2, 0.32, 0.74)
_m = shades(HUE['mixed'], 3, 0.30, 0.82)
IDENT_COL = {}
for k, c in zip(ALPHA_IDS, _a):
    IDENT_COL[k] = c
for k, c in zip(BETA_IDS, _b):
    IDENT_COL[k] = c
for k, c in zip(DELTA_IDS, _d):
    IDENT_COL[k] = c
IDENT_COL['PP cell'] = HUE['pp']
for k, c in zip(MIX_IDS, _m):
    IDENT_COL[k] = c

LINEAGE_OF = {}
for k in ALPHA_IDS:
    LINEAGE_OF[k] = 'Alpha lineage'
for k in BETA_IDS:
    LINEAGE_OF[k] = 'Beta lineage'
LINEAGE_OF['Delta'] = 'Delta'
LINEAGE_OF['Delta-like (unresolved)'] = 'Delta-like (unresolved)'
LINEAGE_OF['PP cell'] = 'PP'
for k in MIX_IDS:
    LINEAGE_OF[k] = 'Alpha/Beta co-expressing'

LIN_ORDER = ['Alpha lineage', 'Beta lineage', 'Alpha/Beta co-expressing', 'Delta',
             'Delta-like (unresolved)', 'PP']
LIN_COL = {'Alpha lineage': HUE['alpha'], 'Beta lineage': HUE['beta'],
           'Alpha/Beta co-expressing': HUE['mixed'], 'Delta': HUE['delta'],
           'Delta-like (unresolved)': '#7fc4ac', 'PP': HUE['pp']}

RENAME = {'Low-quality': 'Delta-like (unresolved)'}

# sequential (single hue, light -> dark) for magnitude
SEQ = LinearSegmentedColormap.from_list('seq', ['#eef4fa', '#a8c8e4', '#4a8fc4', '#0d4f80'])
# diverging: two hues + neutral grey midpoint
DIV = LinearSegmentedColormap.from_list(
    'div', ['#08306b', '#3b7db8', '#a9cbe3', '#eeeeee', '#f4b183', '#d1591f', '#7f2704'])


def panel(ax, letter, dx=-0.13, dy=1.045, size=10.5):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight='bold',
            va='bottom', ha='left', color=INK)


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9a9a9a')
    ax.tick_params(colors=MUTED, labelcolor=INK)
    return ax


def bare(ax):
    """UMAP axes: no ticks, no frame, just a small orientation cue."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def heat(ax, M, vmax=None, cellgap=1.4, fmt='{:.2f}', fontsize=5.4,
         show_x=True, show_y=True, cmap=None, norm=None, values=True):
    """Diverging heatmap with a 2px surface gap between cells and printed values."""
    V = np.asarray(M.values, dtype=float)
    if norm is None:
        vmax = vmax or np.nanmax(np.abs(V))
        norm = TwoSlopeNorm(0, -vmax, vmax)
    im = ax.imshow(V, cmap=cmap or DIV, norm=norm, aspect='auto')
    ax.set_xticks(range(V.shape[1]))
    ax.set_xticklabels(list(M.columns) if show_x else [], rotation=40, ha='right')
    ax.set_yticks(range(V.shape[0]))
    ax.set_yticklabels(list(M.index) if show_y else [])
    ax.set_xticks(np.arange(-.5, V.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, V.shape[0], 1), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=cellgap)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(length=1.6, colors=MUTED, labelcolor=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if values:
        lim = float(norm.vmax) if hasattr(norm, 'vmax') else np.nanmax(np.abs(V))
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                v = V[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        fontsize=fontsize,
                        color='white' if abs(v) > lim * 0.60 else INK)
    return im


def cbar(fig, im, ax, label, shrink=0.8, pad=0.02, fraction=0.035, horiz=False):
    cb = fig.colorbar(im, ax=ax, orientation='horizontal' if horiz else 'vertical',
                      shrink=shrink, pad=pad, fraction=fraction)
    cb.set_label(label, fontsize=6.5, color=INK)
    cb.ax.tick_params(labelsize=6, length=1.6, colors=MUTED, labelcolor=INK)
    cb.outline.set_visible(False)
    return cb


def scatter_umap(ax, x, y, colors, s=0.9, alpha=0.55, rasterize=True):
    ax.scatter(x, y, c=colors, s=s, linewidths=0, alpha=alpha, rasterized=rasterize)
    bare(ax)
    ax.set_xlabel('UMAP 1', fontsize=6.5, color=MUTED, labelpad=1)
    ax.set_ylabel('UMAP 2', fontsize=6.5, color=MUTED, labelpad=1)


def legend_swatches(ax, labels, colors, ncol=1, loc='center left',
                    bbox=(1.01, 0.5), title=None, ms=4.5):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', linestyle='none', markersize=ms,
                markerfacecolor=c, markeredgecolor='none', label=l)
         for l, c in zip(labels, colors)]
    lg = ax.legend(handles=h, loc=loc, bbox_to_anchor=bbox, ncol=ncol,
                   frameon=False, handletextpad=0.4, labelspacing=0.35,
                   columnspacing=0.9, title=title, borderaxespad=0)
    if title:
        lg.get_title().set_fontsize(6.5)
        lg.get_title().set_color(INK)
    return lg


def label_pos(x, y, mask, bins=60):
    """Position for a cluster label: peak of the 2-D density, not the median.
    Median lands outside the cluster whenever a population is bimodal on the
    embedding (e.g. two spatially separate blobs)."""
    xs, ys = x[mask], y[mask]
    if len(xs) < 30:
        return float(np.median(xs)), float(np.median(ys))
    H, xe, ye = np.histogram2d(xs, ys, bins=bins)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return float((xe[i] + xe[i + 1]) / 2), float((ye[j] + ye[j + 1]) / 2)
