# -*- coding: utf-8 -*-
"""New Figure 3: the beta-cell transcriptional signature, tested on the beta
compartment as a whole with per-subpopulation replication."""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = r"d:\Academy\Yasaman\Final result\part2_reanalysis"
sys.path.insert(0, os.path.join(ROOT, "00_scripts"))
import figstyle as F

RES = os.path.join(ROOT, "02_results")
FIG = os.path.join(ROOT, "03_figures")
PADJ, LFC = 0.05, 0.5
SUBS = ["Beta-1", "Beta-2", "Beta-3", "Beta-4", "Beta-5", "Beta-6"]

comb = pd.read_csv(os.path.join(RES, "s16_de_BetaCompartment_compAdjusted.csv"), index_col=0)
sig = comb[(comb.padj < PADJ) & (comb.log2FoldChange.abs() > LFC)]


def sigset(p):
    d = pd.read_csv(os.path.join(RES, "s04_de_noHPAP090_%s.csv" % p), index_col=0)
    return d, d[(d.padj < PADJ) & (d.log2FoldChange.abs() > LFC)]


subd = {p: sigset(p) for p in SUBS}
union = set().union(*[set(s.index) for _, s in subd.values()])

# permutation nulls (15 shuffles, composition-adjusted design)
NULL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
OBSN = len(sig)
pd.DataFrame(dict(permutation=range(1, len(NULL) + 1), n_deg=NULL)).to_csv(
    os.path.join(RES, "s16_de_BetaCompartment_permutation.csv"), index=False)

fig = plt.figure(figsize=(7.09, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                      hspace=0.42, wspace=0.30,
                      left=0.085, right=0.985, top=0.945, bottom=0.075)

# ---------------- A: volcano ----------------
ax = fig.add_subplot(gs[0, 0])
d = comb.dropna(subset=["padj", "log2FoldChange"]).copy()
d["y"] = -np.log10(d.padj.clip(lower=1e-300))
ns = d[~d.index.isin(sig.index)]
ax.scatter(ns.log2FoldChange, ns.y, s=3.0, c=F.FAINT, lw=0, rasterized=True)
up = sig[sig.log2FoldChange > 0]
dnn = sig[sig.log2FoldChange < 0]
ax.scatter(d.loc[up.index].log2FoldChange, d.loc[up.index].y, s=7, c=F.HUE["beta"], lw=0)
ax.scatter(d.loc[dnn.index].log2FoldChange, d.loc[dnn.index].y, s=7, c=F.HUE["alpha"], lw=0)
ax.axhline(-np.log10(PADJ), color=F.MUTED, lw=0.6, ls=(0, (3, 3)))
for x in (-LFC, LFC):
    ax.axvline(x, color=F.MUTED, lw=0.6, ls=(0, (3, 3)))
LAB = {"BARX1": (5, 3), "TBX2": (5, 2), "TBX2-AS1": (-46, 1), "FAIM2": (-36, 3),
       "PCOLCE2": (5, 2), "ELFN1": (5, -1),
       "PPP1R1A": (6, 3), "PCSK9": (6, 3),
       "SLC2A2": (-30, -14), "HNF1A": (-8, -20), "GRB14": (-44, -8),
       "CLTRN": (16, -16)}
for g, (ox, oy) in LAB.items():
    if g not in d.index:
        continue
    r = d.loc[g]
    lead = dict(arrowprops=dict(arrowstyle="-", lw=0.4, color=F.MUTED,
                                shrinkA=0, shrinkB=1.5)) if abs(oy) > 6 else {}
    ax.annotate(g, (r.log2FoldChange, r.y), fontsize=5.4, color=F.INK,
                style="italic", xytext=(ox, oy), textcoords="offset points", **lead)
ax.set_xlabel("log$_2$ fold change (T2D vs control)")
ax.set_ylabel("$-$log$_{10}$ adjusted $p$")
ax.set_title("%d significant genes" % OBSN, fontsize=7.5, color=F.MUTED, pad=3)
F.clean(ax); F.panel(ax, "A")

# ---------------- B: DEG counts ----------------
ax = fig.add_subplot(gs[0, 1])
names = SUBS + ["Union of\nsix", "Beta\ncompartment"]
vals = [len(subd[p][1]) for p in SUBS] + [len(union), OBSN]
cols = [F.FAINT] * 6 + [F.MUTED, F.HUE["beta"]]
b = ax.bar(range(len(vals)), vals, color=cols, width=0.72,
           edgecolor=F.INK, linewidth=0.4)
for i, v in enumerate(vals):
    ax.text(i, v + 3, str(v), ha="center", fontsize=6.2, color=F.INK)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=5.8, rotation=0)
ax.set_ylabel("significant genes")
ax.set_ylim(0, max(vals) * 1.16)
F.clean(ax); F.panel(ax, "B")

# ---------------- C: replication of core genes ----------------
ax = fig.add_subplot(gs[1, 0])
CORE = ["BARX1", "TBX2", "TBX2-AS1", "FAIM2", "TSHR",
        "PPP1R1A", "GOLT1A", "CD82", "ASCL2", "SMIM6"]
M = np.full((len(CORE), len(SUBS) + 1), np.nan)
for j, p in enumerate(SUBS):
    dd = subd[p][0]
    for i, g in enumerate(CORE):
        if g in dd.index:
            M[i, j] = dd.loc[g, "log2FoldChange"]
for i, g in enumerate(CORE):
    if g in comb.index:
        M[i, -1] = comb.loc[g, "log2FoldChange"]
v = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap=F.DIV, vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(len(SUBS) + 1))
ax.set_xticklabels([s.replace("Beta-", "B") for s in SUBS] + ["Comb."], fontsize=6)
ax.set_yticks(range(len(CORE)))
ax.set_yticklabels(CORE, fontsize=6, style="italic")
ax.axvline(len(SUBS) - 0.5, color=F.INK, lw=1.0)
for i in range(len(CORE)):
    for j in range(len(SUBS) + 1):
        if not np.isnan(M[i, j]):
            ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center",
                    fontsize=4.6, color=F.INK if abs(M[i, j]) < v * 0.6 else "white")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0, colors=F.MUTED, labelcolor=F.INK)
ax.set_xlabel("beta subpopulation                    ", fontsize=6.5)
F.cbar(fig, im, ax, "log$_2$ fold change")
F.panel(ax, "C")

# ---------------- D: permutation calibration ----------------
ax = fig.add_subplot(gs[1, 1])
mx = max(max(NULL), 3)
bins = np.arange(-0.5, mx + 1.5, 1)
ax.hist(NULL, bins=bins, color=F.FAINT, edgecolor=F.INK, linewidth=0.4)
ax.axvline(OBSN, color=F.HUE["beta"], lw=1.6)
ax.annotate("observed\n%d genes" % OBSN, (OBSN, len(NULL) * 0.62),
            fontsize=6.2, color=F.HUE["beta"], ha="right",
            xytext=(-6, 0), textcoords="offset points")
ax.set_xlim(-0.6, OBSN * 1.10)
ax.set_xlabel("significant genes under permuted disease labels")
ax.set_ylabel("permutations")
ax.set_title("15 shuffles: median 0, maximum 1", fontsize=7.0, color=F.MUTED, pad=3)
F.clean(ax); F.panel(ax, "D")

out = os.path.join(FIG, "Figure3_de.png")
fig.savefig(out, dpi=400, facecolor="white")
print("wrote", out)
print("  combined %d | union %d | per-subpop %s" % (OBSN, len(union),
      {p: len(subd[p][1]) for p in SUBS}))
