# -*- coding: utf-8 -*-
"""Figure 3, revised after review: panels C and D swapped (C = permutation,
D = per-subpopulation log2FC heatmap) and a new panel E showing the
over-representation analysis of the compartment signature (s26)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

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
NULL = pd.read_csv(os.path.join(RES, "s28_de_BetaCompartment_permutation100.csv")).n_deg.tolist()
OBSN = len(sig)
ORA = pd.read_csv(os.path.join(RES, "s26_ora_compartment.csv"))

fig = plt.figure(figsize=(7.09, 9.3))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.92, 0.80],
                      hspace=0.48, wspace=0.30,
                      left=0.085, right=0.985, top=0.96, bottom=0.05)

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
ax.bar(range(len(vals)), vals, color=cols, width=0.72, edgecolor=F.INK, linewidth=0.4)
for i, v in enumerate(vals):
    ax.text(i, v + 3, str(v), ha="center", fontsize=6.2, color=F.INK)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=5.8, rotation=0)
ax.set_ylabel("significant genes")
ax.set_ylim(0, max(vals) * 1.16)
F.clean(ax); F.panel(ax, "B")

# ---------------- C: permutation calibration (was D) ----------------
ax = fig.add_subplot(gs[1, 0])
NULLa = np.array(NULL)
bins = np.arange(0, max(OBSN, NULLa.max()) + 10, 5)
ax.hist(NULLa, bins=bins, color=F.FAINT, edgecolor=F.INK, linewidth=0.4)
ax.axvline(OBSN, color=F.HUE["beta"], lw=1.6)
n_ge = int((NULLa >= OBSN).sum())
ax.annotate("observed\n%d genes" % OBSN, (OBSN, len(NULLa) * 0.62),
            fontsize=6.2, color=F.HUE["beta"], ha="right",
            xytext=(-6, 0), textcoords="offset points")
ax.text(0.98, 0.30, "%d of %d permutations\n≥ observed\nempirical p = %.2f"
        % (n_ge, len(NULLa), n_ge / len(NULLa)), transform=ax.transAxes,
        ha="right", va="center", fontsize=6.0, color=F.INK)
ax.set_xlim(-3, max(OBSN, NULLa.max()) * 1.10)
ax.set_xlabel("significant genes under permuted disease labels")
ax.set_ylabel("permutations")
ax.set_title("%d shuffles: median %d, %d zeros, maximum %d"
             % (len(NULLa), int(np.median(NULLa)), int((NULLa == 0).sum()), NULLa.max()),
             fontsize=7.0, color=F.MUTED, pad=3)
F.clean(ax); F.panel(ax, "C")

# ---------------- D: replication of core genes (was C) ----------------
ax = fig.add_subplot(gs[1, 1])
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
F.panel(ax, "D")

# ---------------- E: over-representation analysis ----------------
ax = fig.add_subplot(gs[2, :])
NTOP = 6
rows = []
for direction in ("up", "down"):
    sub = ORA[ORA.direction == direction].sort_values("P-value").head(NTOP)
    for _, r in sub.iterrows():
        lib = "Hallmark" if r.Gene_set.startswith("MSigDB") else "KEGG"
        rows.append(dict(term="%s  [%s]" % (r.Term, lib), padj=r["Adjusted P-value"],
                         p=r["P-value"], genes=r.Genes.replace(";", ", "),
                         direction=direction))
E = pd.DataFrame(rows)
E["x"] = -np.log10(E.padj)
# up block on top, down block below, each ordered by adjusted p
E_up = E[E.direction == "up"].sort_values("x")
E_dn = E[E.direction == "down"].sort_values("x")
order = pd.concat([E_dn, E_up])
y = np.arange(len(order))
colors = [F.HUE["beta"] if dd == "up" else F.HUE["alpha"] for dd in order.direction]
ax.barh(y, order.x, color=colors, height=0.66, edgecolor=F.INK, linewidth=0.3)
ax.axvline(-np.log10(0.05), color=F.MUTED, lw=0.7, ls=(0, (3, 3)))
ax.text(-np.log10(0.05) + 0.02, -0.62, "adjusted $p$ = 0.05", fontsize=5.6,
        color=F.MUTED, ha="left", va="bottom", rotation=0)
ax.set_yticks(y)
ax.set_yticklabels(order.term, fontsize=5.8)
for yi, (_, r) in zip(y, order.iterrows()):
    ax.text(r.x + 0.03, yi, r.genes, fontsize=4.8, color=F.MUTED, va="center", style="italic")
ax.set_xlim(0, max(order.x.max() * 1.9, 2.2))
ax.set_ylim(-0.7, len(order) - 0.3)
ax.axhline(len(E_dn) - 0.5, color=F.INK, lw=0.6)
ax.text(ax.get_xlim()[1], len(order) - 1, "upregulated (%d genes)" % ORA[ORA.direction == "up"].n_genes_in.iloc[0],
        fontsize=6.2, color=F.HUE["beta"], ha="right", va="center")
ax.text(ax.get_xlim()[1], 0, "downregulated (%d genes)" % ORA[ORA.direction == "down"].n_genes_in.iloc[0],
        fontsize=6.2, color=F.HUE["alpha"], ha="right", va="center")
ax.set_xlabel("$-$log$_{10}$ adjusted $p$ (Benjamini\u2013Hochberg)")
ax.set_title("over-representation of the compartment signature: no term reaches adjusted $p$ < 0.05",
             fontsize=7.0, color=F.MUTED, pad=3)
F.clean(ax); F.panel(ax, "E", dx=-0.06)

out = os.path.join(FIG, "Figure3_de.png")
fig.savefig(out, dpi=400, facecolor="white")
print("wrote", out)
print(E[["direction", "term", "p", "padj", "genes"]].to_string(index=False))
