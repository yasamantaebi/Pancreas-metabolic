# -*- coding: utf-8 -*-
"""Ranked GSEA using Human-GEM subsystems as gene sets.

Gene sets are built from Human-GEM v1.19 GPR rules (ENSEMBL -> symbol via
genes.tsv). Genes are ranked by the DESeq2 Wald statistic from the donor-level
DE. Run for the beta compartment (new combined, composition-adjusted) and for
Alpha-4, Alpha-6 and Delta using the existing per-population DE, so all six
populations of the metabolic analysis are covered.
"""
import os, re, warnings
import numpy as np, pandas as pd
import gseapy as gp

warnings.filterwarnings("ignore")
ROOT = r"d:\Academy\Yasaman\Final result"
RES = os.path.join(ROOT, "part2_reanalysis", "02_results")
ANN = os.path.join(ROOT, "part2_reanalysis", "06_output_from_colleague",
                   "analysis", "reaction_annotations.csv")
GTSV = os.path.join(ROOT, "part 1", "01_input", "genes.tsv")
OUT = os.path.join(RES, "s17_gsea")
os.makedirs(OUT, exist_ok=True)
SEED = 42

# ---- 1. build subsystem -> gene symbol sets ----
ann = pd.read_csv(ANN)
gmap = pd.read_csv(GTSV, sep="\t")
ens2sym = dict(zip(gmap.genes, gmap.geneSymbols))
print("Human-GEM genes with a symbol: %d / %d" % (gmap.geneSymbols.notna().sum(), len(gmap)))

ENS = re.compile(r"ENSG\d+")
sets = {}
for ss, grp in ann.dropna(subset=["gene_reaction_rule"]).groupby("subsystem"):
    ids = set()
    for rule in grp.gene_reaction_rule:
        ids.update(ENS.findall(str(rule)))
    syms = {ens2sym.get(e) for e in ids}
    syms = {s for s in syms if isinstance(s, str) and s}
    if syms:
        sets[ss] = sorted(syms)

MIN, MAX = 15, 500
gmt = {k: v for k, v in sets.items() if MIN <= len(v) <= MAX}
print("subsystems with >=1 mapped gene: %d" % len(sets))
print("subsystems with %d-%d genes (tested): %d" % (MIN, MAX, len(gmt)))
for k in ["Biopterin metabolism", "Pentose phosphate pathway", "Fatty acid oxidation",
          "Oxidative phosphorylation", "Glycolysis / Gluconeogenesis",
          "Tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism"]:
    n = len(sets.get(k, []))
    print("   %-62s %3d genes  %s" % (k, n, "TESTED" if k in gmt else "excluded by size"))
print("   Biopterin members:", ", ".join(sets.get("Biopterin metabolism", [])))

with open(os.path.join(OUT, "HumanGEM_subsystems.gmt"), "w") as fh:
    for k, v in gmt.items():
        fh.write("%s\t%s\t%s\n" % (k, "Human-GEM v1.19", "\t".join(v)))

# ---- 2. rankings ----
RANKS = {
    "BetaCompartment": ("s16_de_BetaCompartment_compAdjusted.csv", "combined beta, composition-adjusted"),
    "Alpha-4": ("s04_de_noHPAP090_Alpha-4.csv", "per-population"),
    "Alpha-6": ("s04_de_noHPAP090_Alpha-6.csv", "per-population"),
    "Delta": ("s04_de_noHPAP090_Delta.csv", "per-population"),
}

summary, keep = [], {}
for pop, (fn, note) in RANKS.items():
    d = pd.read_csv(os.path.join(RES, fn), index_col=0)
    if "stat" not in d.columns:
        print("  %s: no 'stat' column (%s)" % (pop, list(d.columns))); continue
    r = d["stat"].dropna()
    r = r[~r.index.duplicated()].sort_values(ascending=False)
    res = gp.prerank(rnk=r, gene_sets=os.path.join(OUT, "HumanGEM_subsystems.gmt"),
                     permutation_num=1000, min_size=MIN, max_size=MAX,
                     seed=SEED, threads=4, outdir=None, verbose=False)
    t = res.res2d.copy()
    for c in ["NES", "NOM p-val", "FDR q-val", "ES"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values("NES")
    t.insert(0, "population", pop)
    keep[pop] = t
    nsig = int((t["FDR q-val"] < 0.25).sum())
    nsig05 = int((t["FDR q-val"] < 0.05).sum())
    print("\n=== %s (%s) | %d genes ranked | %d sets tested ==="
          % (pop, note, len(r), len(t)))
    print("  sets at FDR<0.05: %d | FDR<0.25: %d" % (nsig05, nsig))
    show = pd.concat([t.head(6), t.tail(6)])
    for _, x in show.iterrows():
        print("   %-58s NES %+6.2f  p %.4f  q %.4f"
              % (x.Term[:58], x.NES, x["NOM p-val"], x["FDR q-val"]))
    summary.append(dict(population=pop, n_sets=len(t), n_q05=nsig05, n_q25=nsig))

ALL = pd.concat(keep.values())
ALL.to_csv(os.path.join(OUT, "s17_gsea_humangem_all.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(OUT, "s17_gsea_summary.csv"), index=False)

# ---- 3. the a priori hypothesis ----
print("\n" + "=" * 78)
print("A PRIORI SUBSYSTEMS FROM THE ACTIVITY-SCORE ANALYSIS")
AP = ["Biopterin metabolism", "Pentose phosphate pathway", "Fatty acid oxidation",
      "Beta oxidation of even-chain fatty acids (mitochondrial)",
      "Oxidative phosphorylation", "Thiamine metabolism", "Glutathione metabolism"]
print("  %-52s %-12s %7s %8s %8s" % ("subsystem", "population", "NES", "p", "q"))
for a in AP:
    for pop, t in keep.items():
        row = t[t.Term == a]
        if len(row):
            x = row.iloc[0]
            print("  %-52s %-12s %+7.2f %8.4f %8.4f"
                  % (a[:52], pop, x.NES, x["NOM p-val"], x["FDR q-val"]))
        else:
            print("  %-52s %-12s %s" % (a[:52], pop, "not tested (set size)"))
print("\nwrote", OUT)
