# -*- coding: utf-8 -*-
"""Write network-extraction results into part2 02_results for the manuscript build."""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

COL = r"d:\Academy\Yasaman\Final result\part2_reanalysis\06_output_from_colleague\analysis"
RES = r"d:\Academy\Yasaman\Final result\part2_reanalysis\02_results"

meta = pd.read_csv(os.path.join(COL, "per_profile_summary.csv"), index_col=0)
ann = pd.read_csv(os.path.join(COL, "reaction_annotations.csv"))
nrxn = ann.subsystem.value_counts()

meta["disease_bin"] = (meta.disease_state == "T2D").astype(int)
meta["is_beta"] = meta.population.isin(["Beta-1", "Beta-2", "Beta-3"]).astype(int)

drop = {"population", "donor", "n_cells", "disease_state", "disease_bin", "is_beta",
        "n_reactions_total", "fao_centered", "ALL_NONBOUNDARY"}
subs = [c for c in meta.columns if c not in drop]
var = meta[subs].var(numeric_only=True)
testable = [c for c in subs if var.get(c, 0) >= 1e-12]
invariant = [c for c in subs if c not in testable]

def safe(s):
    return "s_" + "".join(ch if ch.isalnum() else "_" for ch in s)
rn = {s: safe(s) for s in testable}
M = meta.rename(columns=rn)

rows = []
for s in testable:
    c = rn[s]
    d = M[[c, "population", "disease_bin", "donor"]].dropna()
    f = "%s ~ C(population) + disease_bin" % c
    o = smf.ols(f, data=d).fit()
    k = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.donor})
    rows.append(dict(subsystem=s, n_rxns=int(nrxn.get(s, 0)),
                     delta_pct_points=o.params["disease_bin"] * 100,
                     p_ols=o.pvalues["disease_bin"],
                     p_clustered=k.pvalues["disease_bin"]))
R = pd.DataFrame(rows)
R["q_ols"] = multipletests(R.p_ols, method="fdr_bh")[1]
R["q_clustered"] = multipletests(R.p_clustered, method="fdr_bh")[1]
R = R.sort_values("p_clustered")
R.to_csv(os.path.join(RES, "s15_network_subsystem_tests.csv"), index=False)

# FAO hypothesis tests
c = rn["Fatty acid oxidation"]
d = M[[c, "population", "disease_bin", "is_beta", "donor"]].dropna()
m1 = smf.ols("%s ~ C(population) + disease_bin" % c, data=d).fit()
m1c = m1.get_robustcov_results(cov_type="cluster", groups=d.donor)
m2 = smf.ols("%s ~ C(population) + disease_bin + disease_bin:is_beta" % c, data=d).fit()
m2c = m2.get_robustcov_results(cov_type="cluster", groups=d.donor)
ik = [k for k in m2.params.index if "is_beta" in k][0]
m3 = smf.ols("%s ~ is_beta + disease_bin" % c, data=d).fit()
m3c = m3.get_robustcov_results(cov_type="cluster", groups=d.donor)
# model size
ms = smf.ols("n_reactions_total ~ C(population) + disease_bin", data=meta).fit()
msc = ms.get_robustcov_results(cov_type="cluster", groups=meta.donor)

def idx(m, name):
    return list(m.params.index).index(name)

fao = pd.DataFrame([
    dict(test="FAO disease main effect", coef_pct_points=m1.params["disease_bin"]*100,
         p_ols=m1.pvalues["disease_bin"], p_clustered=m1c.pvalues[idx(m1, "disease_bin")]),
    dict(test="FAO disease x beta interaction", coef_pct_points=m2.params[ik]*100,
         p_ols=m2.pvalues[ik], p_clustered=m2c.pvalues[idx(m2, ik)]),
    dict(test="FAO beta vs alpha/delta baseline", coef_pct_points=m3.params["is_beta"]*100,
         p_ols=m3.pvalues["is_beta"], p_clustered=m3c.pvalues[idx(m3, "is_beta")]),
    dict(test="Model size, disease effect (reactions)", coef_pct_points=ms.params["disease_bin"],
         p_ols=ms.pvalues["disease_bin"], p_clustered=msc.pvalues[idx(ms, "disease_bin")]),
])
fao.to_csv(os.path.join(RES, "s15_network_fao_tests.csv"), index=False)

qc = pd.DataFrame([dict(
    n_profiles=len(meta), n_donors=meta.donor.nunique(), n_populations=meta.population.nunique(),
    n_optimal=len(meta), mean_reactions=meta.n_reactions_total.mean(),
    sd_reactions=meta.n_reactions_total.std(ddof=1),
    min_reactions=int(meta.n_reactions_total.min()), max_reactions=int(meta.n_reactions_total.max()),
    n_subsystems_total=len(subs), n_testable=len(testable), n_invariant=len(invariant),
    invariant_list="; ".join(sorted(invariant)),
    min_q_clustered=R.q_clustered.min(), min_q_ols=R.q_ols.min(),
    n_q05_clustered=int((R.q_clustered < .05).sum()), n_q10_clustered=int((R.q_clustered < .10).sum()),
    n_q05_ols=int((R.q_ols < .05).sum()), n_q10_ols=int((R.q_ols < .10).sum()))])
qc.to_csv(os.path.join(RES, "s15_network_qc.csv"), index=False)

print("wrote s15_network_subsystem_tests.csv (%d rows)" % len(R))
print("wrote s15_network_fao_tests.csv")
print("wrote s15_network_qc.csv")
print(qc.T.to_string())
