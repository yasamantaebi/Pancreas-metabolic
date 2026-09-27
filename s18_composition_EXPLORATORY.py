# -*- coding: utf-8 -*-
"""Donor-level compositional analysis, corrected.

Estimators (all donor-level, n = 47):
  MW    Mann-Whitney on raw donor proportions (unadjusted, distribution-free)
  QB    quasi-binomial GLM with overdispersion scaling done manually
        (statsmodels' fit(scale='X2') deflates rather than inflates SEs)
  ASIN  arcsine-sqrt transformed proportions + OLS with age + sex (propeller-style)
  CLR   centred log-ratio + OLS with age + sex (compositional)
"""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
import scipy.stats as ss
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
RES = r"d:\Academy\Yasaman\Final result\part2_reanalysis\02_results"
obs = pd.read_csv(os.path.join(RES, "s03_obs_final.csv"), index_col=0)
obs["lineage_grp"] = np.where(obs.identity.str.startswith("Beta"), "Beta",
                     np.where(obs.identity.str.startswith("Alpha"), "Alpha", obs.identity))
dm = obs.groupby("donor_id", observed=True).agg(
    disease_state=("disease_state", "first"), age=("age", "first"), sex=("sex", "first"))


def clr(P, eps=0.5):
    A = P.values + eps
    A = A / A.sum(axis=1, keepdims=True)
    L = np.log(A)
    return pd.DataFrame(L - L.mean(axis=1, keepdims=True), index=P.index, columns=P.columns)


def test(cnt, target, label, exclude=None, verbose=True):
    c = cnt.drop(index=[exclude]) if exclude and exclude in cnt.index else cnt
    meta = dm.loc[c.index]
    tot = c.sum(axis=1)
    k = c[target] if isinstance(target, str) else c[target].sum(axis=1)
    prop = (k / tot).astype(float)
    grp = meta.disease_state.values
    pc, pt = prop[grp == "Control"], prop[grp == "T2D"]

    u, p_mw = mannwhitneyu(pt, pc, alternative="two-sided")
    r_rb = 2 * u / (len(pt) * len(pc)) - 1

    d = pd.DataFrame(dict(k=k.values.astype(float), n=tot.values.astype(float),
                          disease=(meta.disease_state == "T2D").astype(int).values,
                          age=meta.age.astype(float).values,
                          sex=(meta.sex.astype(str) == "M").astype(int).values))
    m = smf.glm("k + I(n - k) ~ disease + age + sex", data=d,
                family=sm.families.Binomial()).fit()
    phi = m.pearson_chi2 / m.df_resid
    se = m.bse["disease"] * np.sqrt(phi)
    tstat = m.params["disease"] / se
    p_qb = 2 * ss.t.sf(abs(tstat), m.df_resid)
    OR = float(np.exp(m.params["disease"]))

    y = np.arcsin(np.sqrt(prop.values))
    a = smf.ols("y ~ disease + age + sex",
                data=pd.DataFrame(dict(y=y, disease=d.disease, age=d.age, sex=d.sex))).fit()
    p_as = a.pvalues["disease"]

    L = clr(c)
    lt = L[target] if isinstance(target, str) else L[target].mean(axis=1)
    o = smf.ols("y ~ disease + age + sex",
                data=pd.DataFrame(dict(y=lt.values, disease=d.disease, age=d.age, sex=d.sex))).fit()
    p_clr = o.pvalues["disease"]

    if verbose:
        rel = 100 * (pt.median() - pc.median()) / pc.median() if pc.median() > 0 else np.nan
        print("\n%s%s" % (label, "   [HPAP090 excluded]" if exclude else ""))
        print("   Control %.4f   T2D %.4f   absolute %+.4f   relative %+.0f%%"
              % (pc.median(), pt.median(), pt.median() - pc.median(), rel))
        print("   Mann-Whitney            p = %.4f    rank-biserial r = %+.3f" % (p_mw, r_rb))
        print("   quasi-binomial +age+sex p = %.4f    OR = %.3f   (overdispersion phi = %.0f)"
              % (p_qb, OR, phi))
        print("   arcsine-sqrt +age+sex   p = %.4f" % p_as)
        print("   CLR +age+sex            p = %.4f" % p_clr)
    return dict(test=str(label).strip(), ctrl=pc.median(), t2d=pt.median(),
                p_mw=p_mw, p_qb=p_qb, p_asin=p_as, p_clr=p_clr, r=r_rb, OR=OR)


print("donors %d (%d C / %d T2D) | cells %d | per donor median %d"
      % (len(dm), (dm.disease_state == "Control").sum(), (dm.disease_state == "T2D").sum(),
         len(obs), obs.groupby("donor_id", observed=True).size().median()))

cl = obs.groupby(["donor_id", "lineage_grp"], observed=True).size().unstack(fill_value=0)
print("\n" + "=" * 78)
print("PRIMARY, PRE-SPECIFIED, SINGLE TEST: BETA CELL FRACTION")
print("=" * 78)
prim = [test(cl, "Beta", "Beta fraction of all islet cells"),
        test(cl, "Beta", "Beta fraction of all islet cells", exclude="HPAP090"),
        test(cl[[c for c in cl.columns if c != "Acinar"]], "Beta",
             "Beta fraction of endocrine cells only")]

print("\n" + "=" * 78)
print("SECONDARY: LINEAGES (BH across %d)" % cl.shape[1])
print("=" * 78)
S = pd.DataFrame([test(cl, c, c, verbose=False) for c in cl.columns])
for col in ["p_mw", "p_qb", "p_asin", "p_clr"]:
    S["q" + col[1:]] = multipletests(S[col], method="fdr_bh")[1]
print("  %-8s %8s %8s %8s %8s %8s %8s"
      % ("lineage", "Ctrl", "T2D", "p(QB)", "q(QB)", "p(MW)", "q(CLR)"))
for _, x in S.sort_values("p_qb").iterrows():
    print("  %-8s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f%s"
          % (x.test, x.ctrl, x.t2d, x.p_qb, x.q_qb, x.p_mw, x.q_clr,
             "  *" if x.q_qb < 0.05 else ""))

print("\n" + "=" * 78)
print("EXPLORATORY: 17 POPULATIONS (BH-corrected)")
print("=" * 78)
cp = obs.groupby(["donor_id", "identity"], observed=True).size().unstack(fill_value=0)
P = pd.DataFrame([test(cp, c, c, verbose=False) for c in cp.columns])
for col in ["p_mw", "p_qb", "p_asin", "p_clr"]:
    P["q" + col[1:]] = multipletests(P[col], method="fdr_bh")[1]
print("  %-11s %8s %8s %8s %8s %8s"
      % ("population", "Ctrl", "T2D", "p(QB)", "q(QB)", "q(MW)"))
for _, x in P.sort_values("p_qb").iterrows():
    print("  %-11s %8.4f %8.4f %8.4f %8.4f %8.4f%s"
          % (x.test, x.ctrl, x.t2d, x.p_qb, x.q_qb, x.q_mw,
             "  *" if x.q_qb < 0.05 else ""))

pd.DataFrame(prim).to_csv(os.path.join(RES, "s18_composition_primary.csv"), index=False)
S.to_csv(os.path.join(RES, "s18_composition_lineage.csv"), index=False)
P.to_csv(os.path.join(RES, "s18_composition_population.csv"), index=False)

# ---- confounder: does the complexity filter bias composition? ----
print("\n" + "=" * 78)
print("CONFOUNDER CHECK: complexity-filter retention")
print("=" * 78)
exc = os.path.join(RES, "s01_excluded_lowcomplexity_cells.csv")
if os.path.exists(exc):
    E = pd.read_csv(exc)
    dcol = [c for c in E.columns if "donor" in c.lower()]
    if dcol:
        nexc = E.groupby(dcol[0]).size()
        kept = obs.groupby("donor_id", observed=True).size()
        ret = (kept / (kept + nexc.reindex(kept.index).fillna(0))).dropna()
        g = dm.loc[ret.index].disease_state
        u, p = mannwhitneyu(ret[g == "T2D"], ret[g == "Control"], alternative="two-sided")
        print("  retention rate  Control %.3f  T2D %.3f  p = %.4f"
              % (ret[g == "Control"].median(), ret[g == "T2D"].median(), p))
        bf = (cl.Beta / cl.sum(axis=1)).reindex(ret.index)
        rho, pr = ss.spearmanr(ret, bf)
        print("  retention vs beta fraction: Spearman rho = %+.3f, p = %.4f" % (rho, pr))
        print("  -> if retention differs AND correlates with beta fraction, composition is biased")
else:
    print("  excluded-cell file not found")
