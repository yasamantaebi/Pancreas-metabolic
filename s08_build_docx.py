# -*- coding: utf-8 -*-
"""Stage 8 - assemble the re-analysis manuscript DOCX (text + tables + figures)."""
import os, sys, re
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, FIG, MS, log

# ── build tables from results ────────────────────────────────────────────────
obs = pd.read_csv(os.path.join(RES, 's03_obs_final.csv'), index_col=0)
summ = pd.read_csv(os.path.join(RES, 's03_identity_summary.csv'), index_col=0)
de_p = pd.read_csv(os.path.join(RES, 's04_de_summary_noHPAP090.csv')).set_index('identity')
de_a = pd.read_csv(os.path.join(RES, 's04_de_summary.csv')).set_index('identity')
SUB = pd.read_csv(os.path.join(RES, 's06_subsystem_scan.csv'))
DT = pd.read_csv(os.path.join(RES, 's05_pt_donor_tests.csv'))

BETA = ['Beta-1', 'Beta-2', 'Beta-3', 'Beta-4', 'Beta-5', 'Beta-6']
ALPHA = ['Alpha-1', 'Alpha-2', 'Alpha-3', 'Alpha-4', 'Alpha-5', 'Alpha-6',
         'Alpha-7', 'Alpha-8']
ORDER = BETA + ALPHA + ['Delta', 'PP', 'Acinar']

d = obs.groupby('donor_id').agg(age=('age', 'first'), ds=('disease_state', 'first'),
                                sex=('sex', 'first'), n=('age', 'size'))
gc = obs.groupby('disease_state')


def f(x, n=0):
    return format(round(x, n) if n else int(round(x)), ',')


T1 = ('Table 1. Donor cohort and cell characteristics after quality filtering.',
      ['Characteristic', 'Control', 'T2D', 'Total'],
      [['Donors, n', '30', '17', '47'],
       ['Sex, female / male', '14 / 16', '10 / 7', '24 / 23'],
       ['Age, years: mean (range)', '31.2 (1–58)', '48.7 (34–59)', '37.5 (1–59)'],
       ['Cells before complexity filter, n', '58,613', '24,949', '83,562'],
       ['Cells retained (≥1,000 genes), n', f(len(obs[obs.disease_state == 'Control'])),
        f(len(obs[obs.disease_state == 'T2D'])), f(len(obs))],
       ['Retention rate', '71.2%', '80.6%', '74.0%'],
       ['Median genes per cell', f(gc.n_genes_by_counts.median()['Control']),
        f(gc.n_genes_by_counts.median()['T2D']), f(obs.n_genes_by_counts.median())],
       ['Median UMI per cell', f(gc.total_counts.median()['Control']),
        f(gc.total_counts.median()['T2D']), f(obs.total_counts.median())]],
      'Ages are donor-level. Donor HPAP027 fell entirely below the complexity threshold and is '
      'excluded. Sequencing chemistry (10x 3′ v2 for 7 donors, v3 for the remainder) is fully '
      'nested within donor and partially confounded with disease state; all v2 donors are '
      'non-diabetic. The 83,562-cell starting point is the quality-controlled matrix distributed '
      'by HPAP.')

rows2 = []
for cid in ORDER:
    r = summ.loc[cid]
    s = obs[obs.identity == cid]
    vc = s.groupby('donor_id').size().sort_values(ascending=False)
    top = 100 * vc.iloc[0] / len(s)
    flag = ' †' if top > 50 else ''
    rows2.append([cid + flag, f(r.n_cells), '%d / %d' % (r.n_ctrl, r.n_t2d),
                  '%.1f' % r.pct_T2D, '%.2f' % r.T2D_enrichment, '%d' % r.n_donors,
                  '%.0f%%' % top, f(r.median_genes)])
T2 = ('Table 2. The seventeen islet populations resolved in this study.',
      ['Population', 'n cells', 'Control / T2D', '% T2D', 'Enrichment', 'Donors',
       'Largest donor', 'Median genes'],
      rows2,
      'Enrichment is the ratio of the population’s T2D cell fraction to the overall T2D cell '
      'fraction (32.5%). "Largest donor" is the percentage of the population contributed by its '
      'single largest donor; † marks populations in which one donor contributes more than half '
      'the cells, which are single-donor states rather than disease-associated ones (Section 3.2). '
      'Every population received an unambiguous lineage call, with a minimum margin of 0.51 between '
      'the best and second-best signature score. Marker expression per population is given in '
      'Supplementary Data.')

PP_, LL_ = 0.05, 0.5


def _sig(fn):
    d = pd.read_csv(os.path.join(RES, fn), index_col=0)
    return d[(d.padj < PP_) & (d.log2FoldChange.abs() > LL_)]


_bc = _sig('s16_de_BetaCompartment_compAdjusted.csv')
_ba = _sig('s16_de_BetaCompartment_compAdjusted_all.csv')
_bup = _bc[_bc.log2FoldChange > 0].sort_values('log2FoldChange', ascending=False)
rows3 = [['Beta compartment (all beta cells)', f(22457), '30 / 16', '%d' % len(_bc),
          '%d / %d' % ((_bc.log2FoldChange > 0).sum(), (_bc.log2FoldChange < 0).sum()),
          '%d' % len(_ba), '; '.join(_bup.index[:5])]]
for cid in ORDER:
    if cid not in de_p.index:
        continue
    p, a = de_p.loc[cid], de_a.loc[cid] if cid in de_a.index else None
    rows3.append([cid, f(p.n_cells), '%d / %d' % (p.n_ctrl_donors, p.n_t2d_donors),
                  '%d' % p.n_deg, '%d / %d' % (p.n_up, p.n_down),
                  '%d' % (a.n_deg if a is not None else 0),
                  str(p.top_up) if isinstance(p.top_up, str) else '—'])
T3 = ('Table 3. Donor-level differential expression: the beta compartment as a whole, and each '
      'population separately.',
      ['Population', 'n cells', 'Donors (C / T2D)', 'DEGs (primary)', 'Up / down',
       'DEGs (all donors)', 'Top upregulated'],
      rows3,
      'Design: ~ age + sex + disease_state, with a maximum of 500 cells per donor per population '
      'sampled before pseudobulk aggregation. Significance: Benjamini–Hochberg-adjusted p < 0.05 '
      'and |log2 fold change| > 0.5. The primary analysis excludes donor HPAP090 (Section 2.6); the '
      'penultimate column gives the sensitivity pass including all donors. Populations with fewer '
      'than five donors per condition were not tested. Signal is confined to the beta compartment. '
      'The first row is the primary beta cell analysis: all of a donor’s beta cells pooled into one '
      'pseudobulk profile with the 500-cell cap applied once, and donor fractions of Beta-1 and '
      'Beta-2 included as covariates because their proportions differ between conditions '
      '(Section 2.6). It recovers 137 genes against 46 for the union of the six per-subpopulation '
      'tests. Permuting donor disease labels 15 times under this design gave a median of 0 '
      'significant genes and a maximum of 1.')

SHORT = {'Beta oxidation of di-unsaturated fatty acids (n-6) (mitochondrial)': 'β-oxidation, di-unsaturated FA (n-6)',
         'Beta oxidation of unsaturated fatty acids (n-7) (mitochondrial)': 'β-oxidation, unsaturated FA (n-7)',
         'Beta oxidation of unsaturated fatty acids (n-9) (mitochondrial)': 'β-oxidation, unsaturated FA (n-9)',
         'Beta oxidation of odd-chain fatty acids (mitochondrial)': 'β-oxidation, odd-chain FA',
         'Beta oxidation of even-chain fatty acids (mitochondrial)': 'β-oxidation, even-chain FA'}
CTS = [c for c in SUB.columns if c in ORDER]
cons = SUB[SUB.all_consistent].sort_values('mean')
rows4 = [[SHORT.get(r.subsystem, r.subsystem), '%d' % r.n_rxns] +
         ['%+.3f' % r[c] for c in CTS] + ['%+.3f' % r['mean']]
         for _, r in cons.iterrows()]
PO = pd.read_csv(os.path.join(RES, 's12_subsystem_tests_pooled.csv'))
tb4 = cons.merge(PO[['subsystem', 'delta', 'effect_r', 'p', 'padj']], on='subsystem',
                 how='left')
rows4 = [[SHORT.get(r.subsystem, r.subsystem), '%d' % r.n_rxns] +
         ['%+.3f' % r[c] for c in CTS] +
         ['%+.3f' % r.delta, '%.3f' % r.p, '%.2f' % r.padj]
         for _, r in tb4.iterrows()]
T4 = ('Table 4. Metabolic subsystems with a concordant direction of change across all six populations, '
      'with donor-level statistics.',
      ['Human-GEM subsystem', 'n rxns'] + CTS + ['Pooled Δ', 'p', 'p adj'],
      rows4,
      'All 98 Human-GEM subsystems containing at least 15 GPR-associated reactions were surveyed. '
      'Shown are the 18 whose direction of change agreed in all six populations, ranked by mean '
      'effect. Per-population columns are changes in mean reaction activity score (T2D − Control) '
      'relative to each sample’s global reaction-score level. "Pooled Δ" is the difference in median '
      'donor-level score between conditions, computed across 224 per-donor profiles with each donor '
      'contributing one value averaged over the populations in which it is represented; p is a '
      'two-sided Mann–Whitney U test on those donor-level values and p adj is the '
      'Benjamini–Hochberg-corrected value across all 98 subsystems. '
      '**No subsystem reaches significance at a 5% false discovery rate.** Directional concordance '
      'across populations is therefore reported as an effect-size observation, not as evidence of a '
      'statistically supported difference (Sections 3.4 and 4.2).')

# ── Table 5: context-specific network extraction ─────────────────────────────
NET = pd.read_csv(os.path.join(RES, 's15_network_subsystem_tests.csv'))
APRIORI = ['Biopterin metabolism', 'Pentose phosphate pathway', 'Fatty acid oxidation',
           'Beta oxidation of even-chain fatty acids (mitochondrial)']
top_net = NET.nsmallest(8, 'p_clustered')
ap_net = NET[NET.subsystem.isin(APRIORI)]
net_tb = pd.concat([top_net, ap_net]).drop_duplicates('subsystem')
rows5 = [[SHORT.get(r.subsystem, r.subsystem), '%d' % r.n_rxns,
          '%+.2f' % r.delta_pct_points, '%.3f' % r.p_clustered, '%.2f' % r.q_clustered,
          'a priori' if r.subsystem in APRIORI else '']
         for _, r in net_tb.iterrows()]
T5 = ('Table 5. Subsystem retention in context-specific metabolic networks extracted for each of '
      '224 donor profiles.',
      ['Human-GEM subsystem', 'n rxns', 'Δ retention (pp)', 'p', 'q', 'Source'],
      rows5,
      'A context-specific model was extracted from Human-GEM v1.19 for every donor profile by '
      'task-constrained integer programming (Section 2.7); all 224 extractions reached proven '
      'optimality. Retention is the fraction of a subsystem’s reactions present in the extracted '
      'model; Δ retention is the T2D − Control difference in percentage points from ordinary least '
      'squares with population as a covariate. Because each donor contributes up to six profiles, '
      'standard errors are clustered on donor and q is the Benjamini–Hochberg value across the 121 '
      'subsystems with non-zero retention variance. Shown are the eight subsystems with the smallest '
      'p-values together with the four processes identified a priori by the activity-score analysis '
      '(Table 4). **No subsystem reaches significance at a 5% false discovery rate** (smallest '
      'q = 0.48), and the a priori processes are all null. Six of the 127 subsystems, including '
      'oxidative phosphorylation, were retained identically in every model and admit no test. '
      'Reaction counts are all reactions annotated to the subsystem in Human-GEM and therefore '
      'differ slightly from Table 4, which counts only GPR-associated reactions.')

rows6 = [[r.population, f(r.n_cells), '%d / %d' % (r.n_ctrl_donors, r.n_t2d_donors),
          '%.3f' % r.median_ctrl, '%.3f' % r.median_t2d, '%+.3f' % r.delta,
          '%.3f' % r.p_twosided] for _, r in DT.iterrows()]
T6 = ('Table 6. Pseudotime position by beta population and disease state.',
      ['Population', 'n cells', 'Donors (C / T2D)', 'Median PT (Control)',
       'Median PT (T2D)', 'Δ median', 'p'],
      rows6,
      'Diffusion pseudotime computed on the marker-verified beta compartment from the centroid-based '
      'root (Root B). Populations are ordered by control median. Medians and p-values are computed at '
      'the donor level (one median per donor), the appropriate unit of analysis given that cells from '
      'the same donor are not independent; p-values are two-sided Mann–Whitney U tests. With six '
      'tests the Bonferroni threshold is 0.008, which none of the nominal values reaches. A '
      'donor-level test of terminal-quintile occupancy is likewise non-significant (control median '
      '20.8% vs T2D 28.3%, p = 0.141).')

TABLES = {'TABLE 1': [T1], 'TABLE 2': [T2], 'TABLE 3': [T3], 'TABLE 4': [T4],
          'TABLE 5': [T5], 'TABLE 6': [T6]}

FIGS = {
 'FIGURE 1': ('Figure 1. The HPAP islet single-cell atlas. UMAP of 61,859 quality-filtered islet '
  'cells after ambient-aware feature selection and Harmony correction, coloured by (A) major lineage '
  'and (B) disease state. (C) The same embedding coloured by the 17 final populations, labelled in '
  'place at each population’s density peak; hue denotes lineage and lightness denotes '
  'subpopulation. (D) Mean expression of canonical markers per population, z-scored across '
  'populations. Every beta population has INS far above GCG with detectable IAPP, MAFA and NKX6-1; '
  'every alpha population has GCG and TTR high with detectable ARX, IRX2 and PCSK2; the delta '
  'population is the only one with high SST and HHEX, and the PP population the only one with high '
  'PPY. Lineage assignments are therefore concordant with the markers, correcting the systematic '
  'mis-assignment in the prior annotation of this dataset (Section 3.1).',
  'Figure1_atlas.png'),
 'FIGURE 2': ('Figure 2. Apparent disease restriction is explained by donor composition. '
  '(A) Percentage of T2D cells in each population; the dashed line marks the 32.5% expected from '
  'the donor ratio. (B) Percentage of each population contributed by its single largest donor; the '
  'dashed line marks 50%. Two populations, Alpha-2 and Alpha-3, exceed it, and both are dominated by '
  'the same donor; these are shown in red in both panels. (C, D) Per-donor cell contributions to the '
  'two populations with the highest T2D fraction. Each derives 93.7% of its cells from donor '
  'HPAP090, and only 5 of the 17 T2D donors contribute at least 1% of either. After accounting for '
  'donor composition, no population in this dataset is disease-restricted, and the largest genuine '
  'compositional differences are the modest expansions of Alpha-7 and Alpha-8 (Section 3.2).',
  'Figure2_donor_composition.png'),
 'FIGURE 3': ('Figure 3. A beta-cell-restricted transcriptional signature, tested on the compartment '
  'as a whole and replicated across the subpopulations it pools. (A) Volcano plot of donor-level '
  'differential expression across all 22,457 beta cells (30 control and 16 T2D donors; design '
  '~ age + sex + Beta-1 fraction + Beta-2 fraction + disease state, HPAP090 excluded). Orange, '
  'significantly upregulated; blue, downregulated; dashed lines mark adjusted p = 0.05 and '
  '|log2 fold change| = 0.5. Labelled are the largest effects and the established beta cell and '
  'diabetes genes recovered only at compartment level. (B) Significant genes recovered by each beta '
  'subpopulation tested separately, by their union, and by the compartment tested once. The '
  'compartment test uses fewer cells than the six separate tests combined, because the 500-cell '
  'donor cap applies once rather than six times. (C) Log2 fold change of the core signature genes '
  'across the six beta subpopulations and the combined test; the direction of effect is consistent '
  'throughout. (D) Significant genes obtained after permuting the donor disease labels, 15 shuffles, '
  'against the 137 observed. The median permutation returned 0 genes and the maximum 1 '
  '(Sections 2.6 and 3.3).',
  'Figure3_de.png'),
 'FIGURE 4': ('Figure 4. Metabolic reprogramming is directionally consistent but not statistically '
  'supported at donor level. (A) The 18 Human-GEM subsystems, of 98 surveyed, whose direction of '
  'change agreed across all six populations, ranked by mean effect, with reaction counts in '
  'parentheses; values are changes in mean reaction activity score relative to each sample’s global '
  'level and blue denotes reduction in T2D. The rightmost column gives the two-sided Mann–Whitney p '
  'from donor-level testing across 224 per-donor profiles. No subsystem reaches a 5% false discovery '
  'rate. (B) Per-donor distributions for selected subsystems, each donor contributing one value '
  'averaged across the populations in which it is represented. Biopterin metabolism, the pathway with '
  'the most consistent direction of effect, gives p = 0.086. (C) Donor-level change for the pentose '
  'phosphate and tetrahydrobiopterin enzymes, with uncorrected p-values; PCBD1 and TPK1 are the '
  'closest to significance (both p = 0.009, adjusted p = 0.052). (D) Per-donor distributions for the '
  'four enzymes central to the proposed mechanism. The direction of effect is consistent and the '
  'enzymes involved are mechanistically coherent, but the cohort is underpowered for effects of this '
  'size (Sections 3.4 and 4.2).',
  'Figure4_metabolic.png'),
 'FIGURE 5': ('Figure 5. Context-specific network extraction shows no disease effect on fatty acid '
  'oxidation, but a constitutive lineage difference. A context-specific metabolic model was extracted '
  'from Human-GEM v1.19 for each of 224 donor profiles by task-constrained integer programming; all '
  'reached proven optimality. (A) Fraction of fatty acid oxidation reactions retained, by population '
  'and disease state; each point is one donor, boxes are quartiles. Control and T2D distributions '
  'overlap in every population, and neither the main disease effect (p = 0.94) nor the disease × '
  'lineage interaction (p = 0.57) is significant. (B) The same profiles pooled across conditions and '
  'grouped by lineage. Beta populations retain 2.95 percentage points fewer fatty acid oxidation '
  'reactions than alpha and delta populations (p = 0.015), a difference present in all three beta '
  'subpopulations. The robust effect at this analytical layer is therefore constitutive rather than '
  'disease-associated (Sections 3.5 and 4.2).',
  'Figure5_network.png'),
 'FIGURE 6': ('Figure 6. The verified beta compartment shows no dedifferentiation trajectory and no '
  'T2D shift. UMAP of the 22,457 marker-verified beta cells coloured by (A) population and (B) '
  'diffusion pseudotime. (C) Donor median pseudotime by population and disease state; each point is '
  'one donor, horizontal bars are group medians, and values above are two-sided Mann-Whitney p at the '
  'donor level. With six tests the Bonferroni threshold is 0.008, which none reaches. (D) Spearman '
  'correlation between pseudotime and canonical beta identity, stress and dedifferentiation genes. '
  'Beta identity genes are flat and, if anything, weakly positive; no gene in the feature set exceeds '
  '|rho| = 0.29. (E) Correlation between pseudotime and technical covariates. The correlation with '
  'total UMI count is now -0.016, against -0.68 before quality filtering, so the axis no longer '
  'tracks library depth — but it correlates with mitochondrial fraction and doublet score about '
  'as strongly as with any transcript of interest. (F) Donor-level terminal-quintile occupancy, which '
  'does not differ between conditions (Section 3.6).',
  'Figure5_trajectory.png'),
}

# ── document ─────────────────────────────────────────────────────────────────
doc = Document()
st = doc.styles['Normal']
st.font.name = 'Times New Roman'
st.font.size = Pt(11)
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(1)
MAXW = 6.5
BOLD_RE = re.compile(r'(\*\*.+?\*\*|\*[^*]+?\*)')


def add_runs(p, text, size=11, bold=False, italic=False):
    for tok in BOLD_RE.split(text):
        if not tok:
            continue
        b, i, t = bold, italic, tok
        if tok.startswith('**') and tok.endswith('**'):
            b, t = True, tok[2:-2]
        elif tok.startswith('*') and tok.endswith('*'):
            i, t = True, tok[1:-1]
        r = p.add_run(t)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(size)
        r.bold = b
        r.italic = i


def para(text, size=11, bold=False, italic=False, align=None, after=6, line=1.5,
         indent=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    if align:
        p.alignment = align
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    add_runs(p, text, size, bold, italic)
    return p


def heading(text, level):
    sizes = {1: 13, 2: 11.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(sizes[level])
    r.bold = True


def add_table(t):
    title, header, rows, note = t
    para(title, size=10, bold=True, after=4, line=1.15)
    tb = doc.add_table(rows=1, cols=len(header))
    tb.style = 'Table Grid'
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    fs = 8.0 if len(header) > 7 else 8.5
    for j, h in enumerate(header):
        c = tb.rows[0].cells[j]
        c.text = ''
        pp = c.paragraphs[0]
        pp.paragraph_format.space_after = Pt(2)
        pp.paragraph_format.line_spacing = 1.0
        r = pp.add_run(str(h)); r.bold = True
        r.font.size = Pt(fs); r.font.name = 'Times New Roman'
    for row in rows:
        cs = tb.add_row().cells
        for j, v in enumerate(row):
            cs[j].text = ''
            pp = cs[j].paragraphs[0]
            pp.paragraph_format.space_after = Pt(2)
            pp.paragraph_format.line_spacing = 1.0
            r = pp.add_run(str(v))
            r.font.size = Pt(fs); r.font.name = 'Times New Roman'
    para(note, size=8.5, italic=True, after=12, line=1.0)


def add_figure(cap, fn):
    path = os.path.join(FIG, fn)
    w, h = Image.open(path).size
    width = MAXW
    if width * h / w > 8.3:
        width = 8.3 * w / h
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(path, width=Inches(width))
    para(cap, size=9.5, after=14, line=1.15)


md = open(os.path.join(MS, 'manuscript.md'), encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(md) if l.strip() == '# ABSTRACT')

# front matter
for txt, sz, bd, al in [
        ('A beta-cell-restricted transcriptional signature in type 2 diabetic human islets, and '
         'the donor-level structure of apparent disease-associated cell states', 15, True,
         WD_ALIGN_PARAGRAPH.CENTER),
        ('Yasaman Taebi ¹,*, [CO-AUTHOR FULL NAME] ¹', 11.5, False, WD_ALIGN_PARAGRAPH.CENTER),
        ('¹ [Department / Faculty], Tehran University, Tehran, Iran', 10, False,
         WD_ALIGN_PARAGRAPH.CENTER),
        ('* Correspondence: [email address]  |  ORCID: [0000-0000-0000-0000]', 10, False,
         WD_ALIGN_PARAGRAPH.CENTER),
        ('Running title: Beta-cell transcriptional reprogramming in T2D islets', 10, False,
         WD_ALIGN_PARAGRAPH.CENTER),
        ('Figures: 5  |  Tables: 5  |  Word count (main text): 6,766', 10, False,
         WD_ALIGN_PARAGRAPH.CENTER)]:
    para(txt, size=sz, bold=bd, align=al, after=6, line=1.15)

i = start
in_refs = False
while i < len(md):
    s = md[i].strip()
    i += 1
    if not s:
        continue
    m = re.match(r'^\*\*\[(TABLE \d|FIGURE \d) HERE\]\*\*$', s)
    if m:
        k = m.group(1)
        if k in TABLES:
            for t in TABLES[k]:
                add_table(t)
        else:
            add_figure(*FIGS[k])
        continue
    if s.startswith('# '):
        t = s[2:].strip()
        in_refs = t.upper().startswith('REFERENCE')
        if t == 'BACK MATTER':
            doc.add_page_break(); heading('Declarations', 1); continue
        heading(t, 1); continue
    if s.startswith('## '):
        heading(s[3:].strip(), 2); continue
    if in_refs and re.match(r'^\d+\.\s', s):
        p = para(s, size=10, after=3, line=1.15)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        continue
    if re.match(r'^\d+\.\s+\*\*', s):
        para(s, after=8, indent=0.25); continue
    para(s, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

out = os.path.join(MS, 'HPAP_reanalysis_manuscript.docx')
doc.save(out)
log('saved %s | paragraphs=%d tables=%d' % (out, len(doc.paragraphs), len(doc.tables)))
