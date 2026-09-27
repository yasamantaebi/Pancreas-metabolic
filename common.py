# -*- coding: utf-8 -*-
"""Shared paths, gene sets and helpers for the HPAP re-analysis."""
import os
import numpy as np

ROOT = r'd:\Academy\Yasaman\Final result'
PART1 = os.path.join(ROOT, 'part 1')
BASE = os.path.join(ROOT, 'part2_reanalysis')
DATA = os.path.join(BASE, '01_data')
RES = os.path.join(BASE, '02_results')
FIG = os.path.join(BASE, '03_figures')
MS = os.path.join(BASE, '04_manuscript')
H5AD = os.path.join(PART1, '01_input', 'hpap_processed.h5ad')
for d in (DATA, RES, FIG, MS):
    os.makedirs(d, exist_ok=True)

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Genes excluded from FEATURE SELECTION only (retained in the expression matrix).
#
# Rationale: islet cell suspensions carry a large ambient ("soup") fraction of
# secreted hormone and exocrine transcripts. In the previous analysis these
# dominated the highly variable gene set, and clustering therefore partitioned
# cells partly by ambient load rather than by lineage. Excluding them from the
# feature set - and annotating on lineage transcription factors instead - is the
# standard mitigation when raw droplet matrices needed for SoupX/decontX are
# unavailable.
# ─────────────────────────────────────────────────────────────────────────────
HORMONE = ['INS', 'IAPP', 'GCG', 'SST', 'PPY', 'GHRL', 'TTR', 'PCSK1N',
           'CHGA', 'CHGB', 'SCG2', 'SCG3', 'SCG5']
EXOCRINE = ['PRSS1', 'PRSS2', 'PRSS3', 'CTRB1', 'CTRB2', 'CTRC', 'CTRL',
            'CELA2A', 'CELA2B', 'CELA3A', 'CELA3B', 'CPA1', 'CPA2', 'CPB1',
            'CLPS', 'PNLIP', 'PNLIPRP1', 'PNLIPRP2', 'PLA2G1B', 'SYCN',
            'AMY2A', 'AMY2B', 'AMY1A', 'CEL', 'GP2', 'SEL1L3',
            'REG1A', 'REG1B', 'REG3A', 'REG3G', 'SPINK1', 'SERPINA3']
SEX_LINKED = ['XIST', 'TSIX', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'EIF1AY', 'UTY',
              'KDM5D', 'USP9Y', 'NLGN4Y', 'TXLNGY', 'ZFY', 'TMSB4Y', 'PRKY']
OTHER_DOMINANT = ['MALAT1', 'NEAT1']


def excluded_features(var_names):
    """Boolean mask of genes to exclude from feature selection."""
    v = np.asarray(var_names)
    named = set(HORMONE + EXOCRINE + SEX_LINKED + OTHER_DOMINANT)
    mask = np.array([g in named for g in v])
    # mitochondrial, ribosomal, haemoglobin, and non-coding scaffolds
    for pref in ('MT-', 'MTRNR', 'RPS', 'RPL', 'MRPS', 'MRPL', 'HB', 'LINC',
                 'MIR', 'SNOR', 'RNU', 'RNA5', 'AC0', 'AL0', 'AP0'):
        mask |= np.array([g.startswith(pref) for g in v])
    mask |= np.array([g.endswith('-AS1') or g.endswith('.1') for g in v])
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Annotation marker panels. Deliberately weighted toward transcription factors
# and low-abundance identity genes rather than secreted hormones, because the
# hormones are the ambient-contaminated transcripts.
# ─────────────────────────────────────────────────────────────────────────────
SIGNATURES = {
    'Beta':        ['INS', 'IAPP', 'MAFA', 'PDX1', 'NKX6-1', 'DLK1', 'G6PC2',
                    'ERO1B', 'PCSK1', 'SLC2A2', 'ADCYAP1', 'RBP4', 'NPTX2',
                    'HADH', 'PFKFB2', 'SIX2', 'SIX3'],
    'Alpha':       ['GCG', 'ARX', 'IRX1', 'IRX2', 'TTR', 'PCSK2', 'SLC7A2',
                    'FEV', 'GC', 'PLCE1', 'CRYBA2', 'FAP', 'SMARCA1', 'LOXL4'],
    'Delta':       ['SST', 'HHEX', 'RBP4', 'LEPR', 'PCSK1', 'BCHE', 'FRZB',
                    'UNC5B', 'CASR', 'EDN3', 'POU3F1'],
    'PP':          ['PPY', 'SERTM1', 'ARX', 'ETV1', 'MEIS2', 'ID4', 'THSD7A',
                    'ABCC9', 'SLITRK6'],
    'Epsilon':     ['GHRL', 'ACSL1', 'ASGR1', 'SERPINA10', 'CORIN', 'PHGR1'],
    'Acinar':      ['PRSS1', 'CTRB1', 'CPA1', 'CELA3A', 'CLPS', 'PNLIP',
                    'CTRC', 'PTF1A', 'RBPJL', 'BHLHA15'],
    'Ductal':      ['KRT19', 'KRT7', 'CFTR', 'SOX9', 'ONECUT1', 'MMP7',
                    'TFPI2', 'SPP1', 'ANXA4', 'SERPING1', 'CLDN1', 'TACSTD2'],
    'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'PLVAP', 'FLT1', 'ESAM', 'EGFL7',
                    'CLEC14A'],
    'Stellate':    ['COL1A1', 'COL1A2', 'COL3A1', 'PDGFRB', 'RGS5', 'DCN',
                    'LUM', 'THY1', 'SPARC', 'BGN'],
    'Immune':      ['PTPRC', 'CD3D', 'CD68', 'AIF1', 'LYZ', 'TYROBP', 'FCER1G',
                    'C1QA', 'HLA-DRA', 'ITGAM'],
}

# markers reported per identity in the manuscript table
REPORT_MARKERS = ['INS', 'IAPP', 'MAFA', 'PDX1', 'NKX6-1', 'G6PC2',
                  'GCG', 'ARX', 'IRX2', 'PCSK2', 'TTR',
                  'SST', 'HHEX', 'LEPR', 'PPY', 'GHRL',
                  'KRT19', 'CFTR', 'PRSS1', 'CPA1',
                  'PECAM1', 'COL1A1', 'PTPRC',
                  'HSPA5', 'DDIT3', 'ATF3', 'VIM', 'ALDH1A3', 'MKI67']

STRESS = ['HSPA5', 'DDIT3', 'ATF3', 'ATF4', 'XBP1', 'HERPUD1', 'HSPA1A',
          'DNAJB1', 'HSPB1', 'JUN', 'FOS', 'SOD2', 'TXNIP']


def log(msg):
    import time
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def mem():
    try:
        import psutil
        return '%.2f GB' % (psutil.Process().memory_info().rss / 1e9)
    except Exception:
        return '?'
