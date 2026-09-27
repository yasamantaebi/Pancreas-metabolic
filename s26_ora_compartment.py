# -*- coding: utf-8 -*-
"""Over-representation analysis of the beta-compartment signature (Figure 3E).

Same method as s09 (Section 2.6 of the manuscript): gseapy enrichr against
MSigDB Hallmark 2020 and KEGG 2021 Human, background = all genes tested in the
compartment model, Benjamini-Hochberg correction, up- and down-regulated genes
tested separately.
"""
import os, sys
import pandas as pd
import gseapy as gp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, log

LIBS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human']
PADJ, LFC = 0.05, 0.5

r = pd.read_csv(os.path.join(RES, 's16_de_BetaCompartment_compAdjusted.csv'),
                index_col=0).dropna(subset=['padj'])
sig = r[(r.padj < PADJ) & (r.log2FoldChange.abs() > LFC)]
log('compartment signature: %d genes (%d up, %d down); background %d genes'
    % (len(sig), (sig.log2FoldChange > 0).sum(), (sig.log2FoldChange < 0).sum(), len(r)))

rows = []
for direction, sub in [('up', sig[sig.log2FoldChange > 0]),
                       ('down', sig[sig.log2FoldChange < 0])]:
    e = gp.enrichr(gene_list=list(sub.index), gene_sets=LIBS,
                   background=list(r.index), outdir=None, no_plot=True)
    d = e.results.copy()
    d['direction'] = direction
    d['n_genes_in'] = len(sub)
    rows.append(d)
    top = d.sort_values('Adjusted P-value').head(5)
    log('  %-4s (%d genes):' % (direction, len(sub)))
    for _, t in top.iterrows():
        log('      %-55s p=%.4f padj=%.4f  %s' % (t.Term[:55], t['P-value'],
                                                  t['Adjusted P-value'], t.Genes))

O = pd.concat(rows)
O.to_csv(os.path.join(RES, 's26_ora_compartment.csv'), index=False)
log('%d terms tested; %d at padj<0.05; %d at padj<0.10'
    % (len(O), (O['Adjusted P-value'] < 0.05).sum(), (O['Adjusted P-value'] < 0.10).sum()))
