# -*- coding: utf-8 -*-
"""Scripted 100-shuffle donor-label permutation for the beta-compartment model.

Re-uses s16_beta_compartment_de.py verbatim up to the pseudobulk step (same
counts extraction, same 500-cell cap, same composition covariates, same
design), then permutes the donor disease labels NPERM times under the
composition-adjusted design used for the primary 137-gene result.

Writes 02_results/s28_de_BetaCompartment_permutation100.csv
       (permutation, n_deg, n_up, n_down, n_t2d_labels_on_true_t2d, r_age)
Does NOT overwrite any s16 output.
"""
import os, sys, io, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NPERM = int(os.environ.get('NPERM', '100'))
PERM_SEED = 2026

src = io.open(os.path.join(HERE, 's16_beta_compartment_de.py'), encoding='utf-8').read()
marker = "log('\\nprimary and sensitivity passes:')"
assert marker in src, 'marker not found in s16'
prefix = src.split(marker)[0]
ns = {'__name__': 's16_prefix', '__file__': os.path.join(HERE, 's16_beta_compartment_de.py')}
exec(compile(prefix, 's16_prefix', 'exec'), ns)

log, RES = ns['log'], ns['RES']
pseudobulk, metadata, run = ns['pseudobulk'], ns['metadata'], ns['run']
donor_meta, PADJ, LFC = ns['donor_meta'], ns['PADJ'], ns['LFC']

C90 = pseudobulk(exclude='HPAP090')
M90 = metadata(C90.index)
labels = donor_meta.loc[C90.index].disease_state.values
age = donor_meta.loc[C90.index].age.astype(float).values
is_t2d = labels == 'T2D'
log('%d donors (%d control / %d T2D), %d genes; %d permutations, seed %d'
    % (len(C90), (~is_t2d).sum(), is_t2d.sum(), C90.shape[1], NPERM, PERM_SEED))

prng = np.random.default_rng(PERM_SEED)
PERMS = [prng.permutation(labels) for _ in range(NPERM)]   # fixed sequence -> resumable
out = os.path.join(RES, 's28_de_BetaCompartment_permutation100.csv')
rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
done = {r['permutation'] for r in rows}
MAX_MIN = float(os.environ.get('MAX_MIN', '1e9'))
log('resuming: %d of %d already done' % (len(done), NPERM))
t0 = time.time()
for i in range(NPERM):
    if (i + 1) in done:
        continue
    if (time.time() - t0) / 60 > MAX_MIN:
        log('time budget reached after %d permutations; re-run to continue' % len(rows))
        sys.exit(0)
    perm = PERMS[i]
    m = metadata(C90.index, labels=perm)
    _, s = run(C90, m, True, '  permutation %3d' % (i + 1))
    p_t2d = perm == 'T2D'
    rows.append(dict(permutation=i + 1, n_deg=len(s),
                     n_up=int((s.log2FoldChange > 0).sum()),
                     n_down=int((s.log2FoldChange < 0).sum()),
                     n_t2d_labels_on_true_t2d=int((p_t2d & is_t2d).sum()),
                     r_age=float(np.corrcoef(p_t2d.astype(float), age)[0, 1])))
    pd.DataFrame(rows).to_csv(out, index=False)   # checkpoint every iteration
    if len(rows) % 10 == 0:
        n = np.array([r['n_deg'] for r in rows])
        log('  [%d/%d] median %d, max %d, >=137: %d  (%.0f s elapsed)'
            % (len(rows), NPERM, np.median(n), n.max(), (n >= 137).sum(), time.time() - t0))

rows = sorted(rows, key=lambda r: r['permutation'])
pd.DataFrame(rows).to_csv(out, index=False)
n = np.array([r['n_deg'] for r in rows])
log('DONE: %d permutations | median %d | mean %.1f | max %d | zeros %d | >= 137: %d  '
    '(empirical p = %.3f; with +1 correction %.3f)'
    % (NPERM, np.median(n), n.mean(), n.max(), (n == 0).sum(), (n >= 137).sum(),
       (n >= 137).mean(), ((n >= 137).sum() + 1) / (NPERM + 1)))
