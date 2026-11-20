"""Penalty-guided candidate search, fixed-order refinement and independent merges."""
import itertools
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .model import Model


def fit(y, times, initial_counts=((7, 7), (10, 10)),
        penalties=(.02, .05, .09), seed=1, with_merging=True, max_iter=150):
    """Fit a row-by-column-by-time array; use NaN for missing measurements.

    times is a common one-dimensional time grid or an array matching y.
    Times must lie in [0,1]. Each starting configuration shares its KMeans
    initialization across penalty pairs. Finite iteration-capped candidates
    remain eligible. The returned state includes memberships, proportions,
    block spline coefficients and variances; records retain every attempt.
    """
    with threadpool_limits(limits=1):
        return _fit(y, times, initial_counts, penalties, seed, with_merging, max_iter)


def _fit(y, times, initial_counts, penalties, seed, with_merging, max_iter):
    model = Model(y, times)
    records, states = {}, {}
    raw_ids, refined_ids = [], []

    def attempt(name, initial, l1, l2, stage, **details):
        started = time.perf_counter()
        record = dict(id=name, lambda1=l1, lambda2=l2, stage=stage, **details)
        try:
            state, info = model.fit(initial, l1, l2, stage, max_iter=max_iter)
            score = model.score(state)
            if not np.isfinite(score['ICL']) or not all(np.isfinite(v).all() for v in state.values()):
                raise ArithmeticError('Nonfinite fitted candidate')
            record.update(status='ok', converged=info['converged'],
                          iterations=info['iterations'], **score)
            states[name] = state
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['elapsed_seconds'] = time.perf_counter() - started
        records[name] = record
        return name in states

    for index, (g, h) in enumerate(initial_counts):
        initial = model.initialize(g, h, seed)
        for pair, (l1, l2) in enumerate(itertools.product(penalties, repeat=2)):
            if g*l1 >= 1 or h*l2 >= 1:
                continue
            name = f'initial_{index}_pair_{pair}'
            if attempt(name+'_raw', initial, l1, l2, 'truncation', initial_G=g, initial_H=h):
                raw_ids.append(name+'_raw')
                if attempt(name+'_ref', states[name+'_raw'], l1, l2, 'refinement', initial_G=g, initial_H=h):
                    refined_ids.append(name+'_ref')
    if not refined_ids:
        raise RuntimeError('No eligible refined candidate; inspect dimensions and penalties')
    basic = max(refined_ids, key=lambda k: records[k]['ICL'])
    eligible = refined_ids.copy()
    parent = max(raw_ids, key=lambda k: records[k]['ICL'])
    if with_merging:
        l1, l2 = records[parent]['lambda1'], records[parent]['lambda2']
        for axis, key in [('row', 'pi'), ('column', 'rho')]:
            if len(states[parent][key]) < 2:
                continue
            child, merge = model.merge(states[parent], axis)
            name = 'merge_'+axis
            if attempt(name, child, l1, l2, 'refinement', parent=parent, merge=merge):
                eligible.append(name)
    winner = max(eligible, key=lambda k: records[k]['ICL'])
    state = states[winner]
    return dict(state=state, selected=records[winner], basic_selected=records[basic],
                row_labels=state['phi'].argmax(1), column_labels=state['gamma'].argmax(1),
                parent=parent, records=list(records.values()))
