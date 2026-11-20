"""Independent, direct-observation checks before the paired pilot."""
from pathlib import Path
import json
import numpy as np
from scipy.special import xlogy
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from funpvlbm.model import Model, CompleteModel


def main():
    rng = np.random.default_rng(219341)
    results = {}
    for irregular in [False, True]:
        y = rng.normal(.5, .4, (8, 6, 30))
        y[rng.uniform(size=y.shape) < .15] = np.nan
        times = np.sort(rng.uniform(size=y.shape), axis=-1) if irregular else np.linspace(0, 1, 30)
        model = Model(y, times)
        s = model.initialize(3, 2, 8)
        mu = np.einsum('ghk,mntk->mnght', s['beta'], model.design)
        direct = norm.logpdf(y[:, :, None, None, :], mu,
                            np.sqrt(s['variance'])[None, None, :, :, None])
        direct = np.nansum(direct, axis=-1)
        np.testing.assert_allclose(model.log_densities(s), direct, atol=1e-9)
        coefficient_errors, variance_errors = [], []
        for g in range(3):
            for h in range(2):
                w = np.broadcast_to((s['phi'][:, g, None] * s['gamma'][None, :, h])[..., None], y.shape)
                a = model.design[model.mask] * np.sqrt(w[model.mask])[:, None]
                target = y[model.mask] * np.sqrt(w[model.mask])
                b = np.linalg.lstsq(a, target, rcond=None)[0]
                v = np.average((y[model.mask] - model.design[model.mask] @ b) ** 2, weights=w[model.mask])
                coefficient_errors.append(float(np.max(np.abs(b - s['beta'][g, h]))))
                variance_errors.append(abs(v - s['variance'][g, h]))
        assert max(coefficient_errors) < 1e-9 and max(variance_errors) < 1e-9
        direct_elbo = np.einsum('mngh,mg,nh->', direct, s['phi'], s['gamma'])
        for p, w, n in [(s['phi'], s['pi'], 8), (s['gamma'], s['rho'], 6)]:
            direct_elbo += np.sum(xlogy(p, w) - xlogy(p, p)) - n * .05 * np.log1p(w / 1e-6).sum()
        np.testing.assert_allclose(model.objective(s, .05, .05), direct_elbo, atol=1e-9)
        r, c = s['phi'].argmax(1), s['gamma'].argmax(1)
        ll = direct[np.arange(8)[:, None], np.arange(6)[None, :], r[:, None], c[None, :]].sum()
        ll += np.log(s['pi'][r]).sum() + np.log(s['rho'][c]).sum()
        np.testing.assert_allclose(model.score(s)['complete_loglik'], ll, atol=1e-9)
        fitted, meta = model.fit(s, .02, .05, 'refinement')
        assert min(v['increment'] for v in meta['trace']) > -1e-7
        results['irregular' if irregular else 'regular'] = dict(
            log_density_error=float(np.max(np.abs(model.log_densities(s) - direct))),
            coefficient_error=max(coefficient_errors), variance_error=max(variance_errors),
            elbo_error=abs(model.objective(s, .05, .05) - direct_elbo),
            min_refinement_gain=min(v['increment'] for v in meta['trace']))
    complete = rng.normal(.5, .2, (8, 6, 30))
    old, new = CompleteModel(complete), Model(complete, np.linspace(0, 1, 30))
    a, b = old.initialize(3, 2, 8), new.initialize(3, 2, 8)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-10)
    for stage in ['truncation', 'refinement']:
        aa, _ = old.fit(a, .02, .05, stage, max_iter=4)
        bb, _ = new.fit(b, .02, .05, stage, max_iter=4)
        for key in aa:
            np.testing.assert_allclose(aa[key], bb[key], atol=1e-7)
    results['complete_data_agrees_with_existing_model'] = True
    bounded = Model(np.full((3, 2, 30), 50.), np.linspace(0, 1, 30))
    bs = bounded.initialize(1, 1, 8)
    assert np.allclose(bs['beta'], 20, atol=1e-7) and np.all(bs['variance'] == 100)
    results['constrained_update_checked'] = True
    Path(__file__).with_name('validation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
