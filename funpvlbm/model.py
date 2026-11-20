"""Observed-data extension of the validated local current-method implementation.

Only the Gaussian sufficient statistics change. The truncation, finite-epsilon
MM updates, stopping rules, initialization and independent merges are inherited.
"""
from copy import deepcopy
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from scipy.special import xlogy, softmax

from .base_model import Model as CompleteModel, mm_weights


def basis(times):
    return BSpline(np.r_[np.zeros(4), [.25, .5, .75], np.ones(4)], np.eye(7), 3)(times)


class Model(CompleteModel):
    def __init__(self, y, times, variance_floor=.01):
        y = np.asarray(y, float)
        self.m, self.n, self.t = y.shape
        self.mask = np.isfinite(y)
        # This zero-filled array is also used by the inherited KMeans initializer.
        self.y = np.where(self.mask, y, 0.)
        self.floor = variance_floor
        self.x = basis(np.linspace(0, 1, 30))
        self.design = np.broadcast_to(basis(times), y.shape + (7,))
        self.count = self.mask.sum(axis=2)
        self.gram = np.einsum('mntk,mntl,mnt->mnkl', self.design, self.design, self.mask)
        self.yx = np.einsum('mntk,mnt->mnk', self.design, self.y)
        self.yy = (self.y ** 2).sum(axis=2)
        self.initial_features = self.y.copy()
        if np.ndim(times) > 1:
            for i in range(self.m):
                for j in range(self.n):
                    keep = self.mask[i, j]
                    self.initial_features[i, j] = np.interp(
                        np.linspace(0, 1, self.t), times[i, j, keep], y[i, j, keep])

    def initialize(self, g, h, seed):
        original = self.y
        try:
            self.y = self.initial_features
            return super().initialize(g, h, seed)
        finally:
            self.y = original

    def fit(self, initial, l1, l2, stage, max_iter=150):
        """Same updates as the local model, with the archived simulation stopping rule."""
        s = deepcopy(initial)
        if stage == 'truncation' and (len(s['pi']) * l1 >= 1 or len(s['rho']) * l2 >= 1):
            raise ValueError('Inadmissible initial count and penalty')
        trace = []
        old_objective = self.objective(s, l1, l2)
        for iteration in range(1, max_iter + 1):
            old = deepcopy(s)
            lr = self.logits(s, 'row')
            s['phi'] = softmax(lr, axis=1)
            lc = self.logits(s, 'column')
            s['gamma'] = softmax(lc, axis=1)
            self.update_blocks(s)
            if stage == 'truncation':
                for key, membership, logits, penalty, axis in [
                    ('pi', 'phi', lr, l1, 0), ('rho', 'gamma', lc, l2, 1)]:
                    a = s[membership].mean(axis=0)
                    p = np.maximum((a - penalty) / (1 - len(a) * penalty), 0)
                    keep = p > 0
                    if not keep.any():
                        raise ArithmeticError('Truncation removed every class')
                    s[key] = p[keep] / p[keep].sum()
                    s[membership] = softmax(logits[:, keep], axis=1)
                    s['beta'] = np.compress(keep, s['beta'], axis=axis)
                    s['variance'] = np.compress(keep, s['variance'], axis=axis)
            else:
                s['pi'] = mm_weights(s['phi'].mean(axis=0), old['pi'], l1)
                s['rho'] = mm_weights(s['gamma'].mean(axis=0), old['rho'], l2)
            value = self.objective(s, l1, l2)
            if not np.isfinite(value):
                raise ArithmeticError('Nonfinite penalized ELBO')
            stable = s['beta'].shape == old['beta'].shape
            beta_norm = float(np.linalg.norm(s['beta'] - old['beta'])) if stable else None
            membership_change = (max(float(np.max(np.abs(s[k] - old[k]))) for k in ['phi', 'gamma'])
                                 if stable else None)
            gain = value - old_objective
            if stage == 'refinement' and gain < -1e-9 * (1 + abs(old_objective)):
                raise ArithmeticError(f'Fixed-order MM decreased ELBO by {gain}')
            converged = stable and beta_norm <= 1e-6
            if stage == 'refinement':
                converged = converged and abs(gain) <= 1e-8 * (1 + abs(old_objective)) and membership_change < 1e-6
            trace.append(dict(iteration=iteration, G=len(s['pi']), H=len(s['rho']),
                              penalized_elbo=value, increment=gain,
                              maximum_membership_change=membership_change, coefficient_norm_change=beta_norm))
            old_objective = value
            if converged:
                break
        return s, dict(converged=bool(converged), iterations=iteration, trace=trace)

    def moments(self, s):
        p, q = s['phi'], s['gamma']
        count = p.T @ self.count @ q
        xx = np.einsum('mg,mnkl,nh->ghkl', p, self.gram, q, optimize=True)
        xy = np.einsum('mg,mnk,nh->ghk', p, self.yx, q, optimize=True)
        yy = p.T @ self.yy @ q
        return count, xx, xy, yy

    def update_blocks(self, s):
        count, xx, xy, yy = self.moments(s)
        active = count > 1e-300
        normalized_xx = xx[active] / count[active, None, None]
        normalized_xy = xy[active] / count[active, None]
        b = np.linalg.solve(normalized_xx, normalized_xy[..., None])[..., 0]
        for index in np.flatnonzero(np.any(np.abs(b) > 20, axis=1)):
            a = np.linalg.cholesky(normalized_xx[index]).T
            target = np.linalg.solve(a.T, normalized_xy[index])
            result = lsq_linear(a, target, bounds=(-20, 20), tol=1e-12, max_iter=1000)
            if not result.success:
                raise ArithmeticError('Constrained block update failed')
            b[index] = result.x
        s['beta'][active] = b
        b = s['beta']
        sse = yy - 2 * np.sum(xy * b, axis=2) + np.einsum('ghk,ghkl,ghl->gh', b, xx, b)
        if sse.min() < -1e-8:
            raise ArithmeticError('Negative observed-data SSE')
        s['variance'][active] = np.clip(sse[active] / count[active], self.floor, 100)

    def log_densities(self, s):
        b = s['beta'].reshape(-1, 7)
        bb = (b[:, :, None] * b[:, None, :]).reshape(-1, 49)
        sse = (self.yy.reshape(-1, 1) - 2 * self.yx.reshape(-1, 7) @ b.T
               + self.gram.reshape(-1, 49) @ bb.T)
        v = s['variance'].ravel()
        return (-.5 * (self.count.reshape(-1, 1) * np.log(2 * np.pi * v)
                       + sse / v)).reshape(self.m, self.n, *s['variance'].shape)

    def logits(self, s, axis):
        logf = self.log_densities(s)
        with np.errstate(divide='ignore'):
            if axis == 'row':
                return np.einsum('mngh,nh->mg', logf, s['gamma']) + np.log(s['pi'])
            return np.einsum('mngh,mg->nh', logf, s['phi']) + np.log(s['rho'])

    def objective(self, s, l1, l2):
        count, xx, xy, yy = self.moments(s)
        b, v = s['beta'], s['variance']
        sse = yy - 2 * np.sum(xy * b, axis=2) + np.einsum('ghk,ghkl,ghl->gh', b, xx, b)
        value = -.5 * np.sum(count * np.log(2 * np.pi * v) + sse / v)
        for memberships, weights, size, penalty in [
            (s['phi'], s['pi'], self.m, l1), (s['gamma'], s['rho'], self.n, l2)]:
            value += np.sum(xlogy(memberships, weights) - xlogy(memberships, memberships))
            value -= size * penalty * np.log1p(weights / 1e-6).sum()
        return float(value)

    def score(self, s):
        r, c = s['phi'].argmax(axis=1), s['gamma'].argmax(axis=1)
        b = s['beta'][r[:, None], c[None, :]]
        prediction = np.einsum('mnk,mntk->mnt', b, self.design)
        v = s['variance'][r[:, None], c[None, :]]
        sse = (((self.y - prediction) * self.mask) ** 2).sum(axis=2)
        ll = -.5 * np.sum(self.count * np.log(2 * np.pi * v) + sse / v)
        ll += np.log(s['pi'][r]).sum() + np.log(s['rho'][c]).sum()
        g, h = s['variance'].shape
        penalty = .5 * ((g - 1) * np.log(self.m) + (h - 1) * np.log(self.n)
                       + g * h * 8 * np.log(self.count.sum()))
        return dict(ICL=float(ll - penalty), complete_loglik=float(ll), G=g, H=h,
                    assigned_G=len(np.unique(r)), assigned_H=len(np.unique(c)),
                    MSE=float(sse.sum() / self.count.sum()),
                    variance_floor_hits=int(np.sum(s['variance'] <= self.floor * (1 + 1e-10))),
                    coefficient_bound_hits=int(np.sum(np.abs(s['beta']) >= 20 - 1e-8)))

