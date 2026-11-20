"""Complete common-grid Gaussian funPVLBM, following current Sections 2 and 5.

This module provides complete-grid updates, initialization and merging.
No fitted labels enter initialization.
"""
from copy import deepcopy
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq, lsq_linear
from scipy.special import softmax, xlogy
from sklearn.cluster import KMeans


def mm_weights(a, old, penalty, epsilon=1e-6):
    """Exact simplex maximum of sum(a log p) - sum(c p), c=lambda/(old+eps).

    Shift the multiplier to avoid cancellation near -min(c). The boundary
    branch also handles numerically zero counts without evaluating 0/0.
    """
    a = np.asarray(a, float) / np.sum(a)
    if penalty == 0:
        return a
    c = penalty / (old + epsilon)
    d = c - c.min()
    if np.all(d == 0):
        return a
    at_min = d == 0
    positive = a > 0
    if not np.any(positive & at_min):
        p = np.zeros_like(a)
        p[~at_min] = a[~at_min] / d[~at_min]
        if p.sum() <= 1:
            p[at_min] = (1 - p.sum()) * old[at_min] / old[at_min].sum()
            return p
    def equation(t):
        if t == 0 and np.any(positive & at_min):
            return np.inf
        return np.sum(a[positive] / (t + d[positive])) - 1
    t = brentq(equation, 0, 2, xtol=1e-300, rtol=1e-14, maxiter=2000)
    p = np.zeros_like(a)
    p[positive] = a[positive] / (t + d[positive])
    return p / p.sum()


class Model:
    def __init__(self, y, variance_floor=0.01):
        self.y = np.asarray(y, float)
        if self.y.ndim != 3 or not np.isfinite(self.y).all():
            raise ValueError('Expected a complete finite station-by-day-by-time array')
        self.m, self.n, self.t = self.y.shape
        self.floor = variance_floor
        knots = np.r_[np.zeros(4), [.25, .5, .75], np.ones(4)]
        self.x = BSpline(knots, np.eye(7), 3)(np.linspace(0, 1, self.t))
        self.q = self.x.T @ self.x
        self.qinv = np.linalg.inv(self.q)
        self.yx = self.y @ self.x
        self.yy = (self.y ** 2).sum(axis=2)

    def initialize(self, g, h, seed):
        def labels(features, count):
            if count == len(features):
                return np.arange(count)
            if count == 1:
                return np.zeros(len(features), dtype=int)
            return KMeans(n_clusters=count, init='k-means++', n_init=1,
                          max_iter=300, tol=1e-4, random_state=seed).fit_predict(features)
        # Same feature ordering as archived bkmeans2, but one start as recorded
        # in the November reply synchronization note. Features are not scaled.
        row_labels = labels(self.y.transpose(0, 2, 1).reshape(self.m, -1), g)
        day_labels = labels(self.y.transpose(1, 2, 0).reshape(self.n, -1), h)
        phi = .99 * np.eye(g)[row_labels] + .01 / g
        gamma = .99 * np.eye(h)[day_labels] + .01 / h
        s = dict(phi=phi, gamma=gamma, pi=phi.mean(axis=0), rho=gamma.mean(axis=0),
                 beta=np.zeros((g, h, 7)), variance=np.ones((g, h)))
        self.update_blocks(s)
        return s

    def moments(self, s):
        phi, gamma = s['phi'], s['gamma']
        mass = np.outer(phi.sum(axis=0), gamma.sum(axis=0))
        sy = np.einsum('mg,mnk,nh->ghk', phi, self.yx, gamma, optimize=True)
        syy = phi.T @ self.yy @ gamma
        return mass, sy, syy

    def update_blocks(self, s):
        mass, sy, syy = self.moments(s)
        # For an empty expected block its objective is flat; retain its old
        # feasible parameters. This does not remove classes in refinement.
        active = mass > 1e-300
        mean_yx = np.zeros_like(sy)
        mean_yx[active] = sy[active] / mass[active, None]
        beta = mean_yx @ self.qinv
        for g, h in np.argwhere(active & np.any(np.abs(beta) > 20, axis=2)):
            # Equivalent least-squares problem expressed on the common grid.
            target = self.x @ beta[g, h]
            fit = lsq_linear(self.x, target, bounds=(-20, 20), tol=1e-12, max_iter=1000)
            if not fit.success:
                raise ArithmeticError('Bounded coefficient fit failed')
            beta[g, h] = fit.x
        s['beta'][active] = beta[active]
        b = s['beta']
        sse = syy - 2 * np.sum(sy * b, axis=2) + mass * np.einsum('ghk,kl,ghl->gh', b, self.q, b)
        if np.min(sse) < -1e-8:
            raise ArithmeticError('Negative block SSE beyond rounding tolerance')
        s['variance'][active] = np.clip(sse[active] / (mass[active] * self.t), self.floor, 100)

    def logits(self, s, axis):
        inv = 1 / s['variance']
        bv = s['beta'] * inv[:, :, None]
        constant = -.5 * (self.t * np.log(2 * np.pi * s['variance'])
                           + np.einsum('ghk,kl,ghl->gh', s['beta'], self.q, s['beta']) * inv)
        with np.errstate(divide='ignore'):
            if axis == 'row':
                a = -.5 * (self.yy @ s['gamma']) @ inv.T
                a += np.einsum('mnk,nh,ghk->mg', self.yx, s['gamma'], bv, optimize=True)
                a += s['gamma'].sum(axis=0) @ constant.T
                return a + np.log(s['pi'])
            a = -.5 * (self.yy.T @ s['phi']) @ inv
            a += np.einsum('mnk,mg,ghk->nh', self.yx, s['phi'], bv, optimize=True)
            a += s['phi'].sum(axis=0) @ constant
            return a + np.log(s['rho'])

    def objective(self, s, l1, l2):
        mass, sy, syy = self.moments(s)
        b, v = s['beta'], s['variance']
        sse = syy - 2 * np.sum(sy * b, axis=2) + mass * np.einsum('ghk,kl,ghl->gh', b, self.q, b)
        value = -.5 * np.sum(self.t * mass * np.log(2 * np.pi * v) + sse / v)
        for memberships, proportions, size, penalty in [
            (s['phi'], s['pi'], self.m, l1), (s['gamma'], s['rho'], self.n, l2)]:
            value += np.sum(xlogy(memberships, proportions) - xlogy(memberships, memberships))
            value -= size * penalty * np.log1p(proportions / 1e-6).sum()
        return float(value)

    def fit(self, initial, l1, l2, stage, max_iter=150):
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
                    # Re-normalize retained log probabilities to avoid a zero
                    # row caused only by underflow in the stored probabilities.
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
            delta = max(float(np.max(np.abs(s[k] - old[k]))) for k in s) if stable else None
            beta_norm = float(np.linalg.norm(s['beta'] - old['beta'])) if stable else None
            gain = value - old_objective
            if stage == 'refinement' and gain < -1e-9 * (1 + abs(old_objective)):
                raise ArithmeticError(f'Fixed-order MM decreased ELBO by {gain}')
            converged = stable and (beta_norm <= 1e-6 if stage == 'truncation'
                                    else abs(gain) <= 1e-8 * (1 + abs(old_objective)) and delta <= 1e-6)
            trace.append(dict(iteration=iteration, G=len(s['pi']), H=len(s['rho']),
                penalized_elbo=value, increment=gain, maximum_change=delta,
                coefficient_norm_change=beta_norm))
            old_objective = value
            if converged:
                break
        return s, dict(converged=bool(converged), iterations=iteration, trace=trace)

    def score(self, s):
        r, c = s['phi'].argmax(axis=1), s['gamma'].argmax(axis=1)
        b = s['beta'][r[:, None], c[None, :]]
        prediction = b @ self.x.T
        v = s['variance'][r[:, None], c[None, :]]
        ll = -.5 * np.sum(self.t * np.log(2 * np.pi * v) + ((self.y - prediction)**2).sum(axis=2) / v)
        ll += np.log(s['pi'][r]).sum() + np.log(s['rho'][c]).sum()
        g, h = len(s['pi']), len(s['rho'])
        penalty = .5 * ((g-1)*np.log(self.m) + (h-1)*np.log(self.n)
                       + g*h*8*np.log(self.y.size))
        return dict(ICL=float(ll-penalty), complete_loglik=float(ll), G=g, H=h,
            assigned_G=len(np.unique(r)), assigned_H=len(np.unique(c)),
            MSE=float(np.mean((self.y-prediction)**2)), parameters=g*h*8+g+h-2,
            variance_floor_hits=int(np.sum(s['variance'] <= self.floor*(1+1e-10))),
            coefficient_bound_hits=int(np.sum(np.abs(s['beta']) >= 20-1e-8)),
            row_sizes=np.bincount(r, minlength=g).tolist(), day_labels=(c+1).tolist())

    def merge(self, parent, axis):
        s = deepcopy(parent)
        phi, gamma = s['phi'], s['gamma']
        curves = s['beta'] @ self.x.T
        if axis == 'row':
            differences = curves[:, None, :, :] - curves[None, :, :, :]
            distances = np.einsum('abht,h->ab', differences**2, gamma.mean(axis=0)) / self.t
            key, memberships, dimension = 'pi', 'phi', 0
        else:
            differences = curves[:, :, None, :] - curves[:, None, :, :]
            distances = np.einsum('gabt,g->ab', differences**2, phi.mean(axis=0)) / self.t
            key, memberships, dimension = 'rho', 'gamma', 1
        # Lexicographic tie-breaking, independent proposals from one parent.
        pairs = [(float(distances[a,b]), a,b) for a in range(len(s[key])) for b in range(a+1,len(s[key]))]
        distance, a, b = min(pairs)
        s[key][a] += s[key][b]
        s[memberships][:, a] += s[memberships][:, b]
        s[key] = np.delete(s[key], b)
        s[memberships] = np.delete(s[memberships], b, axis=1)
        s['beta'] = np.delete(s['beta'], b, axis=dimension)
        s['variance'] = np.delete(s['variance'], b, axis=dimension)
        self.update_blocks(s)
        return s, dict(axis=axis, pair=[a+1,b+1], distance=distance)
