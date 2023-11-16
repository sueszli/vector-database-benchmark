"""
This module implements Brendel and Bethge attack.
"""
from typing import Union, Optional, Tuple, TYPE_CHECKING
import logging
import numpy as np
from numba.experimental import jitclass
from art import config
from art.utils import get_labels_np_array, check_and_transform_label_format, is_probability
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
EPS = 1e-10
logger = logging.getLogger(__name__)

@jitclass(spec=[])
class BFGSB:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def solve(self, fun_and_jac, q0, bounds, args, ftol=1e-10, pgtol=-1e-05, maxiter=None):
        if False:
            for i in range(10):
                print('nop')
        N = q0.shape[0]
        if maxiter is None:
            maxiter = N * 200
        var_l = bounds[:, 0]
        u = bounds[:, 1]
        func_calls = 0
        (old_fval, gfk) = fun_and_jac(q0, *args)
        func_calls += 1
        k = 0
        Hk = np.eye(N)
        qk = q0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2
        _gfk = gfk
        while k < maxiter:
            pg_norm = 0
            for v in range(N):
                if _gfk[v] < 0:
                    gv = max(qk[v] - u[v], _gfk[v])
                else:
                    gv = min(qk[v] - var_l[v], _gfk[v])
                if pg_norm < np.abs(gv):
                    pg_norm = np.abs(gv)
            if pg_norm < pgtol:
                break
            x_cp = self._cauchy_point(qk, var_l, u, _gfk.copy(), Hk)
            qk1 = self._subspace_min(qk, var_l, u, x_cp, _gfk.copy(), Hk)
            pk = qk1 - qk
            (alpha_k, fc, gc, old_fval, old_old_fval, gfkp1, fnev) = self._line_search_wolfe(fun_and_jac, qk, pk, _gfk, old_fval, old_old_fval, var_l, u, args)
            func_calls += fnev
            if alpha_k is None:
                break
            if np.abs(old_fval - old_old_fval) <= ftol + ftol * np.abs(old_fval):
                break
            qkp1 = self._project(qk + alpha_k * pk, var_l, u)
            if gfkp1 is None:
                (_, gfkp1) = fun_and_jac(qkp1, *args)
            sk = qkp1 - qk
            qk = qkp1
            yk = np.zeros_like(qk)
            for k3 in range(N):
                yk[k3] = gfkp1[k3] - _gfk[k3]
                if np.abs(yk[k3]) < 0.0001:
                    yk[k3] = -0.0001
            _gfk = gfkp1
            k += 1
            Hk_sk = Hk.dot(sk)
            sk_yk = 0
            sk_Hk_sk = 0
            for v in range(N):
                sk_yk += sk[v] * yk[v]
                sk_Hk_sk += sk[v] * Hk_sk[v]
            if np.abs(sk_yk) >= 1e-08:
                rhok = 1.0 / sk_yk
            else:
                rhok = 100000.0
            if np.abs(sk_Hk_sk) >= 1e-08:
                rsk_Hk_sk = 1.0 / sk_Hk_sk
            else:
                rsk_Hk_sk = 100000.0
            for v in range(N):
                for w in range(N):
                    Hk[v, w] += yk[v] * yk[w] * rhok - Hk_sk[v] * Hk_sk[w] * rsk_Hk_sk
        return qk

    def _cauchy_point(self, x, var_l, u, g, B):
        if False:
            for i in range(10):
                print('nop')
        n = x.shape[0]
        t = np.zeros_like(x)
        d = np.zeros_like(x)
        for i in range(n):
            if g[i] < 0:
                t[i] = (x[i] - u[i]) / g[i]
            elif g[i] > 0:
                t[i] = (x[i] - var_l[i]) / g[i]
            elif g[i] == 0:
                t[i] = np.inf
            if t[i] == 0:
                d[i] = 0
            else:
                d[i] = -g[i]
        ts = t.copy()
        ts = ts[ts != 0]
        ts = np.sort(ts)
        df = g.dot(d)
        d2f = d.dot(B.dot(d))
        if d2f < 1e-10:
            return x
        dt_min = -df / d2f
        t_old = 0
        i = 0
        z = np.zeros_like(x)
        while i < ts.shape[0] and dt_min >= ts[i] - t_old:
            ind = ts[i] < t
            d[~ind] = 0
            z = z + (ts[i] - t_old) * d
            df = g.dot(d) + d.dot(B.dot(z))
            d2f = d.dot(B.dot(d))
            dt_min = df / (d2f + 1e-08)
            t_old = ts[i]
            i += 1
        dt_min = max(dt_min, 0)
        t_old = t_old + dt_min
        x_cp = x - t_old * g
        temp = x - t * g
        x_cp[t_old > t] = temp[t_old > t]
        return x_cp

    def _subspace_min(self, x, var_l, u, x_cp, d, G):
        if False:
            print('Hello World!')
        n = x.shape[0]
        Z = np.eye(n)
        fixed = (x_cp <= var_l + 1e-08) + (x_cp >= u - 100000000.0)
        if np.all(fixed):
            x = x_cp
            return x
        Z = Z[:, ~fixed]
        rgc = Z.T.dot(d + G.dot(x_cp - x))
        rB = Z.T.dot(G.dot(Z)) + 1e-10 * np.eye(Z.shape[1])
        d[~fixed] = np.linalg.solve(rB, rgc)
        d[~fixed] = -d[~fixed]
        alpha = 1
        temp1 = alpha
        for i in np.arange(n)[~fixed]:
            dk = d[i]
            if dk < 0:
                temp2 = var_l[i] - x_cp[i]
                if temp2 >= 0:
                    temp1 = 0
                elif dk * alpha < temp2:
                    temp1 = temp2 / dk
                else:
                    temp2 = u[i] - x_cp[i]
            else:
                temp2 = u[i] - x_cp[i]
                if temp1 <= 0:
                    temp1 = 0
                elif dk * alpha > temp2:
                    temp1 = temp2 / dk
            alpha = min(temp1, alpha)
        return x_cp + alpha * Z.dot(d[~fixed])

    def _project(self, q, var_l, u):
        if False:
            return 10
        N = q.shape[0]
        for k in range(N):
            if q[k] < var_l[k]:
                q[k] = var_l[k]
            elif q[k] > u[k]:
                q[k] = u[k]
        return q

    def _line_search_armijo(self, fun_and_jac, pt, dpt, func_calls, m, gk, var_l, u, x0, x, b, min_, max_, c, r):
        if False:
            i = 10
            return i + 15
        ls_rho = 0.6
        ls_c = 0.0001
        ls_alpha = 1
        t = m * ls_c
        for k2 in range(100):
            ls_pt = self._project(pt + ls_alpha * dpt, var_l, u)
            (gkp1, dgkp1) = fun_and_jac(ls_pt, x0, x, b, min_, max_, c, r)
            func_calls += 1
            if gk - gkp1 >= ls_alpha * t:
                break
            else:
                ls_alpha *= ls_rho
        return (ls_alpha, ls_pt, gkp1, dgkp1, func_calls)

    def _line_search_wolfe(self, fun_and_jac, xk, pk, gfk, old_fval, old_old_fval, var_l, u, args):
        if False:
            return 10
        "Find alpha that satisfies strong Wolfe conditions.\n        Uses the line search algorithm to enforce strong Wolfe conditions\n        Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60\n        For the zoom phase it uses an algorithm by\n        Outputs: (alpha0, gc, fc)\n        "
        alpha_star = 0.0
        fval_star = 0.0
        fprime_star = None
        c1 = 0.0001
        c2 = 0.9
        N = xk.shape[0]
        _ls_fc = 0
        _ls_ingfk = None
        alpha0 = 0
        phi0 = old_fval
        derphi0 = 0
        for v in range(N):
            derphi0 += gfk[v] * pk[v]
        if derphi0 == 0:
            derphi0 = 1e-08
        elif np.abs(derphi0) < 1e-08:
            derphi0 = np.sign(derphi0) * 1e-08
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_old_fval) / derphi0)
        if alpha1 == 0:
            alpha_star = None
            fval_star = old_fval
            old_fval = old_old_fval
            fprime_star = None
        _xkp1 = self._project(xk + alpha1 * pk, var_l, u)
        (phi_a1, _ls_ingfk) = fun_and_jac(_xkp1, *args)
        _ls_fc += 1
        phi_a0 = phi0
        derphi_a0 = derphi0
        i = 1
        maxiter = 10
        while 1:
            if alpha1 == 0:
                break
            if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and i > 1):
                k = 0
                delta1 = 0.2
                delta2 = 0.1
                phi_rec = phi0
                a_rec = 0
                a_hi = alpha1
                a_lo = alpha0
                phi_lo = phi_a0
                phi_hi = phi_a1
                derphi_lo = derphi_a0
                while 1:
                    dalpha = a_hi - a_lo
                    if dalpha < 0:
                        (a, b) = (a_hi, a_lo)
                    else:
                        (a, b) = (a_lo, a_hi)
                    if k > 0:
                        cchk = delta1 * dalpha
                        a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
                    if k == 0 or a_j is None or a_j > b - cchk or (a_j < a + cchk):
                        qchk = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if a_j is None or a_j > b - qchk or a_j < a + qchk:
                            a_j = a_lo + 0.5 * dalpha
                    _xkp1 = self._project(xk + a_j * pk, var_l, u)
                    (phi_aj, _ls_ingfk) = fun_and_jac(_xkp1, *args)
                    derphi_aj = 0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]
                    if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break
                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                _ls_fc += fnev
                break
            i += 1
            if i > maxiter:
                break
            _xkp1 = self._project(xk + alpha1 * pk, var_l, u)
            (_, _ls_ingfk) = fun_and_jac(_xkp1, *args)
            derphi_a1 = 0
            for v in range(N):
                derphi_a1 += _ls_ingfk[v] * pk[v]
            _ls_fc += 1
            if abs(derphi_a1) <= -c2 * derphi0:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = _ls_ingfk
                break
            if derphi_a1 >= 0:
                maxiter = 10
                k = 0
                delta1 = 0.2
                delta2 = 0.1
                phi_rec = phi0
                a_rec = 0
                a_hi = alpha0
                a_lo = alpha1
                phi_lo = phi_a1
                phi_hi = phi_a0
                derphi_lo = derphi_a1
                while 1:
                    dalpha = a_hi - a_lo
                    if dalpha < 0:
                        (a, b) = (a_hi, a_lo)
                    else:
                        (a, b) = (a_lo, a_hi)
                    if k > 0:
                        cchk = delta1 * dalpha
                        a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
                    if k == 0 or a_j is None or a_j > b - cchk or (a_j < a + cchk):
                        qchk = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if a_j is None or a_j > b - qchk or a_j < a + qchk:
                            a_j = a_lo + 0.5 * dalpha
                    _xkp1 = self._project(xk + a_j * pk, var_l, u)
                    (phi_aj, _ls_ingfk) = fun_and_jac(_xkp1, *args)
                    derphi_aj = 0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]
                    if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break
                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                _ls_fc += fnev
                break
            alpha2 = 2 * alpha1
            i = i + 1
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            _xkp1 = self._project(xk + alpha1 * pk, var_l, u)
            (phi_a1, _ls_ingfk) = fun_and_jac(_xkp1, *args)
            _ls_fc += 1
            derphi_a0 = derphi_a1
            if i > maxiter:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = None
                break
        return (alpha_star, _ls_fc, _ls_fc, fval_star, old_fval, fprime_star, _ls_fc)

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        if False:
            while True:
                i = 10
        C = fpa
        db = b - a
        dc = c - a
        if db == 0 or dc == 0 or b == c:
            return None
        denom = (db * dc) ** 2 * (db - dc)
        A = dc ** 2 * (fb - fa - C * db) - db ** 2 * (fc - fa - C * dc)
        B = -dc ** 3 * (fb - fa - C * db) + db ** 3 * (fc - fa - C * dc)
        A /= denom
        B /= denom
        radical = B * B - 3 * A * C
        if radical < 0:
            return None
        if A == 0:
            return None
        xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        if False:
            for i in range(10):
                print('nop')
        D = fa
        C = fpa
        db = b - a * 1.0
        if db == 0:
            return None
        B = (fb - D - C * db) / (db * db)
        if B <= 0:
            return None
        xmin = a - C / (2.0 * B)
        return xmin

class Optimizer:
    """
    Base class for the trust-region optimization. If feasible, this optimizer solves the problem

    min_delta distance(x0, x + delta) s.t. ||delta||_2 <= r AND delta^T b = c AND min_ <= x + delta <= max_

    where x0 is the original sample, x is the current optimisation state, r is the trust-region radius,
    b is the current estimate of the normal vector of the decision boundary, c is the estimated distance of x
    to the trust region and [min_, max_] are the value constraints of the input. The function distance(.,.)
    is the distance measure to be optimised (e.g. L2, L1, L0).
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.bfgsb = BFGSB()

    def solve(self, x0, x, b, min_, max_, c, r):
        if False:
            for i in range(10):
                print('nop')
        (x0, x, b) = (x0.astype(np.float64), x.astype(np.float64), b.astype(np.float64))
        (cmax, cmaxnorm) = self._max_logit_diff(x, b, min_, max_, c)
        if np.abs(cmax) < np.abs(c):
            if np.sqrt(cmaxnorm) < r:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
            else:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
        elif cmaxnorm < r:
            _delta = self.optimize_distance_s_t_boundary_and_trustregion(x0, x, b, min_, max_, c, r)
        else:
            bnorm = np.linalg.norm(b)
            minnorm = self._minimum_norm_to_boundary(x, b, min_, max_, c, bnorm)
            if minnorm <= r:
                _delta = self.optimize_distance_s_t_boundary_and_trustregion(x0, x, b, min_, max_, c, r)
            else:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
        return _delta

    def _max_logit_diff(self, x, b, _ell, _u, c):
        if False:
            return 10
        '\n        Tests whether the (estimated) boundary can be reached within trust region.\n        '
        N = x.shape[0]
        cmax = 0.0
        norm = 0.0
        if c > 0:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_u - x[n])
                    norm += (_u - x[n]) ** 2
                else:
                    cmax += b[n] * (_ell - x[n])
                    norm += (x[n] - _ell) ** 2
        else:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_ell - x[n])
                    norm += (x[n] - _ell) ** 2
                else:
                    cmax += b[n] * (_u - x[n])
                    norm += (_u - x[n]) ** 2
        return (cmax, np.sqrt(norm))

    def _minimum_norm_to_boundary(self, x, b, _ell, _u, c, bnorm):
        if False:
            i = 10
            return i + 15
        '\n        Computes the minimum norm necessary to reach the boundary. More precisely, we aim to solve the following\n        optimization problem\n\n            min ||delta||_2^2 s.t. lower <= x + delta <= upper AND b.dot(delta) = c\n\n        Lets forget about the box constraints for a second, i.e.\n\n            min ||delta||_2^2 s.t. b.dot(delta) = c\n\n        The dual of this problem is quite straight-forward to solve,\n\n            g(lambda, delta) = ||delta||_2^2 + lambda * (c - b.dot(delta))\n\n        The minimum of this Lagrangian is delta^* = lambda * b / 2, and so\n\n            inf_delta g(lambda, delta) = lambda^2 / 4 ||b||_2^2 + lambda * c\n\n        and so the optimal lambda, which maximizes inf_delta g(lambda, delta), is given by\n\n            lambda^* = 2c / ||b||_2^2\n\n        which in turn yields the optimal delta:\n\n            delta^* = c * b / ||b||_2^2\n\n        To take into account the box-constraints we perform a binary search over lambda and apply the box\n        constraint in each step.\n        '
        N = x.shape[0]
        lambda_lower = 2 * c / (bnorm ** 2 + EPS)
        lambda_upper = np.sign(c) * np.inf
        _lambda = lambda_lower
        k = 0
        while True:
            k += 1
            _c = 0
            norm = 0
            if c > 0:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _u - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
            else:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _u - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
            if np.abs(_c) < np.abs(c):
                if np.isinf(lambda_upper):
                    _lambda *= 2
                else:
                    lambda_lower = _lambda
                    _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower
            else:
                lambda_upper = _lambda
                _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower
            if 0.999 * np.abs(c) - EPS < np.abs(_c) < 1.001 * np.abs(c) + EPS:
                break
        return np.sqrt(norm)

    def optimize_distance_s_t_boundary_and_trustregion(self, x0, x, b, min_, max_, c, r):
        if False:
            while True:
                i = 10
        '\n        Find the solution to the optimization problem\n\n        min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_\n        '
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        args = (x0, x, b, min_, max_, c, r)
        qk = self.bfgsb.solve(self.fun_and_jac, params0, bounds, args)
        return self._get_final_delta(qk[0], qk[1], x0, x, b, min_, max_, c, r, touchup=True)

    def optimize_boundary_s_t_trustregion_fun_and_jac(self, params, x0, x, b, min_, max_, c, r):
        if False:
            for i in range(10):
                print('nop')
        N = x0.shape[0]
        s = -np.sign(c)
        _mu = params[0]
        t = 1 / (2 * _mu + EPS)
        g = -_mu * r ** 2
        grad_mu = -r ** 2
        for n in range(N):
            d = -s * b[n] * t
            if d < min_ - x[n]:
                d = min_ - x[n]
            elif d > max_ - x[n]:
                d = max_ - x[n]
            else:
                grad_mu += (b[n] + 2 * _mu * d) * (b[n] / (2 * _mu ** 2 + EPS))
            grad_mu += d ** 2
            g += (b[n] + _mu * d) * d
        return (-g, -np.array([grad_mu]))

    def safe_div(self, nominator, denominator):
        if False:
            for i in range(10):
                print('nop')
        if np.abs(denominator) > EPS:
            return nominator / denominator
        elif denominator >= 0:
            return nominator / EPS
        else:
            return -nominator / EPS

    def optimize_boundary_s_t_trustregion(self, x0, x, b, min_, max_, c, r):
        if False:
            while True:
                i = 10
        '\n        Find the solution to the optimization problem\n\n        min_delta sign(c) b^T delta s.t. ||delta||_2^2 <= r^2 AND min_ <= x + delta <= max_\n\n        Note: this optimization problem is independent of the Lp norm being optimized.\n\n        Lagrangian: g(delta) = sign(c) b^T delta + mu * (||delta||_2^2 - r^2)\n        Optimal delta: delta = - sign(c) * b / (2 * mu)\n        '
        params0 = np.array([1.0])
        args = (x0, x, b, min_, max_, c, r)
        bounds = np.array([(0, np.inf)])
        qk = self.bfgsb.solve(self.optimize_boundary_s_t_trustregion_fun_and_jac, params0, bounds, args)
        _delta = self.safe_div(-b, 2 * qk[0])
        for n in range(x0.shape[0]):
            if _delta[n] < min_ - x[n]:
                _delta[n] = min_ - x[n]
            elif _delta[n] > max_ - x[n]:
                _delta[n] = max_ - x[n]
        return _delta
spec = [('bfgsb', BFGSB.class_type.instance_type)]

@jitclass(spec=spec)
class L2Optimizer(Optimizer):

    def optimize_distance_s_t_boundary_and_trustregion(self, x0, x, b, min_, max_, c, r):
        if False:
            i = 10
            return i + 15
        '\n        Solves the L2 trust region problem\n\n        min ||x0 - x - delta||_2 s.t. b^top delta = c\n                                    & ell <= x + delta <= u\n                                    & ||delta||_2 <= r\n\n        This is a specialised solver that does not use the generic BFGS-B solver.\n        Instead, this active-set solver computes the active set of indices (those that\n        do not hit the bounds) and then computes that optimal step size in the direction\n        of the boundary and the direction of the original sample over the active indices.\n\n        Parameters\n        ----------\n        x0 : `numpy.ndarray`\n            The original image against which we minimize the perturbation\n            (flattened).\n        x : `numpy.ndarray`\n            The current perturbation (flattened).\n        b : `numpy.ndarray`\n            Normal vector of the local decision boundary (flattened).\n        min_ : float\n            Lower bound on the pixel values.\n        max_ : float\n            Upper bound on the pixel values.\n        c : float\n            Logit difference between the ground truth class of x0 and the\n            leading class different from the ground truth.\n        r : float\n            Size of the trust region.\n        '
        N = x0.shape[0]
        clamp_c = 0
        clamp_norm = 0
        ck = c
        rk = r
        masked_values = 0
        mask = np.zeros(N, dtype=np.uint8)
        delta = np.empty_like(x0)
        dx = x0 - x
        for k in range(20):
            bnorm = 1e-08
            bdotDx = 0
            for i in range(N):
                if mask[i] == 0:
                    bnorm += b[i] * b[i]
                    bdotDx += b[i] * dx[i]
            bdotDx = bdotDx / bnorm
            ck_bnorm = ck / bnorm
            b_scale = -bdotDx + ck / bnorm
            new_masked_values = 0
            delta_norm = 0
            descent_norm = 0
            boundary_step_norm = 0
            for i in range(N):
                if mask[i] == 0:
                    delta[i] = dx[i] + b[i] * b_scale
                    boundary_step_norm = boundary_step_norm + b[i] * ck_bnorm * b[i] * ck_bnorm
                    delta_norm = delta_norm + delta[i] * delta[i]
                    descent_norm = descent_norm + (dx[i] - b[i] * bdotDx) * (dx[i] - b[i] * bdotDx)
            if boundary_step_norm > rk * rk:
                for i in range(N):
                    if mask[i] == 0:
                        delta[i] = b[i] * ck_bnorm
            elif delta_norm > rk * rk:
                region_correct = np.sqrt(rk * rk - boundary_step_norm)
                region_correct = region_correct / (np.sqrt(descent_norm) + 1e-08)
                b_scale = -region_correct * bdotDx + ck / bnorm
                for i in range(N):
                    if mask[i] == 0:
                        delta[i] = region_correct * dx[i] + b[i] * b_scale
            for i in range(N):
                if mask[i] == 0:
                    if x[i] + delta[i] <= min_:
                        mask[i] = 1
                        delta[i] = min_ - x[i]
                        new_masked_values = new_masked_values + 1
                        clamp_norm = clamp_norm + delta[i] * delta[i]
                        clamp_c = clamp_c + b[i] * delta[i]
                    if x[i] + delta[i] >= max_:
                        mask[i] = 1
                        delta[i] = max_ - x[i]
                        new_masked_values = new_masked_values + 1
                        clamp_norm = clamp_norm + delta[i] * delta[i]
                        clamp_c = clamp_c + b[i] * delta[i]
            if new_masked_values == 0:
                break
            masked_values = masked_values + new_masked_values
            if clamp_norm < r * r:
                rk = np.sqrt(r * r - clamp_norm)
            else:
                rk = 0
            ck = c - clamp_c
            if masked_values == N:
                break
        return delta

    def fun_and_jac(self, params, x0, x, b, min_, max_, c, r):
        if False:
            while True:
                i = 10
        (lam, mu) = params
        N = x0.shape[0]
        g = 0
        d_g_d_lam = 0
        d_g_d_mu = 0
        distance = 0
        b_dot_d = 0
        d_norm = 0
        t = 1 / (2 * mu + 2)
        for n in range(N):
            dx = x0[n] - x[n]
            bn = b[n]
            xn = x[n]
            d = (2 * dx - lam * bn) * t
            if d + xn > max_:
                d = max_ - xn
            elif d + xn < min_:
                d = min_ - xn
            else:
                prefac = 2 * (d - dx) + 2 * mu * d + lam * bn
                d_g_d_lam -= prefac * bn * t
                d_g_d_mu -= prefac * 2 * d * t
            distance += (d - dx) ** 2
            b_dot_d += bn * d
            d_norm += d ** 2
            g += (dx - d) ** 2 + mu * d ** 2 + lam * bn * d
            d_g_d_lam += bn * d
            d_g_d_mu += d ** 2
        g += -mu * r ** 2 - lam * c
        d_g_d_lam -= c
        d_g_d_mu -= r ** 2
        return (-g, -np.array([d_g_d_lam, d_g_d_mu]))

    def _get_final_delta(self, lam, mu, x0, x, b, min_, max_, c, r, touchup=True):
        if False:
            while True:
                i = 10
        delta = np.empty_like(x0)
        N = x0.shape[0]
        t = 1 / (2 * mu + 2)
        for n in range(N):
            d = (2 * (x0[n] - x[n]) - lam * b[n]) * t
            if d + x[n] > max_:
                d = max_ - x[n]
            elif d + x[n] < min_:
                d = min_ - x[n]
            delta[n] = d
        return delta

    def _distance(self, x0, x):
        if False:
            for i in range(10):
                print('nop')
        return np.linalg.norm(x0 - x) ** 2

@jitclass(spec=spec)
class L1Optimizer(Optimizer):

    def fun_and_jac(self, params, x0, x, b, min_, max_, c, r):
        if False:
            while True:
                i = 10
        (lam, mu) = params
        N = x0.shape[0]
        g = 0
        d_g_d_lam = 0
        d_g_d_mu = 0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t = 1 / (2 * mu)
                u = -lam * bn * t - dx
                if np.abs(u) - t < 0:
                    d = dx
                else:
                    d = np.sign(u) * (np.abs(u) - t) + dx
                    if d + x[n] < min_:
                        d = min_ - x[n]
                    elif d + x[n] > max_:
                        d = max_ - x[n]
                    else:
                        prefac = np.sign(d - dx) + 2 * mu * d + lam * bn
                        d_g_d_lam -= prefac * bn * t
                        d_g_d_mu -= prefac * 2 * d * t
                g += np.abs(dx - d) + mu * d ** 2 + lam * bn * d
                d_g_d_lam += bn * d
                d_g_d_mu += d ** 2
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                if np.abs(lam * bn) < 1:
                    d = dx
                elif np.sign(lam * bn) < 0:
                    d = max_ - x[n]
                else:
                    d = min_ - x[n]
                g += np.abs(dx - d) + mu * d ** 2 + lam * bn * d
                d_g_d_lam += bn * d
                d_g_d_mu += d ** 2
        g += -mu * r ** 2 - lam * c
        d_g_d_lam -= c
        d_g_d_mu -= r ** 2
        return (-g, -np.array([d_g_d_lam, d_g_d_mu]))

    def _get_final_delta(self, lam, mu, x0, x, b, min_, max_, c, r, touchup=True):
        if False:
            print('Hello World!')
        delta = np.empty_like(x0)
        N = x0.shape[0]
        b_dot_d = 0
        norm_d = 0
        distance = 0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t = 1 / (2 * mu)
                u = -lam * bn * t - dx
                if np.abs(u) - t < 0:
                    d = dx
                else:
                    d = np.sign(u) * (np.abs(u) - t) + dx
                    if d + x[n] < min_:
                        d = min_ - x[n]
                    elif d + x[n] > max_:
                        d = max_ - x[n]
                delta[n] = d
                b_dot_d += b[n] * d
                norm_d += d ** 2
                distance += np.abs(d - dx)
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                if np.abs(lam * bn) < 1:
                    d = dx
                elif np.sign(lam * bn) < 0:
                    d = max_ - x[n]
                else:
                    d = min_ - x[n]
                delta[n] = d
                b_dot_d += b[n] * d
                norm_d += d ** 2
                distance += np.abs(d - dx)
        if touchup:
            dc = c - b_dot_d
            k = 0
            min_distance = np.inf
            min_distance_idx = 0
            for n in range(N):
                if np.abs(b[n]) > 0:
                    dx = x0[n] - x[n]
                    old_d = delta[n]
                    new_d = old_d + dc / b[n]
                    if x[n] + new_d <= max_ and x[n] + new_d >= min_ and (norm_d - old_d ** 2 + new_d ** 2 <= r ** 2):
                        if k == 0:
                            min_distance = distance - np.abs(old_d - dx) + np.abs(new_d - dx)
                            min_distance_idx = n
                            k += 1
                        else:
                            new_distance = distance - np.abs(old_d - dx) + np.abs(new_d - dx)
                            if min_distance > new_distance:
                                min_distance = new_distance
                                min_distance_idx = n
            if k > 0:
                idx = min_distance_idx
                old_d = delta[idx]
                new_d = old_d + dc / b[idx]
                delta[idx] = new_d
        return delta

    def _distance(self, x0, x):
        if False:
            while True:
                i = 10
        return np.abs(x0 - x).sum()

@jitclass(spec=spec)
class LinfOptimizer(Optimizer):

    def optimize_distance_s_t_boundary_and_trustregion(self, x0, x, b, min_, max_, c, r):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the solution to the optimization problem\n\n        min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_\n        '
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        return self.binary_search(params0, bounds, x0, x, b, min_, max_, c, r)

    def binary_search(self, q0, bounds, x0, x, b, min_, max_, c, r, etol=1e-06, maxiter=1000):
        if False:
            i = 10
            return i + 15
        epsilon = (max_ - min_) / 2.0
        eps_low = min_
        eps_high = max_
        func_calls = 0
        bnorm = np.linalg.norm(b)
        lambda0 = 2 * c / bnorm ** 2
        k = 0
        while eps_high - eps_low > etol:
            (fun, nfev, _lambda0) = self.fun(epsilon, x0, x, b, min_, max_, c, r, lambda0=lambda0)
            func_calls += nfev
            if fun > -np.inf:
                eps_high = epsilon
                lambda0 = _lambda0
            else:
                eps_low = epsilon
            k += 1
            epsilon = (eps_high - eps_low) / 2.0 + eps_low
            if k > 20:
                break
        delta = self._get_final_delta(lambda0, eps_high, x0, x, b, min_, max_, c, r, touchup=True)
        return delta

    def _Linf_bounds(self, x0, epsilon, ell, u):
        if False:
            for i in range(10):
                print('nop')
        N = x0.shape[0]
        _ell = np.empty_like(x0)
        _u = np.empty_like(x0)
        for i in range(N):
            (nx, px) = (x0[i] - epsilon, x0[i] + epsilon)
            if nx > ell:
                _ell[i] = nx
            else:
                _ell[i] = ell
            if px < u:
                _u[i] = px
            else:
                _u[i] = u
        return (_ell, _u)

    def fun(self, epsilon, x0, x, b, ell, u, c, r, lambda0=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the minimum norm necessary to reach the boundary. More precisely, we aim to solve the following\n        optimization problem\n\n            min ||delta||_2^2 s.t. lower <= x + delta <= upper AND b.dot(delta) = c\n\n        Lets forget about the box constraints for a second, i.e.\n\n            min ||delta||_2^2 s.t. b.dot(delta) = c\n\n        The dual of this problem is quite straight-forward to solve,\n\n            g(lambda, delta) = ||delta||_2^2 + lambda * (c - b.dot(delta))\n\n        The minimum of this Lagrangian is delta^* = lambda * b / 2, and so\n\n            inf_delta g(lambda, delta) = lambda^2 / 4 ||b||_2^2 + lambda * c\n\n        and so the optimal lambda, which maximizes inf_delta g(lambda, delta), is given by\n\n            lambda^* = 2c / ||b||_2^2\n\n        which in turn yields the optimal delta:\n\n            delta^* = c * b / ||b||_2^2\n\n        To take into account the box-constraints we perform a binary search over lambda and apply the box\n        constraint in each step.\n        '
        N = x.shape[0]
        (_ell, _u) = self._Linf_bounds(x0, epsilon, ell, u)
        _lambda = lambda0
        k = 0
        (lambda_max, lambda_min) = (10000000000.0, -10000000000.0)
        max_c = 0
        min_c = 0
        for n in range(N):
            if b[n] > 0:
                max_c += b[n] * (_u[n] - x[n])
                min_c += b[n] * (_ell[n] - x[n])
            else:
                max_c += b[n] * (_ell[n] - x[n])
                min_c += b[n] * (_u[n] - x[n])
        if c > max_c or c < min_c:
            return (-np.inf, k, _lambda)
        while True:
            k += 1
            _c = 0
            norm = 0
            _active_bnorm = 0
            for n in range(N):
                lam_step = _lambda * b[n] / 2
                if lam_step + x[n] < _ell[n]:
                    delta_step = _ell[n] - x[n]
                elif lam_step + x[n] > _u[n]:
                    delta_step = _u[n] - x[n]
                else:
                    delta_step = lam_step
                    _active_bnorm += b[n] ** 2
                _c += b[n] * delta_step
                norm += delta_step ** 2
            if 0.9999 * np.abs(c) - EPS < np.abs(_c) < 1.0001 * np.abs(c) + EPS:
                if norm > r ** 2:
                    return (-np.inf, k, _lambda)
                else:
                    return (-epsilon, k, _lambda)
            else:
                if _c > c:
                    lambda_max = _lambda
                else:
                    lambda_min = _lambda
                if _active_bnorm == 0:
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min
                else:
                    _lambda += 2 * (c - _c) / _active_bnorm
                dlambda = lambda_max - lambda_min
                if _lambda > lambda_max - 0.1 * dlambda or _lambda < lambda_min + 0.1 * dlambda:
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min

    def _get_final_delta(self, lam, eps, x0, x, b, min_, max_, c, r, touchup=True):
        if False:
            return 10
        N = x.shape[0]
        delta = np.empty_like(x0)
        (_ell, _u) = self._Linf_bounds(x0, eps, min_, max_)
        for n in range(N):
            lam_step = lam * b[n] / 2
            if lam_step + x[n] < _ell[n]:
                delta[n] = _ell[n] - x[n]
            elif lam_step + x[n] > _u[n]:
                delta[n] = _u[n] - x[n]
            else:
                delta[n] = lam_step
        return delta

    def _distance(self, x0, x):
        if False:
            print('Hello World!')
        return np.abs(x0 - x).max()

@jitclass(spec=spec)
class L0Optimizer(Optimizer):

    def optimize_distance_s_t_boundary_and_trustregion(self, x0, x, b, min_, max_, c, r):
        if False:
            i = 10
            return i + 15
        '\n        Find the solution to the optimization problem\n\n        min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_\n        '
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        return self.minimize(params0, bounds, x0, x, b, min_, max_, c, r)

    def minimize(self, q0, bounds, x0, x, b, min_, max_, c, r, ftol=1e-09, xtol=-1e-05, maxiter=1000):
        if False:
            while True:
                i = 10
        (delta, delta_norm) = self.minimize_without_trustregion(x0, x, b, c, r, min_, max_)
        if delta_norm <= r:
            return delta
        else:
            args = (x0, x, b, min_, max_, c, r)
            results = self._nelder_mead_algorithm(q0, bounds, args=args, tol_f=ftol, tol_x=xtol, max_iter=maxiter)
            delta = self._get_final_delta(results[0], results[1], x0, x, b, min_, max_, c, r, touchup=True)
        return delta

    def minimize_without_trustregion(self, x0, x, b, c, r, ell, u):
        if False:
            return 10
        delta = x0 - x
        total = np.empty_like(x0)
        total_b = np.empty_like(x0)
        bdotdelta = b.dot(delta)
        delta_bdotdelta = c - bdotdelta
        for k in range(x0.shape[0]):
            if b[k] > 0 and delta_bdotdelta > 0:
                total_b[k] = (u - x0[k]) * b[k]
                total[k] = u - x0[k]
            elif b[k] > 0 and delta_bdotdelta < 0:
                total_b[k] = (ell - x0[k]) * b[k]
                total[k] = ell - x0[k]
            elif b[k] < 0 and delta_bdotdelta > 0:
                total_b[k] = (ell - x0[k]) * b[k]
                total[k] = ell - x0[k]
            else:
                total_b[k] = (u - x0[k]) * b[k]
                total[k] = u - x0[k]
        b_argsort = np.argsort(np.abs(total_b))[::-1]
        for idx in b_argsort:
            if np.abs(c - bdotdelta) > np.abs(total_b[idx]):
                delta[idx] += total[idx]
                bdotdelta += total_b[idx]
            else:
                delta[idx] += (c - bdotdelta) / (b[idx] + 1e-20)
                break
        delta_norm = np.linalg.norm(delta)
        return (delta, delta_norm)

    def _nelder_mead_algorithm(self, q0, bounds, args=(), ρ=1.0, χ=2.0, γ=0.5, σ=0.5, tol_f=1e-08, tol_x=1e-08, max_iter=1000):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implements the Nelder-Mead algorithm described in Lagarias et al. (1998)\n        modified to maximize instead of minimizing.\n\n        Parameters\n        ----------\n        vertices : ndarray(float, ndim=2)\n            Initial simplex with shape (n+1, n) to be modified in-place.\n\n        args : tuple, optional\n            Extra arguments passed to the objective function.\n\n        ρ : scalar(float), optional(default=1.)\n            Reflection parameter. Must be strictly greater than 0.\n\n        χ : scalar(float), optional(default=2.)\n            Expansion parameter. Must be strictly greater than max(1, ρ).\n\n        γ : scalar(float), optional(default=0.5)\n            Contraction parameter. Must be stricly between 0 and 1.\n\n        σ : scalar(float), optional(default=0.5)\n            Shrinkage parameter. Must be strictly between 0 and 1.\n\n        tol_f : scalar(float), optional(default=1e-10)\n            Tolerance to be used for the function value convergence test.\n\n        tol_x : scalar(float), optional(default=1e-10)\n            Tolerance to be used for the function domain convergence test.\n\n        max_iter : scalar(float), optional(default=1000)\n            The maximum number of allowed iterations.\n\n        Returns\n        ----------\n        x : Approximate solution\n\n        '
        vertices = self._initialize_simplex(q0)
        n = vertices.shape[1]
        self._check_params(ρ, χ, γ, σ, bounds, n)
        nit = 0
        ργ = ρ * γ
        ρχ = ρ * χ
        σ_n = σ ** n
        f_val = np.empty(n + 1, dtype=np.float64)
        for i in range(n + 1):
            f_val[i] = self._neg_bounded_fun(bounds, vertices[i], args=args)
        sort_ind = f_val.argsort()
        LV_ratio = 1
        x_bar = vertices[sort_ind[:n]].sum(axis=0) / n
        while True:
            shrink = False
            fail = nit >= max_iter
            best_val_idx = sort_ind[0]
            worst_val_idx = sort_ind[n]
            term_f = f_val[worst_val_idx] - f_val[best_val_idx] < tol_f
            term_x = LV_ratio < tol_x
            if term_x or term_f or fail:
                break
            x_r = x_bar + ρ * (x_bar - vertices[worst_val_idx])
            f_r = self._neg_bounded_fun(bounds, x_r, args=args)
            if f_r >= f_val[best_val_idx] and f_r < f_val[sort_ind[n - 1]]:
                vertices[worst_val_idx] = x_r
                LV_ratio *= ρ
            elif f_r < f_val[best_val_idx]:
                x_e = x_bar + χ * (x_r - x_bar)
                f_e = self._neg_bounded_fun(bounds, x_e, args=args)
                if f_e < f_r:
                    vertices[worst_val_idx] = x_e
                    LV_ratio *= ρχ
                else:
                    vertices[worst_val_idx] = x_r
                    LV_ratio *= ρ
            else:
                if f_r < f_val[worst_val_idx]:
                    x_c = x_bar + γ * (x_r - x_bar)
                    LV_ratio_update = ργ
                else:
                    x_c = x_bar - γ * (x_r - x_bar)
                    LV_ratio_update = γ
                f_c = self._neg_bounded_fun(bounds, x_c, args=args)
                if f_c < min(f_r, f_val[worst_val_idx]):
                    vertices[worst_val_idx] = x_c
                    LV_ratio *= LV_ratio_update
                else:
                    shrink = True
                    for i in sort_ind[1:]:
                        vertices[i] = vertices[best_val_idx] + σ * (vertices[i] - vertices[best_val_idx])
                        f_val[i] = self._neg_bounded_fun(bounds, vertices[i], args=args)
                    sort_ind[1:] = f_val[sort_ind[1:]].argsort() + 1
                    x_bar = vertices[best_val_idx] + σ * (x_bar - vertices[best_val_idx]) + (vertices[worst_val_idx] - vertices[sort_ind[n]]) / n
                    LV_ratio *= σ_n
            if not shrink:
                f_val[worst_val_idx] = self._neg_bounded_fun(bounds, vertices[worst_val_idx], args=args)
                for (i, j) in enumerate(sort_ind):
                    if f_val[worst_val_idx] < f_val[j]:
                        sort_ind[i + 1:] = sort_ind[i:-1]
                        sort_ind[i] = worst_val_idx
                        break
                x_bar += (vertices[worst_val_idx] - vertices[sort_ind[n]]) / n
            nit += 1
        return vertices[sort_ind[0]]

    def _initialize_simplex(self, x0):
        if False:
            return 10
        '\n        Generates an initial simplex for the Nelder-Mead method.\n\n        Parameters\n        ----------\n        x0 : ndarray(float, ndim=1)\n            Initial guess. Array of real elements of size (n,), where ‘n’ is the\n            number of independent variables.\n\n        bounds: ndarray(float, ndim=2)\n            Sequence of (min, max) pairs for each element in x0.\n\n        Returns\n        ----------\n        vertices : ndarray(float, ndim=2)\n            Initial simplex with shape (n+1, n).\n        '
        n = x0.size
        vertices = np.empty((n + 1, n), dtype=np.float64)
        vertices[:] = x0
        nonzdelt = 0.05
        zdelt = 0.00025
        for i in range(n):
            if vertices[i + 1, i] != 0.0:
                vertices[i + 1, i] *= 1 + nonzdelt
            else:
                vertices[i + 1, i] = zdelt
        return vertices

    def _check_params(self, ρ, χ, γ, σ, bounds, n):
        if False:
            print('Hello World!')
        '\n        Checks whether the parameters for the Nelder-Mead algorithm are valid.\n        JIT-compiled in `nopython` mode using Numba.\n\n        Parameters\n        ----------\n        ρ : scalar(float)\n            Reflection parameter. Must be strictly greater than 0.\n\n        χ : scalar(float)\n            Expansion parameter. Must be strictly greater than max(1, ρ).\n\n        γ : scalar(float)\n            Contraction parameter. Must be stricly between 0 and 1.\n\n        σ : scalar(float)\n            Shrinkage parameter. Must be strictly between 0 and 1.\n\n        bounds: ndarray(float, ndim=2)\n            Sequence of (min, max) pairs for each element in x.\n\n        n : scalar(int)\n            Number of independent variables.\n        '
        if ρ < 0:
            raise ValueError('ρ must be strictly greater than 0.')
        if χ < 1:
            raise ValueError('χ must be strictly greater than 1.')
        if χ < ρ:
            raise ValueError('χ must be strictly greater than ρ.')
        if γ < 0 or γ > 1:
            raise ValueError('γ must be strictly between 0 and 1.')
        if σ < 0 or σ > 1:
            raise ValueError('σ must be strictly between 0 and 1.')
        if bounds.shape not in ((0, 2), (n, 2)):
            raise ValueError('The shape of `bounds` is not valid.')
        if (np.atleast_2d(bounds)[:, 0] > np.atleast_2d(bounds)[:, 1]).any():
            raise ValueError('Lower bounds must be greater than upper bounds.')

    def _check_bounds(self, x, bounds):
        if False:
            print('Hello World!')
        '\n        Checks whether `x` is within `bounds`. JIT-compiled in `nopython` mode\n        using Numba.\n\n        Parameters\n        ----------\n        x : ndarray(float, ndim=1)\n            1-D array with shape (n,) of independent variables.\n\n        bounds: ndarray(float, ndim=2)\n            Sequence of (min, max) pairs for each element in x.\n\n        Returns\n        ----------\n        bool\n            `True` if `x` is within `bounds`, `False` otherwise.\n\n        '
        if bounds.shape == (0, 2):
            return True
        else:
            return (np.atleast_2d(bounds)[:, 0] <= x).all() and (x <= np.atleast_2d(bounds)[:, 1]).all()

    def _neg_bounded_fun(self, bounds, x, args=()):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper for bounding and taking the negative of `fun` for the\n        Nelder-Mead algorithm. JIT-compiled in `nopython` mode using Numba.\n\n        Parameters\n        ----------\n        bounds: ndarray(float, ndim=2)\n            Sequence of (min, max) pairs for each element in x.\n\n        x : ndarray(float, ndim=1)\n            1-D array with shape (n,) of independent variables at which `fun` is\n            to be evaluated.\n\n        args : tuple, optional\n            Extra arguments passed to the objective function.\n\n        Returns\n        ----------\n        scalar\n            `-fun(x, *args)` if x is within `bounds`, `np.inf` otherwise.\n\n        '
        if self._check_bounds(x, bounds):
            return -self.fun(x, *args)
        else:
            return np.inf

    def fun(self, params, x0, x, b, min_, max_, c, r):
        if False:
            return 10
        (lam, mu) = params
        N = x0.shape[0]
        g = -mu * r ** 2 - lam * c
        if mu > 0:
            t = 1 / (2 * mu)
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1 = lam * bn * dx + mu * dx ** 2
                optd = -lam * bn * t
                if optd < min_ - x[n]:
                    optd = min_ - x[n]
                elif optd > max_ - x[n]:
                    optd = max_ - x[n]
                case2 = 1 + lam * bn * optd + mu * optd ** 2
                if case1 <= case2:
                    g += mu * dx ** 2 + lam * bn * dx
                else:
                    g += 1 + mu * optd ** 2 + lam * bn * optd
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1 = lam * bn * dx
                case2 = 1 + lam * bn * (min_ - x[n])
                case3 = 1 + lam * bn * (max_ - x[n])
                if case1 <= case2 and case1 <= case3:
                    g += mu * dx ** 2 + lam * bn * dx
                elif case2 < case3:
                    g += 1 + mu * (min_ - x[n]) ** 2 + lam * bn * (min_ - x[n])
                else:
                    g += 1 + mu * (max_ - x[n]) ** 2 + lam * bn * (max_ - x[n])
        return g

    def _get_final_delta(self, lam, mu, x0, x, b, min_, max_, c, r, touchup=True):
        if False:
            return 10
        if touchup:
            delta = self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r)
            if delta is not None:
                return delta
            else:
                params = [(lam + 1e-05, mu), (lam, mu + 1e-05), (lam - 1e-05, mu), (lam, mu - 1e-05), (lam + 1e-05, mu + 1e-05), (lam - 1e-05, mu - 1e-05), (lam + 1e-05, mu - 1e-05), (lam - 1e-05, mu + 1e-05)]
                for param in params:
                    delta = self.__get_final_delta(param[0], param[1], x0, x, b, min_, max_, c, r)
                    if delta is not None:
                        return delta
                return self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r, False)
        else:
            return self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r, False)

    def __get_final_delta(self, lam, mu, x0, x, b, min_, max_, c, r, touchup=True):
        if False:
            print('Hello World!')
        delta = np.empty_like(x0)
        N = x0.shape[0]
        b_dot_d = 0
        norm_d = 0
        distance = 0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t = 1 / (2 * mu)
                case1 = lam * bn * dx + mu * dx ** 2
                optd = -lam * bn * t
                if optd < min_ - x[n]:
                    optd = min_ - x[n]
                elif optd > max_ - x[n]:
                    optd = max_ - x[n]
                case2 = 1 + lam * bn * optd + mu * optd ** 2
                if case1 <= case2:
                    d = dx
                else:
                    d = optd
                    distance += 1
                delta[n] = d
                b_dot_d += bn * d
                norm_d += d ** 2
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1 = lam * bn * dx
                case2 = 1 + lam * bn * (min_ - x[n])
                case3 = 1 + lam * bn * (max_ - x[n])
                if case1 <= case2 and case1 <= case3:
                    d = dx
                elif case2 < case3:
                    d = min_ - x[n]
                    distance += 1
                else:
                    d = max_ - x[n]
                    distance += 1
                delta[n] = d
                norm_d += d ** 2
                b_dot_d += bn * d
        if touchup:
            dc = c - b_dot_d
            k = 0
            min_distance = np.inf
            min_norm = np.inf
            min_distance_idx = 0
            for n in range(N):
                if np.abs(b[n]) > 0:
                    dx = x0[n] - x[n]
                    old_d = delta[n]
                    new_d = old_d + dc / b[n]
                    if x[n] + new_d <= max_ and x[n] + new_d >= min_ and (norm_d - old_d ** 2 + new_d ** 2 <= r ** 2):
                        if k == 0:
                            min_distance = distance - (np.abs(old_d - dx) > 1e-10) + (np.abs(new_d - dx) > 1e-10)
                            min_distance_idx = n
                            min_norm = norm_d - old_d ** 2 + new_d ** 2
                            k += 1
                        else:
                            new_distance = distance - (np.abs(old_d - dx) > 1e-10) + (np.abs(new_d - dx) > 1e-10)
                            if min_distance > new_distance or (min_distance == new_distance and min_norm > norm_d - old_d ** 2 + new_d ** 2):
                                min_distance = new_distance
                                min_norm = norm_d - old_d ** 2 + new_d ** 2
                                min_distance_idx = n
            if k > 0:
                idx = min_distance_idx
                old_d = delta[idx]
                new_d = old_d + dc / b[idx]
                delta[idx] = new_d
                return delta
            else:
                return None
        return delta

    def _distance(self, x0, x):
        if False:
            while True:
                i = 10
        return np.sum(np.abs(x - x0) > EPS)

class BrendelBethgeAttack(EvasionAttack):
    """
    Base class for the Brendel & Bethge adversarial attack [#Bren19]_, a powerful gradient-based adversarial attack that
    follows the adversarial boundary (the boundary between the space of adversarial and non-adversarial images as
    defined by the adversarial criterion) to find the minimum distance to the clean image.

    This is implementation of the Brendel & Bethge attack follows the reference implementation at
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/brendel_bethge.py.

    Implementation differs from the attack used in the paper in two ways:

    * The initial binary search is always using the full 10 steps (for ease of implementation).
    * The adaptation of the trust region over the course of optimisation is less
      greedy but is more robust, reliable and simpler (decay every K steps)

    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan Ustyuzhaninov, Matthias Bethge,
            "Accurate, reliable and fast robustness evaluation",
            33rd Conference on Neural Information Processing Systems (2019)
            https://arxiv.org/abs/1907.01003
    """
    attack_params = EvasionAttack.attack_params + ['norm', 'targeted', 'init_attack', 'overshoot', 'steps', 'lr', 'lr_decay', 'lr_num_decay', 'momentum', 'binary_search_steps', 'init_size']
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(self, estimator: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', norm: Union[int, float, str]=np.inf, targeted: bool=False, overshoot: float=1.1, steps: int=1000, lr: float=0.001, lr_decay: float=0.5, lr_num_decay: int=20, momentum: float=0.8, binary_search_steps: int=10, init_size: int=100, batch_size: int=32):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param estimator: A trained ART classifier providing loss gradients.\n        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.\n        :param targeted: Flag determining if attack is targeted.\n        :param overshoot: If 1 the attack tries to return exactly to the adversarial boundary in each iteration. For\n                          higher values the attack tries to overshoot over the boundary to ensure that the perturbed\n                          sample in each iteration is adversarial.\n        :param steps: Maximum number of iterations to run. Might converge and stop before that.\n        :param lr: Trust region radius, behaves similar to a learning rate. Smaller values decrease the step size in\n                   each iteration and ensure that the attack follows the boundary more faithfully.\n        :param lr_decay: The trust region lr is multiplied with lr_decay in regular intervals (see lr_num_decay).\n        :param lr_num_decay: Number of learning rate decays in regular intervals of length steps / lr_num_decay.\n        :param momentum: Averaging of the boundary estimation over multiple steps. A momentum of zero would always take\n                         the current estimate while values closer to one average over a larger number of iterations.\n        :param binary_search_steps: Number of binary search steps used to find the adversarial boundary between the\n                                    starting point and the clean image.\n        :param init_size: Maximum number of random search steps to find initial adversarial example.\n        :param batch_size: Batch size for evaluating the model for predictions and gradients.\n        '
        from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
        if isinstance(estimator, TensorFlowV2Classifier):
            import tensorflow as tf
            if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape)))):
                raise ValueError("The provided estimator seems to predict probabilities. If loss_type='difference_logits_ratio' the estimator has to to predict logits.")
            else:

                def logits_difference(y_true, y_pred):
                    if False:
                        for i in range(10):
                            print('nop')
                    i_y_true = tf.cast(tf.math.argmax(tf.cast(y_true, tf.int32), axis=1), tf.int32)
                    i_y_pred_arg = tf.argsort(y_pred, axis=1)
                    i_z_i_list = []
                    for i in range(y_true.shape[0]):
                        if i_y_pred_arg[i, -1] != i_y_true[i]:
                            i_z_i_list.append(i_y_pred_arg[i, -1])
                        else:
                            i_z_i_list.append(i_y_pred_arg[i, -2])
                    i_z_i = tf.stack(i_z_i_list)
                    z_i = tf.gather(y_pred, i_z_i, axis=1, batch_dims=0)
                    z_y = tf.gather(y_pred, i_y_true, axis=1, batch_dims=0)
                    z_i = tf.linalg.diag_part(z_i)
                    z_y = tf.linalg.diag_part(z_y)
                    logits_diff = z_y - z_i
                    return tf.reduce_mean(logits_diff)
                self._loss_fn = logits_difference
                self._loss_object = logits_difference
            estimator_bb: 'CLASSIFIER_LOSS_GRADIENTS_TYPE' = TensorFlowV2Classifier(model=estimator.model, nb_classes=estimator.nb_classes, input_shape=estimator.input_shape, loss_object=self._loss_object, optimizer=estimator.optimizer, train_step=estimator.train_step, channels_first=estimator.channels_first, clip_values=estimator.clip_values, preprocessing_defences=estimator.preprocessing_defences, postprocessing_defences=estimator.postprocessing_defences, preprocessing=estimator.preprocessing)
        elif isinstance(estimator, PyTorchClassifier):
            import torch
            if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=config.ART_NUMPY_DTYPE))):
                raise ValueError("The provided estimator seems to predict probabilities. If loss_type='difference_logits_ratio' the estimator has to to predict logits.")
            else:

                def logits_difference(y_pred, y_true):
                    if False:
                        i = 10
                        return i + 15
                    if isinstance(y_true, np.ndarray):
                        y_true = torch.from_numpy(y_true)
                    if isinstance(y_pred, np.ndarray):
                        y_pred = torch.from_numpy(y_pred)
                    y_true = y_true.float()
                    i_y_true = torch.argmax(y_true, axis=1)
                    i_y_pred_arg = torch.argsort(y_pred, axis=1)
                    i_z_i_list = []
                    for i in range(y_true.shape[0]):
                        if i_y_pred_arg[i, -1] != i_y_true[i]:
                            i_z_i_list.append(i_y_pred_arg[i, -1])
                        else:
                            i_z_i_list.append(i_y_pred_arg[i, -2])
                    i_z_i = torch.stack(i_z_i_list)
                    z_i = y_pred[:, i_z_i]
                    z_y = y_pred[:, i_y_true]
                    z_i = torch.diagonal(z_i)
                    z_y = torch.diagonal(z_y)
                    logits_diff = z_y - z_i
                    return torch.mean(logits_diff.float())
                self._loss_fn = logits_difference
                self._loss_object = logits_difference
            estimator_bb = PyTorchClassifier(model=estimator.model, loss=self._loss_object, input_shape=estimator.input_shape, nb_classes=estimator.nb_classes, optimizer=None, channels_first=estimator.channels_first, clip_values=estimator.clip_values, preprocessing_defences=estimator.preprocessing_defences, postprocessing_defences=estimator.postprocessing_defences, preprocessing=estimator.preprocessing, device_type=str(estimator._device))
        else:
            logger.warning('The type of the provided estimator is not yet support for automated setting of logits difference loss. Therefore, this attack is defaulting to attacking the loss provided by the model in the provided estimator.')
            estimator_bb = estimator
        super().__init__(estimator=estimator_bb)
        self.norm = norm
        self._targeted = targeted
        self.overshoot = overshoot
        self.steps = steps
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_num_decay = lr_num_decay
        self.momentum = momentum
        self.binary_search_steps = binary_search_steps
        self.init_size = init_size
        self.batch_size = batch_size
        self._check_params()
        self._optimizer: Optimizer
        if norm == 0:
            self._optimizer = L0Optimizer()
        if norm == 1:
            self._optimizer = L1Optimizer()
        elif norm == 2:
            self._optimizer = L2Optimizer()
        elif norm in ['inf', np.inf]:
            self._optimizer = LinfOptimizer()
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies the Brendel & Bethge attack.\n\n        :param x: The original clean inputs.\n        :param y: The labels for inputs `x`.\n\n        :Keyword Arguments:\n            * *starting_points* (``np.ndarray``)\n                Optional. Adversarial inputs to use as a starting points, in particular for targeted attacks.\n        '
        starting_points = kwargs.get('starting_points')
        originals = x.copy()
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            logger.info('Using model predictions as correct labels for FGM.')
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        x_adv_init = kwargs.get('x_adv_init')
        if x_adv_init is not None:
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)
        classes = y
        if starting_points is None:
            starting_points = np.zeros_like(x)
            (clip_min, clip_max) = self.estimator.clip_values
            preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
            y_index = np.argmax(y, axis=1)
            for i_x in range(x.shape[0]):
                initial_sample = self._init_sample(x=x[i_x], y=y_index[i_x], y_p=preds[i_x], init_pred=init_preds[i_x], adv_init=x_adv_init[i_x], clip_min=clip_min, clip_max=clip_max)
                if initial_sample is None:
                    starting_points[i_x] = x[i_x]
                else:
                    starting_points[i_x] = initial_sample[0]
        best_advs = starting_points
        if self.targeted:
            assert (np.argmax(self.estimator.predict(x=best_advs), axis=1) == np.argmax(y, axis=1)).all()
        else:
            assert (np.argmax(self.estimator.predict(x=best_advs), axis=1) != np.argmax(y, axis=1)).all()
        N = len(originals)
        rows = range(N)
        bounds = self.estimator.clip_values
        (min_, max_) = bounds
        x0 = originals
        x0_np_flatten = x0.reshape((N, -1))
        x1 = best_advs
        lower_bound = np.zeros(shape=(N,))
        upper_bound = np.ones(shape=(N,))
        for _ in range(self.binary_search_steps):
            epsilons = (lower_bound + upper_bound) / 2
            mid_points = self.mid_points(x0, x1, epsilons, bounds)
            if self.targeted:
                is_advs = (np.argmax(self.estimator.predict(x=mid_points), axis=1) == np.argmax(y, axis=1)).all()
            else:
                is_advs = (np.argmax(self.estimator.predict(x=mid_points), axis=1) != np.argmax(y, axis=1)).all()
            lower_bound = np.where(is_advs, lower_bound, epsilons)
            upper_bound = np.where(is_advs, epsilons, upper_bound)
        starting_points = self.mid_points(x0, x1, upper_bound, bounds)
        x = starting_points.astype(config.ART_NUMPY_DTYPE)
        lrs = self.lr * np.ones(N)
        lr_reduction_interval = max(1, int(self.steps / self.lr_num_decay))
        converged = np.zeros(N, dtype=bool)
        rate_normalization = np.prod(x.shape) * (max_ - min_)
        original_shape = x.shape
        _best_advs = best_advs.copy()
        from tqdm.auto import trange
        for step in trange(1, self.steps + 1):
            if converged.all():
                break
            logits = self.estimator.predict(x=x)
            exclude = classes
            logits_exclude = logits.copy()
            logits_exclude[:, np.argmax(exclude, axis=1)] = -np.inf
            best_other_classes = np.argmax(logits_exclude, axis=1)
            if self.targeted:
                c_minimize = best_other_classes
                c_maximize = np.argmax(classes, axis=1)
            else:
                c_minimize = np.argmax(classes, axis=1)
                c_maximize = best_other_classes
            logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
            _boundary = self.estimator.loss_gradient(x=x, y=y)
            if self.targeted:
                _boundary = -_boundary
            distances = self.norms(originals - x)
            source_norms = self.norms(originals - best_advs)
            closer = distances < source_norms
            closer = np.squeeze(closer)
            is_advs = logits_diffs < 0
            closer = np.logical_and(closer, is_advs)
            x_np_flatten = x.reshape((N, -1))
            if closer.any():
                _best_advs = best_advs.copy()
                _closer = closer.flatten()
                for idx in np.arange(N)[_closer]:
                    _best_advs[idx] = x_np_flatten[idx].reshape(original_shape[1:])
            best_advs = _best_advs.copy()
            if step == 1:
                boundary = _boundary
            else:
                boundary = (1 - self.momentum) * _boundary + self.momentum * boundary
            if (step + 1) % lr_reduction_interval == 0:
                lrs *= self.lr_decay
            x = x.reshape((N, -1))
            region = lrs * rate_normalization
            corr_logits_diffs = np.where(-logits_diffs < 0, -self.overshoot * logits_diffs, -(2 - self.overshoot) * logits_diffs)
            (deltas, k) = ([], 0)
            for sample in range(N):
                if converged[sample]:
                    deltas.append(np.zeros_like(x0_np_flatten[sample]))
                else:
                    _x0 = x0_np_flatten[sample]
                    _x = x_np_flatten[sample]
                    _b = boundary[k].flatten()
                    _c = corr_logits_diffs[k]
                    r = region[sample]
                    delta = self._optimizer.solve(_x0, _x, _b, bounds[0], bounds[1], _c, r)
                    deltas.append(delta)
                    k += 1
            deltas_array = np.stack(deltas).astype(np.float32)
            x = (x + deltas_array).reshape(original_shape)
        return best_advs.astype(config.ART_NUMPY_DTYPE)

    def norms(self, x: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        order = self.norm if self.norm != 'inf' else np.inf
        norm = np.linalg.norm(x=x.reshape(x.shape[0], -1), ord=order, axis=1)
        return norm

    def mid_points(self, x0: np.ndarray, x1: np.ndarray, epsilons: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        returns a point between x0 and x1 where epsilon = 0 returns x0 and epsilon = 1 returns x1\n        '
        if self.norm == 0:
            epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
            threshold = (bounds[1] - bounds[0]) * epsilons
            mask = np.abs(x1 - x0) < threshold
            new_x = np.where(mask, x1, x0)
        elif self.norm == 1:
            epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
            threshold = (bounds[1] - bounds[0]) * (1 - epsilons)
            mask = np.abs(x1 - x0) > threshold
            new_x = np.where(mask, x0 + np.sign(x1 - x0) * (np.abs(x1 - x0) - threshold), x0)
        elif self.norm == 2:
            epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
            new_x = epsilons * x1 + (1 - epsilons) * x0
        elif self.norm in ['inf', np.inf]:
            delta = x1 - x0
            (min_, max_) = bounds
            s = max_ - min_
            epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
            clipped_delta = np.where(delta < -epsilons * s, -epsilons * s, delta)
            clipped_delta = np.where(clipped_delta > epsilons * s, epsilons * s, clipped_delta)
            new_x = x0 + clipped_delta
        else:
            raise ValueError('Value of `norm` is not supported.')
        return new_x.astype(config.ART_NUMPY_DTYPE)

    def _init_sample(self, x: np.ndarray, y: int, y_p: int, init_pred: int, adv_init: np.ndarray, clip_min: float, clip_max: float) -> Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        if False:
            return 10
        '\n        Find initial adversarial example for the attack.\n\n        :param x: An array with 1 original input to be attacked.\n        :param y: If `self.targeted` is true, then `y` represents the target label.\n        :param y_p: The predicted label of x.\n        :param init_pred: The predicted label of the initial image.\n        :param adv_init: Initial array to act as an initial adversarial example.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An adversarial example.\n        '
        nprd = np.random.RandomState()
        initial_sample = None
        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return (adv_init.astype(config.ART_NUMPY_DTYPE), init_pred)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class == y:
                    random_img = self._binary_search(current_sample=random_img, original_sample=x, target=y, norm=2, clip_min=clip_min, clip_max=clip_max, threshold=0.001)
                    initial_sample = (random_img, random_class)
                    logger.info('Found initial adversarial image for targeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        else:
            if adv_init is not None and init_pred != y_p:
                return (adv_init.astype(config.ART_NUMPY_DTYPE), y_p)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class != y_p:
                    random_img = self._binary_search(current_sample=random_img, original_sample=x, target=y_p, norm=2, clip_min=clip_min, clip_max=clip_max, threshold=0.001)
                    initial_sample = (random_img, y_p)
                    logger.info('Found initial adversarial image for untargeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        return initial_sample

    def _binary_search(self, current_sample: np.ndarray, original_sample: np.ndarray, target: int, norm: Union[int, float, str], clip_min: float, clip_max: float, threshold: Optional[float]=None) -> np.ndarray:
        if False:
            return 10
        '\n        Binary search to approach the boundary.\n\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :param target: The target label.\n        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :param threshold: The upper threshold in binary search.\n        :return: an adversarial example.\n        '
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)
            if threshold is None:
                threshold = self.theta
        else:
            (upper_bound, lower_bound) = (np.max(abs(original_sample - current_sample)), 0)
            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)
        while upper_bound - lower_bound > threshold:
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample=current_sample, original_sample=original_sample, alpha=alpha, norm=norm)
            satisfied = self._adversarial_satisfactory(samples=interpolated_sample[None], target=target, clip_min=clip_min, clip_max=clip_max)[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)
        result = self._interpolate(current_sample=current_sample, original_sample=original_sample, alpha=upper_bound, norm=norm)
        return result

    @staticmethod
    def _interpolate(current_sample: np.ndarray, original_sample: np.ndarray, alpha: float, norm: Union[int, float, str]) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Interpolate a new sample based on the original and the current samples.\n\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :param alpha: The coefficient of interpolation.\n        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.\n        :return: An adversarial example.\n        '
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)
        return result

    def _adversarial_satisfactory(self, samples: np.ndarray, target: int, clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Check whether an image is adversarial.\n\n        :param samples: A batch of examples.\n        :param target: The target label.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An array of 0/1.\n        '
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)
        if self.targeted:
            result = preds == target
        else:
            result = preds != target
        return result

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.norm not in [0, 1, 2, np.inf, 'inf']:
            raise ValueError('The argument norm has to be either 1, 2, np.inf, or "inf".')
        if not isinstance(self.targeted, bool):
            raise ValueError('The argument `targeted` has to be of type `bool`.')
        if not isinstance(self.overshoot, float) or self.overshoot < 1.0:
            raise ValueError('The argument `overshoot` has to be of `float` and larger than 1.')
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('The argument `steps` has to be of `int` and larger than 0.')
        if not isinstance(self.lr, float) or self.lr <= 0.0:
            raise ValueError('The argument `lr` has to be of `float` and larger than 0.0.')
        if not isinstance(self.lr_decay, float) or self.lr_decay <= 0.0:
            raise ValueError('The argument `lr_decay` has to be of `float` and larger than 0.0.')
        if not isinstance(self.lr_num_decay, int) or self.lr_num_decay < 1:
            raise ValueError('The argument `lr_num_decay` has to be of `int` and larger than 0.')
        if not isinstance(self.momentum, float) or self.momentum <= 0.0:
            raise ValueError('The argument `momentum` has to be of `float` and larger than 0.0.')
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 1:
            raise ValueError('The argument `binary_search_steps` has to be of `int` and larger than 0.')
        if not isinstance(self.init_size, int) or self.init_size < 1:
            raise ValueError('The argument `init_size` has to be of `int` and larger than 0.')