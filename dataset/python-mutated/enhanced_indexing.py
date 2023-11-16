import numpy as np
import cvxpy as cp
from typing import Union, Optional, Dict, Any, List
from qlib.log import get_module_logger
from .base import BaseOptimizer
logger = get_module_logger('EnhancedIndexingOptimizer')

class EnhancedIndexingOptimizer(BaseOptimizer):
    """
    Portfolio Optimizer for Enhanced Indexing

    Notations:
        w0: current holding weights
        wb: benchmark weight
        r: expected return
        F: factor exposure
        cov_b: factor covariance
        var_u: residual variance (diagonal)
        lamb: risk aversion parameter
        delta: total turnover limit
        b_dev: benchmark deviation limit
        f_dev: factor deviation limit

    Also denote:
        d = w - wb: benchmark deviation
        v = d @ F: factor deviation

    The optimization problem for enhanced indexing:
        max_w  d @ r - lamb * (v @ cov_b @ v + var_u @ d**2)
        s.t.   w >= 0
               sum(w) == 1
               sum(|w - w0|) <= delta
               d >= -b_dev
               d <= b_dev
               v >= -f_dev
               v <= f_dev
    """

    def __init__(self, lamb: float=1, delta: Optional[float]=0.2, b_dev: Optional[float]=0.01, f_dev: Optional[Union[List[float], np.ndarray]]=None, scale_return: bool=True, epsilon: float=5e-05, solver_kwargs: Optional[Dict[str, Any]]={}):
        if False:
            while True:
                i = 10
        '\n        Args:\n            lamb (float): risk aversion parameter (larger `lamb` means more focus on risk)\n            delta (float): total turnover limit\n            b_dev (float): benchmark deviation limit\n            f_dev (list): factor deviation limit\n            scale_return (bool): whether scale return to match estimated volatility\n            epsilon (float): minimum weight\n            solver_kwargs (dict): kwargs for cvxpy solver\n        '
        assert lamb >= 0, 'risk aversion parameter `lamb` should be positive'
        self.lamb = lamb
        assert delta >= 0, 'turnover limit `delta` should be positive'
        self.delta = delta
        assert b_dev is None or b_dev >= 0, 'benchmark deviation limit `b_dev` should be positive'
        self.b_dev = b_dev
        if isinstance(f_dev, float):
            assert f_dev >= 0, 'factor deviation limit `f_dev` should be positive'
        elif f_dev is not None:
            f_dev = np.array(f_dev)
            assert all(f_dev >= 0), 'factor deviation limit `f_dev` should be positive'
        self.f_dev = f_dev
        self.scale_return = scale_return
        self.epsilon = epsilon
        self.solver_kwargs = solver_kwargs

    def __call__(self, r: np.ndarray, F: np.ndarray, cov_b: np.ndarray, var_u: np.ndarray, w0: np.ndarray, wb: np.ndarray, mfh: Optional[np.ndarray]=None, mfs: Optional[np.ndarray]=None) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Args:\n            r (np.ndarray): expected returns\n            F (np.ndarray): factor exposure\n            cov_b (np.ndarray): factor covariance\n            var_u (np.ndarray): residual variance\n            w0 (np.ndarray): current holding weights\n            wb (np.ndarray): benchmark weights\n            mfh (np.ndarray): mask force holding\n            mfs (np.ndarray): mask force selling\n\n        Returns:\n            np.ndarray: optimized portfolio allocation\n        '
        if self.scale_return:
            r = r / r.std()
            r *= np.sqrt(np.mean(np.diag(F @ cov_b @ F.T) + var_u))
        w = cp.Variable(len(r), nonneg=True)
        w.value = wb
        d = w - wb
        v = d @ F
        ret = d @ r
        risk = cp.quad_form(v, cov_b) + var_u @ d ** 2
        obj = cp.Maximize(ret - self.lamb * risk)
        lb = np.zeros_like(wb)
        ub = np.ones_like(wb)
        if self.b_dev is not None:
            lb = np.maximum(lb, wb - self.b_dev)
            ub = np.minimum(ub, wb + self.b_dev)
        if mfh is not None:
            lb[mfh] = w0[mfh]
            ub[mfh] = w0[mfh]
        if mfs is not None:
            lb[mfs] = 0
            ub[mfs] = 0
        cons = [cp.sum(w) == 1, w >= lb, w <= ub]
        if self.f_dev is not None:
            cons.extend([v >= -self.f_dev, v <= self.f_dev])
        t_cons = []
        if self.delta is not None:
            if w0 is not None and w0.sum() > 0:
                t_cons.extend([cp.norm(w - w0, 1) <= self.delta])
        success = False
        try:
            prob = cp.Problem(obj, cons + t_cons)
            prob.solve(solver=cp.ECOS, warm_start=True, **self.solver_kwargs)
            assert prob.status == 'optimal'
            success = True
        except Exception as e:
            logger.warning(f'trial 1 failed {e} (status: {prob.status})')
        if not success and len(t_cons):
            logger.info('try removing turnover constraint as the last optimization failed')
            try:
                w.value = wb
                prob = cp.Problem(obj, cons)
                prob.solve(solver=cp.ECOS, warm_start=True, **self.solver_kwargs)
                assert prob.status in ['optimal', 'optimal_inaccurate']
                success = True
            except Exception as e:
                logger.warning(f'trial 2 failed {e} (status: {prob.status})')
        if not success:
            logger.warning('optimization failed, will return current holding weight')
            return w0
        if prob.status == 'optimal_inaccurate':
            logger.warning(f'the optimization is inaccurate')
        w = np.asarray(w.value)
        w[w < self.epsilon] = 0
        w /= w.sum()
        return w