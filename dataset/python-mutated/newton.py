from __future__ import annotations
from river import optim, utils
__all__ = ['Newton']

def sherman_morrison(A_inv: dict, u: dict, v: dict) -> dict:
    if False:
        while True:
            i = 10
    'Sherman–Morrison formula.\n\n    This modifies `A_inv` inplace.\n\n    Parameters\n    ----------\n    A_inv\n    u\n    v\n\n    Examples\n    --------\n\n    >>> import pprint\n\n    >>> A_inv = {\n    ...     (0, 0): 0.2,\n    ...     (1, 1): 1,\n    ...     (2, 2): 1\n    ... }\n    >>> u = {0: 1, 1: 2, 2: 3}\n    >>> v = {0: 4}\n\n    >>> inv = sherman_morrison(A_inv, u, v)\n    >>> pprint.pprint(inv)\n    {(0, 0): 0.111111,\n        (1, 0): -0.888888,\n        (1, 1): 1,\n        (2, 0): -1.333333,\n        (2, 2): 1}\n\n    References\n    ----------\n    [^1]: [Wikipedia article on the Sherman-Morrison formula](https://www.wikiwand.com/en/Sherman%E2%80%93Morrison_formula)s\n\n    '
    den = 1 + utils.math.dot(utils.math.dotvecmat(u, A_inv), v)
    for (k, v) in utils.math.matmul2d(utils.math.matmul2d(A_inv, utils.math.outer(u, v)), A_inv).items():
        A_inv[k] = A_inv.get(k, 0) - v / den
    return A_inv

class Newton(optim.base.Optimizer):
    """Online Newton Step (ONS) optimizer.

    This optimizer uses second-order information (i.e. the Hessian of the cost function) in
    addition to first-order information (i.e. the gradient of the cost function).

    Parameters
    ----------
    lr
    eps

    References
    ----------
    [^1]: [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3), pp.169-192](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)

    """

    def __init__(self, lr=0.1, eps=1e-05):
        if False:
            i = 10
            return i + 15
        super().__init__(lr)
        self.eps = eps
        self.H_inv = {}

    def _step_with_dict(self, w, g):
        if False:
            while True:
                i = 10
        for i in g:
            if (i, i) not in self.H_inv:
                self.H_inv[i, i] = self.eps
        self.H = sherman_morrison(A_inv=self.H_inv, u=g, v=g)
        step = utils.math.dotvecmat(x=g, A=self.H_inv)
        for (i, s) in step.items():
            w[i] -= self.learning_rate * s
        return w