"""
This module implements the Feature Adversaries attack.

| Paper link: https://arxiv.org/abs/1511.05122
"""
import logging
from typing import TYPE_CHECKING, Optional
import numpy as np
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class FeatureAdversariesNumpy(EvasionAttack):
    """
    This class represent a Feature Adversaries evasion attack.

    | Paper link: https://arxiv.org/abs/1511.05122
    """
    attack_params = EvasionAttack.attack_params + ['delta', 'layer', 'batch_size']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', delta: Optional[float]=None, layer: Optional[int]=None, batch_size: int=32):
        if False:
            return 10
        '\n        Create a :class:`.FeatureAdversaries` instance.\n\n        :param classifier: A trained classifier.\n        :param delta: The maximum deviation between source and guide images.\n        :param layer: Index of the representation layer.\n        :param batch_size: Batch size.\n        '
        super().__init__(estimator=classifier)
        self.delta = delta
        self.layer = layer
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: Source samples.\n        :param y: Guide samples.\n        :param kwargs: The kwargs are used as `options` for the minimisation with `scipy.optimize.minimize` using\n                       `method="L-BFGS-B"`. Valid options are based on the output of\n                       `scipy.optimize.show_options(solver=\'minimize\', method=\'L-BFGS-B\')`:\n                       Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.\n\n                       disp : None or int\n                           If `disp is None` (the default), then the supplied version of `iprint`\n                           is used. If `disp is not None`, then it overrides the supplied version\n                           of `iprint` with the behaviour you outlined.\n                       maxcor : int\n                           The maximum number of variable metric corrections used to\n                           define the limited memory matrix. (The limited memory BFGS\n                           method does not store the full hessian but uses this many terms\n                           in an approximation to it.)\n                       ftol : float\n                           The iteration stops when ``(f^k -\n                           f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.\n                       gtol : float\n                           The iteration will stop when ``max{|proj g_i | i = 1, ..., n}\n                           <= gtol`` where ``pg_i`` is the i-th component of the\n                           projected gradient.\n                       eps : float\n                           Step size used for numerical approximation of the Jacobian.\n                       maxfun : int\n                           Maximum number of function evaluations.\n                       maxiter : int\n                           Maximum number of iterations.\n                       iprint : int, optional\n                           Controls the frequency of output. ``iprint < 0`` means no output;\n                           ``iprint = 0``    print only one line at the last iteration;\n                           ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;\n                           ``iprint = 99``   print details of every iteration except n-vectors;\n                           ``iprint = 100``  print also the changes of active set and final x;\n                           ``iprint > 100``  print details of every iteration including x and g.\n                       callback : callable, optional\n                           Called after each iteration, as ``callback(xk)``, where ``xk`` is the\n                           current parameter vector.\n                       maxls : int, optional\n                           Maximum number of line search steps (per iteration). Default is 20.\n\n                       The option `ftol` is exposed via the `scipy.optimize.minimize` interface,\n                       but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The\n                       relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.\n                       I.e., `factr` multiplies the default machine floating-point precision to\n                       arrive at `ftol`.\n        :return: Adversarial examples.\n        :raises KeyError: The argument {} in kwargs is not allowed as option for `scipy.optimize.minimize` using\n                          `method="L-BFGS-B".`\n        '
        from scipy.linalg import norm
        from scipy.optimize import Bounds, minimize
        if y is None:
            raise ValueError('The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.')
        if x.shape != y.shape:
            raise ValueError('The shape of source `x` and guide `y` must be of same shape.')
        if x.shape[1:] != self.estimator.input_shape:
            raise ValueError('Source and guide inputs must match `input_shape` of estimator.')
        l_b = x.flatten() - self.delta
        l_b[l_b < self.estimator.clip_values[0]] = self.estimator.clip_values[0]
        u_b = x.flatten() + self.delta
        u_b[u_b > self.estimator.clip_values[1]] = self.estimator.clip_values[1]
        bound = Bounds(lb=l_b, ub=u_b, keep_feasible=False)
        guide_representation = self.estimator.get_activations(x=y.reshape(-1, *self.estimator.input_shape), layer=self.layer, batch_size=self.batch_size)

        def func(x_i):
            if False:
                return 10
            x_i = x_i.astype(x.dtype)
            source_representation = self.estimator.get_activations(x=x_i.reshape(-1, *self.estimator.input_shape), layer=self.layer, batch_size=self.batch_size)
            n = norm(source_representation.flatten() - guide_representation.flatten(), ord=2) ** 2
            return n
        x_0 = x.copy().flatten()
        options = {'eps': 0.001, 'ftol': 0.001}
        options_allowed_keys = ['disp', 'maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter', 'iprint', 'callback', 'maxls']
        for key in kwargs:
            if key not in options_allowed_keys:
                raise KeyError(f'The argument `{key}` in kwargs is not allowed as option for `scipy.optimize.minimize` using `method="L-BFGS-B".`')
        options.update(kwargs)
        res = minimize(func, x_0, method='L-BFGS-B', bounds=bound, options=options)
        x_adv = res.x
        logger.info(res)
        return x_adv.reshape(-1, *self.estimator.input_shape).astype(x.dtype)

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Apply attack-specific checks.\n        '
        if self.delta is None:
            raise ValueError('The delta cannot be None.')
        if self.delta is not None and self.delta <= 0:
            raise ValueError('The maximum deviation `delta` has to be positive.')
        if not isinstance(self.layer, int):
            raise ValueError('The index of the representation layer `layer` has to be integer.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')