"""
This module implements the total variance minimization defence `TotalVarMin`.

| Paper link: https://openreview.net/forum?id=SyJ7ClWCb

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE
logger = logging.getLogger(__name__)

class TotalVarMin(Preprocessor):
    """
    Implement the total variance minimization defence approach.

    | Paper link: https://openreview.net/forum?id=SyJ7ClWCb

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """
    params = ['prob', 'norm', 'lamb', 'solver', 'max_iter', 'clip_values', 'verbose']

    def __init__(self, prob: float=0.3, norm: int=2, lamb: float=0.5, solver: str='L-BFGS-B', max_iter: int=10, clip_values: Optional['CLIP_VALUES_TYPE']=None, apply_fit: bool=False, apply_predict: bool=True, verbose: bool=False):
        if False:
            return 10
        '\n        Create an instance of total variance minimization.\n\n        :param prob: Probability of the Bernoulli distribution.\n        :param norm: The norm (positive integer).\n        :param lamb: The lambda parameter in the objective function.\n        :param solver: Current support: `L-BFGS-B`, `CG`, `Newton-CG`.\n        :param max_iter: Maximum number of iterations when performing optimization.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param verbose: Show progress bars.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.prob = prob
        self.norm = norm
        self.lamb = lamb
        self.solver = solver
        self.max_iter = max_iter
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            i = 10
            return i + 15
        '\n        Apply total variance minimization to sample `x`.\n\n        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Similar samples.\n        '
        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. Variance minimization can only be applied to data with spatial dimensions.')
        x_preproc = x.copy()
        for (i, x_i) in enumerate(tqdm(x_preproc, desc='Variance minimization', disable=not self.verbose)):
            mask = (np.random.rand(*x_i.shape) < self.prob).astype('int')
            x_preproc[i] = self._minimize(x_i, mask)
        if self.clip_values is not None:
            np.clip(x_preproc, self.clip_values[0], self.clip_values[1], out=x_preproc)
        return (x_preproc.astype(ART_NUMPY_DTYPE), y)

    def _minimize(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Minimize the total variance objective function.\n\n        :param x: Original image.\n        :param mask: A matrix that decides which points are kept.\n        :return: A new image.\n        '
        z_min = x.copy()
        for i in range(x.shape[2]):
            res = minimize(self._loss_func, z_min[:, :, i].flatten(), (x[:, :, i], mask[:, :, i], self.norm, self.lamb), method=self.solver, jac=self._deri_loss_func, options={'maxiter': self.max_iter})
            z_min[:, :, i] = np.reshape(res.x, z_min[:, :, i].shape)
        return z_min

    @staticmethod
    def _loss_func(z_init: np.ndarray, x: np.ndarray, mask: np.ndarray, norm: int, lamb: float) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Loss function to be minimized.\n\n        :param z_init: Initial guess.\n        :param x: Original image.\n        :param mask: A matrix that decides which points are kept.\n        :param norm: The norm (positive integer).\n        :param lamb: The lambda parameter in the objective function.\n        :return: Loss value.\n        '
        res = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        z_init = np.reshape(z_init, x.shape)
        res += lamb * np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1).sum()
        res += lamb * np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0).sum()
        return res

    @staticmethod
    def _deri_loss_func(z_init: np.ndarray, x: np.ndarray, mask: np.ndarray, norm: int, lamb: float) -> float:
        if False:
            while True:
                i = 10
        '\n        Derivative of loss function to be minimized.\n\n        :param z_init: Initial guess.\n        :param x: Original image.\n        :param mask: A matrix that decides which points are kept.\n        :param norm: The norm (positive integer).\n        :param lamb: The lambda parameter in the objective function.\n        :return: Derivative value.\n        '
        nor1 = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        nor1 = max(nor1, 1e-06)
        der1 = (z_init - x.flatten()) * mask.flatten() / (nor1 * 1.0)
        z_init = np.reshape(z_init, x.shape)
        if norm == 1:
            z_d1 = np.sign(z_init[1:, :] - z_init[:-1, :])
            z_d2 = np.sign(z_init[:, 1:] - z_init[:, :-1])
        else:
            z_d1_norm = np.power(np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1), norm - 1)
            z_d2_norm = np.power(np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0), norm - 1)
            z_d1_norm[z_d1_norm < 1e-06] = 1e-06
            z_d2_norm[z_d2_norm < 1e-06] = 1e-06
            z_d1_norm = np.repeat(z_d1_norm[:, np.newaxis], z_init.shape[1], axis=1)
            z_d2_norm = np.repeat(z_d2_norm[np.newaxis, :], z_init.shape[0], axis=0)
            z_d1 = norm * np.power(z_init[1:, :] - z_init[:-1, :], norm - 1) / z_d1_norm
            z_d2 = norm * np.power(z_init[:, 1:] - z_init[:, :-1], norm - 1) / z_d2_norm
        der2 = np.zeros(z_init.shape)
        der2[:-1, :] -= z_d1
        der2[1:, :] += z_d1
        der2[:, :-1] -= z_d2
        der2[:, 1:] += z_d2
        der2 = lamb * der2.flatten()
        return der1 + der2

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self.prob, (float, int)) or self.prob < 0.0 or self.prob > 1.0:
            logger.error('Probability must be between 0 and 1.')
            raise ValueError('Probability must be between 0 and 1.')
        if not isinstance(self.norm, int) or self.norm <= 0:
            logger.error('Norm must be a positive integer.')
            raise ValueError('Norm must be a positive integer.')
        if self.solver not in ('L-BFGS-B', 'CG', 'Newton-CG'):
            logger.error('Current support only L-BFGS-B, CG, Newton-CG.')
            raise ValueError('Current support only L-BFGS-B, CG, Newton-CG.')
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            logger.error('Number of iterations must be a positive integer.')
            raise ValueError('Number of iterations must be a positive integer.')
        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError('`clip_values` should be a tuple of 2 floats containing the allowed data range.')
            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError('Invalid `clip_values`: min >= max.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')