"""
This module implements the Threshold Attack and Pixel Attack.
The Pixel Attack is a generalisation of One Pixel Attack.

| One Pixel Attack Paper link: https://arxiv.org/ans/1710.08864
| Pixel and Threshold Attack Paper link: https://arxiv.org/abs/1906.06026
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from itertools import product
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from six import string_types
import scipy
from scipy._lib._util import check_random_state
scipy_version = list(map(int, scipy.__version__.lower().split('.')))
if scipy_version[1] >= 8:
    from scipy.optimize._optimize import _status_message
else:
    from scipy.optimize.optimize import _status_message
from scipy.optimize import OptimizeResult, minimize
from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class PixelThreshold(EvasionAttack):
    """
    These attacks were originally implemented by Vargas et al. (2019) & Su et al.(2019).

    | One Pixel Attack Paper link: https://arxiv.org/abs/1710.08864
    | Pixel and Threshold Attack Paper link: https://arxiv.org/abs/1906.06026
    """
    attack_params = EvasionAttack.attack_params + ['th', 'es', 'max_iter', 'targeted', 'verbose', 'verbose_es']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', th: Optional[int]=None, es: int=0, max_iter: int=100, targeted: bool=False, verbose: bool=True, verbose_es: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a :class:`.PixelThreshold` instance.\n\n        :param classifier: A trained classifier.\n        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.\n        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.\n        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param verbose: Print verbose messages of ES and show progress bars.\n        '
        super().__init__(estimator=classifier)
        self._project = True
        self.type_attack = -1
        self.th = th
        self.es = es
        self.max_iter = max_iter
        self._targeted = targeted
        self.verbose = verbose
        self.verbose_es = verbose_es
        self.rescale = False
        PixelThreshold._check_params(self)
        if self.estimator.channels_first:
            self.img_rows = self.estimator.input_shape[-2]
            self.img_cols = self.estimator.input_shape[-1]
            self.img_channels = self.estimator.input_shape[-3]
        else:
            self.img_rows = self.estimator.input_shape[-3]
            self.img_cols = self.estimator.input_shape[-2]
            self.img_channels = self.estimator.input_shape[-1]

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.th is not None:
            if self.th <= 0:
                raise ValueError('The perturbation size `eps` has to be positive.')
        if not isinstance(self.es, int):
            raise ValueError('The flag `es` has to be of type int.')
        if not isinstance(self.targeted, bool):
            raise ValueError('The flag `targeted` has to be of type bool.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The flag `verbose` has to be of type bool.')
        if not isinstance(self.verbose_es, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')
        if self.estimator.clip_values is None:
            raise ValueError('This attack requires estimator clip values to be defined.')

    def rescale_input(self, x):
        if False:
            print('Hello World!')
        'Rescale inputs'
        x = x.astype(ART_NUMPY_DTYPE) / 255.0
        x = x * (self.estimator.clip_values[1] - self.estimator.clip_values[0]) + self.estimator.clip_values[0]
        return x

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param max_iter: Maximum number of optimisation iterations.\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            y = np.argmax(self.estimator.predict(x), axis=1)
        else:
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:
                raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
            if y.ndim > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
        y = y.flatten()
        if self.th is None:
            logger.info('Performing minimal perturbation Attack.                 This could take long time to process.                 For sanity check, pass th=10 to the Attack instance.')
        if self.estimator.clip_values[1] != 255.0:
            self.rescale = True
            x = (x - self.estimator.clip_values[0]) / (self.estimator.clip_values[1] - self.estimator.clip_values[0])
            x = x * 255.0
        x = x.astype(ART_NUMPY_DTYPE)
        adv_x_best = []
        self.adv_th = []
        for (image, target_class) in tqdm(zip(x, y), desc='Pixel threshold', disable=not self.verbose):
            if self.th is None:
                min_th = -1
                (start, end) = (1, 127)
                image_result = image
                while True:
                    threshold = (start + end) // 2
                    (success, trial_image_result) = self._attack(image, target_class, threshold)
                    if success:
                        image_result = trial_image_result
                        end = threshold - 1
                        min_th = threshold
                    else:
                        start = threshold + 1
                    if end < start:
                        break
                self.adv_th = [min_th]
            else:
                (success, image_result) = self._attack(image, target_class, self.th)
                if not success:
                    image_result = image
            adv_x_best += [image_result]
        adv_x_best_array = np.array(adv_x_best)
        if self.rescale:
            adv_x_best_array = self.rescale_input(adv_x_best_array)
        return adv_x_best_array

    def _get_bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        if False:
            i = 10
            return i + 15
        '\n        Define the bounds for the image `img` within the limits `limit`.\n        '

        def bound_limit(value):
            if False:
                return 10
            return (np.clip(value - limit, 0, 255), np.clip(value + limit, 0, 255))
        (minbounds, maxbounds, bounds, initial) = ([], [], [], [])
        for (i, j, k) in product(range(img.shape[-3]), range(img.shape[-2]), range(img.shape[-1])):
            temp = img[i, j, k]
            initial += [temp]
            bound = bound_limit(temp)
            if self.es == 0:
                minbounds += [bound[0]]
                maxbounds += [bound[1]]
            else:
                bounds += [bound]
        if self.es == 0:
            bounds = [minbounds, maxbounds]
        return (bounds, initial)

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Perturbs the given image `img` with the given perturbation `x`.\n        '
        return img

    def _attack_success(self, adv_x, x, target_class):
        if False:
            i = 10
            return i + 15
        '\n        Checks whether the given perturbation `adv_x` for the image `img` is successful.\n        '
        adv = self._perturb_image(adv_x, x)
        if self.rescale:
            adv = self.rescale_input(adv)
        predicted_class = np.argmax(self.estimator.predict(adv)[0])
        return bool(self.targeted and predicted_class == target_class or (not self.targeted and predicted_class != target_class))

    def _attack(self, image: np.ndarray, target_class: np.ndarray, limit: int) -> Tuple[bool, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Attack the given image `image` with the threshold `limit` for the `target_class` which is true label for\n        untargeted attack and targeted label for targeted attack.\n        '
        (bounds, initial) = self._get_bounds(image, limit)

        def predict_fn(x):
            if False:
                i = 10
                return i + 15
            adv = self._perturb_image(x, image)
            if self.rescale:
                adv = self.rescale_input(adv)
            predictions = self.estimator.predict(adv)[:, target_class]
            return predictions if not self.targeted else 1 - predictions

        def callback_fn(x, convergence=None):
            if False:
                while True:
                    i = 10
            if self.es == 0:
                if self._attack_success(x.result[0], image, target_class):
                    raise CMAEarlyStoppingException('Attack Completed :) Earlier than expected')
            else:
                return self._attack_success(x, image, target_class)
        if self.es == 0:
            from cma import CMAOptions
            opts = CMAOptions()
            if not self.verbose_es:
                opts.set('verbose', -9)
                opts.set('verb_disp', 40000)
                opts.set('verb_log', 40000)
                opts.set('verb_time', False)
            opts.set('bounds', bounds)
            if self.type_attack == 0:
                std = 63
            else:
                std = limit
            from cma import CMAEvolutionStrategy
            strategy = CMAEvolutionStrategy(initial, std / 4, opts)
            try:
                strategy.optimize(predict_fn, maxfun=max(1, 400 // len(bounds)) * len(bounds) * 100, callback=callback_fn, iterations=self.max_iter)
            except CMAEarlyStoppingException as err:
                if self.verbose_es:
                    logger.info(err)
            adv_x = strategy.result[0]
        else:
            strategy = differential_evolution(predict_fn, bounds, disp=self.verbose_es, maxiter=self.max_iter, popsize=max(1, 400 // len(bounds)), recombination=1, atol=-1, callback=callback_fn, polish=False)
            adv_x = strategy.x
        if self._attack_success(adv_x, image, target_class):
            return (True, self._perturb_image(adv_x, image)[0])
        return (False, image)

class PixelAttack(PixelThreshold):
    """
    This attack was originally implemented by Vargas et al. (2019). It is generalisation of One Pixel Attack originally
    implemented by Su et al. (2019).

    | One Pixel Attack Paper link: https://arxiv.org/abs/1710.08864
    | Pixel Attack Paper link: https://arxiv.org/abs/1906.06026
    """

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', th: Optional[int]=None, es: int=1, max_iter: int=100, targeted: bool=False, verbose: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a :class:`.PixelAttack` instance.\n\n        :param classifier: A trained classifier.\n        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.\n        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.\n        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param verbose: Indicates whether to print verbose messages of ES used.\n        '
        super().__init__(classifier, th, es, max_iter, targeted, verbose)
        self.type_attack = 0

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perturbs the given image `img` with the given perturbation `x`.\n        '
        if x.ndim < 2:
            x = np.array([x])
        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for (adv, image) in zip(x, imgs):
            for pixel in np.split(adv, len(adv) // (2 + self.img_channels)):
                (x_pos, y_pos, *rgb) = pixel
                if not self.estimator.channels_first:
                    image[x_pos % self.img_rows, y_pos % self.img_cols] = rgb
                else:
                    image[:, x_pos % self.img_rows, y_pos % self.img_cols] = rgb
        return imgs

    def _get_bounds(self, img: np.ndarray, limit) -> Tuple[List[list], list]:
        if False:
            return 10
        '\n        Define the bounds for the image `img` within the limits `limit`.\n        '
        initial: List[int] = []
        bounds: List[List[int]]
        if self.es == 0:
            for (count, (i, j)) in enumerate(product(range(self.img_rows), range(self.img_cols))):
                initial += [i, j]
                for k in range(self.img_channels):
                    if not self.estimator.channels_first:
                        initial += [img[i, j, k]]
                    else:
                        initial += [img[k, i, j]]
                if count == limit - 1:
                    break
            min_bounds = [0, 0]
            for _ in range(self.img_channels):
                min_bounds += [0]
            min_bounds = min_bounds * limit
            max_bounds = [self.img_rows, self.img_cols]
            for _ in range(self.img_channels):
                max_bounds += [255]
            max_bounds = max_bounds * limit
            bounds = [min_bounds, max_bounds]
        else:
            bounds = [[0, self.img_rows], [0, self.img_cols]]
            for _ in range(self.img_channels):
                bounds += [[0, 255]]
            bounds = bounds * limit
        return (bounds, initial)

class ThresholdAttack(PixelThreshold):
    """
    This attack was originally implemented by Vargas et al. (2019).

    | Paper link: https://arxiv.org/abs/1906.06026
    """

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', th: Optional[int]=None, es: int=0, max_iter: int=100, targeted: bool=False, verbose: bool=False) -> None:
        if False:
            return 10
        '\n        Create a :class:`.PixelThreshold` instance.\n\n        :param classifier: A trained classifier.\n        :param th: threshold value of the Pixel/ Threshold attack. th=None indicates finding a minimum threshold.\n        :param es: Indicates whether the attack uses CMAES (0) or DE (1) as Evolutionary Strategy.\n        :param max_iter: Sets the Maximum iterations to run the Evolutionary Strategies for optimisation.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param verbose: Indicates whether to print verbose messages of ES used.\n        '
        super().__init__(classifier, th, es, max_iter, targeted, verbose)
        self.type_attack = 1

    def _perturb_image(self, x: np.ndarray, img: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perturbs the given image `img` with the given perturbation `x`.\n        '
        if x.ndim < 2:
            x = x[None, ...]
        imgs = np.tile(img, [len(x)] + [1] * (x.ndim + 1))
        x = x.astype(int)
        for (adv, image) in zip(x, imgs):
            for (count, (i, j, k)) in enumerate(product(range(image.shape[-3]), range(image.shape[-2]), range(image.shape[-1]))):
                image[i, j, k] = adv[count]
        return imgs

class CMAEarlyStoppingException(Exception):
    """Raised when CMA is stopping early after successful optimisation."""
    pass
"\nA slight modification to Scipy's implementation of differential evolution.\nTo speed up predictions, the entire parameters array is passed to `self.func`,\nwhere a neural network model can batch its computations and execute in parallel\nSearch for `CHANGES` to find all code changes.\n\nDan Kondratyuk 2018\n\nOriginal code adapted from\nhttps://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py\n----------\ndifferential_evolution:The differential evolution global optimization algorithm\nAdded by Andrew Nelson 2014\n"
__all__ = ['differential_evolution']
_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0):
    if False:
        i = 10
        return i + 15
    "Finds the global minimum of a multivariate function.\n    Differential Evolution is stochastic in nature (does not use gradient\n    methods) to find the minimium, and can search large areas of candidate\n    space, but often requires larger numbers of function evaluations than\n    conventional gradient based techniques.\n    The algorithm is due to Storn and Price [1]_.\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized.  Must be in the form\n        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array\n        and ``args`` is a  tuple of any additional fixed parameters needed to\n        completely specify the function.\n    bounds : sequence\n        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,\n        defining the lower and upper bounds for the optimizing argument of\n        `func`. It is required to have ``len(bounds) == len(x)``.\n        ``len(bounds)`` is used to determine the number of parameters in ``x``.\n    args : tuple, optional\n        Any additional fixed parameters needed to\n        completely specify the objective function.\n    strategy : str, optional\n        The differential evolution strategy to use. Should be one of:\n            - 'best1bin'\n            - 'best1exp'\n            - 'rand1exp'\n            - 'randtobest1exp'\n            - 'currenttobest1exp'\n            - 'best2exp'\n            - 'rand2exp'\n            - 'randtobest1bin'\n            - 'currenttobest1bin'\n            - 'best2bin'\n            - 'rand2bin'\n            - 'rand1bin'\n        The default is 'best1bin'.\n    maxiter : int, optional\n        The maximum number of generations over which the entire population is\n        evolved. The maximum number of function evaluations (with no polishing)\n        is: ``(maxiter + 1) * popsize * len(x)``\n    popsize : int, optional\n        A multiplier for setting the total population size.  The population has\n        ``popsize * len(x)`` individuals (unless the initial population is\n        supplied via the `init` keyword).\n    tol : float, optional\n        Relative tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    mutation : float or tuple(float, float), optional\n        The mutation constant. In the literature this is also known as\n        differential weight, being denoted by F.\n        If specified as a float it should be in the range [0, 2].\n        If specified as a tuple ``(min, max)`` dithering is employed. Dithering\n        randomly changes the mutation constant on a generation by generation\n        basis. The mutation constant for that generation is taken from\n        ``U[min, max)``. Dithering can help speed convergence significantly.\n        Increasing the mutation constant increases the search radius, but will\n        slow down convergence.\n    recombination : float, optional\n        The recombination constant, should be in the range [0, 1]. In the\n        literature this is also known as the crossover probability, being\n        denoted by CR. Increasing this value allows a larger number of mutants\n        to progress into the next generation, but at the risk of population\n        stability.\n    seed : int or `np.random.RandomState`, optional\n        If `seed` is not specified the `np.RandomState` singleton is used.\n        If `seed` is an int, a new `np.random.RandomState` instance is used,\n        seeded with seed.\n        If `seed` is already a `np.random.RandomState instance`, then that\n        `np.random.RandomState` instance is used.\n        Specify `seed` for repeatable minimizations.\n    disp : bool, optional\n        Display status messages\n    callback : callable, `callback(xk, convergence=val)`, optional\n        A function to follow the progress of the minimization. ``xk`` is\n        the current value of ``x0``. ``val`` represents the fractional\n        value of the population convergence.  When ``val`` is greater than one\n        the function halts. If callback returns `True`, then the minimization\n        is halted (any polishing is still carried out).\n    polish : bool, optional\n        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`\n        method is used to polish the best population member at the end, which\n        can improve the minimization slightly.\n    init : str or array-like, optional\n        Specify which type of population initialization is performed. Should be\n        one of:\n            - 'latinhypercube'\n            - 'random'\n            - array specifying the initial population. The array should have\n              shape ``(M, len(x))``, where len(x) is the number of parameters.\n              `init` is clipped to `bounds` before use.\n        The default is 'latinhypercube'. Latin Hypercube sampling tries to\n        maximize coverage of the available parameter space. 'random'\n        initializes the population randomly - this has the drawback that\n        clustering can occur, preventing the whole of parameter space being\n        covered. Use of an array to specify a population subset could be used,\n        for example, to create a tight bunch of initial guesses in an location\n        where the solution is known to exist, thereby reducing time for\n        convergence.\n    atol : float, optional\n        Absolute tolerance for convergence, the solving stops when\n        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,\n        where and `atol` and `tol` are the absolute and relative tolerance\n        respectively.\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a `OptimizeResult` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.  If `polish`\n        was employed, and a lower minimum was obtained by the polishing, then\n        OptimizeResult also contains the ``jac`` attribute.\n    Notes\n    -----\n    Differential evolution is a stochastic population based method that is\n    useful for global optimization problems. At each pass through the\n    population the algorithm mutates each candidate solution by mixing with\n    other candidate solutions to create a trial candidate. There are several\n    strategies [2]_ for creating trial candidates, which suit some problems\n    more than others. The 'best1bin' strategy is a good starting point for many\n    systems. In this strategy two members of the population are randomly\n    chosen. Their difference is used to mutate the best member (the `best` in\n    `best1bin`), :math:`b_0`,\n    so far:\n    .. math::\n        b' = b_0 + mutation * (population[rand0] - population[rand1])\n    A trial vector is then constructed. Starting with a randomly chosen 'i'th\n    parameter the trial is sequentially filled (in modulo) with parameters from\n    `b'` or the original candidate. The choice of whether to use `b'` or the\n    original candidate is made with a binomial distribution (the 'bin' in\n    'best1bin') - a random number in [0, 1) is generated.  If this number is\n    less than the `recombination` constant then the parameter is loaded from\n    `b'`, otherwise it is loaded from the original candidate.  The final\n    parameter is always loaded from `b'`.  Once the trial candidate is built\n    its fitness is assessed. If the trial is better than the original candidate\n    then it takes its place. If it is also better than the best overall\n    candidate it also replaces that.\n    To improve your chances of finding a global minimum use higher `popsize`\n    values, with higher `mutation` and (dithering), but lower `recombination`\n    values. This has the effect of widening the search radius, but slowing\n    convergence.\n    .. versionadded:: 0.15.0\n    Examples\n    --------\n    Let us consider the problem of minimizing the Rosenbrock function. This\n    function is implemented in `rosen` in `scipy.optimize`.\n    >>> from scipy.optimize import rosen, differential_evolution\n    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]\n    >>> result = differential_evolution(rosen, bounds)\n    >>> result.x, result.fun\n    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)\n    Next find the minimum of the Ackley function\n    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).\n    >>> from scipy.optimize import differential_evolution\n    >>> import numpy as np\n    >>> def ackley(x):\n    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))\n    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi *x[1]))\n    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e\n    >>> bounds = [(-5, 5), (-5, 5)]\n    >>> result = differential_evolution(ackley, bounds)\n    >>> result.x, result.fun\n    (array([ 0.,  0.]), 4.4408920985006262e-16)\n    References\n    ----------\n    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and\n           Efficient Heuristic for Global Optimization over Continuous Spaces,\n           Journal of Global Optimization, 1997, 11, 341 - 359.\n    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html\n    .. [3] http://en.wikipedia.org/wiki/Differential_evolution\n    "
    solver = DifferentialEvolutionSolver(func, bounds, args=args, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, recombination=recombination, seed=seed, polish=polish, callback=callback, disp=disp, init=init, atol=atol)
    return solver.solve()

class DifferentialEvolutionSolver:
    """This class implements the differential evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population could be used, for
        example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    """
    _binomial = {'best1bin': '_best1', 'randtobest1bin': '_randtobest1', 'currenttobest1bin': '_currenttobest1', 'best2bin': '_best2', 'rand2bin': '_rand2', 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1', 'rand1exp': '_rand1', 'randtobest1exp': '_randtobest1', 'currenttobest1exp': '_currenttobest1', 'best2exp': '_best2', 'rand2exp': '_rand2'}
    __init_error_msg = "The population initialization method must be one of 'latinhypercube' or 'random', or an array of shape (M, N) where N is the number of parameters and M>5"

    def __init__(self, func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, maxfun=np.inf, callback=None, disp=False, polish=True, init='latinhypercube', atol=0):
        if False:
            print('Hello World!')
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError('Please select a valid mutation strategy')
        self.strategy = strategy
        self.callback = callback
        self.polish = polish
        (self.tol, self.atol) = (tol, atol)
        self.scale = mutation
        if not np.all(np.isfinite(mutation)) or np.any(np.array(mutation) >= 2) or np.any(np.array(mutation) < 0):
            raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()
        self.cross_over_probability = recombination
        self.func = func
        self.args = args
        self.limits = np.array(bounds, dtype='float').T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError('bounds should be a sequence containing real valued (min, max) pairs for each value in x')
        if maxiter is None:
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)
        self.num_population_members = max(5, popsize * self.parameter_count)
        self.population_shape = (self.num_population_members, self.parameter_count)
        self._nfev = 0
        if isinstance(init, string_types):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)
        self.disp = disp

    def init_population_lhs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes the population with Latin Hypercube Sampling.\n        Latin Hypercube Sampling ensures that each parameter is uniformly\n        sampled over its range.\n        '
        rng = self.random_number_generator
        segsize = 1.0 / self.num_population_members
        samples = segsize * rng.random_sample(self.population_shape) + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[:, np.newaxis]
        self.population = np.zeros_like(samples)
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    def init_population_random(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialises the population at random.  This type of initialization\n        can possess clustering, Latin Hypercube sampling is generally better.\n        '
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    def init_population_array(self, init):
        if False:
            while True:
                i = 10
        '\n        Initialises the population with a user specified population.\n        Parameters\n        ----------\n        init : np.ndarray\n            Array specifying subset of the initial population. The array should\n            have shape (M, len(x)), where len(x) is the number of parameters.\n            The population is clipped to the lower and upper `bounds`.\n        '
        popn = np.asfarray(init)
        if np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2:
            raise ValueError('The population supplied needs to have shape (M, len(x)), where M > 4.')
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = np.size(self.population, 0)
        self.population_shape = (self.num_population_members, self.parameter_count)
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    @property
    def x(self):
        if False:
            print('Hello World!')
        '\n        The best solution from the solver\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        '
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        if False:
            i = 10
            return i + 15
        '\n        The standard deviation of the population energies divided by their\n        mean.\n        '
        return np.std(self.population_energies) / np.abs(np.mean(self.population_energies) + _MACHEPS)

    def solve(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Runs the DifferentialEvolutionSolver.\n        Returns\n        -------\n        res : OptimizeResult\n            The optimization result represented as a ``OptimizeResult`` object.\n            Important attributes are: ``x`` the solution array, ``success`` a\n            Boolean flag indicating if the optimizer exited successfully and\n            ``message`` which describes the cause of the termination. See\n            `OptimizeResult` for a description of other attributes. If `polish`\n            was employed, and a lower minimum was obtained by the polishing,\n            then OptimizeResult also contains the ``jac`` attribute.\n        '
        (nit, warning_flag) = (0, False)
        status_message = _status_message['success']
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        for nit in range(1, self.maxiter + 1):
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break
            if self.disp:
                print(f'differential_evolution step {nit}: f(x)= {self.population_energies[0]}')
            convergence = self.convergence
            if self.callback and self.callback(self._scale_parameters(self.population[0]), convergence=self.tol / convergence) is True:
                warning_flag = True
                status_message = 'callback function requested stop early by returning True'
                break
            intol = np.std(self.population_energies) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))
            if warning_flag or intol:
                break
        else:
            status_message = _status_message['maxiter']
            warning_flag = True
        de_result = OptimizeResult(x=self.x, fun=self.population_energies[0], nfev=self._nfev, nit=nit, message=status_message, success=warning_flag is not True)
        if self.polish:
            result = minimize(self.func, np.copy(de_result.x), method='L-BFGS-B', bounds=self.limits.T, args=self.args)
            self._nfev += result.nfev
            de_result.nfev = self._nfev
            if result.fun < de_result.fun:
                de_result.fun = result.fun
                de_result.x = result.x
                de_result.jac = result.jac
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)
        return de_result

    def _calculate_population_energies(self):
        if False:
            return 10
        '\n        Calculate the energies of all the population members at the same time.\n        Puts the best member in first place. Useful if the population has just\n        been initialised.\n        '
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates])
        energies = self.func(parameters, *self.args)
        self.population_energies = energies
        self._nfev += itersize
        minval = np.argmin(self.population_energies)
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy
        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        '\n        Evolve the population by a single generation\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        fun : float\n            Value of objective function obtained from the best solution.\n        '
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        if self.dither is not None:
            self.scale = self.random_number_generator.rand() * (self.dither[1] - self.dither[0]) + self.dither[0]
        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))
        trials = np.array([self._mutate(c) for c in range(itersize)])
        for trial in trials:
            self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize
        for (candidate, (energy, trial)) in enumerate(zip(energies, trials)):
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial
        return (self.x, self.population_energies[0])

    def next(self):
        if False:
            while True:
                i = 10
        '\n        Evolve the population by a single generation\n        Returns\n        -------\n        x : ndarray\n            The best solution from the solver.\n        fun : float\n            Value of objective function obtained from the best solution.\n        '
        return self.__next__()

    def _scale_parameters(self, trial):
        if False:
            return 10
        '\n        scale from a number between 0 and 1 to parameters.\n        '
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        if False:
            i = 10
            return i + 15
        '\n        scale from parameters to a number between 0 and 1.\n        '
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        if False:
            return 10
        '\n        make sure the parameters lie between the limits\n        '
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        if False:
            return 10
        '\n        create a trial vector based on a mutation strategy\n        '
        trial = np.copy(self.population[candidate])
        rng = self.random_number_generator
        fill_point = rng.randint(0, self.parameter_count)
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))
        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial
        if self.strategy in self._exponential:
            i = 0
            while i < self.parameter_count and rng.rand() < self.cross_over_probability:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1
            return trial

    def _best1(self, samples):
        if False:
            while True:
                i = 10
        '\n        best1bin, best1exp\n        '
        (r_0, r_1) = samples[:2]
        return self.population[0] + self.scale * (self.population[r_0] - self.population[r_1])

    def _rand1(self, samples):
        if False:
            return 10
        '\n        rand1bin, rand1exp\n        '
        (r_0, r_1, r_2) = samples[:3]
        return self.population[r_0] + self.scale * (self.population[r_1] - self.population[r_2])

    def _randtobest1(self, samples):
        if False:
            for i in range(10):
                print('nop')
        '\n        randtobest1bin, randtobest1exp\n        '
        (r_0, r_1, r_2) = samples[:3]
        bprime = np.copy(self.population[r_0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r_1] - self.population[r_2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        if False:
            for i in range(10):
                print('nop')
        '\n        currenttobest1bin, currenttobest1exp\n        '
        (r_0, r_1) = samples[:2]
        bprime = self.population[candidate] + self.scale * (self.population[0] - self.population[candidate] + self.population[r_0] - self.population[r_1])
        return bprime

    def _best2(self, samples):
        if False:
            return 10
        '\n        best2bin, best2exp\n        '
        (r_0, r_1, r_2, r_3) = samples[:4]
        bprime = self.population[0] + self.scale * (self.population[r_0] + self.population[r_1] - self.population[r_2] - self.population[r_3])
        return bprime

    def _rand2(self, samples):
        if False:
            for i in range(10):
                print('nop')
        '\n        rand2bin, rand2exp\n        '
        (r_0, r_1, r_2, r_3, r_4) = samples
        bprime = self.population[r_0] + self.scale * (self.population[r_1] + self.population[r_2] - self.population[r_3] - self.population[r_4])
        return bprime

    def _select_samples(self, candidate, number_samples):
        if False:
            for i in range(10):
                print('nop')
        "\n        obtain random integers from range(self.num_population_members),\n        without replacement.  You can't have the original candidate either.\n        "
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs