"""
This module implements the `LowProFool` attack. This is a white-box attack.

Its main objective is to take a valid tabular sample and transform it, so that a given classifier predicts it to be some
target class.

`LowProFool` attack transforms the provided real-valued tabular data into adversaries of the specified target classes.
The generated adversaries have to be as close as possible to the original samples in terms of the weighted Lp-norm,
where the weights determine each feature's importance.

| Paper link: https://arxiv.org/abs/1911.03274
"""
import logging
from typing import Callable, Optional, Union, TYPE_CHECKING
import numpy as np
from scipy.stats import pearsonr
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import LossGradientsMixin
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class LowProFool(EvasionAttack):
    """
    `LowProFool` attack.

    | Paper link: https://arxiv.org/abs/1911.03274
    """
    attack_params = EvasionAttack.attack_params + ['n_steps', 'threshold', 'lambd', 'eta', 'eta_decay', 'eta_min', 'norm', 'importance', 'verbose']
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', n_steps: int=100, threshold: Union[float, None]=0.5, lambd: float=1.5, eta: float=0.2, eta_decay: float=0.98, eta_min: float=1e-07, norm: Union[int, float, str]=2, importance: Union[Callable, str, np.ndarray]='pearson', verbose: bool=False) -> None:
        if False:
            print('Hello World!')
        "\n        Create a LowProFool instance.\n\n        :param classifier: Appropriate classifier's instance\n        :param n_steps: Number of iterations to follow\n        :param threshold: Lowest prediction probability of a valid adversary\n        :param lambd: Amount of lp-norm impact on objective function\n        :param eta: Rate of updating the perturbation vectors\n        :param eta_decay: Step-by-step decrease of eta\n        :param eta_min: Minimal eta value\n        :param norm: Parameter `p` for Lp-space norm (norm=2 - euclidean norm)\n        :param importance: Function to calculate feature importance with\n            or vector of those precomputed; possibilities:\n            > 'pearson' - Pearson correlation (string)\n            > function  - Custom function (callable object)\n            > vector    - Vector of feature importance (np.ndarray)\n        :param verbose: Verbose mode / Show progress bars.\n        "
        super().__init__(estimator=classifier)
        self.n_steps = n_steps
        self.threshold = threshold
        self.lambd = lambd
        self.eta = eta
        self.eta_decay = eta_decay
        self.eta_min = eta_min
        self.norm = norm
        self.importance = importance
        self.verbose = verbose
        self._targeted = True
        self.n_classes = self.estimator.nb_classes
        self.n_features = self.estimator.input_shape[0]
        self.importance_vec = None
        if self.estimator.clip_values is None:
            logger.warning('The `clip_values` attribute of the estimator is `None`, therefore this instance of LowProFool will by default generate adversarial perturbations without clipping them.')
        self._check_params()
        if isinstance(self.importance, np.ndarray):
            self.importance_vec = self.importance
        if eta_decay < 1 and eta_min > 0:
            steps_before_min_eta_reached = np.ceil(np.log(eta_min / eta) / np.log(eta_decay))
            if steps_before_min_eta_reached / self.n_steps < 0.8:
                logger.warning("The given combination of 'n_steps', 'eta', 'eta_decay' and 'eta_min' effectively sets learning rate to its minimal value after about %d steps out of all %d.", steps_before_min_eta_reached, self.n_steps)

    def __weighted_lp_norm(self, perturbations: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Lp-norm of perturbation vectors weighted by feature importance.\n\n        :param perturbations: Perturbations of samples towards being adversarial.\n        :return: Array with weighted Lp-norm of perturbations.\n        '
        return self.lambd * np.linalg.norm(self.importance_vec * perturbations, axis=1, ord=np.inf if self.norm == 'inf' else self.norm).reshape(-1, 1)

    def __weighted_lp_norm_gradient(self, perturbations: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Gradient of the weighted Lp-space norm with regards to the data vector.\n\n        :param perturbations: Perturbations of samples towards being adversarial.\n        :return: Weighted Lp-norm gradients array.\n        '
        norm = self.norm
        if isinstance(norm, (int, float)) and norm < np.inf and (self.importance_vec is not None):
            numerator = self.importance_vec * self.importance_vec * perturbations * np.power(np.abs(perturbations), norm - 2)
            denominator = np.power(np.sum(np.power(self.importance_vec * perturbations, norm)), (norm - 1) / norm)
            numerator = np.where(denominator > 1e-10, numerator, np.zeros(numerator.shape[1]))
            denominator = np.where(denominator <= 1e-10, 1.0, denominator)
            return numerator / denominator
        numerator = np.array(self.importance_vec * perturbations)
        optimum = np.max(np.abs(numerator))
        return np.where(abs(numerator) == optimum, np.sign(numerator), 0)

    def __get_gradients(self, samples: np.ndarray, perturbations: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Gradient of the objective function with regards to the data vector, i.e. sum of the classifier's loss gradient\n        and weighted lp-space norm gradient, both with regards to data vector.\n\n        :param samples: Base design matrix.\n        :param perturbations: Perturbations of samples towards being adversarial.\n        :param targets: The target labels for the attack.\n        :return: Aggregate gradient of objective function.\n        "
        clf_loss_grad = self.estimator.loss_gradient((samples + perturbations).astype(np.float32), targets.astype(np.float32))
        norm_grad = self.lambd * self.__weighted_lp_norm_gradient(perturbations)
        return clf_loss_grad + norm_grad

    def __apply_clipping(self, samples: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Function for clipping perturbation vectors to forbid the adversary vectors to go beyond the allowed ranges of\n        values.\n\n        :param samples: Base design matrix.\n        :param perturbations: Perturbations of samples towards being adversarial.\n        :return: Clipped perturbation array.\n        '
        if self.estimator.clip_values is None:
            return perturbations
        mins = self.estimator.clip_values[0]
        maxs = self.estimator.clip_values[1]
        np.clip(perturbations, mins - samples, maxs - samples, perturbations)
        return perturbations

    def __calculate_feature_importances(self, x: np.ndarray, y: np.ndarray) -> None:
        if False:
            while True:
                i = 10
        '\n        This function calculates feature importances using a specified built-in function or applies a provided custom\n        function (callable object). It calculates those values on the passed training data.\n\n        :param x: Design matrix of the dataset used to train the classifier.\n        :param y: Labels of the dataset used to train the classifier.\n        :return: None.\n        '
        if self.importance == 'pearson':
            pearson_correlations = [pearsonr(x[:, col], y)[0] for col in range(x.shape[1])]
            absolutes = np.abs(np.array(pearson_correlations))
            self.importance_vec = absolutes / np.power(np.sum(absolutes ** 2), 0.5)
        elif callable(self.importance):
            try:
                self.importance_vec = np.array(self.importance(x, y))
            except Exception as exception:
                logger.exception('Provided importance function has failed.')
                raise exception
            if not isinstance(self.importance_vec, np.ndarray):
                self.importance_vec = None
                raise TypeError('Feature importance vector should be of type np.ndarray or any convertible to that.')
            if self.importance_vec.shape != (self.n_features,):
                self.importance_vec = None
                raise ValueError('Feature has to be one-dimensional array of size (n_features, ).')
        else:
            raise TypeError(f'Unrecognized feature importance function: {self.importance}')

    def fit_importances(self, x: Optional[np.ndarray]=None, y: Optional[np.ndarray]=None, importance_array: Optional[np.ndarray]=None, normalize: Optional[bool]=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function allows one to easily calculate the feature importance vector using the pre-specified function,\n        in case it wasn't passed at initialization.\n\n        :param x: Design matrix of the dataset used to train the classifier.\n        :param y: Labels of the dataset used to train the classifier.\n        :param importance_array: Array providing features' importance score.\n        :param normalize: Assure that feature importance values sum to 1.\n        :return: LowProFool instance itself.\n        "
        if importance_array is not None:
            if np.array(importance_array).shape == (self.n_features,):
                self.importance_vec = np.array(importance_array)
            else:
                raise ValueError('Feature has to be one-dimensional array of size (n_features, ).')
        elif self.importance_vec is None:
            self.__calculate_feature_importances(np.array(x), np.array(y))
        if normalize:
            if self.importance_vec is not None:
                self.importance_vec = np.array(self.importance_vec) / np.sum(self.importance_vec)
            else:
                raise ValueError('Unexpected `None` detected.')
        return self

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate adversaries for the samples passed in the `x` data matrix, whose targets are specified in `y`,\n        one-hot-encoded target matrix. This procedure makes use of the LowProFool algorithm. In the case of failure,\n        the resulting array will contain the initial samples on the problematic positions - which otherwise should\n        contain the best adversary found in the process.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: One-hot-encoded target classes of shape (nb_samples, nb_classes).\n        :param kwargs:\n        :return: An array holding the adversarial examples.\n        '
        if self.importance_vec is None:
            raise ValueError('No feature importance vector has been provided yet.')
        if y is None:
            raise ValueError('It is required to pass target classes as `y` parameter.')
        samples = np.array(x, dtype=np.float64)
        targets = np.array(y, dtype=np.float64)
        targets_integer = np.argmax(y, axis=1)
        if targets.shape[1] != self.n_classes:
            raise ValueError('Targets shape is not compatible with number of classes.')
        if samples.shape[1] != self.n_features:
            raise ValueError('Samples shape is not compatible with number of features.')
        perturbations = np.zeros(samples.shape, dtype=np.float64)
        eta = self.eta
        best_norm_losses = np.inf * np.ones(samples.shape[0], dtype=np.float64)
        best_perturbations = perturbations.copy()
        success_indicators = np.zeros(samples.shape[0], dtype=np.float64)

        def met_target(probas, target_class):
            if False:
                print('Hello World!')
            if self.threshold is None:
                return np.argmax(probas) == target_class
            return probas[target_class] > self.threshold
        for _ in trange(self.n_steps, desc='LowProFool', disable=not self.verbose):
            grad = self.__get_gradients(samples, perturbations, targets)
            perturbations -= eta * grad
            perturbations = self.__apply_clipping(samples, perturbations)
            eta = max(eta * self.eta_decay, self.eta_min)
            y_probas = self.estimator.predict((samples + perturbations).astype(np.float32))
            for (j, target_int) in enumerate(targets_integer):
                if met_target(y_probas[j], target_int):
                    success_indicators[j] = 1.0
                    norm_loss = self.__weighted_lp_norm(perturbations[j:j + 1])[0, 0]
                    if norm_loss < best_norm_losses[j]:
                        best_norm_losses[j] = norm_loss
                        best_perturbations[j] = perturbations[j].copy()
        logger.info('Success rate of LowProFool attack: %.2f}%%', 100 * np.sum(success_indicators) / success_indicators.size)
        return samples + best_perturbations

    def _check_params(self) -> None:
        if False:
            return 10
        '\n        Check correctness of parameters.\n\n        :return: None.\n        '
        if not (isinstance(self.n_classes, int) and self.n_classes > 0):
            raise ValueError('The argument `n_classes` has to be positive integer.')
        if not (isinstance(self.n_features, int) and self.n_features > 0):
            raise ValueError('The argument `n_features` has to be positive integer.')
        if not (isinstance(self.n_steps, int) and self.n_steps > 0):
            raise ValueError('The argument `n_steps` (number of iterations) has to be positive integer.')
        if not (isinstance(self.threshold, float) and 0 < self.threshold < 1 or self.threshold is None):
            raise ValueError('The argument `threshold` has to be either float in range (0, 1) or None.')
        if not (isinstance(self.lambd, (float, int)) and self.lambd >= 0):
            raise ValueError('The argument `lambd` has to be non-negative float or integer.')
        if not (isinstance(self.eta, (float, int)) and self.eta > 0):
            raise ValueError('The argument `eta` has to be positive float or integer.')
        if not (isinstance(self.eta_decay, (float, int)) and 0 < self.eta_decay <= 1):
            raise ValueError('The argument `eta_decay` has to be float or integer in range (0, 1].')
        if not (isinstance(self.eta_min, (float, int)) and self.eta_min >= 0):
            raise ValueError('The argument `eta_min` has to be non-negative float or integer.')
        if not (isinstance(self.norm, (float, int)) and self.norm > 0 or (isinstance(self.norm, str) and self.norm == 'inf') or self.norm == np.inf):
            raise ValueError('The argument `norm` has to be either positive-valued float or integer, np.inf, or "inf".')
        if not (isinstance(self.importance, str) or callable(self.importance) or (isinstance(self.importance, np.ndarray) and self.importance.shape == (self.n_features,))):
            raise ValueError('The argument `importance` has to be either string, ' + 'callable or np.ndarray of the shape (n_features, ).')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')