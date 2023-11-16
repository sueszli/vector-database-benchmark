"""
This module implements the abstract base classes for all attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from art.exceptions import EstimatorError
from art.summary_writer import SummaryWriter, SummaryWriterDefault
from art.utils import get_feature_index
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, GENERATOR_TYPE
logger = logging.getLogger(__name__)

class InputFilter(abc.ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls
    """

    def __init__(cls, name, bases, clsdict):
        if False:
            i = 10
            return i + 15
        '\n        This function overrides any existing generate or extract methods with a new method that\n        ensures the input is an `np.ndarray`. There is an assumption that the input object has implemented\n        __array__ with np.array calls.\n        '

        def make_replacement(fdict, func_name):
            if False:
                print('Hello World!')
            '\n            This function overrides creates replacement functions dynamically\n            '

            def replacement_function(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if len(args) > 0:
                    lst = list(args)
                else:
                    lst = []
                if 'x' in kwargs:
                    if not isinstance(kwargs['x'], np.ndarray):
                        kwargs['x'] = np.array(kwargs['x'])
                elif not isinstance(args[0], np.ndarray):
                    lst[0] = np.array(args[0])
                if 'y' in kwargs:
                    if kwargs['y'] is not None and (not isinstance(kwargs['y'], np.ndarray)):
                        kwargs['y'] = np.array(kwargs['y'])
                elif len(args) == 2:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])
                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)
            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = 'new_' + func_name
            return replacement_function
        replacement_list = ['generate', 'extract']
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)

class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """
    attack_params: List[str] = []
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(self, estimator, summary_writer: Union[str, bool, SummaryWriter]=False):
        if False:
            print('Hello World!')
        '\n        :param estimator: An estimator.\n        :param summary_writer: Activate summary writer for TensorBoard.\n                               Default is `False` and deactivated summary writer.\n                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.\n                               If of type `str` save in path.\n                               If of type `SummaryWriter` apply provided custom summary writer.\n                               Use hierarchical folder structure to compare between runs easily. e.g. pass in\n                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.\n        '
        super().__init__()
        if self.estimator_requirements is None:
            raise ValueError('Estimator requirements have not been defined in `_estimator_requirements`.')
        if not self.is_estimator_valid(estimator, self._estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)
        self._estimator = estimator
        self._summary_writer_arg = summary_writer
        self._summary_writer: Optional[SummaryWriter] = None
        if isinstance(summary_writer, SummaryWriter):
            self._summary_writer = summary_writer
        elif summary_writer:
            self._summary_writer = SummaryWriterDefault(summary_writer)
        Attack._check_params(self)

    @property
    def estimator(self):
        if False:
            i = 10
            return i + 15
        'The estimator.'
        return self._estimator

    @property
    def summary_writer(self):
        if False:
            while True:
                i = 10
        'The summary writer.'
        return self._summary_writer

    @property
    def estimator_requirements(self):
        if False:
            return 10
        'The estimator requirements.'
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.\n\n        :param kwargs: A dictionary of attack-specific parameters.\n        '
        for (key, value) in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
            else:
                raise ValueError(f'The attribute "{key}" cannot be set for this attack.')
        self._check_params()

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError('The argument `summary_writer` has to be either of type bool or str.')

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if the given estimator satisfies the requirements for this attack.\n\n        :param estimator: The estimator to check.\n        :param estimator_requirements: Estimator requirements.\n        :return: True if the estimator is valid for the attack.\n        '
        for req in estimator_requirements:
            if isinstance(req, tuple):
                if all((p not in type(estimator).__mro__ for p in req)):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string describing the attack class and attack_params\n        '
        param_str = ''
        for param in self.attack_params:
            if hasattr(self, param):
                param_str += f'{param}={getattr(self, param)}, '
            elif hasattr(self, '_attack'):
                if hasattr(self._attack, param):
                    param_str += f'{param}={getattr(self._attack, param)}, '
        return f'{type(self).__name__}({param_str})'

class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Generate adversarial examples and return them as an array. This method should be overridden by all concrete\n        evasion attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Correct labels or target labels for `x`, depending if the attack is targeted\n                  or not. This parameter is only used by some of the attacks.\n        :return: An array holding the adversarial examples.\n        '
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return Boolean if attack is targeted. Return None if not applicable.\n        '
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        if False:
            return 10
        self._targeted = targeted

class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes
    """

    def __init__(self, classifier: Optional['CLASSIFIER_TYPE']) -> None:
        if False:
            return 10
        '\n        :param classifier: A trained classifier (or none if no classifier is needed)\n        '
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Generate poisoning examples and return them as an array. This method should be overridden by all concrete\n        poisoning attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y:  Target labels for `x`. Untargeted attacks set this value to None.\n        :return: An tuple holding the (poisoning examples, poisoning labels).\n        '
        raise NotImplementedError

class PoisoningAttackGenerator(Attack):
    """
    Abstract base class for poisoning attack classes that return a transformed generator.
    These attacks have an additional method, `poison_estimator`, that returns the poisoned generator.
    """

    def __init__(self, generator: 'GENERATOR_TYPE') -> None:
        if False:
            print('Hello World!')
        '\n        :param generator: A generator\n        '
        super().__init__(generator)

    @abc.abstractmethod
    def poison_estimator(self, z_trigger: np.ndarray, x_target: np.ndarray, batch_size: int, max_iter: int, lambda_p: float, verbose: int, **kwargs) -> 'GENERATOR_TYPE':
        if False:
            return 10
        '\n        Returns a poisoned version of the generator used to initialize the attack\n        :return: A poisoned generator\n        '
        raise NotImplementedError

    @property
    def z_trigger(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the secret attacker trigger\n        '
        return self._z_trigger

    @property
    def x_target(self):
        if False:
            print('Hello World!')
        '\n        Returns the secret attacker target which the poisoned generator should produce\n        '
        return self._x_target

class PoisoningAttackTransformer(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that return a transformed classifier.
    These attacks have an additional method, `poison_estimator`, that returns the poisoned classifier.
    """

    def __init__(self, classifier: Optional['CLASSIFIER_TYPE']) -> None:
        if False:
            return 10
        '\n        :param classifier: A trained classifier (or none if no classifier is needed)\n        '
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Generate poisoning examples and return them as an array. This method should be overridden by all concrete\n        poisoning attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y:  Target labels for `x`. Untargeted attacks set this value to None.\n        :return: An tuple holding the (poisoning examples, poisoning labels).\n        :rtype: `(np.ndarray, np.ndarray)`\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> 'CLASSIFIER_TYPE':
        if False:
            print('Hello World!')
        '\n        Returns a poisoned version of the classifier used to initialize the attack\n        :param x: Training data\n        :param y: Training labels\n        :return: A poisoned classifier\n        '
        raise NotImplementedError

class PoisoningAttackObjectDetector(Attack):
    """
    Abstract base class for poisoning attack classes on object detection models.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initializes object detector poisoning attack.\n        '
        super().__init__(None)

    @abc.abstractmethod
    def poison(self, x: Union[np.ndarray, List[np.ndarray]], y: List[Dict[str, np.ndarray]], **kwargs) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[Dict[str, np.ndarray]]]:
        if False:
            return 10
        '\n        Generate poisoning examples and return them as an array. This method should be overridden by all concrete\n        poisoning attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: True labels of type `List[Dict[np.ndarray]]`, one dictionary per input image.\n                  The keys and values of the dictionary are:\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image\n                  - scores [N]: the scores or each prediction.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        raise NotImplementedError

class PoisoningAttackBlackBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have no access to the model (classifier object).
    """

    def __init__(self):
        if False:
            return 10
        '\n        Initializes black-box data poisoning attack.\n        '
        super().__init__(None)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Generate poisoning examples and return them as an array. This method should be overridden by all concrete\n        poisoning attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y:  Target labels for `x`. Untargeted attacks set this value to None.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        raise NotImplementedError

class PoisoningAttackWhiteBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have white-box access to the model (classifier object).
    """

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Generate poisoning examples and return them as an array. This method should be overridden by all concrete\n        poisoning attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Correct labels or target labels for `x`, depending if the attack is targeted\n               or not. This parameter is only used by some of the attacks.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        raise NotImplementedError

class ExtractionAttack(Attack):
    """
    Abstract base class for extraction attack classes.
    """

    @abc.abstractmethod
    def extract(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> 'CLASSIFIER_TYPE':
        if False:
            i = 10
            return i + 15
        '\n        Extract models and return them as an ART classifier. This method should be overridden by all concrete extraction\n        attack implementations.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Correct labels or target labels for `x`, depending if the attack is targeted\n               or not. This parameter is only used by some of the attacks.\n        :return: ART classifier of the extracted model.\n        '
        raise NotImplementedError

class InferenceAttack(Attack):
    """
    Abstract base class for inference attack classes.
    """

    def __init__(self, estimator):
        if False:
            return 10
        '\n        :param estimator: A trained estimator targeted for inference attack.\n        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`\n        '
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer sensitive attributes from the targeted estimator. This method\n        should be overridden by all concrete inference attack implementations.\n\n        :param x: An array with reference inputs to be used in the attack.\n        :param y: Labels for `x`. This parameter is only used by some of the attacks.\n        :return: An array holding the inferred attribute values.\n        '
        raise NotImplementedError

class AttributeInferenceAttack(InferenceAttack):
    """
    Abstract base class for attribute inference attack classes.
    """
    attack_params = InferenceAttack.attack_params + ['attack_feature']

    def __init__(self, estimator, attack_feature: Union[int, slice]=0):
        if False:
            return 10
        '\n        :param estimator: A trained estimator targeted for inference attack.\n        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`\n        :param attack_feature: The index of the feature to be attacked.\n        '
        super().__init__(estimator)
        self._check_attack_feature(attack_feature)
        self.attack_feature = get_feature_index(attack_feature)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Infer sensitive attributes from the targeted estimator. This method\n        should be overridden by all concrete inference attack implementations.\n\n        :param x: An array with reference inputs to be used in the attack.\n        :param y: Labels for `x`. This parameter is only used by some of the attacks.\n        :return: An array holding the inferred attribute values.\n        '
        raise NotImplementedError

    @staticmethod
    def _check_attack_feature(attack_feature: Union[int, slice]) -> None:
        if False:
            return 10
        if not isinstance(attack_feature, int) and (not isinstance(attack_feature, slice)):
            raise ValueError('Attack feature must be either an integer or a slice object.')
        if isinstance(attack_feature, int) and attack_feature < 0:
            raise ValueError('Attack feature index must be non-negative.')

    def _check_params(self) -> None:
        if False:
            return 10
        self._check_attack_feature(self.attack_feature)

class MembershipInferenceAttack(InferenceAttack):
    """
    Abstract base class for membership inference attack classes.
    """

    def __init__(self, estimator):
        if False:
            while True:
                i = 10
        '\n        :param estimator: A trained estimator targeted for inference attack.\n        :type estimator: :class:`.art.estimators.estimator.BaseEstimator`\n        :param attack_feature: The index of the feature to be attacked.\n        '
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Infer membership status of samples from the target estimator. This method\n        should be overridden by all concrete inference attack implementations.\n\n        :param x: An array with reference inputs to be used in the attack.\n        :param y: Labels for `x`. This parameter is only used by some of the attacks.\n        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just\n                              the predicted class.\n        :return: An array holding the inferred membership status (1 indicates member of training set,\n                 0 indicates non-member) or class probabilities.\n        '
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.\n        '
        super().set_params(**kwargs)
        self._check_params()

class ReconstructionAttack(Attack):
    """
    Abstract base class for reconstruction attack classes.
    """
    attack_params = InferenceAttack.attack_params

    def __init__(self, estimator):
        if False:
            print('Hello World!')
        '\n        :param estimator: A trained estimator targeted for reconstruction attack.\n        '
        super().__init__(estimator)

    @abc.abstractmethod
    def reconstruct(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Reconstruct the training dataset of and from the targeted estimator. This method\n        should be overridden by all concrete inference attack implementations.\n\n        :param x: An array with known records of the training set of `estimator`.\n        :param y: An array with known labels of the training set of `estimator`, if None predicted labels will be used.\n        :return: A tuple of two arrays for the reconstructed training input and labels.\n        '
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.\n        '
        super().set_params(**kwargs)
        self._check_params()