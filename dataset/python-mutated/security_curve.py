"""
This module implements the evaluation of Security Curves.

Examples of Security Curves can be found in Figure 6 of Madry et al., 2017 (https://arxiv.org/abs/1706.06083).
"""
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from art.evaluations.evaluation import Evaluation
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

class SecurityCurve(Evaluation):
    """
    This class implements the evaluation of Security Curves.

    Examples of Security Curves can be found in Figure 6 of Madry et al., 2017 (https://arxiv.org/abs/1706.06083).
    """

    def __init__(self, eps: Union[int, List[float], List[int]]):
        if False:
            print('Hello World!')
        '\n        Create an instance of a Security Curve evaluation.\n\n        :param eps: Defines the attack budgets `eps` for Projected Gradient Descent used for evaluation.\n        '
        self.eps = eps
        self.eps_list: List[float] = []
        self.accuracy_adv_list: List[float] = []
        self.accuracy: Optional[float] = None

    def evaluate(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', x: np.ndarray, y: np.ndarray, **kwargs: Union[str, bool, int, float]) -> Tuple[List[float], List[float], float]:
        if False:
            return 10
        '\n        Evaluate the Security Curve of a classifier using Projected Gradient Descent.\n\n        :param classifier: A trained classifier that provides loss gradients.\n        :param x: Input data to classifier for evaluation.\n        :param y: True labels for input data `x`.\n        :param kwargs: Keyword arguments for the Projected Gradient Descent attack used for evaluation, except keywords\n                       `classifier` and `eps`.\n        :return: List of evaluated `eps` values, List of adversarial accuracies, and benign accuracy.\n        '
        kwargs.pop('classifier', None)
        kwargs.pop('eps', None)
        self.eps_list.clear()
        self.accuracy_adv_list.clear()
        self.accuracy = None
        if isinstance(self.eps, int):
            if classifier.clip_values is not None:
                eps_increment = (classifier.clip_values[1] - classifier.clip_values[0]) / self.eps
            else:
                eps_increment = (np.max(x) - np.min(x)) / self.eps
            for i in range(1, self.eps + 1):
                self.eps_list.append(float(i * eps_increment))
        else:
            self.eps_list = [float(eps) for eps in self.eps]
        y_pred = classifier.predict(x=x, y=y)
        self.accuracy = self._get_accuracy(y=y, y_pred=y_pred)
        for eps in self.eps_list:
            attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, **kwargs)
            x_adv = attack_pgd.generate(x=x, y=y)
            y_pred_adv = classifier.predict(x=x_adv, y=y)
            accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)
            self.accuracy_adv_list.append(accuracy_adv)
        self._check_gradient(classifier=classifier, x=x, y=y, **kwargs)
        return (self.eps_list, self.accuracy_adv_list, self.accuracy)

    @property
    def detected_obfuscating_gradients(self) -> bool:
        if False:
            print('Hello World!')
        '\n        This property describes if the previous call to method `evaluate` identified potential gradient obfuscation.\n        '
        return self._detected_obfuscating_gradients

    def _check_gradient(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', x: np.ndarray, y: np.ndarray, **kwargs: Union[str, bool, int, float]) -> None:
        if False:
            while True:
                i = 10
        '\n        Check if potential gradient obfuscation can be detected. Projected Gradient Descent with 100 iterations is run\n        with maximum attack budget `eps` being equal to upper clip value of input data and `eps_step` of\n        `eps / (max_iter / 2)`.\n\n        :param classifier: A trained classifier that provides loss gradients.\n        :param x: Input data to classifier for evaluation.\n        :param y: True labels for input data `x`.\n        :param kwargs: Keyword arguments for the Projected Gradient Descent attack used for evaluation, except keywords\n                       `classifier` and `eps`.\n        '
        max_iter = 100
        kwargs['max_iter'] = max_iter
        if classifier.clip_values is not None:
            clip_value_max = classifier.clip_values[1]
        else:
            clip_value_max = np.max(x)
        kwargs['eps'] = float(clip_value_max)
        kwargs['eps_step'] = float(clip_value_max / (max_iter / 2))
        attack_pgd = ProjectedGradientDescent(estimator=classifier, **kwargs)
        x_adv = attack_pgd.generate(x=x, y=y)
        y_pred_adv = classifier.predict(x=x_adv, y=y)
        accuracy_adv = self._get_accuracy(y=y, y_pred=y_pred_adv)
        if accuracy_adv > 1 / classifier.nb_classes:
            self._detected_obfuscating_gradients = True
        else:
            self._detected_obfuscating_gradients = False

    def plot(self) -> None:
        if False:
            print('Hello World!')
        '\n        Plot the Security Curve of adversarial accuracy as function opf attack budget `eps` together with the accuracy\n        on benign samples.\n        '
        from matplotlib import pyplot as plt
        plt.plot(self.eps_list, self.accuracy_adv_list, label='adversarial', marker='o')
        plt.plot([self.eps_list[0], self.eps_list[-1]], [self.accuracy, self.accuracy], linestyle='--', label='benign')
        plt.legend()
        plt.xlabel('Attack budget eps')
        plt.ylabel('Accuracy')
        if self.detected_obfuscating_gradients:
            plt.title('Potential gradient obfuscation detected.')
        else:
            plt.title('No gradient obfuscation detected')
        plt.ylim([0, 1.05])
        plt.show()

    @staticmethod
    def _get_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate accuracy of predicted labels.\n\n        :param y: True labels.\n        :param y_pred: Predicted labels.\n        :return: Accuracy.\n        '
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).item()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        repr_ = f"{self.__module__ + '.' + self.__class__.__name__}(eps={self.eps})"
        return repr_