"""
This module implements the copycat cnn attack `CopycatCNN`.

| Paper link: https://arxiv.org/abs/1806.05476
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import ExtractionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import to_categorical
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class CopycatCNN(ExtractionAttack):
    """
    Implementation of the Copycat CNN attack from Rodrigues Correia-Silva et al. (2018).

    | Paper link: https://arxiv.org/abs/1806.05476
    """
    attack_params = ExtractionAttack.attack_params + ['batch_size_fit', 'batch_size_query', 'nb_epochs', 'nb_stolen', 'use_probability']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_TYPE', batch_size_fit: int=1, batch_size_query: int=1, nb_epochs: int=10, nb_stolen: int=1, use_probability: bool=False) -> None:
        if False:
            return 10
        '\n        Create a Copycat CNN attack instance.\n\n        :param classifier: A victim classifier.\n        :param batch_size_fit: Size of batches for fitting the thieved classifier.\n        :param batch_size_query: Size of batches for querying the victim classifier.\n        :param nb_epochs: Number of epochs to use for training.\n        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.\n        '
        super().__init__(estimator=classifier)
        self.batch_size_fit = batch_size_fit
        self.batch_size_query = batch_size_query
        self.nb_epochs = nb_epochs
        self.nb_stolen = nb_stolen
        self.use_probability = use_probability
        self._check_params()

    def extract(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> 'CLASSIFIER_TYPE':
        if False:
            return 10
        '\n        Extract a thieved classifier.\n\n        :param x: An array with the source input to the victim classifier.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). Not used in this attack.\n        :param thieved_classifier: A classifier to be stolen, currently always trained on one-hot labels.\n        :type thieved_classifier: :class:`.Classifier`\n        :return: The stolen classifier.\n        '
        if y is not None:
            logger.warning('This attack does not use the provided label y.')
        if x.shape[0] < self.nb_stolen:
            logger.warning('The size of the source input is smaller than the expected number of queries submitted to the victim classifier.')
        thieved_classifier = kwargs['thieved_classifier']
        if thieved_classifier is None or not isinstance(thieved_classifier, ClassifierMixin):
            raise ValueError('A thieved classifier is needed.')
        selected_x = self._select_data(x)
        fake_labels = self._query_label(selected_x)
        thieved_classifier.fit(x=selected_x, y=fake_labels, batch_size=self.batch_size_fit, nb_epochs=self.nb_epochs)
        return thieved_classifier

    def _select_data(self, x: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Select data to attack.\n\n        :param x: An array with the source input to the victim classifier.\n        :return: An array with the selected input to the victim classifier.\n        '
        nb_stolen = np.minimum(self.nb_stolen, x.shape[0])
        rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)
        return x[rnd_index].astype(ART_NUMPY_DTYPE)

    def _query_label(self, x: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Query the victim classifier.\n\n        :param x: An array with the source input to the victim classifier.\n        :return: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).\n        '
        labels = self.estimator.predict(x=x, batch_size=self.batch_size_query)
        if not self.use_probability:
            labels = np.argmax(labels, axis=1)
            labels = to_categorical(labels=labels, nb_classes=self.estimator.nb_classes)
        return labels

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self.batch_size_fit, int) or self.batch_size_fit <= 0:
            raise ValueError('The size of batches for fitting the thieved classifier must be a positive integer.')
        if not isinstance(self.batch_size_query, int) or self.batch_size_query <= 0:
            raise ValueError('The size of batches for querying the victim classifier must be a positive integer.')
        if not isinstance(self.nb_epochs, int) or self.nb_epochs <= 0:
            raise ValueError('The number of epochs must be a positive integer.')
        if not isinstance(self.nb_stolen, int) or self.nb_stolen <= 0:
            raise ValueError('The number of queries submitted to the victim classifier must be a positive integer.')
        if not isinstance(self.use_probability, bool):
            raise ValueError('The argument `use_probability` has to be of type bool.')