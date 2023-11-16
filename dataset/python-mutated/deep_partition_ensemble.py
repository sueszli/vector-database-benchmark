"""
Creates a Deep Partition Aggregation ensemble classifier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import warnings
from typing import List, Optional, Union, Callable, Dict, TYPE_CHECKING
import copy
import numpy as np
from art.estimators.classification.ensemble import EnsembleClassifier
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE, CLASSIFIER_NEURALNETWORK_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class DeepPartitionEnsemble(EnsembleClassifier):
    """
    Implementation of Deep Partition Aggregation Defense. Training data is partitioned into
    disjoint buckets based on a hash function and a classifier is trained on each bucket.

    | Paper link: https://arxiv.org/abs/2006.14768
    """
    estimator_params = EnsembleClassifier.estimator_params + ['hash_function', 'ensemble_size']

    def __init__(self, classifiers: Union['CLASSIFIER_NEURALNETWORK_TYPE', List['CLASSIFIER_NEURALNETWORK_TYPE']], hash_function: Optional[Callable]=None, ensemble_size: int=50, channels_first: bool=False, clip_values: Optional['CLIP_VALUES_TYPE']=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0.0, 1.0)) -> None:
        if False:
            i = 10
            return i + 15
        '\n        :param classifiers: The base model definition to use for defining the ensemble.\n               If a list, the list must be the same size as the ensemble size.\n        :param hash_function: The function used to partition the training data. If empty, the hash function\n               will use the sum of the input values modulo the ensemble size for partitioning.\n        :param ensemble_size: The number of models in the ensemble.\n        :param channels_first: Set channels first or last.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier. Not applicable\n               in this classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one. Not applicable in this classifier.\n        '
        self.can_fit = False
        if not isinstance(classifiers, list):
            warnings.warn('If a single classifier is passed, it should not have been loaded                 from disk due to cloning errors with models loaded from disk. If you are                 using pre-trained model(s), create a list of Estimator objects the same                 length as the ensemble size')
            self.can_fit = True
            if hasattr(classifiers, 'clone_for_refitting'):
                try:
                    classifiers = [classifiers.clone_for_refitting() for _ in range(ensemble_size)]
                except ValueError as error:
                    warnings.warn('Switching to deepcopy due to ART Cloning Error: ' + str(error))
                    classifiers = [copy.deepcopy(classifiers) for _ in range(ensemble_size)]
            else:
                classifiers = [copy.deepcopy(classifiers) for _ in range(ensemble_size)]
        elif isinstance(classifiers, list) and len(classifiers) != ensemble_size:
            raise ValueError('The length of the classifier list must be the same as the ensemble size')
        super().__init__(classifiers=classifiers, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        if hash_function is None:

            def default_hash(x):
                if False:
                    print('Hello World!')
                return int(np.sum(x)) % ensemble_size
            self.hash_function = default_hash
        else:
            self.hash_function = hash_function
        self.ensemble_size = ensemble_size

    def predict(self, x: np.ndarray, batch_size: int=128, raw: bool=False, max_aggregate: bool=True, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Perform prediction for a batch of inputs. Aggregation will be performed on the prediction from\n        each classifier if max_aggregate is True. Otherwise, the probabilities will be summed instead.\n        For logits output set max_aggregate=True, as logits are not comparable between models and should\n        not be aggregated using a sum.\n\n        :param x: Input samples.\n        :param batch_size: Size of batches.\n        :param raw: Return the individual classifier raw outputs (not aggregated).\n        :param max_aggregate: Aggregate the predicted classes of each classifier if True. If false, aggregation\n               is done using a sum. If raw is true, this arg is ignored\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`, or of shape\n                 `(nb_classifiers, nb_inputs, nb_classes)` if `raw=True`.\n        '
        if raw:
            return super().predict(x, batch_size=batch_size, raw=True, **kwargs)
        if max_aggregate:
            preds = super().predict(x, batch_size=batch_size, raw=True, **kwargs)
            aggregated_preds = np.zeros_like(preds, shape=preds.shape[1:])
            for i in range(preds.shape[0]):
                aggregated_preds[np.arange(len(aggregated_preds)), np.argmax(preds[i], axis=1)] += 1
            return aggregated_preds
        return super().predict(x, batch_size=batch_size, raw=False, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=20, train_dict: Dict=None, **kwargs) -> None:
        if False:
            return 10
        "\n        Fit the classifier on the training set `(x, y)`. Each classifier will be trained with the\n        same parameters unless train_dict is provided. If train_dict is provided, the model id's\n        specified will use the training parameters in train_dict instead.\n\n        :param x: Training data.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param train_dict: A dictionary of training args if certain models need specialized arguments.\n               The key should be the model's partition id and this will override any default training\n               parameters including batch_size and nb_epochs.\n        :param kwargs: Dictionary of framework-specific arguments.\n        "
        if self.can_fit:
            partition_ind = [[] for _ in range(self.ensemble_size)]
            for (i, p_x) in enumerate(x):
                partition_id = int(self.hash_function(p_x))
                partition_ind[partition_id].append(i)
            for i in range(self.ensemble_size):
                current_x = x[np.array(partition_ind[i])]
                current_y = y[np.array(partition_ind[i])]
                if train_dict is not None and i in train_dict.keys():
                    self.classifiers[i].fit(current_x, current_y, **train_dict[i])
                else:
                    self.classifiers[i].fit(current_x, current_y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)
        else:
            warnings.warn('Cannot call fit() for an ensemble of pre-trained classifiers.')