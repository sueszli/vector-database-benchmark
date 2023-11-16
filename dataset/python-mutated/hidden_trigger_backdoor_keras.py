"""
This module implements a Hidden Trigger Backdoor attack on Neural Networks.

| Paper link: https://arxiv.org/abs/1910.00033
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import six
from tqdm.auto import trange
from art.attacks.attack import PoisoningAttackWhiteBox
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier
from art.attacks.poisoning.hidden_trigger_backdoor.loss_meter import LossMeter
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier
logger = logging.getLogger(__name__)

class HiddenTriggerBackdoorKeras(PoisoningAttackWhiteBox):
    """
    Implementation of Hidden Trigger Backdoor Attack by Saha et al 2019.
    "Hidden Trigger Backdoor Attacks

    | Paper link: https://arxiv.org/abs/1910.00033
    """
    attack_params = PoisoningAttackWhiteBox.attack_params + ['target']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(self, classifier: Union['KerasClassifier', 'TensorFlowV2Classifier'], target: np.ndarray, source: np.ndarray, feature_layer: Union[str, int], backdoor: PoisoningAttackBackdoor, eps: float=0.1, learning_rate: float=0.001, decay_coeff: float=0.95, decay_iter: Union[int, List[int]]=2000, stopping_threshold: float=10, max_iter: int=5000, batch_size: float=100, poison_percent: float=0.1, is_index: bool=False, verbose: bool=True, print_iter: int=100) -> None:
        if False:
            return 10
        '\n        Creates a new Hidden Trigger Backdoor poisoning attack for Keras and TensorflowV2.\n\n        :param classifier: A trained neural network classifier.\n        :param target: The target class/indices to poison. Triggers added to inputs not in the target class will\n                       result in misclassifications to the target class. If an int, it represents a label.\n                       Otherwise, it is an array of indices.\n        :param source: The class/indices which will have a trigger added to cause misclassification\n                       If an int, it represents a label. Otherwise, it is an array of indices.\n        :param feature_layer: The name of the feature representation layer.\n        :param backdoor: A PoisoningAttackBackdoor that adds a backdoor trigger to the input.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param learning_rate: The learning rate of clean-label attack optimization.\n        :param decay_coeff: The decay coefficient of the learning rate.\n        :param decay_iter: The number of iterations before the learning rate decays\n        :param stopping_threshold: Stop iterations after loss is less than this threshold.\n        :param max_iter: The maximum number of iterations for the attack.\n        :param batch_size: The number of samples to draw per batch.\n        :param poison_percent: The percentage of the data to poison. This is ignored if indices are provided\n        :param is_index: If true, the source and target params are assumed to represent indices rather\n                         than a class label. poison_percent is ignored if true.\n        :param verbose: Show progress bars.\n        :print iter: The number of iterations to print the current loss progress.\n        '
        super().__init__(classifier=classifier)
        self.target = target
        self.source = source
        self.feature_layer = feature_layer
        self.backdoor = backdoor
        self.eps = eps
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.decay_iter = decay_iter
        self.stopping_threshold = stopping_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.poison_percent = poison_percent
        self.is_index = is_index
        self.verbose = verbose
        self.print_iter = print_iter

    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Calls perturbation function on the dataset x and returns only the perturbed input and their\n        indices in the dataset.\n        :param x: An array in the shape NxWxHxC with the points to draw source and target samples from.\n                  Source indicates the class(es) that the backdoor would be added to to cause\n                  misclassification into the target label.\n                  Target indicates the class that the backdoor should cause misclassification into.\n        :param y: The labels of the provided samples. If none, we will use the classifier to label the\n                  data.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        import tensorflow as tf
        from scipy.spatial import distance
        if isinstance(self.estimator, KerasClassifier):
            if not self.estimator.is_tensorflow:
                import keras.backend as k
            else:
                import tensorflow.keras.backend as k
        data = np.copy(x)
        if y is None:
            estimated_labels = self.estimator.predict(data)
        else:
            estimated_labels = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if not self.is_index:
            poison_class = self.target
            poison_indices = np.where(np.all(estimated_labels == poison_class, axis=1))[0]
            num_poison = int(np.ceil(self.poison_percent * len(poison_indices)))
            if num_poison == 0:
                raise ValueError('No data points with target label found')
            poison_indices = np.random.choice(poison_indices, num_poison, replace=False)
        else:
            poison_class = estimated_labels[self.target[0]]
            poison_indices = self.target
            if not np.all(np.all(estimated_labels[poison_indices] == poison_class, axis=1)):
                raise ValueError('The target indices do not share the same label')
            num_poison = len(poison_indices)
        if not self.is_index:
            trigger_class = self.source
            trigger_indices = np.where(np.all(estimated_labels == trigger_class, axis=1))[0]
            num_trigger = min(len(trigger_indices), num_poison)
            if num_trigger == 0:
                raise ValueError('No data points with source label found')
            if num_trigger < num_poison:
                raise ValueError('There must be at least as many images with the source label as the target.')
            trigger_indices = np.random.choice(trigger_indices, num_poison, replace=False)
            num_trigger = len(trigger_indices)
        else:
            trigger_indices = self.source
            num_trigger = len(trigger_indices)
            if np.any(np.all(estimated_labels[poison_indices] == poison_class, axis=1)):
                raise ValueError('There is a source class that is labeled as the target indices')
            if num_trigger < num_poison:
                raise ValueError('There must be at least as many images with the source label as the target.')
        logger.info('Number of poison inputs: %d', num_poison)
        logger.info('Number of trigger inputs: %d', num_trigger)
        batches = int(np.ceil(num_poison / float(self.batch_size)))
        losses = LossMeter()
        final_poison = np.copy(data[poison_indices])
        original_images = np.copy(data[poison_indices])
        for batch_id in trange(batches, desc='Hidden Trigger', disable=not self.verbose):
            cur_index = self.batch_size * batch_id
            offset = min(self.batch_size, num_poison - cur_index)
            poison_batch_indices = poison_indices[cur_index:cur_index + offset]
            trigger_batch_indices = trigger_indices[cur_index:cur_index + offset]
            poison_samples = data[poison_batch_indices]
            (trigger_samples, _) = self.backdoor.poison(data[trigger_batch_indices], self.target, broadcast=True)
            feat1 = self.estimator.get_activations(trigger_samples, self.feature_layer)
            feat1 = np.copy(feat1)
            for i in range(self.max_iter):
                if isinstance(self.decay_iter, int):
                    decay_exp = i // self.decay_iter
                else:
                    max_index = [ii for (ii, _) in enumerate(self.decay_iter) if self.decay_iter[ii] <= i]
                    if len(max_index) == 0:
                        decay_exp = 0
                    else:
                        decay_exp = max(max_index) + 1
                learning_rate = self.learning_rate * self.decay_coeff ** decay_exp
                feat2 = self.estimator.get_activations(poison_samples, self.feature_layer)
                feat11 = np.copy(feat1)
                dist = distance.cdist(feat1, feat2, 'minkowski')
                for _ in range(len(feat2)):
                    min_index = np.squeeze((dist == np.min(dist)).nonzero())
                    feat1[min_index[1]] = feat11[min_index[0]]
                    dist[min_index[0], min_index[1]] = 100000.0
                loss = np.linalg.norm(feat1 - feat2) ** 2
                losses.update(loss, len(trigger_samples))
                if isinstance(self.estimator, KerasClassifier):
                    if not hasattr(self, '_custom_loss'):
                        self._custom_loss = {}
                        feat1_var = k.variable(feat1)
                        self._custom_loss['feat_var'] = feat1_var
                        output_tensor = self._get_keras_tensor()
                        attack_loss = tf.math.square(tf.norm(feat1_var - output_tensor))
                        attack_grad_f = k.gradients(attack_loss, self.estimator._input)[0]
                        self._custom_loss['loss_function'] = k.function([self.estimator._input, k.learning_phase()], [attack_grad_f])
                    else:
                        feat1_var = self._custom_loss['feat_var']
                    k.set_value(feat1_var, feat1)
                    preprocessed_poison_samples = self._apply_preprocessing(poison_samples)
                    attack_grad = self._custom_loss['loss_function']([preprocessed_poison_samples, 0])[0]
                else:
                    poison_tensor = tf.convert_to_tensor(poison_samples)
                    with tf.GradientTape() as tape:
                        tape.watch(poison_tensor)
                        feat2_tensor = self.estimator.get_activations(poison_tensor, 9, 1, framework=True)
                        attack_loss = tf.math.square(tf.norm(feat1 - feat2_tensor))
                    attack_grad = tape.gradient(attack_loss, poison_tensor).numpy()
                poison_samples = poison_samples - learning_rate * attack_grad
                pert = poison_samples - original_images[cur_index:cur_index + offset]
                pert = np.clip(pert, -self.eps, self.eps)
                poison_samples = pert + original_images[cur_index:cur_index + offset]
                poison_samples = np.clip(poison_samples, *self.estimator.clip_values)
                if i % self.print_iter == 0:
                    print(f'Batch: {batch_id} | i: {i:5d} |                         LR: {learning_rate:2.5f} |                         Loss Val: {losses.val:5.3f} | Loss Avg: {losses.avg:5.3f}')
                if loss < self.stopping_threshold or i == self.max_iter - 1:
                    print(f'Max_Loss: {loss}')
                    final_poison[cur_index:cur_index + offset] = poison_samples
                    break
        return (final_poison, poison_indices)

    def _get_keras_tensor(self):
        if False:
            i = 10
            return i + 15
        '\n        Helper function to get the feature layer output tensor in the keras graph\n        :return: Output tensor\n        '
        if self.estimator._layer_names is None:
            raise ValueError('No layer names identified.')
        if isinstance(self.feature_layer, six.string_types):
            if self.feature_layer not in self.estimator._layer_names:
                raise ValueError(f'Layer name {self.feature_layer} is not part of the graph.')
            layer_name = self.feature_layer
        elif isinstance(self.feature_layer, int):
            if self.feature_layer < 0 or self.feature_layer >= len(self.estimator._layer_names):
                raise ValueError(f'Layer index {self.feature_layer} is outside of range [0 to {len(self.estimator._layer_names) - 1}]).')
            layer_name = self.estimator._layer_names[self.feature_layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')
        keras_layer = self.estimator._model.get_layer(layer_name)
        num_inbound_nodes = len(getattr(keras_layer, '_inbound_nodes', []))
        if num_inbound_nodes > 1:
            layer_output = keras_layer.get_output_at(0)
        else:
            layer_output = keras_layer.output
        return layer_output

    def _apply_preprocessing(self, x: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Helper function to preprocess the input for use with computing the loss gradient.\n        :param x: The input to preprocess\n        :return: Preprocessed input\n        '
        if x.shape == self.estimator.input_shape:
            x_expanded = np.expand_dims(x, 0)
        else:
            x_expanded = x
        (x_preprocessed, _) = self.estimator._apply_preprocessing(x=x_expanded, y=None, fit=False)
        return x_preprocessed