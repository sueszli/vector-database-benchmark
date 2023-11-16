"""
This module implements Sleeper Agent attack on Neural Networks.

| Paper link: https://arxiv.org/abs/2106.08970
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Tuple, TYPE_CHECKING, List, Union
import random
import numpy as np
from tqdm.auto import trange
from art.attacks.poisoning.gradient_matching_attack import GradientMatchingAttack
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification import TensorFlowV2Classifier
from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch
from art.preprocessing.standardisation_mean_std.tensorflow import StandardisationMeanStdTensorFlow
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class SleeperAgentAttack(GradientMatchingAttack):
    """
    Implementation of Sleeper Agent Attack

    | Paper link: https://arxiv.org/pdf/2106.08970.pdf
    """

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', percent_poison: float, patch: np.ndarray, indices_target: List[int], epsilon: float=0.1, max_trials: int=8, max_epochs: int=250, learning_rate_schedule: Tuple[List[float], List[int]]=([0.1, 0.01, 0.001, 0.0001], [100, 150, 200, 220]), batch_size: int=128, clip_values: Tuple[float, float]=(0, 1.0), verbose: int=1, patching_strategy: str='random', selection_strategy: str='random', retraining_factor: int=1, model_retrain: bool=False, model_retraining_epoch: int=1, class_source: int=0, class_target: int=1, device_name: str='cpu', retrain_batch_size: int=128):
        if False:
            while True:
                i = 10
        '\n        Initialize a Sleeper Agent poisoning attack.\n\n        :param classifier: The proxy classifier used for the attack.\n        :param percent_poison: The ratio of samples to poison among x_train, with range [0,1].\n        :param patch: The patch to be applied as trigger.\n        :param indices_target: The indices of training data having target label.\n        :param epsilon: The L-inf perturbation budget.\n        :param max_trials: The maximum number of restarts to optimize the poison.\n        :param max_epochs: The maximum number of epochs to optimize the train per trial.\n        :param learning_rate_schedule: The learning rate schedule to optimize the poison.\n            A List of (learning rate, epoch) pairs. The learning rate is used\n            if the current epoch is less than the specified epoch.\n        :param batch_size: Batch size.\n        :param clip_values: The range of the input features to the classifier.\n        :param verbose: Show progress bars.\n        :param patching_strategy: Patching strategy to be used for adding trigger, either random/fixed.\n        :param selection_strategy: Selection strategy for getting the indices of\n                             poison examples - either random/maximum gradient norm.\n        :param retraining_factor: The factor for which retraining needs to be applied.\n        :param model_retrain: True, if retraining has to be applied, else False.\n        :param model_retraining_epoch: The epochs for which retraining has to be applied.\n        :param class_source: The source class from which triggers were selected.\n        :param class_target: The target label to which the poisoned model needs to misclassify.\n        :param retrain_batch_size: Batch size required for model retraining.\n        '
        if isinstance(classifier.preprocessing, (StandardisationMeanStdPyTorch, StandardisationMeanStdTensorFlow)):
            clip_values_normalised = (classifier.clip_values - classifier.preprocessing.mean) / classifier.preprocessing.std
            clip_values_normalised = (clip_values_normalised[0], clip_values_normalised[1])
            epsilon_normalised = epsilon * (clip_values_normalised[1] - clip_values_normalised[0])
            patch_normalised = (patch - classifier.preprocessing.mean) / classifier.preprocessing.std
        else:
            raise ValueError('classifier.preprocessing not an instance of pytorch/tensorflow')
        super().__init__(classifier, percent_poison, epsilon_normalised, max_trials, max_epochs, learning_rate_schedule, batch_size, clip_values_normalised, verbose)
        self.indices_target = indices_target
        self.selection_strategy = selection_strategy
        self.patching_strategy = patching_strategy
        self.retraining_factor = retraining_factor
        self.model_retrain = model_retrain
        self.model_retraining_epoch = model_retraining_epoch
        self.indices_poison: np.ndarray
        self.patch = patch_normalised
        self.class_target = class_target
        self.class_source = class_source
        self.device_name = device_name
        self.initial_epoch = 0
        self.retrain_batch_size = retrain_batch_size

    def poison(self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Optimizes a portion of poisoned samples from x_train to make a model classify x_target\n        as y_target by matching the gradients.\n\n        :param x_trigger: A list of samples to use as triggers.\n        :param y_trigger: A list of target classes to classify the triggers into.\n        :param x_train: A list of training data to poison a portion of.\n        :param y_train: A list of labels for x_train.\n        :return: x_train, y_train and indices of poisoned samples.\n                 Here, x_train are the samples selected from target class\n                 in training data.\n        '
        x_train = np.copy(x_train)
        if isinstance(self.substitute_classifier.preprocessing, (StandardisationMeanStdPyTorch, StandardisationMeanStdTensorFlow)):
            x_trigger = (x_trigger - self.substitute_classifier.preprocessing.mean) / self.substitute_classifier.preprocessing.std
            x_train = (x_train - self.substitute_classifier.preprocessing.mean) / self.substitute_classifier.preprocessing.std
        (x_train_target_samples, y_train_target_samples) = self._select_target_train_samples(x_train, y_train)
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self._poison__pytorch
            finish_poisoning = self._finish_poison_pytorch
            initializer = self._initialize_poison_pytorch
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            poisoner = self._poison__tensorflow
            finish_poisoning = self._finish_poison_tensorflow
            initializer = self._initialize_poison_tensorflow
        else:
            raise NotImplementedError('SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.')
        x_trigger = self._apply_trigger_patch(x_trigger)
        if len(np.shape(y_trigger)) == 2:
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:
            classes_target = set(y_trigger)
        num_poison_samples = int(self.percent_poison * len(x_train_target_samples))
        best_B = np.finfo(np.float32).max
        best_x_poisoned: np.ndarray
        best_indices_poison: np.ndarray
        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train_target_samples, axis=-1)
        else:
            y_train_classes = y_train_target_samples
        for _ in trange(self.max_trials):
            if self.selection_strategy == 'random':
                self.indices_poison = np.random.permutation(np.where([y in classes_target for y in y_train_classes])[0])[:num_poison_samples]
            else:
                self.indices_poison = self._select_poison_indices(self.substitute_classifier, x_train_target_samples, y_train_target_samples, num_poison_samples)
            x_poison = x_train_target_samples[self.indices_poison]
            y_poison = y_train_target_samples[self.indices_poison]
            initializer(x_trigger, y_trigger, x_poison, y_poison)
            if self.model_retrain:
                retrain_epochs = self.max_epochs // self.retraining_factor
                self.max_epochs = retrain_epochs
                for i in range(self.retraining_factor):
                    if i == self.retraining_factor - 1:
                        (x_poisoned, B_) = poisoner(x_poison, y_poison)
                    else:
                        (x_poisoned, B_) = poisoner(x_poison, y_poison)
                        self._model_retraining(x_poisoned, x_train, y_train, x_test, y_test)
                    self.initial_epoch = self.max_epochs
                    self.max_epochs = self.max_epochs + retrain_epochs
            else:
                (x_poisoned, B_) = poisoner(x_poison, y_poison)
            finish_poisoning()
            B_ = np.mean(B_)
            if B_ < best_B:
                best_B = B_
                best_x_poisoned = x_poisoned
                best_indices_poison = self.indices_poison
        if best_B == np.finfo(np.float32).max:
            logger.warning('Attack unsuccessful: all loss values were non-finite. Defaulting to final trial.')
            best_B = B_
            best_x_poisoned = x_poisoned
            best_indices_poison = self.indices_poison
        self.indices_poison = best_indices_poison
        if isinstance(self.substitute_classifier.preprocessing, (StandardisationMeanStdPyTorch, StandardisationMeanStdTensorFlow)):
            x_train = x_train * self.substitute_classifier.preprocessing.std + self.substitute_classifier.preprocessing.mean
            best_x_poisoned = best_x_poisoned * self.substitute_classifier.preprocessing.std + self.substitute_classifier.preprocessing.mean
        if self.verbose > 0:
            logger.info('Best B-score: %s', best_B)
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            x_train[self.indices_target[best_indices_poison]] = best_x_poisoned
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            x_train[self.indices_target[best_indices_poison]] = best_x_poisoned
        else:
            raise NotImplementedError('SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.')
        return (x_train, y_train)

    def _select_target_train_samples(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Used for selecting train samples from target class\n        :param x_train: clean training data\n        :param y_train: labels fo clean training data\n        :return x_train_target_samples, y_train_target_samples:\n        samples and labels selected from target class in train data\n        '
        x_train_samples = np.copy(x_train)
        index_target = np.where(y_train.argmax(axis=1) == self.class_target)[0]
        x_train_target_samples = x_train_samples[index_target]
        y_train_target_samples = y_train[index_target]
        return (x_train_target_samples, y_train_target_samples)

    def get_poison_indices(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: indices of best poison index\n        '
        return self.indices_poison

    def _model_retraining(self, poisoned_samples: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies retraining to substitute model\n\n        :param poisoned_samples: poisoned array.\n        :param x_train: clean training data.\n        :param y_train: labels for training data.\n        :param x_test: clean test data.\n        :param y_test: labels for test data.\n        '
        if isinstance(self.substitute_classifier.preprocessing, (StandardisationMeanStdPyTorch, StandardisationMeanStdTensorFlow)):
            x_train_un = np.copy(x_train)
            x_train_un[self.indices_target[self.indices_poison]] = poisoned_samples
            x_train_un = x_train_un * self.substitute_classifier.preprocessing.std
            x_train_un += self.substitute_classifier.preprocessing.mean
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            check_train = self.substitute_classifier.model.training
            model_pt = self._create_model(x_train_un, y_train, x_test, y_test, batch_size=self.retrain_batch_size, epochs=self.model_retraining_epoch)
            self.substitute_classifier = model_pt
            self.substitute_classifier.model.training = check_train
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            check_train = self.substitute_classifier.model.trainable
            model_tf = self._create_model(x_train_un, y_train, x_test, y_test, batch_size=self.retrain_batch_size, epochs=self.model_retraining_epoch)
            self.substitute_classifier = model_tf
            self.substitute_classifier.model.trainable = check_train
        else:
            raise NotImplementedError('SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.')

    def _create_model(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, batch_size: int=128, epochs: int=80) -> Union['TensorFlowV2Classifier', 'PyTorchClassifier']:
        if False:
            print('Hello World!')
        '\n        Creates a new model.\n\n        :param x_train: Samples of train data.\n        :param y_train: Labels of train data.\n        :param x_test: Samples of test data.\n        :param y_test: Labels of test data.\n        :param num_classes: Number of classes of labels in train data.\n        :param batch_size: The size of batch used for training.\n        :param epochs: The number of epochs for which training need to be applied.\n        :return model, loss_fn, optimizer - trained model, loss function used to train the model and optimizer used.\n        '
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            model_pt = self.substitute_classifier.clone_for_refitting()
            for layer in model_pt.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            model_pt.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=1)
            predictions = model_pt.predict(x_test)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            logger.info('Accuracy of retrained model : %s', accuracy * 100.0)
            return model_pt
        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            self.substitute_classifier.model.trainable = True
            model_tf = self.substitute_classifier.clone_for_refitting()
            model_tf.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs, verbose=0)
            predictions = model_tf.predict(x_test)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            logger.info('Accuracy of retrained model : %s', accuracy * 100.0)
            return model_tf
        raise ValueError('SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.')

    def _select_poison_indices(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', x_samples: np.ndarray, y_samples: np.ndarray, num_poison: int) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Select indices of poisoned samples\n\n        :classifier: Substitute Model.\n        :x_samples: Samples of poison. [x_samples are normalised]\n        :y_samples: Labels of samples of poison.\n        :num_poison: Number of poisoned samples to be selected out of all x_samples.\n        :return indices - Indices of samples to be poisoned.\n        '
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            import torch
            device = torch.device(self.device_name)
            grad_norms = []
            criterion = torch.nn.CrossEntropyLoss()
            model = classifier.model
            model.eval()
            differentiable_params = [p for p in classifier.model.parameters() if p.requires_grad]
            for (x, y) in zip(x_samples, y_samples):
                image = torch.tensor(x, dtype=torch.float32).float().to(device)
                label = torch.tensor(y).to(device)
                loss_pt = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = list(torch.autograd.grad(loss_pt, differentiable_params, only_inputs=True))
                grad_norm = torch.tensor(0, dtype=torch.float32).to(device)
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt())
        elif isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            import tensorflow as tf
            model_trainable = classifier.model.trainable
            classifier.model.trainable = False
            grad_norms = []
            for i in range(len(x_samples) - 1):
                image = tf.constant(x_samples[i:i + 1])
                label = tf.constant(y_samples[i:i + 1])
                with tf.GradientTape() as t:
                    t.watch(classifier.model.weights)
                    output = classifier.model(image, training=False)
                    loss_tf = classifier.loss_object(label, output)
                    gradients = list(t.gradient(loss_tf, classifier.model.weights))
                    gradients = [w for w in gradients if w is not None]
                    grad_norm = tf.constant(0, dtype=tf.float32)
                    for grad in gradients:
                        grad_norm += tf.reduce_sum(tf.math.square(grad))
                    grad_norms.append(tf.math.sqrt(grad_norm))
            classifier.model.trainable = model_trainable
        else:
            raise NotImplementedError('SleeperAgentAttack is currently implemented only for PyTorch and TensorFlowV2.')
        indices = sorted(range(len(grad_norms)), key=lambda k: grad_norms[k])
        indices = indices[-num_poison:]
        return np.array(indices)

    def _apply_trigger_patch(self, x_trigger: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Select indices of poisoned samples\n\n        :x_trigger: Samples to be used for trigger.\n        :return tensor with applied trigger patches.\n        '
        patch_size = self.patch.shape[1]
        if self.patching_strategy == 'fixed':
            if self.estimator.channels_first:
                x_trigger[:, :, -patch_size:, -patch_size:] = self.patch
            else:
                x_trigger[:, -patch_size:, -patch_size:, :] = self.patch
        else:
            for x in x_trigger:
                if self.estimator.channels_first:
                    x_cord = random.randrange(0, x.shape[1] - self.patch.shape[1] + 1)
                    y_cord = random.randrange(0, x.shape[2] - self.patch.shape[2] + 1)
                    x[:, x_cord:x_cord + patch_size, y_cord:y_cord + patch_size] = self.patch
                else:
                    x_cord = random.randrange(0, x.shape[0] - self.patch.shape[0] + 1)
                    y_cord = random.randrange(0, x.shape[1] - self.patch.shape[1] + 1)
                    x[x_cord:x_cord + patch_size, y_cord:y_cord + patch_size, :] = self.patch
        return x_trigger