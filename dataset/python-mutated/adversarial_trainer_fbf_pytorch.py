"""
This is a PyTorch implementation of the Fast is better than free protocol.

| Paper link: https://openreview.net/forum?id=BJx040EFvH
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import time
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.config import ART_NUMPY_DTYPE
from art.defences.trainer.adversarial_trainer_fbf import AdversarialTrainerFBF
from art.utils import random_sphere
if TYPE_CHECKING:
    from art.data_generators import DataGenerator
    from art.estimators.classification.pytorch import PyTorchClassifier
logger = logging.getLogger(__name__)

class AdversarialTrainerFBFPyTorch(AdversarialTrainerFBF):
    """
    Class performing adversarial training following Fast is Better Than Free protocol.

    | Paper link: https://openreview.net/forum?id=BJx040EFvH

    | The effectiveness of this protocol is found to be sensitive to the use of techniques like
        data augmentation, gradient clipping and learning rate schedules. Optionally, the use of
        mixed precision arithmetic operation via apex library can significantly reduce the training
        time making this one of the fastest adversarial training protocol.
    """

    def __init__(self, classifier: 'PyTorchClassifier', eps: Union[int, float]=8, use_amp: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Create an :class:`.AdversarialTrainerFBFPyTorch` instance.\n\n        :param classifier: Model to train adversarially.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param use_amp: Boolean that decides if apex should be used for mixed precision arithmetic during training\n        '
        super().__init__(classifier, eps)
        self._classifier: 'PyTorchClassifier'
        self._use_amp = use_amp

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, batch_size: int=128, nb_epochs: int=20, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Train a model adversarially with FBF protocol.\n        See class documentation for more information on the exact procedure.\n\n        :param x: Training set.\n        :param y: Labels for the training set.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n                                  the target classifier.\n        '
        logger.info('Performing adversarial training with Fast is better than Free protocol')
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        def lr_schedule(step_t):
            if False:
                print('Hello World!')
            return np.interp([step_t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]
        logger.info('Adversarial Training FBF')
        for i_epoch in trange(nb_epochs, desc='Adversarial Training FBF - Epochs'):
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0
            for batch_id in range(nb_batches):
                l_r = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)
                x_batch = x[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]]
                (_train_loss, _train_acc, _train_n) = self._batch_process(x_batch, y_batch, l_r)
                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n
            train_time = time.time()
            if validation_data is not None:
                (x_test, y_test) = validation_data
                output = np.argmax(self.predict(x_test), axis=1)
                nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
                logger.info('epoch: %s time(s): %.1f l_r: %.4f loss: %.4f acc(tr): %.4f acc(val): %.4f', i_epoch, train_time - start_time, l_r, train_loss / train_n, train_acc / train_n, nb_correct_pred / x_test.shape[0])
            else:
                logger.info('epoch: %s time(s): %.1f l_r %.4f loss: %.4f acc: %.4f', i_epoch, train_time - start_time, l_r, train_loss / train_n, train_acc / train_n)

    def fit_generator(self, generator: 'DataGenerator', nb_epochs: int=20, **kwargs):
        if False:
            print('Hello World!')
        '\n        Train a model adversarially with FBF protocol using a data generator.\n        See class documentation for more information on the exact procedure.\n\n        :param generator: Data generator.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of\n                                  the target classifier.\n        '
        logger.info('Performing adversarial training with Fast is better than Free protocol')
        size = generator.size
        batch_size = generator.batch_size
        if size is not None:
            nb_batches = int(np.ceil(size / batch_size))
        else:
            raise ValueError('Size is None.')

        def lr_schedule(step_t):
            if False:
                print('Hello World!')
            return np.interp([step_t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]
        logger.info('Adversarial Training FBF')
        for i_epoch in trange(nb_epochs, desc='Adversarial Training FBF - Epochs'):
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0
            for batch_id in range(nb_batches):
                l_r = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)
                (x_batch, y_batch) = generator.get_batch()
                x_batch = x_batch.copy()
                (_train_loss, _train_acc, _train_n) = self._batch_process(x_batch, y_batch, l_r)
                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n
            train_time = time.time()
            logger.info('epoch: %s time(s): %.1f l_r: %.4f loss: %.4f acc: %.4f', i_epoch, train_time - start_time, l_r, train_loss / train_n, train_acc / train_n)

    def _batch_process(self, x_batch: np.ndarray, y_batch: np.ndarray, l_r: float) -> Tuple[float, float, float]:
        if False:
            while True:
                i = 10
        '\n        Perform the operations of FBF for a batch of data.\n        See class documentation for more information on the exact procedure.\n\n        :param x_batch: batch of x.\n        :param y_batch: batch of y.\n        :param l_r: learning rate for the optimisation step.\n        '
        import torch
        if self._classifier._optimizer is None:
            raise ValueError('Optimizer of classifier is currently None, but is required for adversarial training.')
        n = x_batch.shape[0]
        m = np.prod(x_batch.shape[1:]).item()
        delta = random_sphere(n, m, self._eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
        delta_grad = self._classifier.loss_gradient(x_batch + delta, y_batch)
        delta = np.clip(delta + 1.25 * self._eps * np.sign(delta_grad), -self._eps, +self._eps)
        if self._classifier.clip_values is not None:
            x_batch_pert = np.clip(x_batch + delta, self._classifier.clip_values[0], self._classifier.clip_values[1])
        else:
            x_batch_pert = x_batch + delta
        (x_preprocessed, y_preprocessed) = self._classifier._apply_preprocessing(x_batch_pert, y_batch, fit=True)
        if self._classifier._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)
        i_batch = torch.from_numpy(x_preprocessed).to(self._classifier._device)
        o_batch = torch.from_numpy(y_preprocessed).to(self._classifier._device)
        self._classifier._optimizer.zero_grad()
        model_outputs = self._classifier._model(i_batch)
        loss = self._classifier._loss(model_outputs[-1], o_batch)
        self._classifier._optimizer.param_groups[0].update(lr=l_r)
        if self._use_amp:
            from apex import amp
            with amp.scale_loss(loss, self._classifier._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self._classifier._model.parameters(), 0.5)
        self._classifier._optimizer.step()
        train_loss = loss.item() * o_batch.size(0)
        train_acc = (model_outputs[0].max(1)[1] == o_batch).sum().item()
        train_n = o_batch.size(0)
        return (train_loss, train_acc, train_n)