"""
This module implements DP-InstaHide training method.

| Paper link: https://arxiv.org/abs/2103.02079

| This training method is dependent to the choice of data augmentation and noise parameters. Consequently, framework
    specific implementations are being provided in ART.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import sys
import time
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.defences.trainer.trainer import Trainer
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
if TYPE_CHECKING:
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, CLIP_VALUES_TYPE
logger = logging.getLogger(__name__)

class DPInstaHideTrainer(Trainer):
    """
    Class performing adversarial training following the DP-InstaHide protocol.

    Uses data augmentation methods in conjunction with some type of additive noise.

    | Paper link: https://arxiv.org/abs/2103.02079
    """

    def __init__(self, classifier: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', augmentations: Union['Preprocessor', List['Preprocessor']], noise: Literal['gaussian', 'laplacian', 'exponential']='laplacian', loc: Union[int, float]=0.0, scale: Union[int, float]=0.03, clip_values: 'CLIP_VALUES_TYPE'=(0.0, 1.0)):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create an :class:`.DPInstaHideTrainer` instance.\n\n        :param classifier: The model to train using the protocol.\n        :param augmentations: The preprocessing data augmentation defence(s) to be applied.\n        :param noise: The type of additive noise to use: 'gaussian' | 'laplacian' | 'exponential'.\n        :param loc: The location or mean parameter of the distribution to sample.\n        :param scale: The scale or standard deviation parameter of the distribution to sample.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        "
        from art.defences.preprocessor import Preprocessor
        super().__init__(classifier)
        if isinstance(augmentations, Preprocessor):
            self.augmentations = [augmentations]
        else:
            self.augmentations = augmentations
        self.noise = noise
        self.loc = loc
        self.scale = scale
        self.clip_values = clip_values

    def _generate_noise(self, x: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        if self.noise == 'gaussian':
            noise = np.random.normal(loc=self.loc, scale=self.scale, size=x.shape)
        elif self.noise == 'laplacian':
            noise = np.random.laplace(loc=self.loc, scale=self.scale, size=x.shape)
        elif self.noise == 'exponential':
            noise = np.random.exponential(scale=self.scale, size=x.shape)
        else:
            raise ValueError('The provided noise type is not supported:', self.noise)
        x_noise = x + noise
        x_noise = np.clip(x_noise, self.clip_values[0], self.clip_values[1])
        return x_noise.astype(x.dtype)

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]=None, batch_size: int=128, nb_epochs: int=20, **kwargs):
        if False:
            return 10
        '\n        Train a model adversarially with the DP-InstaHide protocol.\n        See class documentation for more information on the exact procedure.\n\n        :param x: Training set.\n        :param y: Labels for the training set.\n        :param validation_data: Tuple consisting of validation data, (x_val, y_val)\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function\n               of the target classifier.\n        '
        logger.info('Performing adversarial training with DP-InstaHide protocol')
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))
        logger.info('Adversarial Training DP-InstaHide')
        for i_epoch in trange(nb_epochs, desc='DP-InstaHide training epochs'):
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0
            for batch_id in range(nb_batches):
                x_batch = x[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]]
                y_batch = y[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]]
                x_aug = x_batch.copy()
                y_aug = y_batch.copy()
                for augmentation in self.augmentations:
                    (x_aug, y_aug) = augmentation(x_aug, y_aug)
                x_aug = self._generate_noise(x_aug)
                self._classifier.fit(x_aug, y_aug, nb_epochs=1, batch_size=x_aug.shape[0], verbose=0, **kwargs)
                loss = self._classifier.compute_loss(x_aug, y_aug, reduction='mean')
                output = np.argmax(self.predict(x_batch), axis=1)
                acc = np.sum(output == np.argmax(y_batch, axis=1))
                n = len(x_aug)
                train_loss += np.sum(loss)
                train_acc += acc
                train_n += n
            train_time = time.time()
            if validation_data is not None:
                (x_test, y_test) = validation_data
                output = np.argmax(self.predict(x_test), axis=1)
                test_loss = self._classifier.compute_loss(x_test, y_test, reduction='mean')
                test_acc = np.mean(output == np.argmax(y_test, axis=1))
                logger.info('epoch: %s time(s): %.1f, loss(tr): %.4f, acc(tr): %.4f, loss(val): %.4f, acc(val): %.4f', i_epoch, train_time - start_time, train_loss / train_n, train_acc / train_n, test_loss, test_acc)
            else:
                logger.info('epoch: %s time(s): %.1f, loss: %.4f, acc: %.4f', i_epoch, train_time - start_time, train_loss / train_n, train_acc / train_n)

    def fit_generator(self, generator: 'DataGenerator', nb_epochs: int=20, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train a model adversarially with the DP-InstaHide protocol using a data generator.\n        See class documentation for more information on the exact procedure.\n\n        :param generator: Data generator.\n        :param nb_epochs: Number of epochs to use for trainings.\n        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function\n               of the target classifier.\n        '
        logger.info('Performing adversarial training with DP-InstaHide protocol')
        size = generator.size
        batch_size = generator.batch_size
        if size is not None:
            nb_batches = int(np.ceil(size / batch_size))
        else:
            raise ValueError('Size is None.')
        logger.info('Adversarial Training DP-InstaHide')
        for i_epoch in trange(nb_epochs, desc='DP-InstaHide training epochs'):
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0
            for _ in range(nb_batches):
                (x_batch, y_batch) = generator.get_batch()
                x_aug = x_batch.copy()
                y_aug = y_batch.copy()
                for augmentation in self.augmentations:
                    (x_aug, y_aug) = augmentation(x_aug, y_aug)
                x_aug = self._generate_noise(x_aug)
                self._classifier.fit(x_aug, y_aug, nb_epochs=1, batch_size=x_aug.shape[0], verbose=0, **kwargs)
                loss = self._classifier.compute_loss(x_aug, y_aug, reduction='mean')
                output = np.argmax(self.predict(x_batch), axis=1)
                acc = np.sum(output == np.argmax(y_batch, axis=1))
                n = len(x_aug)
                train_loss += np.sum(loss)
                train_acc += acc
                train_n += n
            train_time = time.time()
            logger.info('epoch: %s time(s): %.1f, loss: %.4f, acc: %.4f', i_epoch, train_time - start_time, train_loss / train_n, train_acc / train_n)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Perform prediction using the adversarially trained classifier.\n\n        :param x: Input samples.\n        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.\n        :return: Predictions for test set.\n        '
        return self._classifier.predict(x, **kwargs)