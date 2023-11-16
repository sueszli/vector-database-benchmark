"""
This module implements the InverseGAN defence.

| Paper link: https://arxiv.org/abs/1911.10291
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize
from art.defences.preprocessor.preprocessor import Preprocessor
if TYPE_CHECKING:
    import tensorflow as tf
    from art.estimators.encoding.tensorflow import TensorFlowEncoder
    from art.estimators.generation.tensorflow import TensorFlowGenerator
logger = logging.getLogger(__name__)

class InverseGAN(Preprocessor):
    """
    Given a latent variable generating a given adversarial sample, either inferred by an inverse GAN or randomly
    generated, the InverseGAN optimizes that latent variable to project a sample as close as possible to the adversarial
    sample without the adversarial noise.
    """
    params = ['sess', 'gan', 'inverse_gan']

    def __init__(self, sess: 'tf.compat.v1.Session', gan: 'TensorFlowGenerator', inverse_gan: Optional['TensorFlowEncoder'], apply_fit: bool=False, apply_predict: bool=False):
        if False:
            while True:
                i = 10
        '\n        Create an instance of an InverseGAN.\n\n        :param sess: TF session for computations.\n        :param gan: GAN model.\n        :param inverse_gan: Inverse GAN model.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        import tensorflow as tf
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.gan = gan
        self.inverse_gan = inverse_gan
        self.sess = sess
        self._image_adv = tf.placeholder(tf.float32, shape=self.gan.model.get_shape().as_list(), name='image_adv_ph')
        num_dim = len(self._image_adv.get_shape())
        image_loss = tf.reduce_mean(tf.square(self.gan.model - self._image_adv), axis=list(range(1, num_dim)))
        self._loss = tf.reduce_sum(image_loss)
        self._grad = tf.gradients(self._loss, self.gan.input_ph)
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            i = 10
            return i + 15
        '\n        Applies the :class:`.InverseGAN` defence upon the sample input.\n\n        :param x: Sample input.\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Defended input.\n        '
        batch_size = x.shape[0]
        iteration_count = 0
        if self.inverse_gan is not None:
            logger.info('Encoding x_adv into starting z encoding')
            initial_z_encoding = self.inverse_gan.predict(x)
        else:
            logger.info('Choosing a random starting z encoding')
            initial_z_encoding = np.random.rand(batch_size, self.gan.encoding_length)

        def func_gen_gradients(z_i):
            if False:
                print('Hello World!')
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            grad = self.estimate_gradient(z_i_reshaped, x)
            grad = np.float64(grad)
            return grad.flatten()

        def func_loss(z_i):
            if False:
                return 10
            nonlocal iteration_count
            iteration_count += 1
            logging.info('Iteration: %d', iteration_count)
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            loss = self.compute_loss(z_i_reshaped, x)
            return loss
        options_allowed_keys = ['disp', 'maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter', 'iprint', 'callback', 'maxls']
        for key in kwargs:
            if key not in options_allowed_keys:
                raise KeyError(f'The argument `{key}` in kwargs is not allowed as option for `scipy.optimize.minimize` using `method="L-BFGS-B".`')
        options = kwargs.copy()
        optimized_z_encoding_flat = minimize(func_loss, initial_z_encoding, jac=func_gen_gradients, method='L-BFGS-B', options=options)
        optimized_z_encoding = np.reshape(optimized_z_encoding_flat.x, [batch_size, self.gan.encoding_length])
        y = self.gan.predict(optimized_z_encoding)
        return (x, y)

    def compute_loss(self, z_encoding: np.ndarray, image_adv: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Given a encoding z, computes the loss between the projected sample and the original sample.\n\n        :param z_encoding: The encoding z.\n        :param image_adv: The adversarial image.\n        :return: The loss value\n        '
        logging.info('Calculating Loss')
        loss = self.sess.run(self._loss, feed_dict={self.gan.input_ph: z_encoding, self._image_adv: image_adv})
        return loss

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Compute the gradient of the loss function w.r.t. a `z_encoding` input within a GAN against a\n        corresponding adversarial sample.\n\n        :param x: The encoding z.\n        :param grad: Target values of shape `(nb_samples, nb_classes)`.\n        :return: Array of gradients of the same shape as `z_encoding`.\n        '
        logging.info('Calculating Gradients')
        gradient = self.sess.run(self._grad, feed_dict={self._image_adv: grad, self.gan.input_ph: x})
        return gradient

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.inverse_gan is not None and self.gan.encoding_length != self.inverse_gan.encoding_length:
            raise ValueError('Both GAN and InverseGAN must use the same size encoding.')

class DefenseGAN(InverseGAN):
    """
    Implementation of DefenseGAN.
    """

    def __init__(self, sess, gan):
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of DefenseGAN.\n        '
        super().__init__(sess=sess, gan=gan, inverse_gan=None)