"""
This module implements the adversarial and imperceptible attack on automatic speech recognition systems of Qin et al.
(2019). It generates an adversarial audio example.

| Paper link: http://proceedings.mlr.press/v97/qin19a.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import scipy.signal as ss
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import pad_sequence_input
if TYPE_CHECKING:
    from tensorflow.compat.v1 import Tensor
    from torch import Tensor as PTensor
    from art.utils import SPEECH_RECOGNIZER_TYPE
logger = logging.getLogger(__name__)

class ImperceptibleASR(EvasionAttack):
    """
    Implementation of the imperceptible attack against a speech recognition model.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    """
    attack_params = EvasionAttack.attack_params + ['masker', 'eps', 'learning_rate_1', 'max_iter_1', 'alpha', 'learning_rate_2', 'max_iter_2', 'batch_size', 'loss_theta_min', 'decrease_factor_eps', 'num_iter_decrease_eps', 'increase_factor_alpha', 'num_iter_increase_alpha', 'decrease_factor_alpha', 'num_iter_decrease_alpha']
    _estimator_requirements = (NeuralNetworkMixin, LossGradientsMixin, BaseEstimator, SpeechRecognizerMixin)

    def __init__(self, estimator: 'SPEECH_RECOGNIZER_TYPE', masker: 'PsychoacousticMasker', eps: float=2000.0, learning_rate_1: float=100.0, max_iter_1: int=1000, alpha: float=0.05, learning_rate_2: float=1.0, max_iter_2: int=4000, loss_theta_min: float=0.05, decrease_factor_eps: float=0.8, num_iter_decrease_eps: int=10, increase_factor_alpha: float=1.2, num_iter_increase_alpha: int=20, decrease_factor_alpha: float=0.8, num_iter_decrease_alpha: int=50, batch_size: int=1) -> None:
        if False:
            return 10
        '\n        Create an instance of the :class:`.ImperceptibleASR`.\n\n        The default parameters assume that audio input is in `int16` range. If using normalized audio input, parameters\n        `eps` and `learning_rate_{1,2}` need to be scaled with a factor `2^-15`\n\n        :param estimator: A trained speech recognition estimator.\n        :param masker: A Psychoacoustic masker.\n        :param eps: Initial max norm bound for adversarial perturbation.\n        :param learning_rate_1: Learning rate for stage 1 of attack.\n        :param max_iter_1: Number of iterations for stage 1 of attack.\n        :param alpha: Initial alpha value for balancing stage 2 loss.\n        :param learning_rate_2: Learning rate for stage 2 of attack.\n        :param max_iter_2: Number of iterations for stage 2 of attack.\n        :param loss_theta_min: If imperceptible loss reaches minimum, stop early. Works best with `batch_size=1`.\n        :param decrease_factor_eps: Decrease factor for epsilon (Paper default: 0.8).\n        :param num_iter_decrease_eps: Iterations after which to decrease epsilon if attack succeeds (Paper default: 10).\n        :param increase_factor_alpha: Increase factor for alpha (Paper default: 1.2).\n        :param num_iter_increase_alpha: Iterations after which to increase alpha if attack succeeds (Paper default: 20).\n        :param decrease_factor_alpha: Decrease factor for alpha (Paper default: 0.8).\n        :param num_iter_decrease_alpha: Iterations after which to decrease alpha if attack fails (Paper default: 50).\n        :param batch_size: Batch size.\n        '
        super().__init__(estimator=estimator)
        self.masker = masker
        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self.alpha = alpha
        self.learning_rate_2 = learning_rate_2
        self.max_iter_2 = max_iter_2
        self._targeted = True
        self.batch_size = batch_size
        self.loss_theta_min = loss_theta_min
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        self._check_params()
        self._window_size = masker.window_size
        self._hop_size = masker.hop_size
        self._sample_rate = masker.sample_rate
        self._framework: Optional[str] = None
        if isinstance(self.estimator, TensorFlowV2Estimator):
            import tensorflow.compat.v1 as tf1
            self._framework = 'tensorflow'
            tf1.disable_eager_execution()
            self._delta = tf1.placeholder(tf1.float32, shape=[None, None], name='art_delta')
            self._power_spectral_density_maximum_tf = tf1.placeholder(tf1.float32, shape=[None], name='art_psd_max')
            self._masking_threshold_tf = tf1.placeholder(tf1.float32, shape=[None, None, None], name='art_masking_threshold')
            self._loss_gradient_masking_threshold_op_tf = self._loss_gradient_masking_threshold_tf(self._delta, self._power_spectral_density_maximum_tf, self._masking_threshold_tf)
        elif isinstance(self.estimator, PyTorchEstimator):
            self._framework = 'pytorch'

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Generate imperceptible, adversarial examples.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different\n            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :return: An array holding the adversarial examples.\n        "
        if y is None:
            raise ValueError('The target values `y` cannot be None. Please provide a `np.ndarray` of target labels.')
        nb_samples = x.shape[0]
        x_imperceptible = [None] * nb_samples
        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            (begin, end) = (m * self.batch_size, min((m + 1) * self.batch_size, nb_samples))
            x_imperceptible[begin:end] = self._generate_batch(x[begin:end], y[begin:end])
        dtype = np.float32 if x.ndim != 1 else object
        return np.array(x_imperceptible, dtype=dtype)

    def _generate_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Create imperceptible, adversarial sample.\n\n        This is a helper method that calls the methods to create an adversarial (`ImperceptibleASR._create_adversarial`)\n        and imperceptible (`ImperceptibleASR._create_imperceptible`) example subsequently.\n        '
        x_adversarial = self._create_adversarial(x, y)
        if self.max_iter_2 == 0:
            return x_adversarial
        x_imperceptible = self._create_imperceptible(x, x_adversarial, y)
        return x_imperceptible

    def _create_adversarial(self, x, y) -> np.ndarray:
        if False:
            return 10
        "\n        Create adversarial example with small perturbation that successfully deceives the estimator.\n\n        The method implements the part of the paper by Qin et al. (2019) that is referred to as the first stage of the\n        attack. The authors basically follow Carlini and Wagner (2018).\n\n        | Paper link: https://arxiv.org/abs/1801.01944.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different\n            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :return: An array with the adversarial outputs.\n        "
        batch_size = x.shape[0]
        dtype = np.float32 if x.ndim != 1 else object
        epsilon = [self.eps] * batch_size
        x_adversarial = [None] * batch_size
        x_perturbed = x.copy()
        for i in range(1, self.max_iter_1 + 1):
            gradients = self.estimator.loss_gradient(x_perturbed, y, batch_mode=True)
            x_perturbed = x_perturbed - self.learning_rate_1 * np.array([np.sign(g) for g in gradients], dtype=dtype)
            perturbation = x_perturbed - x
            perturbation = np.array([np.clip(p, -e, e) for (p, e) in zip(perturbation, epsilon)], dtype=dtype)
            x_perturbed = x + perturbation
            if i % self.num_iter_decrease_eps == 0:
                prediction = self.estimator.predict(x_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    if prediction[j] == y[j].upper():
                        perturbation_norm = np.max(np.abs(perturbation[j]))
                        if epsilon[j] > perturbation_norm:
                            epsilon[j] = perturbation_norm
                        epsilon[j] *= self.decrease_factor_eps
                        x_adversarial[j] = x_perturbed[j]
                logger.info('Current iteration %s, epsilon %s', i, epsilon)
        for j in range(batch_size):
            if x_adversarial[j] is None:
                logger.critical('Adversarial attack stage 1 for x_%s was not successful', j)
                x_adversarial[j] = x_perturbed[j]
        return np.array(x_adversarial, dtype=dtype)

    def _create_imperceptible(self, x: np.ndarray, x_adversarial: np.ndarray, y: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        "\n        Create imperceptible, adversarial example with small perturbation.\n\n        This method implements the part of the paper by Qin et al. (2019) that is described as the second stage of the\n        attack. The resulting adversarial audio samples are able to successfully deceive the ASR estimator and are\n        imperceptible to the human ear.\n\n        :param x: An array with the original inputs to be attacked.\n        :param x_adversarial: An array with the adversarial examples.\n        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different\n            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :return: An array with the imperceptible, adversarial outputs.\n        "
        batch_size = x.shape[0]
        alpha_min = 0.0005
        dtype = np.float32 if x.ndim != 1 else object
        early_stop = [False] * batch_size
        alpha = np.array([self.alpha] * batch_size, dtype=np.float32)
        loss_theta_previous = [np.inf] * batch_size
        x_imperceptible = [None] * batch_size
        if x.ndim != 1:
            alpha = np.expand_dims(alpha, axis=-1)
        (masking_threshold, psd_maximum) = self._stabilized_threshold_and_psd_maximum(x)
        x_perturbed = x_adversarial.copy()
        for i in range(1, self.max_iter_2 + 1):
            perturbation = x_perturbed - x
            gradients_net = self.estimator.loss_gradient(x_perturbed, y, batch_mode=True)
            (gradients_theta, loss_theta) = self._loss_gradient_masking_threshold(perturbation, x, masking_threshold, psd_maximum)
            assert gradients_net.shape == gradients_theta.shape
            x_perturbed = x_perturbed - self.learning_rate_2 * (gradients_net + alpha * gradients_theta)
            if i % self.num_iter_increase_alpha == 0 or i % self.num_iter_decrease_alpha == 0:
                prediction = self.estimator.predict(x_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    if i % self.num_iter_increase_alpha == 0 and prediction[j] == y[j].upper():
                        alpha[j] *= self.increase_factor_alpha
                        if loss_theta[j] < loss_theta_previous[j]:
                            x_imperceptible[j] = x_perturbed[j]
                            loss_theta_previous[j] = loss_theta[j]
                    if i % self.num_iter_decrease_alpha == 0 and prediction[j] != y[j].upper():
                        alpha[j] = max(alpha[j] * self.decrease_factor_alpha, alpha_min)
                logger.info('Current iteration %s, alpha %s, loss theta %s', i, alpha, loss_theta)
            for j in range(batch_size):
                if loss_theta[j] < self.loss_theta_min and (not early_stop[j]):
                    logger.warning('Batch sample %s reached minimum threshold of %s for theta loss.', j, self.loss_theta_min)
                    early_stop[j] = True
            if all(early_stop):
                logger.warning('All batch samples reached minimum threshold for theta loss. Stopping early at iteration %s.', i)
                break
        for j in range(batch_size):
            if x_imperceptible[j] is None:
                logger.critical('Adversarial attack stage 2 for x_%s was not successful', j)
                x_imperceptible[j] = x_perturbed[j]
        return np.array(x_imperceptible, dtype=dtype)

    def _stabilized_threshold_and_psd_maximum(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Return batch of stabilized masking thresholds and PSD maxima.\n\n        :param x: An array with the original inputs to be attacked.\n        :return: Tuple consisting of stabilized masking thresholds and PSD maxima.\n        '
        masking_threshold = []
        psd_maximum = []
        (x_padded, _) = pad_sequence_input(x)
        for x_i in x_padded:
            (m_t, p_m) = self.masker.calculate_threshold_and_psd_maximum(x_i)
            masking_threshold.append(m_t)
            psd_maximum.append(p_m)
        masking_threshold_stabilized = 10 ** (np.array(masking_threshold) * 0.1)
        psd_maximum_stabilized = 10 ** (np.array(psd_maximum) * 0.1)
        return (masking_threshold_stabilized, psd_maximum_stabilized)

    def _loss_gradient_masking_threshold(self, perturbation: np.ndarray, x: np.ndarray, masking_threshold_stabilized: np.ndarray, psd_maximum_stabilized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute loss gradient of the global masking threshold w.r.t. the PSD approximate of the perturbation.\n\n        The loss is defined as the hinge loss w.r.t. to the frequency masking threshold of the original audio input `x`\n        and the normalized power spectral density estimate of the perturbation. In order to stabilize the optimization\n        problem during back-propagation, the `10*log`-terms are canceled out.\n\n        :param perturbation: Adversarial perturbation.\n        :param x: An array with the original inputs to be attacked.\n        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.\n        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of\n            the original unnormalized PSD of `x`.\n        :return: Tuple consisting of the loss gradient, which has same shape as `perturbation`, and loss value.\n        '
        (perturbation_padded, delta_mask) = pad_sequence_input(perturbation)
        if self._framework == 'tensorflow':
            feed_dict = {self._delta: perturbation_padded, self._power_spectral_density_maximum_tf: psd_maximum_stabilized, self._masking_threshold_tf: masking_threshold_stabilized}
            (gradients_padded, loss) = self.estimator._sess.run(self._loss_gradient_masking_threshold_op_tf, feed_dict)
        elif self._framework == 'pytorch':
            (gradients_padded, loss) = self._loss_gradient_masking_threshold_torch(perturbation_padded, psd_maximum_stabilized, masking_threshold_stabilized)
        else:
            raise NotImplementedError
        lengths = delta_mask.sum(axis=1)
        gradients = []
        for (gradient_padded, length) in zip(gradients_padded, lengths):
            gradient = gradient_padded[:length]
            gradients.append(gradient)
        dtype = np.float32 if x.ndim != 1 else object
        return (np.array(gradients, dtype=dtype), loss)

    def _loss_gradient_masking_threshold_tf(self, perturbation: 'Tensor', psd_maximum_stabilized: 'Tensor', masking_threshold_stabilized: 'Tensor') -> Union['Tensor', 'Tensor']:
        if False:
            return 10
        '\n        Compute loss gradient of the masking threshold loss in TensorFlow.\n\n        Note that the PSD maximum and masking threshold are required to be stabilized, i.e. have the `10*log10`-term\n        canceled out. Following Qin et al (2019) this mitigates optimization instabilities.\n\n        :param perturbation: Adversarial perturbation.\n        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of\n            the original unnormalized PSD of `x`.\n        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.\n        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.\n        '
        import tensorflow.compat.v1 as tf1
        psd_perturbation = self._approximate_power_spectral_density_tf(perturbation, psd_maximum_stabilized)
        loss = tf1.reduce_mean(tf1.nn.relu(psd_perturbation - masking_threshold_stabilized), axis=[1, 2], keepdims=False)
        loss_gradient = tf1.gradients(loss, [perturbation])[0]
        return (loss_gradient, loss)

    def _loss_gradient_masking_threshold_torch(self, perturbation: np.ndarray, psd_maximum_stabilized: np.ndarray, masking_threshold_stabilized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute loss gradient of the masking threshold loss in PyTorch.\n\n        See also `ImperceptibleASR._loss_gradient_masking_threshold_tf`.\n        '
        import torch
        perturbation_torch = torch.from_numpy(perturbation).to(self.estimator._device)
        masking_threshold_stabilized_torch = torch.from_numpy(masking_threshold_stabilized).to(self.estimator._device)
        psd_maximum_stabilized_torch = torch.from_numpy(psd_maximum_stabilized).to(self.estimator._device)
        perturbation_torch.requires_grad = True
        psd_perturbation = self._approximate_power_spectral_density_torch(perturbation_torch, psd_maximum_stabilized_torch)
        loss = torch.mean(torch.nn.functional.relu(psd_perturbation - masking_threshold_stabilized_torch), dim=(1, 2), keepdims=False)
        loss.sum().backward()
        if perturbation_torch.grad is not None:
            loss_gradient = perturbation_torch.grad.cpu().numpy()
        else:
            raise ValueError('Gradient tensor in PyTorch model is `None`.')
        loss_value = loss.detach().cpu().numpy()
        return (loss_gradient, loss_value)

    def _approximate_power_spectral_density_tf(self, perturbation: 'Tensor', psd_maximum_stabilized: 'Tensor') -> 'Tensor':
        if False:
            while True:
                i = 10
        '\n        Approximate the power spectral density for a perturbation `perturbation` in TensorFlow.\n\n        Note that a stabilized PSD approximate is returned, where the `10*log10`-term has been canceled out.\n        Following Qin et al (2019) this mitigates optimization instabilities.\n\n        :param perturbation: Adversarial perturbation.\n        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of\n            the original unnormalized PSD of `x`.\n        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.\n        '
        import tensorflow.compat.v1 as tf1
        stft_matrix = tf1.signal.stft(perturbation, self._window_size, self._hop_size, fft_length=self._window_size)
        gain_factor = np.sqrt(8.0 / 3.0)
        psd_matrix = tf1.square(tf1.abs(gain_factor * stft_matrix / self._window_size))
        psd_matrix_approximated = tf1.pow(10.0, 9.6) / tf1.reshape(psd_maximum_stabilized, [-1, 1, 1]) * psd_matrix
        return tf1.transpose(psd_matrix_approximated, [0, 2, 1])

    def _approximate_power_spectral_density_torch(self, perturbation: 'PTensor', psd_maximum_stabilized: 'PTensor') -> 'PTensor':
        if False:
            for i in range(10):
                print('nop')
        '\n        Approximate the power spectral density for a perturbation `perturbation` in PyTorch.\n\n        See also `ImperceptibleASR._approximate_power_spectral_density_tf`.\n        '
        import torch
        stft_matrix = torch.stft(perturbation, n_fft=self._window_size, hop_length=self._hop_size, win_length=self._window_size, center=False, window=torch.hann_window(self._window_size).to(self.estimator._device)).to(self.estimator._device)
        gain_factor = np.sqrt(8.0 / 3.0)
        stft_matrix_abs = torch.sqrt(torch.sum(torch.square(gain_factor * stft_matrix / self._window_size), -1))
        psd_matrix = torch.square(stft_matrix_abs)
        psd_matrix_approximated = pow(10.0, 9.6) / psd_maximum_stabilized.reshape(-1, 1, 1) * psd_matrix
        return psd_matrix_approximated

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply attack-specific checks.\n        '
        if self.eps <= 0:
            raise ValueError('The perturbation max norm bound `eps` has to be positive.')
        if not isinstance(self.alpha, float):
            raise ValueError('The value of alpha must be of type float.')
        if self.alpha <= 0.0:
            raise ValueError('The value of alpha must be positive')
        if not isinstance(self.max_iter_1, int):
            raise ValueError('The maximum number of iterations for stage 1 must be of type int.')
        if self.max_iter_1 <= 0:
            raise ValueError('The maximum number of iterations for stage 1 must be greater than 0.')
        if not isinstance(self.max_iter_2, int):
            raise ValueError('The maximum number of iterations for stage 2 must be of type int.')
        if self.max_iter_2 < 0:
            raise ValueError('The maximum number of iterations for stage 2 must be non-negative.')
        if not isinstance(self.learning_rate_1, float):
            raise ValueError('The learning rate for stage 1 must be of type float.')
        if self.learning_rate_1 <= 0.0:
            raise ValueError('The learning rate for stage 1 must be greater than 0.0.')
        if not isinstance(self.learning_rate_2, float):
            raise ValueError('The learning rate for stage 2 must be of type float.')
        if self.learning_rate_2 <= 0.0:
            raise ValueError('The learning rate for stage 2 must be greater than 0.0.')
        if not isinstance(self.loss_theta_min, float):
            raise ValueError('The loss_theta_min threshold must be of type float.')
        if not isinstance(self.decrease_factor_eps, float):
            raise ValueError('The factor to decrease eps must be of type float.')
        if self.decrease_factor_eps <= 0.0:
            raise ValueError('The factor to decrease eps must be greater than 0.0.')
        if not isinstance(self.num_iter_decrease_eps, int):
            raise ValueError('The number of iterations must be of type int.')
        if self.num_iter_decrease_eps <= 0:
            raise ValueError('The number of iterations must be greater than 0.')
        if not isinstance(self.num_iter_decrease_alpha, int):
            raise ValueError('The number of iterations must be of type int.')
        if self.num_iter_decrease_alpha <= 0:
            raise ValueError('The number of iterations must be greater than 0.')
        if not isinstance(self.increase_factor_alpha, float):
            raise ValueError('The factor to increase alpha must be of type float.')
        if self.increase_factor_alpha <= 0.0:
            raise ValueError('The factor to increase alpha must be greater than 0.0.')
        if not isinstance(self.num_iter_increase_alpha, int):
            raise ValueError('The number of iterations must be of type int.')
        if self.num_iter_increase_alpha <= 0:
            raise ValueError('The number of iterations must be greater than 0.')
        if not isinstance(self.decrease_factor_alpha, float):
            raise ValueError('The factor to decrease alpha must be of type float.')
        if self.decrease_factor_alpha <= 0.0:
            raise ValueError('The factor to decrease alpha must be greater than 0.0.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.

    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int=2048, hop_size: int=512, sample_rate: int=16000) -> None:
        if False:
            print('Hello World!')
        '\n        Initialization.\n\n        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.\n        :param hop_size: Number of audio samples between adjacent STFT columns.\n        :param sample_rate: Sampling frequency of audio inputs.\n        '
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate
        self._fft_frequencies: Optional[np.ndarray] = None
        self._bark: Optional[np.ndarray] = None
        self._absolute_threshold_hearing: Optional[np.ndarray] = None

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the global masking threshold for an audio input and also return its maximum power spectral density.\n\n        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also\n        returns the maximum power spectral density (PSD) for each frame. Given an audio input, the following steps are\n        performed:\n\n        1. STFT analysis and sound pressure level normalization\n        2. Identification and filtering of maskers\n        3. Calculation of individual masking thresholds\n        4. Calculation of global masking thresholds\n\n        :param audio: Audio samples of shape `(length,)`.\n        :return: Global masking thresholds of shape `(window_size // 2 + 1, frame_length)` and the PSD maximum for each\n            frame of shape `(frame_length)`.\n        '
        (psd_matrix, psd_max) = self.power_spectral_density(audio)
        threshold = np.zeros_like(psd_matrix)
        for frame in range(psd_matrix.shape[1]):
            (maskers, masker_idx) = self.filter_maskers(*self.find_maskers(psd_matrix[:, frame]))
            threshold[:, frame] = self.calculate_global_threshold(self.calculate_individual_threshold(maskers, masker_idx))
        return (threshold, psd_max)

    @property
    def window_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: Window size of the masker.\n        '
        return self._window_size

    @property
    def hop_size(self) -> int:
        if False:
            print('Hello World!')
        '\n        :return: Hop size of the masker.\n        '
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        if False:
            print('Hello World!')
        '\n        :return: Sample rate of the masker.\n        '
        return self._sample_rate

    @property
    def fft_frequencies(self) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        :return: Discrete fourier transform sample frequencies.\n        '
        if self._fft_frequencies is None:
            self._fft_frequencies = np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> np.ndarray:
        if False:
            return 10
        '\n        :return: Bark scale for discrete fourier transform sample frequencies.\n        '
        if self._bark is None:
            self._bark = 13 * np.arctan(0.00076 * self.fft_frequencies) + 3.5 * np.arctan(np.square(self.fft_frequencies / 7500.0))
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> np.ndarray:
        if False:
            return 10
        '\n        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.\n        '
        if self._absolute_threshold_hearing is None:
            valid_domain = np.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 20000.0)
            freq = self.fft_frequencies[valid_domain] * 0.001
            self._absolute_threshold_hearing = np.ones(valid_domain.shape) * -np.inf
            self._absolute_threshold_hearing[valid_domain] = 3.64 * pow(freq, -0.8) - 6.5 * np.exp(-0.6 * np.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
        return self._absolute_threshold_hearing

    def power_spectral_density(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Compute the power spectral density matrix for an audio input.\n\n        :param audio: Audio sample of shape `(length,)`.\n        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape\n        `(frame_length)`.\n        '
        import librosa
        audio_float = audio.astype(np.float32)
        stft_params = {'n_fft': self.window_size, 'hop_length': self.hop_size, 'win_length': self.window_size, 'window': ss.get_window('hann', self.window_size, fftbins=True), 'center': False}
        stft_matrix = librosa.core.stft(audio_float, **stft_params)
        with np.errstate(divide='ignore'):
            gain_factor = np.sqrt(8.0 / 3.0)
            psd_matrix = 20 * np.log10(np.abs(gain_factor * stft_matrix / self.window_size))
            psd_matrix = psd_matrix.clip(min=-200)
        psd_matrix_max = np.max(psd_matrix)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix
        return (psd_matrix_normalized, psd_matrix_max)

    @staticmethod
    def find_maskers(psd_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Identify maskers.\n\n        Possible maskers are local PSD maxima. Following Qin et al., all maskers are treated as tonal. Thus neglecting\n        the nontonal type.\n\n        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.\n        :return: Possible PSD maskers and indices.\n        '
        masker_idx = ss.argrelmax(psd_vector)[0]
        psd_maskers = 10 * np.log10(np.sum([10 ** (psd_vector[masker_idx + i] / 10) for i in range(-1, 2)], axis=0))
        return (psd_maskers, masker_idx)

    def filter_maskers(self, maskers: np.ndarray, masker_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Filter maskers.\n\n        First, discard all maskers that are below the absolute threshold of hearing. Second, reduce pairs of maskers\n        that are within 0.5 bark distance of each other by keeping the larger masker.\n\n        :param maskers: Masker PSD values.\n        :param masker_idx: Masker indices.\n        :return: Filtered PSD maskers and indices.\n        '
        ath_condition = maskers > self.absolute_threshold_hearing[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]
        bark_condition = np.ones(masker_idx.shape, dtype=bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            if self.bark[i] - self.bark[i_prev] < 0.5:
                (i_todelete, i_prev) = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]
        return (maskers, masker_idx)

    def calculate_individual_threshold(self, maskers: np.ndarray, masker_idx: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Calculate individual masking threshold with frequency denoted at bark scale.\n\n        :param maskers: Masker PSD values.\n        :param masker_idx: Masker indices.\n        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.\n        '
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = np.zeros(masker_idx.shape + self.bark.shape)
        for (k, (masker_j, masker)) in enumerate(zip(masker_idx, maskers)):
            z_j = self.bark[masker_j]
            delta_z = self.bark - z_j
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function
        return threshold

    def calculate_global_threshold(self, individual_threshold):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate global masking threshold.\n\n        :param individual_threshold: Individual masking threshold vector.\n        :return: Global threshold vector of shape `(window_size // 2 + 1)`.\n        '
        with np.errstate(divide='ignore'):
            return 10 * np.log10(np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10))