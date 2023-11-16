"""
This module implements task-specific estimators for automatic speech recognition in TensorFlow.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from art import config
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import get_file, make_directory
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
    from tensorflow.compat.v1 import Tensor
    from tensorflow.compat.v1 import Session
logger = logging.getLogger(__name__)

class TensorFlowLingvoASR(SpeechRecognizerMixin, TensorFlowV2Estimator):
    """
    This class implements the task-specific Lingvo ASR model of Qin et al. (2019).

    The estimator uses a pre-trained model provided by Qin et al., which is trained using the Lingvo library and the
    LibriSpeech dataset.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html, https://arxiv.org/abs/1902.08295

    .. warning:: In order to calculate loss gradients, this estimator requires a user-patched Lingvo module. A patched
                 source file for the `lingvo.tasks.asr.decoder` module will be automatically applied. The original
                 source file can be found in `<PYTHON_SITE_PACKAGES>/lingvo/tasks/asr/decoder.py` and will be patched as
                 outlined in the following commit diff:
                 https://github.com/yaq007/lingvo/commit/414e035b2c60372de732c9d67db14d1003be6dd6

    The patched `decoder_patched.py` can be found in `ART_DATA_PATH/lingvo/asr`.

    Note: Run `python -m site` to obtain a list of possible candidates where to find the `<PYTHON_SITE_PACKAGES` folder.
    """
    _LINGVO_CFG: Dict[str, Any] = {'path': os.path.join(config.ART_DATA_PATH, 'lingvo'), 'model_data': {'uri': 'http://cseweb.ucsd.edu/~yaq007/ckpt-00908156.data-00000-of-00001', 'basename': 'ckpt-00908156.data-00000-of-00001'}, 'model_index': {'uri': 'https://github.com/tensorflow/cleverhans/blob/6ef939059172901db582c7702eb803b7171e3db5/examples/adversarial_asr/model/ckpt-00908156.index?raw=true', 'basename': 'ckpt-00908156.index'}, 'params': {'uri': 'https://raw.githubusercontent.com/tensorflow/lingvo/9961306adf66f7340e27f109f096c9322d4f9636/lingvo/tasks/asr/params/librispeech.py', 'basename': 'librispeech.py'}, 'vocab': {'uri': 'https://raw.githubusercontent.com/tensorflow/lingvo/9961306adf66f7340e27f109f096c9322d4f9636/lingvo/tasks/asr/wpm_16k_librispeech.vocab', 'basename': 'wpm_16k_librispeech.vocab'}, 'decoder': {'uri': 'https://raw.githubusercontent.com/Trusted-AI/adversarial-robustness-toolbox/4dabf5fcfb55502316ad48abbdc1a26033db1da5/contrib/lingvo-patched-decoder.py', 'basename': 'decoder_patched.py'}}
    estimator_params = TensorFlowV2Estimator.estimator_params + ['random_seed', 'sess']

    def __init__(self, clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: Optional[bool]=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, random_seed: Optional[int]=None, sess: Optional['Session']=None):
        if False:
            i = 10
            return i + 15
        '\n        Initialization.\n\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n                used for data preprocessing. The first value will be subtracted from the input. The input will then\n                be divided by the second one.\n        :param random_seed: Specify a random seed.\n        '
        import pkg_resources
        import tensorflow.compat.v1 as tf1
        super().__init__(model=None, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        self.random_seed = random_seed
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self._input_shape = None
        if tf1.__version__ != '2.1.0':
            raise AssertionError('The Lingvo estimator only supports TensorFlow 2.1.0.')
        if sys.version_info[:2] != (3, 6):
            raise AssertionError('The Lingvo estimator only supports Python 3.6.')
        if pkg_resources.get_distribution('lingvo').version != '0.6.4':
            raise AssertionError('The Lingvo estimator only supports Lingvo 0.6.4')
        tf1.disable_eager_execution()
        sys.path.append(self._LINGVO_CFG['path'])
        tf1.flags.FLAGS(tuple(sys.argv[0]))
        _ = self._check_and_download_file(self._LINGVO_CFG['params']['uri'], self._LINGVO_CFG['params']['basename'], self._LINGVO_CFG['path'], 'asr')
        self._x_padded: 'Tensor' = tf1.placeholder(tf1.float32, shape=[None, None], name='art_x_padded')
        self._y_target: 'Tensor' = tf1.placeholder(tf1.string, name='art_y_target')
        self._mask_frequency: 'Tensor' = tf1.placeholder(tf1.float32, shape=[None, None, 80], name='art_mask_frequency')
        self._sess: 'Session' = tf1.Session() if sess is None else sess
        (model, task, cluster) = self._load_model()
        self._model = model
        self._task = task
        self._cluster = cluster
        self._metrics: Optional[Tuple[Union[Dict[str, 'Tensor'], Dict[str, Tuple['Tensor', 'Tensor']]], ...]] = None
        self._predict_batch_op: Dict[str, 'Tensor'] = self._predict_batch(self._x_padded, self._y_target, self._mask_frequency)
        self._loss_gradient_op: 'Tensor' = self._loss_gradient(self._x_padded, self._y_target, self._mask_frequency)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def sess(self) -> 'Session':
        if False:
            while True:
                i = 10
        '\n        Get current TensorFlow session.\n\n        :return: The current TensorFlow session.\n        '
        return self._sess

    @staticmethod
    def _check_and_download_file(uri: str, basename: str, *paths: str) -> str:
        if False:
            while True:
                i = 10
        'Check and download the file from given URI.'
        dir_path = os.path.join(*paths)
        file_path = os.path.join(dir_path, basename)
        if not os.path.isdir(dir_path):
            make_directory(dir_path)
        if not os.path.isfile(file_path):
            logger.info('Could not find %s. Downloading it now...', basename)
            get_file(basename, uri, path=dir_path)
        return file_path

    def _load_model(self):
        if False:
            while True:
                i = 10
        '\n        Define and instantiate the computation graph.\n        '
        import tensorflow.compat.v1 as tf1
        from lingvo import model_registry, model_imports
        from lingvo.core import cluster_factory
        from asr.librispeech import Librispeech960Wpm
        _ = self._check_and_download_file(self._LINGVO_CFG['decoder']['uri'], self._LINGVO_CFG['decoder']['basename'], self._LINGVO_CFG['path'], 'asr')
        from lingvo.tasks.asr import decoder
        from asr import decoder_patched
        decoder.AsrDecoderBase._ComputeMetrics = decoder_patched.AsrDecoderBase._ComputeMetrics
        vocab_path = self._check_and_download_file(self._LINGVO_CFG['vocab']['uri'], self._LINGVO_CFG['vocab']['basename'], self._LINGVO_CFG['path'], 'asr')
        Librispeech960Wpm.WPM_SYMBOL_TABLE_FILEPATH = vocab_path
        model_name = 'asr.librispeech.Librispeech960Wpm'
        model_imports.ImportParams(model_name)
        params = model_registry._ModelRegistryHelper.GetParams(model_name, 'Test')
        if self.random_seed is not None:
            params.random_seed = self.random_seed
        cluster = cluster_factory.Cluster(params.cluster)
        with cluster, tf1.device(cluster.GetPlacer()):
            model = params.Instantiate()
            task = model.GetTask()
        _ = self._check_and_download_file(self._LINGVO_CFG['model_data']['uri'], self._LINGVO_CFG['model_data']['basename'], self._LINGVO_CFG['path'], 'asr', 'model')
        model_index_path = self._check_and_download_file(self._LINGVO_CFG['model_index']['uri'], self._LINGVO_CFG['model_index']['basename'], self._LINGVO_CFG['path'], 'asr', 'model')
        self.sess.run(tf1.global_variables_initializer())
        saver = tf1.train.Saver([var for var in tf1.global_variables() if var.name.startswith('librispeech')])
        saver.restore(self.sess, os.path.splitext(model_index_path)[0])
        tf1.flags.FLAGS.enable_asserts = False
        return (model, task, cluster)

    def _create_decoder_input(self, x: 'Tensor', y: 'Tensor', mask_frequency: 'Tensor') -> 'Tensor':
        if False:
            return 10
        'Create decoder input per batch.'
        import tensorflow.compat.v1 as tf1
        from lingvo.core.py_utils import NestedMap
        source_features = self._create_log_mel_features(x)
        source_features *= tf1.expand_dims(mask_frequency, dim=-1)
        source_paddings = 1.0 - mask_frequency[:, :, 0]
        target = self._task.input_generator.StringsToIds(y)
        decoder_inputs = NestedMap({'src': NestedMap({'src_inputs': source_features, 'paddings': source_paddings}), 'sample_ids': tf1.zeros(tf1.shape(source_features)[0]), 'tgt': NestedMap(zip(('ids', 'labels', 'paddings'), target))})
        decoder_inputs.tgt['weights'] = 1.0 - decoder_inputs.tgt['paddings']
        return decoder_inputs

    @staticmethod
    def _create_log_mel_features(x: 'Tensor') -> 'Tensor':
        if False:
            print('Hello World!')
        'Extract Log-Mel features from audio samples of shape (batch_size, max_length).'
        from lingvo.core.py_utils import NestedMap
        import tensorflow.compat.v1 as tf1

        def _create_asr_frontend():
            if False:
                i = 10
                return i + 15
            'Parameters corresponding to default ASR frontend.'
            from lingvo.tasks.asr import frontend
            params = frontend.MelAsrFrontend.Params()
            params.sample_rate = 16000.0
            params.frame_size_ms = 25.0
            params.frame_step_ms = 10.0
            params.num_bins = 80
            params.lower_edge_hertz = 125.0
            params.upper_edge_hertz = 7600.0
            params.preemph = 0.97
            params.noise_scale = 0.0
            params.pad_end = False
            params.stack_left_context = 2
            params.frame_stride = 3
            return params.Instantiate()
        mel_frontend = _create_asr_frontend()
        log_mel = mel_frontend.FPropDefaultTheta(NestedMap(src_inputs=x, paddings=tf1.zeros_like(x)))
        features = log_mel.src_inputs
        features_shape = (tf1.shape(x)[0], -1, 80, features.shape[-1])
        features = tf1.reshape(features, features_shape)
        return features

    @staticmethod
    def _pad_audio_input(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        'Apply padding to a batch of audio samples such that it has shape of (batch_size, max_length).'
        max_length = max(map(len, x))
        batch_size = x.shape[0]
        assert max_length >= 480, 'Maximum length of audio input must be at least 480.'
        frequency_length = [(len(item) // 2 + 1) // 240 * 3 for item in x]
        max_frequency_length = max(frequency_length)
        x_padded = np.zeros((batch_size, max_length))
        x_mask = np.zeros((batch_size, max_length), dtype=bool)
        mask_frequency = np.zeros((batch_size, max_frequency_length, 80))
        for (i, x_i) in enumerate(x):
            x_padded[i, :len(x_i)] = x_i
            x_mask[i, :len(x_i)] = 1
            mask_frequency[i, :frequency_length[i], :] = 1
        return (x_padded, x_mask, mask_frequency)

    def _predict_batch(self, x: 'Tensor', y: 'Tensor', mask_frequency: 'Tensor') -> Dict[str, 'Tensor']:
        if False:
            for i in range(10):
                print('nop')
        'Create prediction operation for a batch of padded inputs.'
        import tensorflow.compat.v1 as tf1
        decoder_inputs = self._create_decoder_input(x, y, mask_frequency)
        if self._metrics is None:
            with self._cluster, tf1.device(self._cluster.GetPlacer()):
                self._metrics = self._task.FPropDefaultTheta(decoder_inputs)
        predictions = self._task.Decode(decoder_inputs)
        return predictions

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Perform batch-wise prediction for given inputs.\n\n        :param x: Samples of shape `(nb_samples)` with values in range `[-32768, 32767]`. Note that it is allowable\n                  that sequences in the batch could have different lengths. A possible example of `x` could be:\n                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.\n        :param batch_size: Size of batches.\n        :return: Array of predicted transcriptions of shape `(nb_samples)`. A possible example of a transcription\n                 return is `np.array(['SIXTY ONE', 'HELLO'])`.\n        "
        if x[0].ndim != 1:
            raise ValueError('The LingvoASR estimator can only be used temporal data of type mono. Please remove any channeldimension.')
        is_normalized = max(map(max, np.abs(x))) <= 1.0
        if is_normalized and self.preprocessing is None:
            raise ValueError('The LingvoASR estimator requires input values in the range [-32768, 32767] or normalized input values with correct preprocessing argument (mean=0, stddev=1/normalization_factor).')
        nb_samples = x.shape[0]
        assert nb_samples % batch_size == 0, 'Number of samples must be divisible by batch_size'
        (x, _) = self._apply_preprocessing(x, y=None, fit=False)
        y = []
        nb_batches = int(np.ceil(nb_samples / float(batch_size)))
        for m in range(nb_batches):
            (begin, end) = (m * batch_size, min((m + 1) * batch_size, nb_samples))
            (x_batch_padded, _, mask_frequency) = self._pad_audio_input(x[begin:end])
            feed_dict = {self._x_padded: x_batch_padded, self._y_target: np.array(['DUMMY'] * batch_size), self._mask_frequency: mask_frequency}
            y_batch = self.sess.run(self._predict_batch_op, feed_dict)
            y += y_batch['topk_decoded'][:, 0].tolist()
        y_decoded = [item.decode('utf-8').upper() for item in y]
        return np.array(y_decoded, dtype=str)

    def _loss_gradient(self, x: 'Tensor', y: 'Tensor', mask: 'Tensor') -> 'Tensor':
        if False:
            for i in range(10):
                print('nop')
        'Define loss gradients computation operation for a batch of padded inputs.'
        import tensorflow.compat.v1 as tf1
        decoder_inputs = self._create_decoder_input(x, y, mask)
        if self._metrics is None:
            with self._cluster, tf1.device(self._cluster.GetPlacer()):
                self._metrics = self._task.FPropDefaultTheta(decoder_inputs)
        loss = tf1.get_collection('per_loss')[0]
        loss_gradient = tf1.gradients(loss, [x])[0]
        return loss_gradient

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, batch_mode: bool=False, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        "\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape `(nb_samples)`. Note that, it is allowable that sequences in the batch\n                  could have different lengths. A possible example of `x` could be:\n                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.\n        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different\n                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :param batch_mode: If `True` calculate gradient per batch or otherwise per sequence.\n        :return: Loss gradients of the same shape as `x`.\n        "
        is_normalized = max(map(max, np.abs(x))) <= 1.0
        if is_normalized and self.preprocessing is None:
            raise ValueError('The LingvoASR estimator requires input values in the range [-32768, 32767] or normalized input values with correct preprocessing argument (mean=0, stddev=1/normalization_factor).')
        y = np.array([y_i.lower() for y_i in y])
        (x_preprocessed, y_preprocessed) = self._apply_preprocessing(x, y, fit=False)
        if batch_mode:
            gradients = self._loss_gradient_per_batch(x_preprocessed, y_preprocessed)
        else:
            gradients = self._loss_gradient_per_sequence(x_preprocessed, y_preprocessed)
        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

    def _loss_gradient_per_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Compute the gradient of the loss function w.r.t. `x` per batch.\n        '
        assert x.shape[0] == y.shape[0], 'Number of samples in x and y differ.'
        (x_padded, mask, mask_frequency) = self._pad_audio_input(x)
        feed_dict = {self._x_padded: x_padded, self._y_target: y, self._mask_frequency: mask_frequency}
        gradients_padded = self.sess.run(self._loss_gradient_op, feed_dict)
        lengths = mask.sum(axis=1)
        gradients = []
        for (gradient_padded, length) in zip(gradients_padded, lengths):
            gradient = gradient_padded[:length]
            gradients.append(gradient)
        dtype = np.float32 if x.ndim != 1 else object
        return np.array(gradients, dtype=dtype)

    def _loss_gradient_per_sequence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the gradient of the loss function w.r.t. `x` per sequence.\n        '
        assert x.shape[0] == y.shape[0], 'Number of samples in x and y differ.'
        (_, _, mask_frequency) = self._pad_audio_input(x)
        gradients = []
        for (x_i, y_i, mask_frequency_i) in zip(x, y, mask_frequency):
            frequency_length = (len(x_i) // 2 + 1) // 240 * 3
            feed_dict = {self._x_padded: np.expand_dims(x_i, 0), self._y_target: np.array([y_i]), self._mask_frequency: np.expand_dims(mask_frequency_i[:frequency_length], 0)}
            gradient = self.sess.run(self._loss_gradient_op, feed_dict)
            gradients.append(np.squeeze(gradient))
        dtype = np.float32 if x.ndim != 1 else object
        return np.array(gradients, dtype=dtype)

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        raise NotImplementedError