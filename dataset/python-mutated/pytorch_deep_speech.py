"""
This module implements the task specific estimator for DeepSpeech, an end-to-end speech recognition in English and
Mandarin in PyTorch.

| Paper link: https://arxiv.org/abs/1512.02595
"""
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from pkg_resources import packaging
import numpy as np
from art import config
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin, PytorchSpeechRecognizerMixin
from art.utils import get_file
if TYPE_CHECKING:
    import torch
    from deepspeech_pytorch.model import DeepSpeech
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
logger = logging.getLogger(__name__)

class PyTorchDeepSpeech(PytorchSpeechRecognizerMixin, SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using the end-to-end speech recognizer
    DeepSpeech and PyTorch. It supports both version 2 and version 3 of DeepSpeech models as released at
    https://github.com/SeanNaren/deepspeech.pytorch.

    | Paper link: https://arxiv.org/abs/1512.02595
    """
    estimator_params = PyTorchEstimator.estimator_params + ['optimizer', 'use_amp', 'opt_level', 'lm_config', 'verbose']

    def __init__(self, model: Optional['DeepSpeech']=None, pretrained_model: Optional[str]=None, filename: Optional[str]=None, url: Optional[str]=None, use_half: bool=False, optimizer: Optional['torch.optim.Optimizer']=None, use_amp: bool=False, opt_level: str='O1', decoder_type: str='greedy', lm_path: str='', top_paths: int=1, alpha: float=0.0, beta: float=0.0, cutoff_top_n: int=40, cutoff_prob: float=1.0, beam_width: int=10, lm_workers: int=4, clip_values: Optional['CLIP_VALUES_TYPE']=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, device_type: str='gpu', verbose: bool=True):
        if False:
            print('Hello World!')
        '\n        Initialization of an instance PyTorchDeepSpeech.\n\n        :param model: DeepSpeech model.\n        :param pretrained_model: The choice of pretrained model if a pretrained model is required. Currently this\n                                 estimator supports 3 different pretrained models consisting of `an4`, `librispeech`\n                                 and `tedlium`.\n        :param filename: Name of the file.\n        :param url: Download URL.\n        :param use_half: Whether to use FP16 for pretrained model.\n        :param optimizer: The optimizer used to train the estimator.\n        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or\n                        gradient computation, e.g. with loss gradient computation. When set to True, this option is\n                        only triggered if there are GPUs available.\n        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted\n                          values are `O0`, `O1`, `O2`, and `O3`.\n        :param decoder_type: Decoder type. Either `greedy` or `beam`. This parameter is only used when users want\n                             transcription outputs.\n        :param lm_path: Path to an (optional) kenlm language model for use with beam search. This parameter is only\n                        used when users want transcription outputs.\n        :param top_paths: Number of beams to be returned. This parameter is only used when users want transcription\n                          outputs.\n        :param alpha: The weight used for the language model. This parameter is only used when users want transcription\n                      outputs.\n        :param beta: Language model word bonus (all words). This parameter is only used when users want transcription\n                     outputs.\n        :param cutoff_top_n: Cutoff_top_n characters with highest probs in vocabulary will be used in beam search. This\n                             parameter is only used when users want transcription outputs.\n        :param cutoff_prob: Cutoff probability in pruning. This parameter is only used when users want transcription\n                            outputs.\n        :param beam_width: The width of beam to be used. This parameter is only used when users want transcription\n                           outputs.\n        :param lm_workers: Number of language model processes to use. This parameter is only used when users want\n                           transcription outputs.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU\n                            if available otherwise run on CPU.\n        '
        import torch
        from deepspeech_pytorch.model import DeepSpeech
        from deepspeech_pytorch.configs.inference_config import LMConfig
        from deepspeech_pytorch.enums import DecoderType
        from deepspeech_pytorch.utils import load_decoder, load_model
        super().__init__(model=None, clip_values=clip_values, channels_first=None, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        if str(DeepSpeech.__base__) == "<class 'torch.nn.modules.module.Module'>":
            self._version = 2
        elif str(DeepSpeech.__base__) == "<class 'pytorch_lightning.core.lightning.LightningModule'>":
            self._version = 3
        else:
            raise NotImplementedError('Only DeepSpeech version 2 and DeepSpeech version 3 are currently supported.')
        self.verbose = verbose
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):
                raise ValueError('This estimator requires normalized input audios with clip_vales=(-1, 1).')
            if not np.all(self.clip_values[1] == 1):
                raise ValueError('This estimator requires normalized input audios with clip_vales=(-1, 1).')
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self._device: torch.device
        if device_type == 'cpu' or not torch.cuda.is_available():
            self._device = torch.device('cpu')
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f'cuda:{cuda_idx}')
        self._input_shape = None
        if model is None:
            if self._version == 2:
                if pretrained_model == 'an4':
                    (filename, url) = ('an4_pretrained_v2.pth', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth')
                elif pretrained_model == 'librispeech':
                    (filename, url) = ('librispeech_pretrained_v2.pth', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth')
                elif pretrained_model == 'tedlium':
                    (filename, url) = ('ted_pretrained_v2.pth', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth')
                elif pretrained_model is None:
                    if filename is None or url is None:
                        (filename, url) = ('librispeech_pretrained_v2.pth', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth')
                else:
                    raise ValueError(f'The input pretrained model {pretrained_model} is not supported.')
                model_path = get_file(filename=filename, path=config.ART_DATA_PATH, url=url, extract=False, verbose=self.verbose)
                self._model = load_model(device=self._device, model_path=model_path, use_half=use_half)
            else:
                if pretrained_model == 'an4':
                    (filename, url) = ('an4_pretrained_v3.ckpt', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt')
                elif pretrained_model == 'librispeech':
                    (filename, url) = ('librispeech_pretrained_v3.ckpt', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt')
                elif pretrained_model == 'tedlium':
                    (filename, url) = ('ted_pretrained_v3.ckpt', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt')
                elif pretrained_model is None:
                    if filename is None or url is None:
                        (filename, url) = ('librispeech_pretrained_v3.ckpt', 'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt')
                else:
                    raise ValueError(f'The input pretrained model {pretrained_model} is not supported.')
                model_path = get_file(filename=filename, path=config.ART_DATA_PATH, url=url, extract=False, verbose=self.verbose)
                self._model = load_model(device=self._device, model_path=model_path)
        else:
            self._model = model
            self._model.to(self._device)
        if self._version == 2:
            from warpctc_pytorch import CTCLoss
            self.criterion = CTCLoss()
        else:
            self.criterion = self._model.criterion
        self._optimizer = optimizer
        self._use_amp = use_amp
        self._opt_level = opt_level
        lm_config = LMConfig()
        if decoder_type == 'greedy':
            lm_config.decoder_type = DecoderType.greedy
        elif decoder_type == 'beam':
            lm_config.decoder_type = DecoderType.beam
        else:
            raise ValueError(f'Decoder type {decoder_type} currently not supported.')
        lm_config.lm_path = lm_path
        lm_config.top_paths = top_paths
        lm_config.alpha = alpha
        lm_config.beta = beta
        lm_config.cutoff_top_n = cutoff_top_n
        lm_config.cutoff_prob = cutoff_prob
        lm_config.beam_width = beam_width
        lm_config.lm_workers = lm_workers
        self.lm_config = lm_config
        self.decoder = load_decoder(labels=self._model.labels, cfg=lm_config)
        if self.use_amp:
            from apex import amp
            if self.optimizer is None:
                logger.warning('An optimizer is needed to use the automatic mixed precision tool, but none for provided. A default optimizer is used.')
                parameters = self._model.parameters()
                self._optimizer = torch.optim.SGD(parameters, lr=0.01)
            if self._device.type == 'cpu':
                enabled = False
            else:
                enabled = True
            (self._model, self._optimizer) = amp.initialize(models=self._model, optimizers=self._optimizer, enabled=enabled, opt_level=opt_level, loss_scale=1.0)

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if False:
            while True:
                i = 10
        "\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch\n                  could have different lengths. A possible example of `x` could be:\n                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.\n        :param batch_size: Batch size.\n        :param transcription_output: Indicate whether the function will produce probability or transcription as\n                                     prediction output. If transcription_output is not available, then probability\n                                     output is returned. Default: True\n        :return: Predicted probability (if transcription_output False) or transcription (default, if\n                 transcription_output is True):\n                 - Probability return is a tuple of (probs, sizes), where `probs` is the probability of characters of\n                 shape (nb_samples, seq_length, nb_classes) and `sizes` is the real sequence length of shape\n                 (nb_samples,).\n                 - Transcription return is a numpy array of characters. A possible example of a transcription return\n                 is `np.array(['SIXTY ONE', 'HELLO'])`.\n        "
        import torch
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        x_in = np.empty(len(x_preprocessed), dtype=object)
        x_in[:] = list(x_preprocessed)
        self._model.eval()
        (inputs, _, input_rates, _, batch_idx) = self._transform_model_input(x=x_in)
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()
        results = []
        result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            (begin, end) = (m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0]))
            with torch.no_grad():
                (outputs, output_sizes) = self._model(inputs[begin:end].to(self._device), input_sizes[begin:end].to(self._device))
            results.append(outputs)
            result_output_sizes[begin:end] = output_sizes.detach().cpu().numpy()
        result_outputs = np.zeros(shape=(x_preprocessed.shape[0], result_output_sizes.max(), results[0].shape[-1]), dtype=config.ART_NUMPY_DTYPE)
        for m in range(num_batch):
            (begin, end) = (m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0]))
            result_outputs[begin:end, :results[m].shape[1], :results[m].shape[-1]] = results[m].cpu().numpy()
        result_output_sizes_ = result_output_sizes.copy()
        result_outputs_ = result_outputs.copy()
        result_output_sizes[batch_idx] = result_output_sizes_
        result_outputs[batch_idx] = result_outputs_
        transcription_output = kwargs.get('transcription_output', True)
        if transcription_output is False:
            return (result_outputs, result_output_sizes)
        (decoded_output, _) = self.decoder.decode(torch.tensor(result_outputs, device=self._device), torch.tensor(result_output_sizes, device=self._device))
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)
        return decoded_output

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        "\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch\n                  could have different lengths. A possible example of `x` could be:\n                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.\n        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different\n                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :return: Loss gradients of the same shape as `x`.\n        "
        (x_preprocessed, _) = self._apply_preprocessing(x, None, fit=False)
        x_in = np.empty(len(x_preprocessed), dtype=object)
        x_in[:] = list(x_preprocessed)
        self._model.train()
        self.set_batchnorm(train=False)
        (inputs, targets, input_rates, target_sizes, _) = self._transform_model_input(x=x_in, y=y, compute_gradient=True)
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()
        (outputs, output_sizes) = self._model(inputs.to(self._device), input_sizes.to(self._device))
        outputs = outputs.transpose(0, 1)
        if self._version == 2:
            outputs = outputs.float()
        else:
            outputs = outputs.log_softmax(-1)
        loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        if self._version == 2:
            loss = loss / inputs.size(0)
        if self.use_amp:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        results_list = []
        for (i, _) in enumerate(x_in):
            results_list.append(x_in[i].grad.cpu().numpy().copy())
        results = np.array(results_list)
        if results.shape[0] == 1:
            results_ = np.empty(len(results), dtype=object)
            results_[:] = list(results)
            results = results_
        results = self._apply_preprocessing_gradient(x, results)
        if x.dtype != object:
            results = np.array([i for i in results], dtype=x.dtype)
            assert results.shape == x.shape and results.dtype == x.dtype
        self.set_batchnorm(train=True)
        return results

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=10, **kwargs) -> None:
        if False:
            return 10
        "\n        Fit the estimator on the training set `(x, y)`.\n\n        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch\n                  could have different lengths. A possible example of `x` could be:\n                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.\n        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different\n                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch\n               and providing it takes no effect.\n        "
        import random
        (x_preprocessed, _) = self._apply_preprocessing(x, None, fit=True)
        y_preprocessed = y
        x_in = np.empty(len(x_preprocessed), dtype=object)
        x_in[:] = list(x_preprocessed)
        self._model.train()
        if self.optimizer is None:
            raise ValueError('An optimizer is required to train the model, but none was provided.')
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))
        for _ in range(nb_epochs):
            random.shuffle(ind)
            for m in range(num_batch):
                (begin, end) = (m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0]))
                i_batch = np.empty(len(x_preprocessed[ind[begin:end]]), dtype=object)
                i_batch[:] = list(x_preprocessed[ind[begin:end]])
                o_batch = y_preprocessed[ind[begin:end]]
                (inputs, targets, input_rates, target_sizes, _) = self._transform_model_input(x=i_batch, y=o_batch, compute_gradient=False)
                input_sizes = input_rates.mul_(inputs.size(-1)).int()
                self.optimizer.zero_grad()
                (outputs, output_sizes) = self._model(inputs.to(self._device), input_sizes.to(self._device))
                outputs = outputs.transpose(0, 1)
                if self._version == 2:
                    outputs = outputs.float()
                else:
                    outputs = outputs.log_softmax(-1)
                loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
                if self._version == 2:
                    loss = loss / inputs.size(0)
                if self.use_amp:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

    def compute_loss_and_decoded_output(self, masked_adv_input: 'torch.Tensor', original_output: np.ndarray, **kwargs) -> Tuple['torch.Tensor', np.ndarray]:
        if False:
            i = 10
            return i + 15
        "\n        Compute loss function and decoded output.\n\n        :param masked_adv_input: The perturbed inputs.\n        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and\n                                it may possess different lengths. A possible example of `original_output` could be:\n                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.\n        :param real_lengths: Real lengths of original sequences.\n        :return: The loss and the decoded output.\n        "
        real_lengths = kwargs.get('real_lengths')
        if real_lengths is None:
            raise ValueError('The PyTorchDeepSpeech estimator needs information about the real lengths of input sequences to compute loss and decoded output.')
        (inputs, targets, input_rates, target_sizes, batch_idx) = self._preprocess_transform_model_input(x=masked_adv_input.to(self.device), y=original_output, real_lengths=real_lengths)
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()
        (outputs, output_sizes) = self.model(inputs.to(self.device), input_sizes.to(self.device))
        outputs_ = outputs.transpose(0, 1)
        if self._version == 2:
            outputs_ = outputs_.float()
        else:
            outputs_ = outputs_.log_softmax(-1)
        loss = self.criterion(outputs_, targets, output_sizes, target_sizes).to(self._device)
        if self._version == 2:
            loss = loss / inputs.size(0)
        (decoded_output, _) = self.decoder.decode(outputs, output_sizes)
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)
        decoded_output_ = decoded_output.copy()
        decoded_output[batch_idx] = decoded_output_
        return (loss, decoded_output)

    def _preprocess_transform_model_input(self, x: 'torch.Tensor', y: np.ndarray, real_lengths: np.ndarray) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', List]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Apply preprocessing and then transform the user input space into the model input space. This function is used\n        by the ASR attack to attack into the PyTorchDeepSpeech estimator whose defences are called with the\n        `_apply_preprocessing` function.\n\n        :param x: Samples of shape (nb_samples, seq_length).\n        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different\n                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :param real_lengths: Real lengths of original sequences.\n        :return: A tuple of inputs and targets in the model space with the original index\n                 `(inputs, targets, input_percentages, target_sizes, batch_idx)`, where:\n                 - inputs: model inputs of shape (nb_samples, nb_frequencies, seq_length).\n                 - targets: ground truth targets of shape (sum over nb_samples of real seq_lengths).\n                 - input_percentages: percentages of real inputs in inputs.\n                 - target_sizes: list of real seq_lengths.\n                 - batch_idx: original index of inputs.\n        "
        import torch
        x_batch = []
        for (i, _) in enumerate(x):
            (preprocessed_x_i, _) = self._apply_preprocessing(x=x[i], y=None, no_grad=False)
            x_batch.append(preprocessed_x_i)
        x = torch.stack(x_batch)
        (inputs, targets, input_rates, target_sizes, batch_idx) = self._transform_model_input(x=x, y=y, compute_gradient=False, tensor_input=True, real_lengths=real_lengths)
        return (inputs, targets, input_rates, target_sizes, batch_idx)

    def _transform_model_input(self, x: Union[np.ndarray, 'torch.Tensor'], y: Optional[np.ndarray]=None, compute_gradient: bool=False, tensor_input: bool=False, real_lengths: Optional[np.ndarray]=None) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', List]:
        if False:
            return 10
        "\n        Transform the user input space into the model input space.\n\n        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch\n                  could have different lengths. A possible example of `x` could be:\n                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.\n        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different\n                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.\n        :param compute_gradient: Indicate whether to compute gradients for the input `x`.\n        :param tensor_input: Indicate whether input is tensor.\n        :param real_lengths: Real lengths of original sequences.\n        :return: A tuple of inputs and targets in the model space with the original index\n                 `(inputs, targets, input_percentages, target_sizes, batch_idx)`, where:\n                 - inputs: model inputs of shape (nb_samples, nb_frequencies, seq_length).\n                 - targets: ground truth targets of shape (sum over nb_samples of real seq_lengths).\n                 - input_percentages: percentages of real inputs in inputs.\n                 - target_sizes: list of real seq_lengths.\n                 - batch_idx: original index of inputs.\n        "
        import torch
        import torchaudio
        from deepspeech_pytorch.loader.data_loader import _collate_fn
        if self._version == 2:
            window_name = self.model.audio_conf.window.value
            sample_rate = self.model.audio_conf.sample_rate
            window_size = self.model.audio_conf.window_size
            window_stride = self.model.audio_conf.window_stride
        else:
            window_name = self.model.spect_cfg['window'].value
            sample_rate = self.model.spect_cfg['sample_rate']
            window_size = self.model.spect_cfg['window_size']
            window_stride = self.model.spect_cfg['window_stride']
        n_fft = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        win_length = n_fft
        if window_name == 'hamming':
            window_fn = torch.hamming_window
        elif window_name == 'hann':
            window_fn = torch.hann_window
        elif window_name == 'blackman':
            window_fn = torch.blackman_window
        elif window_name == 'bartlett':
            window_fn = torch.bartlett_window
        else:
            raise NotImplementedError(f'Spectrogram window {window_name} not supported.')
        transformer = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None)
        transformer.to(self._device)
        label_map = {self._model.labels[i]: i for i in range(len(self._model.labels))}
        batch = []
        for (i, _) in enumerate(x):
            if y is None:
                target = []
            else:
                target = list(filter(None, [label_map.get(letter) for letter in list(y[i])]))
            if isinstance(x, np.ndarray) and (not tensor_input):
                x[i] = x[i].astype(config.ART_NUMPY_DTYPE)
                x[i] = torch.tensor(x[i]).to(self._device)
            if compute_gradient:
                x[i].requires_grad = True
            if tensor_input and real_lengths is not None:
                transformed_input = transformer(x[i][:real_lengths[i]])
            else:
                transformed_input = transformer(x[i])
            if self._version == 3 and packaging.version.parse(torch.__version__) >= packaging.version.parse('1.10.0'):
                spectrogram = torch.abs(transformed_input)
            else:
                (spectrogram, _) = torchaudio.functional.magphase(transformed_input)
            spectrogram = torch.log1p(spectrogram)
            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram = spectrogram - mean
            spectrogram = spectrogram / std
            batch.append((spectrogram, target))
        batch_idx = sorted(range(len(batch)), key=lambda i: batch[i][0].size(1), reverse=True)
        (inputs, targets, input_percentages, target_sizes) = _collate_fn(batch)
        return (inputs, targets, input_percentages, target_sizes, batch_idx)

    def to_training_mode(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Put the estimator in the training mode.\n        '
        self.model.train()

    @property
    def sample_rate(self) -> int:
        if False:
            return 10
        '\n        Get the sampling rate.\n\n        :return: The audio sampling rate.\n        '
        if self._version == 2:
            sample_rate = self.model.audio_conf.sample_rate
        else:
            sample_rate = self.model.spect_cfg['sample_rate']
        return sample_rate

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def model(self) -> 'DeepSpeech':
        if False:
            return 10
        '\n        Get current model.\n\n        :return: Current model.\n        '
        return self._model

    @property
    def device(self) -> 'torch.device':
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current used device.\n\n        :return: Current used device.\n        '
        return self._device

    @property
    def use_amp(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return a boolean indicating whether to use the automatic mixed precision tool.\n\n        :return: Whether to use the automatic mixed precision tool.\n        '
        return self._use_amp

    @property
    def optimizer(self) -> 'torch.optim.Optimizer':
        if False:
            print('Hello World!')
        '\n        Return the optimizer.\n\n        :return: The optimizer.\n        '
        return self._optimizer

    @property
    def opt_level(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Return a string specifying a pure or mixed precision optimization level.\n\n        :return: A string specifying a pure or mixed precision optimization level. Possible\n                 values are `O0`, `O1`, `O2`, and `O3`.\n        '
        return self._opt_level

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        raise NotImplementedError