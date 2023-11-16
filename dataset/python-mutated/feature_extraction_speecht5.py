"""Feature extractor class for SpeechT5."""
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
logger = logging.get_logger(__name__)

class SpeechT5FeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a SpeechT5 feature extractor.

    This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by
    the SpeechT5 speech encoder prenet.

    This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder
    prenet.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel-frequency bins in the extracted spectrogram features.
        hop_length (`int`, *optional*, defaults to 16):
            Number of ms between windows. Otherwise referred to as "shift" in many papers.
        win_length (`int`, *optional*, defaults to 64):
            Number of ms per window.
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        frame_signal_scale (`float`, *optional*, defaults to 1.0):
            Constant multiplied in creating the frames before applying DFT. This argument is deprecated.
        fmin (`float`, *optional*, defaults to 80):
            Minimum mel frequency in Hz.
        fmax (`float`, *optional*, defaults to 7600):
            Maximum mel frequency in Hz.
        mel_floor (`float`, *optional*, defaults to 1e-10):
            Minimum value of mel frequency banks.
        reduction_factor (`int`, *optional*, defaults to 2):
            Spectrogram length reduction factor. This argument is deprecated.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~SpeechT5FeatureExtractor.__call__`] should return `attention_mask`.
    """
    model_input_names = ['input_values', 'attention_mask']

    def __init__(self, feature_size: int=1, sampling_rate: int=16000, padding_value: float=0.0, do_normalize: bool=False, num_mel_bins: int=80, hop_length: int=16, win_length: int=64, win_function: str='hann_window', frame_signal_scale: float=1.0, fmin: float=80, fmax: float=7600, mel_floor: float=1e-10, reduction_factor: int=2, return_attention_mask: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.do_normalize = do_normalize
        self.return_attention_mask = return_attention_mask
        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.frame_signal_scale = frame_signal_scale
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor
        self.reduction_factor = reduction_factor
        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000
        self.n_fft = optimal_fft_length(self.sample_size)
        self.n_freqs = self.n_fft // 2 + 1
        self.window = window_function(window_length=self.sample_size, name=self.win_function, periodic=True)
        self.mel_filters = mel_filter_bank(num_frequency_bins=self.n_freqs, num_mel_filters=self.num_mel_bins, min_frequency=self.fmin, max_frequency=self.fmax, sampling_rate=self.sampling_rate, norm='slaney', mel_scale='slaney')
        if frame_signal_scale != 1.0:
            warnings.warn('The argument `frame_signal_scale` is deprecated and will be removed in version 4.30.0 of Transformers', FutureWarning)
        if reduction_factor != 2.0:
            warnings.warn('The argument `reduction_factor` is deprecated and will be removed in version 4.30.0 of Transformers', FutureWarning)

    @staticmethod
    def zero_mean_unit_var_norm(input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float=0.0) -> List[np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Every array in the list is normalized to have zero mean and unit variance\n        '
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []
            for (vector, length) in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-07)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value
                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-07) for x in input_values]
        return normed_input_values

    def _extract_mel_features(self, one_waveform: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Extracts log-mel filterbank features for one waveform array (unbatched).\n        '
        log_mel_spec = spectrogram(one_waveform, window=self.window, frame_length=self.sample_size, hop_length=self.sample_stride, fft_length=self.n_fft, mel_filters=self.mel_filters, mel_floor=self.mel_floor, log_mel='log10')
        return log_mel_spec.T

    def __call__(self, audio: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]]=None, audio_target: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]]=None, padding: Union[bool, str, PaddingStrategy]=False, max_length: Optional[int]=None, truncation: bool=False, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None, sampling_rate: Optional[int]=None, **kwargs) -> BatchFeature:
        if False:
            while True:
                i = 10
        "\n        Main method to featurize and prepare for the model one or several sequence(s).\n\n        Pass in a value for `audio` to extract waveform features. Pass in a value for `audio_target` to extract log-mel\n        spectrogram features.\n\n        Args:\n            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, *optional*):\n                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float\n                values, a list of numpy arrays or a list of list of float values. This outputs waveform features. Must\n                be mono channel audio, not stereo, i.e. single float per timestep.\n            audio_target (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, *optional*):\n                The sequence or batch of sequences to be processed as targets. Each sequence can be a numpy array, a\n                list of float values, a list of numpy arrays or a list of list of float values. This outputs log-mel\n                spectrogram features.\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):\n                Select a strategy to pad the returned sequences (according to the model's padding side and padding\n                index) among:\n\n                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            max_length (`int`, *optional*):\n                Maximum length of the returned list and optionally padding length (see above).\n            truncation (`bool`):\n                Activates truncation to cut input sequences longer than *max_length* to *max_length*.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.\n            return_attention_mask (`bool`, *optional*):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific feature_extractor's default.\n\n                [What are attention masks?](../glossary#attention-mask)\n\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n            sampling_rate (`int`, *optional*):\n                The sampling rate at which the `audio` or `audio_target` input was sampled. It is strongly recommended\n                to pass `sampling_rate` at the forward call to prevent silent errors.\n        "
        if audio is None and audio_target is None:
            raise ValueError('You must provide either `audio` or `audio_target` values.')
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f'The model corresponding to this feature extractor: {self} was trained using a sampling rate of {self.sampling_rate}. Please make sure that the provided audio input was sampled with {self.sampling_rate} and not {sampling_rate}.')
        else:
            logger.warning('It is strongly recommended to pass the ``sampling_rate`` argument to this function. Failing to do so can result in silent errors that might be hard to debug.')
        if audio is not None:
            inputs = self._process_audio(audio, False, padding, max_length, truncation, pad_to_multiple_of, return_attention_mask, return_tensors, **kwargs)
        else:
            inputs = None
        if audio_target is not None:
            inputs_target = self._process_audio(audio_target, True, padding, max_length, truncation, pad_to_multiple_of, return_attention_mask, return_tensors, **kwargs)
            if inputs is None:
                return inputs_target
            else:
                inputs['labels'] = inputs_target['input_values']
                decoder_attention_mask = inputs_target.get('attention_mask')
                if decoder_attention_mask is not None:
                    inputs['decoder_attention_mask'] = decoder_attention_mask
        return inputs

    def _process_audio(self, speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], is_target: bool=False, padding: Union[bool, str, PaddingStrategy]=False, max_length: Optional[int]=None, truncation: bool=False, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchFeature:
        if False:
            return 10
        is_batched_numpy = isinstance(speech, np.ndarray) and len(speech.shape) > 1
        if is_batched_numpy and len(speech.shape) > 2:
            raise ValueError(f'Only mono-channel audio is supported for input to {self}')
        is_batched = is_batched_numpy or (isinstance(speech, (list, tuple)) and isinstance(speech[0], (np.ndarray, tuple, list)))
        if is_batched:
            speech = [np.asarray(speech, dtype=np.float32) for speech in speech]
        elif not is_batched and (not isinstance(speech, np.ndarray)):
            speech = np.asarray(speech, dtype=np.float32)
        elif isinstance(speech, np.ndarray) and speech.dtype is np.dtype(np.float64):
            speech = speech.astype(np.float32)
        if not is_batched:
            speech = [speech]
        feature_size_hack = self.feature_size
        if is_target:
            features = [self._extract_mel_features(waveform) for waveform in speech]
            encoded_inputs = BatchFeature({'input_values': features})
            self.feature_size = self.num_mel_bins
        else:
            encoded_inputs = BatchFeature({'input_values': speech})
        padded_inputs = self.pad(encoded_inputs, padding=padding, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, **kwargs)
        self.feature_size = feature_size_hack
        input_values = padded_inputs['input_values']
        if not isinstance(input_values[0], np.ndarray):
            padded_inputs['input_values'] = [np.asarray(array, dtype=np.float32) for array in input_values]
        elif not isinstance(input_values, np.ndarray) and isinstance(input_values[0], np.ndarray) and (input_values[0].dtype is np.dtype(np.float64)):
            padded_inputs['input_values'] = [array.astype(np.float32) for array in input_values]
        elif isinstance(input_values, np.ndarray) and input_values.dtype is np.dtype(np.float64):
            padded_inputs['input_values'] = input_values.astype(np.float32)
        attention_mask = padded_inputs.get('attention_mask')
        if attention_mask is not None:
            padded_inputs['attention_mask'] = [np.asarray(array, dtype=np.int32) for array in attention_mask]
        if not is_target and self.do_normalize:
            attention_mask = attention_mask if self._get_padding_strategies(padding, max_length=max_length) is not PaddingStrategy.DO_NOT_PAD else None
            padded_inputs['input_values'] = self.zero_mean_unit_var_norm(padded_inputs['input_values'], attention_mask=attention_mask, padding_value=self.padding_value)
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        return padded_inputs

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        output = super().to_dict()
        names = ['window', 'mel_filters', 'sample_size', 'sample_stride', 'n_fft', 'n_freqs']
        for name in names:
            if name in output:
                del output[name]
        return output