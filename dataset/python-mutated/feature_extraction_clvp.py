"""
Feature extractor class for CLVP
"""
from typing import List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class ClvpFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        default_audio_length (`int`, *optional*, defaults to 6):
            The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
            automatically be set to default_audio_length * `self.sampling_rate`.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        mel_norms (`list` of length `feature_size`, *optional*):
            If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
            mel-filter.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)
    """
    model_input_names = ['input_features', 'attention_mask']

    def __init__(self, feature_size=80, sampling_rate=22050, default_audio_length=6, hop_length=256, chunk_length=30, n_fft=1024, padding_value=0.0, mel_norms=None, return_attention_mask=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, return_attention_mask=return_attention_mask, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.default_audio_length = default_audio_length
        self.mel_norms = mel_norms
        self.mel_filters = mel_filter_bank(num_frequency_bins=1 + n_fft // 2, num_mel_filters=feature_size, min_frequency=0.0, max_frequency=8000.0, sampling_rate=sampling_rate, norm='slaney', mel_scale='htk')

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        This method first computes the log-mel spectrogram of the provided audio then applies normalization along the\n        each mel-filterbank, if `mel_norms` is provided.\n        '
        log_spec = spectrogram(waveform, window_function(self.n_fft, 'hann'), frame_length=self.n_fft, hop_length=self.hop_length, power=2.0, mel_filters=self.mel_filters, log_mel=None)
        log_spec = np.log(np.clip(log_spec, a_min=1e-05, a_max=None))
        if self.mel_norms is not None:
            log_spec = log_spec / np.array(self.mel_norms)[:, None]
        return log_spec

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], sampling_rate: Optional[int]=None, truncation: bool=True, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_attention_mask: Optional[bool]=True, padding: Optional[str]='max_length', max_length: Optional[int]=None, **kwargs) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        "\n        `ClvpFeatureExtractor` is used to extract various voice specific properties such as the pitch and tone of the\n        voice, speaking speed, and even speaking defects like a lisp or stuttering from a sample voice or `raw_speech`.\n\n        First the voice is padded or truncated in a way such that it becomes a waveform of `self.default_audio_length`\n        seconds long and then the log-mel spectrogram is extracted from it.\n\n        Args:\n            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):\n                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float\n                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not\n                stereo, i.e. single float per timestep.\n            sampling_rate (`int`, *optional*):\n                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass\n                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition\n                pipeline.\n            truncation (`bool`, *optional*, default to `True`):\n                Activates truncation to cut input sequences longer than *max_length* to *max_length*.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.\n            return_attention_mask (`bool`, *optional*, defaults to `True`):\n                Whether to return the attention mask. If left to the default, it will return the attention mask.\n\n                [What are attention masks?](../glossary#attention-mask)\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n            padding_value (`float`, defaults to 0.0):\n                The value that is used to fill the padding values / vectors.\n            max_length (`int`, *optional*):\n                The maximum input length of the inputs.\n        "
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f'The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with {self.sampling_rate} and not {sampling_rate}.')
        else:
            logger.warning('It is strongly recommended to pass the `sampling_rate` argument to this function. Failing to do so can result in silent errors that might be hard to debug.')
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f'Only mono-channel audio is supported for input to {self}')
        is_batched = is_batched_numpy or (isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and (not isinstance(raw_speech, np.ndarray)):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]
        batched_speech = BatchFeature({'input_features': raw_speech})
        max_length = self.default_audio_length * self.sampling_rate if max_length is None else max_length
        padded_inputs = self.pad(batched_speech, padding=padding, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask)
        input_features = padded_inputs.get('input_features').transpose(2, 0, 1)
        input_features = [self._np_extract_fbank_features(waveform).astype(np.float32) for waveform in input_features[0]]
        if isinstance(input_features[0], List):
            padded_inputs['input_features'] = [np.asarray(feature) for feature in input_features]
        else:
            padded_inputs['input_features'] = input_features
        return padded_inputs.convert_to_tensors(return_tensors)