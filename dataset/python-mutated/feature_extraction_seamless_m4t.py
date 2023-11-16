"""
Feature extractor class for SeamlessM4T
"""
from typing import List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
logger = logging.get_logger(__name__)

class SeamlessM4TFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a SeamlessM4T feature extractor.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
    """
    model_input_names = ['input_features', 'attention_mask']

    def __init__(self, feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=0.0, stride=2, **kwargs):
        if False:
            print('Hello World!')
        self.num_mel_bins = num_mel_bins
        self.return_attention_mask = True
        self.stride = stride
        mel_filters = mel_filter_bank(num_frequency_bins=256, num_mel_filters=self.num_mel_bins, min_frequency=20, max_frequency=sampling_rate // 2, sampling_rate=sampling_rate, norm=None, mel_scale='kaldi', triangularize_in_mel_space=True)
        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
        self.window = window_function(400, 'povey', periodic=False)
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

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

    def _extract_fbank_features(self, waveform: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs\n        and hence the waveform should not be normalized before feature extraction.\n        '
        if len(waveform.shape) == 2:
            waveform = waveform[0]
        waveform = np.squeeze(waveform) * 2 ** 15
        features = spectrogram(waveform, self.window, frame_length=400, hop_length=160, fft_length=512, power=2.0, center=False, preemphasis=0.97, mel_filters=self.mel_filters, log_mel='log', mel_floor=1.192092955078125e-07, remove_dc_offset=True).T
        return features

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], padding: Union[bool, str, PaddingStrategy]=True, pad_to_multiple_of: Optional[int]=2, max_length: Optional[int]=None, truncation: bool=False, return_tensors: Optional[Union[str, TensorType]]=None, sampling_rate: Optional[int]=None, return_attention_mask: Optional[bool]=None, do_normalize_per_mel_bins: Optional[bool]=True, **kwargs) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        "\n        Main method to featurize and prepare for the model one or several sequence(s).\n\n        Args:\n            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, `List[List[List[float]]]`):\n                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float\n                values, a list of numpy arrays, a list of list of float values or a list of a list of list of float\n                values. If `raw_speech` is a one-dimensional `np.ndarray` or a `List[float]`, `raw_speech` is\n                considered a single-channel, single-sample sound. In all other cases, the first dimension of\n                `raw_speech`, whether from an `np.ndarray` or a `List[...]`, corresponds to the number of samples in\n                the batch, and the number of channels (i.e. mono or stereo character) is derived from the other\n                dimensions (1D -> single-channel waveform batches; 2D-> stereo-channel waveform batches).\n            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):\n                Select a strategy to pad the returned sequences (according to the model's padding side and padding\n                index) among:\n\n                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            pad_to_multiple_of (`int`, *optional*, defaults to 2):\n                If set will pad the sequence to a multiple of the provided value.\n\n                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.\n            max_length (`int`, *optional*):\n                Maximum length of the returned list and optionally padding length (see above).\n            truncation (`bool`):\n                Activates truncation to cut input sequences longer than *max_length* to *max_length*.\n            return_attention_mask (`bool`, *optional*):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific feature_extractor's default.\n\n                [What are attention masks?](../glossary#attention-mask)\n\n                <Tip>\n\n                For SeamlessM4T models, `attention_mask` should always be passed for batched inference, to avoid subtle\n                bugs.\n\n                </Tip>\n\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n            sampling_rate (`int`, *optional*):\n                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass\n                `sampling_rate` at the forward call to prevent silent errors.\n            do_normalize_per_mel_bins (`bool`, *optional*, defaults to `True`):\n                Whether or not to zero-mean unit-variance normalize the input per mel-channel.\n            kwargs (*optional*):\n                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature\n                extractor.\n        "
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f'The model corresponding to this feature extractor: {self} was trained using a sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with {self.sampling_rate} and not {sampling_rate}.')
        else:
            logger.warning('It is strongly recommended to pass the `sampling_rate` argument to this function. Failing to do so can result in silent errors that might be hard to debug.')
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 3:
            raise ValueError(f'Only mono-channel or stereo-channel audio is supported for input to {self}')
        is_batched = is_batched_numpy or (isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and (not isinstance(raw_speech, np.ndarray)):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)
        if not is_batched:
            raw_speech = [raw_speech]
        features = [self._extract_fbank_features(waveform) for waveform in raw_speech]
        if do_normalize_per_mel_bins:
            features = [(x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-07) for x in features]
        encoded_inputs = BatchFeature({'input_features': features})
        padded_inputs = self.pad(encoded_inputs, padding=padding, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, return_tensors='np')
        input_features = padded_inputs.get('input_features')
        attention_mask = padded_inputs.get('attention_mask')
        (batch_size, num_frames, num_channels) = input_features.shape
        remainder = num_frames % self.stride
        if remainder != 0:
            input_features = input_features[:, :num_frames, :]
            attention_mask = attention_mask[:, :num_frames]
        input_features = np.reshape(input_features, (batch_size, num_frames // self.stride, num_channels * self.stride))
        indices = np.arange(0, num_frames)
        attention_mask = attention_mask[:, indices % self.stride == 1]
        padded_inputs['input_features'] = input_features
        padded_inputs['attention_mask'] = attention_mask
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
        return padded_inputs