"""Feature extractor class for TVLT."""
from math import ceil
from typing import List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class TvltFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        spectrogram_length (`Dict[str, int]` *optional*, defaults to 2048):
            The time length of each audio spectrogram.
        num_channels (`int` *optional*, defaults to 1):
            Number of audio channels.
        patch_size (`List[int]` *optional*, defaults to `[16, 16]`):
            The patch size of audio patch embedding.
        feature_size (`int`, *optional*, defaults to 128):
            The frequency length of audio spectrogram.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).
        hop_length_to_sampling_rate (`int`, *optional*, defaults to 86):
            Hop length is length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
            For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86
        n_fft (`int`, *optional*, defaults to 2048):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """
    model_input_names = ['audio_values', 'audio_mask']

    def __init__(self, spectrogram_length=2048, num_channels=1, patch_size=[16, 16], feature_size=128, sampling_rate=44100, hop_length_to_sampling_rate=86, n_fft=2048, padding_value=0.0, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.spectrogram_length = spectrogram_length
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.freq_len = feature_size // self.patch_size[1]
        self.n_fft = n_fft
        self.hop_length = sampling_rate // hop_length_to_sampling_rate
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.mel_filters = mel_filter_bank(num_frequency_bins=1 + n_fft // 2, num_mel_filters=feature_size, min_frequency=0.0, max_frequency=22050.0, sampling_rate=sampling_rate, norm='slaney', mel_scale='slaney').T

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        if False:
            return 10
        "\n        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch\n        implementation with 1e-5 tolerance.\n        "
        log_spec = spectrogram(waveform, window_function(self.n_fft, 'hann'), frame_length=self.n_fft, hop_length=self.hop_length, power=2.0, mel_filters=self.mel_filters.T, log_mel='dB', db_range=80.0)
        log_spec = log_spec[:, :-1]
        log_spec = log_spec - 20.0
        log_spec = np.clip(log_spec / 40.0, -2.0, 0.0) + 1.0
        return log_spec

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], return_tensors: Optional[Union[str, TensorType]]=None, return_attention_mask: Optional[bool]=True, sampling_rate: Optional[int]=None, resample: bool=False, mask_audio: bool=False, **kwargs) -> BatchFeature:
        if False:
            print('Hello World!')
        "\n        Main method to prepare one or several audio(s) for the model.\n\n        Args:\n            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):\n                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float\n                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not\n                stereo, i.e. single float per timestep.\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n                - `'pt'`: Return PyTorch `torch.Tensor` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n            return_attention_mask (`bool`, *optional*, default to `True`):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific feature_extractor's default. [What are attention masks?](../glossary#attention-mask)\n\n                <Tip>\n\n                For TvltTransformer models, `attention_mask` should alwys be passed for batched inference, to avoid\n                subtle bugs.\n\n                </Tip>\n\n            sampling_rate (`int`, *optional*):\n                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass\n                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition\n                pipeline. Current model supports sampling rate 16000 and 44100.\n            resample (`bool`, *optional*, defaults to `False`):\n                If the sampling rate is not matched, resample the input audio to match.\n            mask_audio (`bool`, *optional*, defaults to `False`):\n                Whether or not to mask input audio for MAE task.\n\n        Returns:\n            [`BatchFeature`]: A [`BatchFeature`] with the following fields:\n\n            - **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height,\n              width).\n\n            - **audio_mask** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_patches).\n        "
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f'This feature extractor is set to support sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with {self.sampling_rate} and not {sampling_rate}.')
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
        audio_features = [self._np_extract_fbank_features(waveform.squeeze()).T[:self.spectrogram_length] for waveform in raw_speech]
        if isinstance(audio_features[0], List):
            audio_features = [np.asarray(feature, dtype=np.float32) for feature in audio_features]
        max_patch_len = max([ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len for feature in audio_features])
        if return_attention_mask:
            audio_mask = [ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len * [1] + (max_patch_len - ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len) * [0] for feature in audio_features]
            audio_mask = np.array(audio_mask).astype(np.float32)
        max_time_len = max_patch_len // self.freq_len * self.patch_size[0]
        padded_audio_features = np.ones([len(audio_features), 1, max_time_len, self.feature_size]).astype(np.float32)
        padded_audio_features = padded_audio_features * self.padding_value
        for i in range(len(audio_features)):
            feature = audio_features[i]
            padded_audio_features[i, :, :feature.shape[0], :] = feature
        if return_attention_mask:
            data = {'audio_values': padded_audio_features, 'audio_mask': audio_mask}
        else:
            data = {'audio_values': padded_audio_features}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return encoded_inputs