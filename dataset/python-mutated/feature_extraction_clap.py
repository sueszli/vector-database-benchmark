"""Feature extractor class for CLAP."""
import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
logger = logging.get_logger(__name__)

class ClapFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a CLAP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the *Short Time
    Fourier Transform* (STFT) which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 64):
            The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters
            (`n_mels`).
        sampling_rate (`int`, *optional*, defaults to 48000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves
            to warn users if the audio fed to the feature extractor does not have the same sampling rate.
        hop_length (`int`,*optional*, defaults to 480):
            Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split
            in smaller `frames` with a step of `hop_length` between each frame.
        max_length_s (`int`, *optional*, defaults to 10):
            The maximum input length of the model in seconds. This is used to pad the audio.
        fft_window_size (`int`, *optional*, defaults to 1024):
            Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency
            resolution of the spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the attention masks coresponding to the input.
        frequency_min (`float`, *optional*, defaults to 0):
            The lowest frequency of interest. The STFT will not be computed for values below this.
        frequency_max (`float`, *optional*, defaults to 14000):
            The highest frequency of interest. The STFT will not be computed for values above this.
        top_db (`float`, *optional*):
            The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the
            `audio_utils.power_to_db` function
        truncation (`str`, *optional*, defaults to `"fusion"`):
            Truncation pattern for long audio inputs. Two patterns are available:
                - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a
                  downsampled version of the entire mel spectrogram.
            If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy
            of the original mel obtained from the padded audio.
                - `rand_trunc` will select a random crop of the mel spectrogram.
        padding (`str`, *optional*, defaults to `"repeatpad"`):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                - `repeat`: the audio is repeated and then cut to fit the `max_length`
                - `pad`: the audio is padded.
    """
    model_input_names = ['input_features', 'is_longer']

    def __init__(self, feature_size=64, sampling_rate=48000, hop_length=480, max_length_s=10, fft_window_size=1024, padding_value=0.0, return_attention_mask=False, frequency_min: float=0, frequency_max: float=14000, top_db: int=None, truncation: str='fusion', padding: str='repeatpad', **kwargs):
        if False:
            return 10
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, return_attention_mask=return_attention_mask, **kwargs)
        self.top_db = top_db
        self.truncation = truncation
        self.padding = padding
        self.fft_window_size = fft_window_size
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        self.hop_length = hop_length
        self.max_length_s = max_length_s
        self.nb_max_samples = max_length_s * sampling_rate
        self.sampling_rate = sampling_rate
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.mel_filters = mel_filter_bank(num_frequency_bins=self.nb_frequency_bins, num_mel_filters=feature_size, min_frequency=frequency_min, max_frequency=frequency_max, sampling_rate=sampling_rate, norm=None, mel_scale='htk')
        self.mel_filters_slaney = mel_filter_bank(num_frequency_bins=self.nb_frequency_bins, num_mel_filters=feature_size, min_frequency=frequency_min, max_frequency=frequency_max, sampling_rate=sampling_rate, norm='slaney', mel_scale='slaney')

    def to_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Serializes this instance to a Python dictionary.\n\n        Returns:\n            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the\n            mel filter banks, which do not need to be saved or printed as they are too long.\n        '
        output = copy.deepcopy(self.__dict__)
        output['feature_extractor_type'] = self.__class__.__name__
        if 'mel_filters' in output:
            del output['mel_filters']
        if 'mel_filters_slaney' in output:
            del output['mel_filters_slaney']
        return output

    def _np_extract_fbank_features(self, waveform: np.array, mel_filters: Optional[np.array]=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the log-mel spectrogram of the provided `waveform` using the Hann window. In CLAP, two different filter\n        banks are used depending on the truncation pattern:\n            - `self.mel_filters`: they correspond to the default parameters of `torchaudio` which can be obtained from\n              calling `torchaudio.transforms.MelSpectrogram().mel_scale.fb`. These filters are used when `truncation`\n              is set to `"fusion"`.\n            - `self.mel_filteres_slaney` : they correspond to the default parameters of `librosa` which used\n              `librosa.filters.mel` when computing the mel spectrogram. These filters were only used in the original\n              implementation when the truncation mode is not `"fusion"`.\n        '
        log_mel_spectrogram = spectrogram(waveform, window_function(self.fft_window_size, 'hann'), frame_length=self.fft_window_size, hop_length=self.hop_length, power=2.0, mel_filters=mel_filters, log_mel='dB')
        return log_mel_spectrogram.T

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        if False:
            print('Hello World!')
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            ranges[2] = [0]
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])
        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
        mel = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(mel, size=[chunk_frames, 64], mode='bilinear', align_corners=False)
        mel_shrink = mel_shrink[0][0].numpy()
        mel_fusion = np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)
        return mel_fusion

    def _get_input_mel(self, waveform: np.array, max_length, truncation, padding) -> np.array:
        if False:
            i = 10
            return i + 15
        '\n        Extracts the mel spectrogram and prepares it for the mode based on the `truncation` and `padding` arguments.\n        Four different path are possible:\n            - `truncation="fusion"` and the length of the waveform is greater than the max length: the mel spectrogram\n              will be computed on the entire audio. 3 random crops and a dowsampled version of the full mel spectrogram\n              are then stacked together. They will later be used for `feature_fusion`.\n            - `truncation="rand_trunc"` and the length of the waveform is smaller than the max length: the audio is\n              padded based on `padding`.\n            - `truncation="fusion"` and the length of the waveform is smaller than the max length: the audio is padded\n              based on `padding`, and is repeated `4` times.\n            - `truncation="rand_trunc"` and the length of the waveform is greater than the max length: the mel\n              spectrogram will be computed on a random crop of the waveform.\n\n        '
        if waveform.shape[0] > max_length:
            if truncation == 'rand_trunc':
                longer = True
                overflow = len(waveform) - max_length
                idx = np.random.randint(0, overflow + 1)
                waveform = waveform[idx:idx + max_length]
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters_slaney)[None, :]
            elif truncation == 'fusion':
                mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                chunk_frames = max_length // self.hop_length + 1
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    input_mel = np.stack([mel, mel, mel, mel], axis=0)
                    longer = False
                else:
                    input_mel = self._random_mel_fusion(mel, total_frames, chunk_frames)
                    longer = True
            else:
                raise NotImplementedError(f'data_truncating {truncation} not implemented')
        else:
            longer = False
            if waveform.shape[0] < max_length:
                if padding == 'repeat':
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.tile(waveform, n_repeat + 1)[:max_length]
                if padding == 'repeatpad':
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.tile(waveform, n_repeat)
                waveform = np.pad(waveform, (0, max_length - waveform.shape[0]), mode='constant', constant_values=0)
            if truncation == 'fusion':
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                input_mel = np.stack([input_mel, input_mel, input_mel, input_mel], axis=0)
            else:
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters_slaney)[None, :]
        return (input_mel, longer)

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], truncation: str=None, padding: Optional[str]=None, max_length: Optional[int]=None, sampling_rate: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchFeature:
        if False:
            return 10
        "\n        Main method to featurize and prepare for the model one or several sequence(s).\n\n        Args:\n            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):\n                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float\n                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not\n                stereo, i.e. single float per timestep.\n            truncation (`str`, *optional*):\n                Truncation pattern for long audio inputs. Two patterns are available:\n                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and\n                      a downsampled version of the entire mel spectrogram.\n                If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a\n                copy of the original mel obtained from the padded audio.\n                    - `rand_trunc` will select a random crop of the mel spectrogram.\n            padding (`str`, *optional*):\n               Padding pattern for shorter audio inputs. Three patterns were originally implemented:\n                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.\n                    - `repeat`: the audio is repeated and then cut to fit the `max_length`\n                    - `pad`: the audio is padded.\n            return_tensors (`str` or [`~utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `'tf'`: Return TensorFlow `tf.constant` objects.\n                - `'pt'`: Return PyTorch `torch.np.array` objects.\n                - `'np'`: Return Numpy `np.ndarray` objects.\n            sampling_rate (`int`, *optional*):\n                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass\n                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition\n                pipeline.\n        "
        truncation = truncation if truncation is not None else self.truncation
        padding = padding if padding else self.padding
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
            raw_speech = [np.asarray(speech, dtype=np.float64) for speech in raw_speech]
        elif not is_batched and (not isinstance(raw_speech, np.ndarray)):
            raw_speech = np.asarray(raw_speech, dtype=np.float64)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float64)
        if not is_batched:
            raw_speech = [np.asarray(raw_speech)]
        padded_inputs = [self._get_input_mel(waveform, max_length if max_length else self.nb_max_samples, truncation, padding) for waveform in raw_speech]
        input_mel = []
        is_longer = []
        for (mel, longer) in padded_inputs:
            input_mel.append(mel)
            is_longer.append(longer)
        if truncation == 'fusion' and sum(is_longer) == 0:
            rand_idx = np.random.randint(0, len(input_mel))
            is_longer[rand_idx] = True
        if isinstance(input_mel[0], List):
            input_mel = [np.asarray(feature, dtype=np.float64) for feature in input_mel]
        is_longer = [[longer] for longer in is_longer]
        input_features = {'input_features': input_mel, 'is_longer': is_longer}
        input_features = BatchFeature(input_features)
        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)
        return input_features