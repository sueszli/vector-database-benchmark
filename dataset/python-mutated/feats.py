""" 
This code is modified from https://github.com/wenet-e2e/wetts. 
"""
import librosa
import numpy as np
import pyworld
from scipy.interpolate import interp1d
from librosa.filters import mel as librosa_mel_fn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util

class LogMelFBank:

    def __init__(self, sr=24000, n_fft=2048, hop_length=300, win_length=None, window='hann', n_mels=80, fmin=80, fmax=7600):
        if False:
            return 10
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = True
        self.pad_mode = 'reflect'
        self.n_mels = n_mels
        self.fmin = 0 if fmin is None else fmin
        self.fmax = sr / 2 if fmax is None else fmax
        self.mel_filter = self._create_mel_filter()

    def _create_mel_filter(self):
        if False:
            return 10
        mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return mel_filter

    def _stft(self, wav):
        if False:
            print('Hello World!')
        D = librosa.core.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, pad_mode=self.pad_mode)
        return D

    def _spectrogram(self, wav):
        if False:
            while True:
                i = 10
        D = self._stft(wav)
        return np.abs(D)

    def _mel_spectrogram(self, wav):
        if False:
            print('Hello World!')
        S = self._spectrogram(wav)
        mel = np.dot(self.mel_filter, S)
        return mel

    def get_log_mel_fbank(self, wav):
        if False:
            while True:
                i = 10
        mel = self._mel_spectrogram(wav)
        mel = np.clip(mel, a_min=1e-10, a_max=float('inf'))
        mel = np.log(mel.T)
        return mel

class Pitch:

    def __init__(self, sr=24000, hop_length=300, pitch_min=80, pitch_max=7600):
        if False:
            while True:
                i = 10
        self.sr = sr
        self.hop_length = hop_length
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

    def _convert_to_continuous_pitch(self, pitch: np.array) -> np.array:
        if False:
            while True:
                i = 10
        if (pitch == 0).all():
            return pitch
        start_pitch = pitch[pitch != 0][0]
        end_pitch = pitch[pitch != 0][-1]
        start_idx = np.where(pitch == start_pitch)[0][0]
        end_idx = np.where(pitch == end_pitch)[0][-1]
        pitch[:start_idx] = start_pitch
        pitch[end_idx:] = end_pitch
        nonzero_idxs = np.where(pitch != 0)[0]
        interp_fn = interp1d(nonzero_idxs, pitch[nonzero_idxs])
        pitch = interp_fn(np.arange(0, pitch.shape[0]))
        return pitch

    def _calculate_pitch(self, input: np.array, use_continuous_pitch=True, use_log_pitch=False) -> np.array:
        if False:
            return 10
        input = input.astype(float)
        frame_period = 1000 * self.hop_length / self.sr
        (pitch, timeaxis) = pyworld.dio(input, fs=self.sr, frame_period=frame_period)
        pitch = pyworld.stonemask(input, pitch, timeaxis, self.sr)
        if use_continuous_pitch:
            pitch = self._convert_to_continuous_pitch(pitch)
        if use_log_pitch:
            nonzero_idxs = np.where(pitch != 0)[0]
            pitch[nonzero_idxs] = np.log(pitch[nonzero_idxs])
        return pitch.reshape(-1)

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        if False:
            for i in range(10):
                print('nop')
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for (start, end) in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            mask = arr == 0
            arr[mask] = 0
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        arr_list = np.array(arr_list)
        return arr_list

    def get_pitch(self, wav, use_continuous_pitch=True, use_log_pitch=False, use_token_averaged_pitch=False, duration=None):
        if False:
            i = 10
            return i + 15
        pitch = self._calculate_pitch(wav, use_continuous_pitch, use_log_pitch)
        if use_token_averaged_pitch and duration is not None:
            pitch = self._average_by_duration(pitch, duration)
        return pitch

class Energy:

    def __init__(self, sr=24000, n_fft=2048, hop_length=300, win_length=None, window='hann', center=True, pad_mode='reflect'):
        if False:
            print('Hello World!')
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def _stft(self, wav):
        if False:
            i = 10
            return i + 15
        D = librosa.core.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, pad_mode=self.pad_mode)
        return D

    def _calculate_energy(self, input):
        if False:
            for i in range(10):
                print('nop')
        input = input.astype(np.float32)
        input_stft = self._stft(input)
        input_power = np.abs(input_stft) ** 2
        energy = np.sqrt(np.clip(np.sum(input_power, axis=0), a_min=1e-10, a_max=float('inf')))
        return energy

    def _average_by_duration(self, input: np.array, d: np.array) -> np.array:
        if False:
            print('Hello World!')
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for (start, end) in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        arr_list = np.array(arr_list)
        return arr_list

    def get_energy(self, wav, use_token_averaged_energy=True, duration=None):
        if False:
            for i in range(10):
                print('nop')
        energy = self._calculate_energy(wav)
        if use_token_averaged_energy and duration is not None:
            energy = self._average_by_duration(energy, duration)
        return energy

def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    if False:
        return 10
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def griffin_lim(magnitudes, stft_fn, n_iters=30):
    if False:
        for i in range(10):
            print('nop')
    '\n    PARAMS\n    ------\n    magnitudes: spectrogram magnitudes\n    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods\n    '
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    for i in range(n_iters):
        (_, angles) = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal

def dynamic_range_compression(x, C=1, clip_val=1e-05):
    if False:
        print('Hello World!')
    '\n    PARAMS\n    ------\n    C: compression factor\n    '
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    if False:
        while True:
            i = 10
    '\n    PARAMS\n    ------\n    C: compression factor used to compress\n    '
    return torch.exp(x) / C

class STFT(torch.nn.Module):

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        if False:
            return 10
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(data=fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        if False:
            while True:
                i = 10
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return (magnitude, phase)

    def inverse(self, magnitude, phase):
        if False:
            while True:
                i = 10
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        if False:
            print('Hello World!')
        (self.magnitude, self.phase) = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
        if False:
            i = 10
            return i + 15
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        if False:
            return 10
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        if False:
            for i in range(10):
                print('nop')
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        if False:
            for i in range(10):
                print('nop')
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        (magnitudes, phases) = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output