"""
@author: nl8590687
ASRT语音识别内置声学特征提取模块，定义了几个常用的声学特征类
"""
import random
import numpy as np
from scipy.fftpack import fft
from .base import mfcc, delta, logfbank

class SpeechFeatureMeta:
    """
    ASRT语音识别中所有声学特征提取类的基类
    """

    def __init__(self, framesamplerate=16000):
        if False:
            print('Hello World!')
        self.framesamplerate = framesamplerate

    def run(self, wavsignal, fs=16000):
        if False:
            for i in range(10):
                print('nop')
        '\n        run method\n        '
        raise NotImplementedError('[ASRT] `run()` method is not implemented.')

class MFCC(SpeechFeatureMeta):
    """
    ASRT语音识别内置的mfcc声学特征提取类

    Compute MFCC features from an audio signal.

    :param framesamplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    """

    def __init__(self, framesamplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97):
        if False:
            while True:
                i = 10
        self.framesamplerate = framesamplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.preemph = preemph
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs=16000):
        if False:
            i = 10
            return i + 15
        '\n        计算mfcc声学特征，包含静态特征、一阶差分和二阶差分\n\n        :returns: A numpy array of size (NUMFRAMES by numcep * 3) containing features. Each row holds 1 feature vector.\n        '
        wavsignal = np.array(wavsignal, dtype=np.float64)
        feat_mfcc = mfcc(wavsignal[0], samplerate=self.framesamplerate, winlen=self.winlen, winstep=self.winstep, numcep=self.numcep, nfilt=self.nfilt, preemph=self.preemph)
        feat_mfcc_d = delta(feat_mfcc, 2)
        feat_mfcc_dd = delta(feat_mfcc_d, 2)
        wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
        return wav_feature

class Logfbank(SpeechFeatureMeta):
    """
    ASRT语音识别内置的logfbank声学特征提取类
    """

    def __init__(self, framesamplerate=16000, nfilt=26):
        if False:
            print('Hello World!')
        self.nfilt = nfilt
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs=16000):
        if False:
            print('Hello World!')
        wavsignal = np.array(wavsignal, dtype=np.float64)
        wav_feature = logfbank(wavsignal, fs, nfilt=self.nfilt)
        return wav_feature

class Spectrogram(SpeechFeatureMeta):
    """
    ASRT语音识别内置的语谱图声学特征提取类
    """

    def __init__(self, framesamplerate=16000, timewindow=25, timeshift=10):
        if False:
            i = 10
            return i + 15
        self.time_window = timewindow
        self.window_length = int(framesamplerate / 1000 * self.time_window)
        self.timeshift = timeshift
        '\n        # 保留将来用于不同采样频率\n        self.x=np.linspace(0, self.window_length - 1, self.window_length, dtype = np.int64)\n        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (self.window_length - 1) ) # 汉明窗\n        '
        self.x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * self.x / (400 - 1))
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs=16000):
        if False:
            i = 10
            return i + 15
        if fs != 16000:
            raise ValueError(f'[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is {fs} Hz.')
        time_window = 25
        window_length = int(fs / 1000 * time_window)
        wav_arr = np.array(wavsignal)
        range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10 + 1
        data_input = np.zeros((range0_end, window_length // 2), dtype=np.float64)
        data_line = np.zeros((1, window_length), dtype=np.float64)
        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            data_line = wav_arr[0, p_start:p_end]
            data_line = data_line * self.w
            data_line = np.abs(fft(data_line))
            data_input[i] = data_line[0:window_length // 2]
        data_input = np.log(data_input + 1)
        return data_input

class SpecAugment(SpeechFeatureMeta):
    """
    复现谷歌SpecAugment数据增强特征算法，基于Spectrogram语谱图基础特征
    """

    def __init__(self, framesamplerate=16000, timewindow=25, timeshift=10):
        if False:
            print('Hello World!')
        self.time_window = timewindow
        self.window_length = int(framesamplerate / 1000 * self.time_window)
        self.timeshift = timeshift
        '\n        # 保留将来用于不同采样频率\n        self.x=np.linspace(0, self.window_length - 1, self.window_length, dtype = np.int64)\n        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (self.window_length - 1) ) # 汉明窗\n        '
        self.x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * self.x / (400 - 1))
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs=16000):
        if False:
            for i in range(10):
                print('nop')
        if fs != 16000:
            raise ValueError(f'[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is {fs} Hz.')
        time_window = 25
        window_length = int(fs / 1000 * time_window)
        wav_arr = np.array(wavsignal)
        range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10 + 1
        data_input = np.zeros((range0_end, window_length // 2), dtype=np.float64)
        data_line = np.zeros((1, window_length), dtype=np.float64)
        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            data_line = wav_arr[0, p_start:p_end]
            data_line = data_line * self.w
            data_line = np.abs(fft(data_line))
            data_input[i] = data_line[0:window_length // 2]
        data_input = np.log(data_input + 1)
        mode = random.randint(1, 100)
        h_start = random.randint(1, data_input.shape[0])
        h_width = random.randint(1, 100)
        v_start = random.randint(1, data_input.shape[1])
        v_width = random.randint(1, 100)
        if mode <= 60:
            pass
        elif 60 < mode <= 75:
            data_input[h_start:h_start + h_width, :] = 0
        elif 75 < mode <= 90:
            data_input[:, v_start:v_start + v_width] = 0
        else:
            data_input[h_start:h_start + h_width, :v_start:v_start + v_width] = 0
        return data_input