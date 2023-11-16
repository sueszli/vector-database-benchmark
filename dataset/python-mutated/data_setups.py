import os
import librosa
import torch
import numpy as np
from torchaudio.transforms import Resample
SAMPLE_RATE = 44100
AUDIO_LEN = 2.9
N_MELS = 128
F_MIN = 20
F_MAX = 16000
N_FFT = 1024
HOP_LEN = 512

def find_classes(directory: str):
    if False:
        while True:
            i = 10
    classes = sorted((entry.name for entry in os.scandir(directory) if entry.is_dir()))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name: i for (i, cls_name) in enumerate(classes)}
    return (classes, class_to_idx)

def resample(wav, sample_rate, new_sample_rate):
    if False:
        print('Hello World!')
    if wav.shape[0] >= 2:
        wav = torch.mean(wav, dim=0)
    else:
        wav = wav.squeeze(0)
    if sample_rate > new_sample_rate:
        resampler = Resample(sample_rate, new_sample_rate)
        wav = resampler(wav)
    return wav

def mono_to_color(X, eps=1e-06, mean=None, std=None):
    if False:
        while True:
            i = 10
    X = np.stack([X, X, X], axis=-1)
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    (_min, _max) = (X.min(), X.max())
    if _max - _min > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V

def normalize(image, mean=None, std=None):
    if False:
        return 10
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)

def compute_melspec(wav, sample_rate=SAMPLE_RATE):
    if False:
        i = 10
        return i + 15
    melspec = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_fft=N_FFT, fmin=F_MIN, fmax=F_MAX, n_mels=N_MELS, hop_length=HOP_LEN)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec

def audio_preprocess(wav, sample_rate):
    if False:
        print('Hello World!')
    wav = wav.numpy()
    melspec = compute_melspec(wav, sample_rate)
    image = mono_to_color(melspec)
    image = normalize(image, mean=None, std=None)
    image = torch.from_numpy(image)
    return image