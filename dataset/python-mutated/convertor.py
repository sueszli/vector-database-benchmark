""" This module provides audio data convertion functions. """
import numpy as np
import tensorflow as tf
from ..utils.tensor import from_float32_to_uint8, from_uint8_to_float32
__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'

def to_n_channels(waveform: tf.Tensor, n_channels: int) -> tf.Tensor:
    if False:
        return 10
    '\n    Convert a waveform to n_channels by removing or duplicating channels if\n    needed (in tensorflow).\n\n    Parameters:\n        waveform (tf.Tensor):\n            Waveform to transform.\n        n_channels (int):\n            Number of channel to reshape waveform in.\n\n    Returns:\n        tf.Tensor:\n            Reshaped waveform.\n    '
    return tf.cond(tf.shape(waveform)[1] >= n_channels, true_fn=lambda : waveform[:, :n_channels], false_fn=lambda : tf.tile(waveform, [1, n_channels])[:, :n_channels])

def to_stereo(waveform: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Convert a waveform to stereo by duplicating if mono, or truncating\n    if too many channels.\n\n    Parameters:\n        waveform (np.ndarray):\n            a `(N, d)` numpy array.\n\n    Returns:\n        np.ndarray:\n            A stereo waveform as a `(N, 1)` numpy array.\n    '
    if waveform.shape[1] == 1:
        return np.repeat(waveform, 2, axis=-1)
    if waveform.shape[1] > 2:
        return waveform[:, :2]
    return waveform

def gain_to_db(tensor: tf.Tensor, espilon: float=1e-09) -> tf.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Convert from gain to decibel in tensorflow.\n\n    Parameters:\n        tensor (tf.Tensor):\n            Tensor to convert\n        epsilon (float):\n            Operation constant.\n\n    Returns:\n        tf.Tensor:\n            Converted tensor.\n    '
    return 20.0 / np.log(10) * tf.math.log(tf.maximum(tensor, espilon))

def db_to_gain(tensor: tf.Tensor) -> tf.Tensor:
    if False:
        return 10
    '\n    Convert from decibel to gain in tensorflow.\n\n    Parameters:\n        tensor (tf.Tensor):\n            Tensor to convert\n\n    Returns:\n        tf.Tensor:\n            Converted tensor.\n    '
    return tf.pow(10.0, tensor / 20.0)

def spectrogram_to_db_uint(spectrogram: tf.Tensor, db_range: float=100.0, **kwargs) -> tf.Tensor:
    if False:
        while True:
            i = 10
    '\n    Encodes given spectrogram into uint8 using decibel scale.\n\n    Parameters:\n        spectrogram (tf.Tensor):\n            Spectrogram to be encoded as TF float tensor.\n        db_range (float):\n            Range in decibel for encoding.\n\n    Returns:\n        tf.Tensor:\n            Encoded decibel spectrogram as `uint8` tensor.\n    '
    db_spectrogram: tf.Tensor = gain_to_db(spectrogram)
    max_db_spectrogram: tf.Tensor = tf.reduce_max(db_spectrogram)
    int_db_spectrogram: tf.Tensor = tf.maximum(db_spectrogram, max_db_spectrogram - db_range)
    return from_float32_to_uint8(int_db_spectrogram, **kwargs)

def db_uint_spectrogram_to_gain(db_uint_spectrogram: tf.Tensor, min_db: tf.Tensor, max_db: tf.Tensor) -> tf.Tensor:
    if False:
        print('Hello World!')
    '\n    Decode spectrogram from uint8 decibel scale.\n\n    Paramters:\n        db_uint_spectrogram (tf.Tensor):\n            Decibel spectrogram to decode.\n        min_db (tf.Tensor):\n            Lower bound limit for decoding.\n        max_db (tf.Tensor):\n            Upper bound limit for decoding.\n\n    Returns:\n        tf.Tensor:\n            Decoded spectrogram as `float32` tensor.\n    '
    db_spectrogram: tf.Tensor = from_uint8_to_float32(db_uint_spectrogram, min_db, max_db)
    return db_to_gain(db_spectrogram)