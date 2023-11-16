"""Unit tests for just intonation and friends"""
import os
import sys
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass
import warnings
import numpy as np
import pytest
import librosa

def test_pythagorean():
    if False:
        i = 10
        return i + 15
    ivals = librosa.pythagorean_intervals(bins_per_octave=6, sort=False)
    assert np.allclose(ivals, [1, 3 / 2, 9 / 8, 27 / 16, 81 / 64, 243 / 128])
    ivals2 = librosa.pythagorean_intervals(bins_per_octave=6, sort=True)
    assert np.allclose(sorted(ivals), ivals2)

def test_plimit3():
    if False:
        while True:
            i = 10
    intervals = librosa.plimit_intervals(primes=[3], bins_per_octave=24, sort=False)
    intervals_s = librosa.plimit_intervals(primes=[3], bins_per_octave=24, sort=True)
    assert np.allclose(sorted(intervals), intervals_s)
    assert np.allclose(intervals, [1, 3 / 2, 4 / 3, 9 / 8, 16 / 9, 32 / 27, 27 / 16, 128 / 81, 81 / 64, 256 / 243, 243 / 128, 729 / 512, 2187 / 2048, 6561 / 4096, 1024 / 729, 4096 / 2187, 8192 / 6561, 7625 / 6347, 10037 / 6029, 13654 / 7577, 7577 / 6827, 12457 / 9217, 2006 / 1979, 12497 / 8445])

def test_plimit5():
    if False:
        for i in range(10):
            print('nop')
    intervals = librosa.plimit_intervals(primes=[3, 5], bins_per_octave=24, sort=False)
    intervals_s = librosa.plimit_intervals(primes=[3, 5], bins_per_octave=24, sort=True)
    assert np.allclose(sorted(intervals), intervals_s)
    assert np.allclose(intervals, [1, 3 / 2, 4 / 3, 9 / 8, 5 / 4, 15 / 8, 5 / 3, 45 / 32, 8 / 5, 6 / 5, 16 / 15, 9 / 5, 16 / 9, 27 / 16, 10 / 9, 64 / 45, 32 / 27, 25 / 16, 75 / 64, 25 / 24, 135 / 128, 225 / 128, 40 / 27, 25 / 18])

def test_plimit7():
    if False:
        print('Hello World!')
    intervals = librosa.plimit_intervals(primes=[3, 5, 7], bins_per_octave=24, sort=False)
    intervals_s = librosa.plimit_intervals(primes=[3, 5, 7], bins_per_octave=24, sort=True)
    assert np.allclose(sorted(intervals), intervals_s)
    assert np.allclose(intervals, [1, 3 / 2, 4 / 3, 9 / 8, 5 / 4, 15 / 8, 5 / 3, 45 / 32, 8 / 5, 6 / 5, 16 / 15, 9 / 5, 16 / 9, 27 / 16, 7 / 4, 21 / 16, 8 / 7, 12 / 7, 9 / 7, 32 / 21, 7 / 6, 63 / 32, 35 / 32, 105 / 64])

@pytest.mark.parametrize('n_bins', [6, 12, 24, 30])
@pytest.mark.parametrize('intervals', ['equal', 'pythagorean', 'ji3', 'ji5', 'ji7', [1, 4 / 3, 3 / 2, 5 / 4]])
@pytest.mark.parametrize('bins_per_octave', [6, 12, 15])
def test_interval_frequencies(n_bins, intervals, bins_per_octave):
    if False:
        for i in range(10):
            print('nop')
    freqs = librosa.interval_frequencies(n_bins, fmin=10, intervals=intervals, bins_per_octave=bins_per_octave)
    assert len(freqs) == n_bins
    assert min(freqs) == 10

@pytest.mark.parametrize('intervals', ['pythagorean', 'ji3', 'ji5', 'ji7', [1, 3 / 2, 4 / 3, 5 / 4]])
def test_intervals_sorted(intervals):
    if False:
        for i in range(10):
            print('nop')
    freqs = librosa.interval_frequencies(12, fmin=1, intervals=intervals, sort=False)
    freqs_s = librosa.interval_frequencies(12, fmin=1, intervals=intervals, sort=True)
    assert not np.allclose(freqs, freqs_s)
    assert np.allclose(sorted(freqs), freqs_s)

@pytest.mark.parametrize('sort', [False, True])
def test_pythagorean_factorizations(sort):
    if False:
        i = 10
        return i + 15
    intervals = librosa.pythagorean_intervals(bins_per_octave=20, sort=sort, return_factors=False)
    factors = librosa.pythagorean_intervals(bins_per_octave=20, sort=sort, return_factors=True)
    assert len(intervals) == len(factors)
    for (ival, facts) in zip(intervals, factors):
        value = 0.0
        for prime in facts:
            value += facts[prime] * np.log2(prime)
        assert np.isclose(ival, np.power(2, value))

@pytest.mark.parametrize('sort', [False, True])
@pytest.mark.parametrize('primes', [[3], [3, 5], [3, 5, 7]])
def test_plimit_factorizations(sort, primes):
    if False:
        while True:
            i = 10
    intervals = librosa.plimit_intervals(primes=primes, bins_per_octave=20, sort=sort, return_factors=False)
    factors = librosa.plimit_intervals(primes=primes, bins_per_octave=20, sort=sort, return_factors=True)
    assert len(intervals) == len(factors)
    for (ival, facts) in zip(intervals, factors):
        value = 0.0
        for prime in facts:
            value += facts[prime] * np.log2(prime)
        assert np.isclose(ival, np.power(2, value))