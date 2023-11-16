import numpy as np
from aubio import pitch
import pylab as plt
buf_size = 2048 * 1
hop_size = buf_size // 4
samplerate = 44100
minfreq = 40
maxfreq = 6000

def sinewave(freq, duration, samplerate=samplerate):
    if False:
        while True:
            i = 10
    ' generate a sinewave '
    length = hop_size
    while length < duration * samplerate:
        length += hop_size
    return np.sin(2.0 * np.pi * np.arange(length) * freq / samplerate).astype('float32')

def get_stats_for_pitch_method(method, freqs, samplerate=samplerate):
    if False:
        for i in range(10):
            print('nop')
    ' for a given pitch method and a list of frequency, generate a sinewave\n    and get mean deviation '
    means = np.zeros(len(freqs))
    medians = np.zeros(len(freqs))
    for (freq, fn) in zip(freqs, range(len(freqs))):
        s = sinewave(freq, 0.5).reshape(-1, hop_size)
        p = pitch(method, buf_size, hop_size, samplerate=samplerate)
        candidates = np.zeros(len(s))
        for (frame, i) in zip(s, range(len(s))):
            candidates[i] = p(frame)[0]
        candidates = candidates[4:]
        means[fn] = np.mean(candidates[candidates != 0] - freq)
        medians[fn] = np.median(candidates[candidates != 0] - freq)
        print(freq, means[fn], medians[fn])
    return (means, medians)
if __name__ == '__main__':
    freqs = np.arange(minfreq, maxfreq, 1.0)
    modes = ['yin', 'yinfft']
    for mode in modes:
        (means, medians) = get_stats_for_pitch_method(mode, freqs)
        plt.figure()
        plt.plot(freqs, means, 'g-')
        plt.plot(freqs, medians, 'r--')
        plt.show()