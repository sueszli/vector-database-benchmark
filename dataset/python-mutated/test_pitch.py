from numpy.testing import TestCase, assert_equal
from numpy import sin, arange, mean, median, isnan, pi
from aubio import fvec, pitch, freqtomidi, float_type

class aubio_pitch_Good_Values(TestCase):

    def skip_test_new_default(self):
        if False:
            print('Hello World!')
        ' creating a pitch object without parameters '
        p = pitch()
        assert_equal([p.method, p.buf_size, p.hop_size, p.samplerate], ['default', 1024, 512, 44100])

    def test_run_on_silence(self):
        if False:
            while True:
                i = 10
        ' creating a pitch object with parameters '
        p = pitch('default', 2048, 512, 32000)
        assert_equal([p.method, p.buf_size, p.hop_size, p.samplerate], ['default', 2048, 512, 32000])

    def test_run_on_zeros(self):
        if False:
            print('Hello World!')
        ' running on silence gives 0 '
        p = pitch('default', 2048, 512, 32000)
        f = fvec(512)
        for _ in range(10):
            assert_equal(p(f), 0.0)

    def test_run_on_ones(self):
        if False:
            for i in range(10):
                print('nop')
        ' running on ones gives 0 '
        p = pitch('default', 2048, 512, 32000)
        f = fvec(512)
        f[:] = 1
        for _ in range(10):
            assert_equal(p(f), 0.0)

class aubio_pitch_Sinusoid(TestCase):

    def run_pitch_on_sinusoid(self, method, buf_size, hop_size, samplerate, freq):
        if False:
            print('Hello World!')
        p = pitch(method, buf_size, hop_size, samplerate)
        seconds = 0.3
        duration = seconds * samplerate
        duration = duration - duration % hop_size + hop_size
        sinvec = self.build_sinusoid(duration, freq, samplerate)
        self.run_pitch(p, sinvec, freq)

    def build_sinusoid(self, length, freq, samplerate):
        if False:
            return 10
        return sin(2.0 * pi * arange(length).astype(float_type) * freq / samplerate)

    def run_pitch(self, p, input_vec, freq):
        if False:
            print('Hello World!')
        (pitches, errors) = ([], [])
        input_blocks = input_vec.reshape((-1, p.hop_size))
        for new_block in input_blocks:
            pitch = p(new_block)[0]
            pitches.append(pitch)
            errors.append(1.0 - freqtomidi(pitch) / freqtomidi(freq))
        assert_equal(len(input_blocks), len(pitches))
        assert_equal(isnan(pitches), False)
        pitches = pitches[2:]
        errors = errors[2:]
        assert abs(median(errors)) < 0.06, 'median of relative errors is bigger than 0.06 (%f)\n found %s\n errors %s' % (mean(errors), pitches, errors)
pitch_algorithms = ['default', 'yinfft', 'yin', 'yinfast', 'schmitt', 'mcomb', 'fcomb', 'specacf']
pitch_algorithms = ['default', 'yinfft', 'yin', 'yinfast', 'schmitt', 'mcomb', 'fcomb']
freqs = [110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0]
signal_modes = []
for freq in freqs:
    signal_modes += [(4096, 2048, 44100, freq), (2048, 512, 44100, freq), (2048, 1024, 44100, freq), (2048, 1024, 32000, freq)]
freqs = []
for freq in freqs:
    signal_modes += [(2048, 1024, 22050, freq), (1024, 256, 16000, freq), (1024, 256, 8000, freq), (1024, 512 + 256, 8000, freq)]
'\nsignal_modes = [\n        ( 4096,  512, 44100, 2.*882. ),\n        ( 4096,  512, 44100, 882. ),\n        ( 4096,  512, 44100, 440. ),\n        ( 2048,  512, 44100, 440. ),\n        ( 2048, 1024, 44100, 440. ),\n        ( 2048, 1024, 44100, 440. ),\n        ( 2048, 1024, 32000, 440. ),\n        ( 2048, 1024, 22050, 440. ),\n        ( 1024,  256, 16000, 440. ),\n        ( 1024,  256, 8000,  440. ),\n        ( 1024, 512+256, 8000, 440. ),\n        ]\n'

def create_test(algo, mode):
    if False:
        return 10

    def do_test_pitch(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_pitch_on_sinusoid(algo, mode[0], mode[1], mode[2], mode[3])
    return do_test_pitch
for algo in pitch_algorithms:
    for mode in signal_modes:
        _test_method = create_test(algo, mode)
        _test_method.__name__ = 'test_pitch_%s_%d_%d_%dHz_sin_%.0f' % (algo, mode[0], mode[1], mode[2], mode[3])
        setattr(aubio_pitch_Sinusoid, _test_method.__name__, _test_method)
if __name__ == '__main__':
    from unittest import main
    main()