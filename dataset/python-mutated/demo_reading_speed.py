"""

Compare the speed of several methods for reading and loading a sound file.

Optionally, this file can make use of the following packages:

    - audioread     https://github.com/beetbox/audioread
    - scipy         https://scipy.org
    - librosa       https://github.com/bmcfee/librosa
    - pydub         https://github.com/jiaaro/pydub

Uncomment the function names below and send us your speed results!

"""
test_functions = ['read_file_aubio', 'load_file_aubio']
import numpy as np

def read_file_audioread(filename):
    if False:
        for i in range(10):
            print('nop')
    import audioread

    def convert_buffer_to_float(buf, n_bytes=2, dtype=np.float32):
        if False:
            print('Hello World!')
        scale = 1.0 / float(1 << 8 * n_bytes - 1)
        fmt = '<i{:d}'.format(n_bytes)
        out = scale * np.frombuffer(buf, fmt).astype(dtype)
        return out
    with audioread.audio_open(filename) as f:
        total_frames = 0
        for buf in f:
            samples = convert_buffer_to_float(buf)
            samples = samples.reshape(f.channels, -1)
            total_frames += samples.shape[1]
        return (total_frames, f.samplerate)

def load_file_librosa(filename):
    if False:
        for i in range(10):
            print('nop')
    import librosa
    (y, sr) = librosa.load(filename, sr=None)
    return (len(y), sr)

def load_file_scipy(filename):
    if False:
        print('Hello World!')
    import scipy.io.wavfile
    (sr, y) = scipy.io.wavfile.read(filename)
    y = y.astype('float32') / 32767
    return (len(y), sr)

def load_file_scipy_mmap(filename):
    if False:
        print('Hello World!')
    import scipy.io.wavfile
    (sr, y) = scipy.io.wavfile.read(filename, mmap=True)
    return (len(y), sr)

def read_file_pydub(filename):
    if False:
        while True:
            i = 10
    from pydub import AudioSegment
    song = AudioSegment.from_file(filename)
    song.get_array_of_samples()
    return (song.frame_count(), song.frame_rate)

def load_file_pydub(filename):
    if False:
        print('Hello World!')
    from pydub import AudioSegment
    song = AudioSegment.from_file(filename)
    y = np.asarray(song.get_array_of_samples(), dtype='float32')
    y = y.reshape(song.channels, -1) / 32767.0
    return (song.frame_count(), song.frame_rate)

def read_file_aubio(filename):
    if False:
        print('Hello World!')
    import aubio
    f = aubio.source(filename, hop_size=1024)
    total_frames = 0
    while True:
        (_, read) = f()
        total_frames += read
        if read < f.hop_size:
            break
    return (total_frames, f.samplerate)

def load_file_aubio(filename):
    if False:
        return 10
    import aubio
    f = aubio.source(filename, hop_size=1024)
    y = np.zeros(f.duration, dtype=aubio.float_type)
    total_frames = 0
    while True:
        (samples, read) = f()
        y[total_frames:total_frames + read] = samples[:read]
        total_frames += read
        if read < f.hop_size:
            break
    assert len(y) == total_frames
    return (total_frames, f.samplerate)

def test_speed(function, filename):
    if False:
        return 10
    times = []
    for _ in range(10):
        start = time.time()
        try:
            (total_frames, samplerate) = function(filename)
        except ImportError as e:
            print('error: failed importing {:s}'.format(e))
            return
        elapsed = time.time() - start
        times.append(elapsed)
    times = np.array(times)
    duration_min = int(total_frames / float(samplerate) // 60)
    str_format = '{:25s} took {:5f} seconds avg (±{:5f}) to run on a ~ {:d} minutes long file'
    print(str_format.format(function.__name__, times.mean(), times.std(), duration_min))
if __name__ == '__main__':
    import sys, time
    if len(sys.argv) < 2:
        print('not enough arguments')
        sys.exit(1)
    filename = sys.argv[1]
    for f in test_functions:
        test_function = globals()[f]
        test_speed(test_function, filename)