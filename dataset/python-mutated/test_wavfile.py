import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_equal, assert_, assert_array_equal, break_cycles, suppress_warnings, IS_PYPY
import pytest
from pytest import raises, warns
from scipy.io import wavfile

def datafile(fn):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(os.path.dirname(__file__), 'data', fn)

def test_read_1():
    if False:
        i = 10
        return i + 15
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.int32))
        assert_equal(data.shape, (4410,))
        del data

def test_read_2():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-2ch-1byteu.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.uint8))
        assert_equal(data.shape, (800, 2))
        del data

def test_read_3():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-le.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.float32))
        assert_equal(data.shape, (441, 2))
        del data

def test_read_4():
    if False:
        i = 10
        return i + 15
    for mmap in [False, True]:
        with suppress_warnings() as sup:
            sup.filter(wavfile.WavFileWarning, 'Chunk .non-data. not understood, skipping it')
            filename = 'test-48000Hz-2ch-64bit-float-le-wavex.wav'
            (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 48000)
        assert_(np.issubdtype(data.dtype, np.float64))
        assert_equal(data.shape, (480, 2))
        del data

def test_read_5():
    if False:
        print('Hello World!')
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-be.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.float32))
        assert_(data.dtype.byteorder == '>' or (sys.byteorder == 'big' and data.dtype.byteorder == '='))
        assert_equal(data.shape, (441, 2))
        del data

def test_5_bit_odd_size_no_pad():
    if False:
        print('Hello World!')
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-5ch-9S-5bit.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.uint8))
        assert_equal(data.shape, (9, 5))
        assert_equal(data & 7, 0)
        assert_equal(data.max(), 248)
        assert_equal(data[0, 0], 128)
        assert_equal(data.min(), 0)
        del data

def test_12_bit_even_size():
    if False:
        print('Hello World!')
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-4ch-9S-12bit.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.int16))
        assert_equal(data.shape, (9, 4))
        assert_equal(data & 15, 0)
        assert_equal(data.max(), 32752)
        assert_equal(data[0, 0], 0)
        assert_equal(data.min(), -32768)
        del data

def test_24_bit_odd_size_with_pad():
    if False:
        return 10
    filename = 'test-8000Hz-le-3ch-5S-24bit.wav'
    (rate, data) = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int32))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 255, 0)
    assert_equal(data, [[-2147483648, -2147483392, -512], [-1073741824, -1073741568, -256], [+0, +0, +0], [+1073741824, +1073741568, +256], [+2147483392, +2147483392, +512]])

def test_20_bit_extra_data():
    if False:
        return 10
    filename = 'test-8000Hz-le-1ch-10S-20bit-extra.wav'
    (rate, data) = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 1234)
    assert_(np.issubdtype(data.dtype, np.int32))
    assert_equal(data.shape, (10,))
    assert_equal(data & 255, 0)
    assert_((data & 3840).any())
    assert_equal(data, [+2147479552, -2147479552, +2147479552 >> 1, -2147479552 >> 1, +2147479552 >> 2, -2147479552 >> 2, +2147479552 >> 3, -2147479552 >> 3, +2147479552 >> 4, -2147479552 >> 4])

def test_36_bit_odd_size():
    if False:
        while True:
            i = 10
    filename = 'test-8000Hz-le-3ch-5S-36bit.wav'
    (rate, data) = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int64))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 268435455, 0)
    correct = [[-9223372036854775808, -9223372036586340352, -536870912], [-4611686018427387904, -4611686018158952448, -268435456], [+0, +0, +0], [+4611686018427387904, +4611686018158952448, +268435456], [+9223372036586340352, +9223372036586340352, +536870912]]
    assert_equal(data, correct)

def test_45_bit_even_size():
    if False:
        while True:
            i = 10
    filename = 'test-8000Hz-le-3ch-5S-45bit.wav'
    (rate, data) = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int64))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 524287, 0)
    correct = [[-9223372036854775808, -9223372036854251520, -1048576], [-4611686018427387904, -4611686018426863616, -524288], [+0, +0, +0], [+4611686018427387904, +4611686018426863616, +524288], [+9223372036854251520, +9223372036854251520, +1048576]]
    assert_equal(data, correct)

def test_53_bit_odd_size():
    if False:
        return 10
    filename = 'test-8000Hz-le-3ch-5S-53bit.wav'
    (rate, data) = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int64))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 2047, 0)
    correct = [[-9223372036854775808, -9223372036854773760, -4096], [-4611686018427387904, -4611686018427385856, -2048], [+0, +0, +0], [+4611686018427387904, +4611686018427385856, +2048], [+9223372036854773760, +9223372036854773760, +4096]]
    assert_equal(data, correct)

def test_64_bit_even_size():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-3ch-5S-64bit.wav'
        (rate, data) = wavfile.read(datafile(filename), mmap=False)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.int64))
        assert_equal(data.shape, (5, 3))
        correct = [[-9223372036854775808, -9223372036854775807, -2], [-4611686018427387904, -4611686018427387903, -1], [+0, +0, +0], [+4611686018427387904, +4611686018427387903, +1], [+9223372036854775807, +9223372036854775807, +2]]
        assert_equal(data, correct)
        del data

def test_unsupported_mmap():
    if False:
        return 10
    for filename in {'test-8000Hz-le-3ch-5S-24bit.wav', 'test-8000Hz-le-3ch-5S-36bit.wav', 'test-8000Hz-le-3ch-5S-45bit.wav', 'test-8000Hz-le-3ch-5S-53bit.wav', 'test-8000Hz-le-1ch-10S-20bit-extra.wav'}:
        with raises(ValueError, match='mmap.*not compatible'):
            (rate, data) = wavfile.read(datafile(filename), mmap=True)

def test_rifx():
    if False:
        i = 10
        return i + 15
    for (rifx, riff) in {('test-44100Hz-be-1ch-4bytes.wav', 'test-44100Hz-le-1ch-4bytes.wav'), ('test-8000Hz-be-3ch-5S-24bit.wav', 'test-8000Hz-le-3ch-5S-24bit.wav')}:
        (rate1, data1) = wavfile.read(datafile(rifx), mmap=False)
        (rate2, data2) = wavfile.read(datafile(riff), mmap=False)
        assert_equal(rate1, rate2)
        assert_equal(data1, data2)

def test_read_unknown_filetype_fail():
    if False:
        for i in range(10):
            print('nop')
    for mmap in [False, True]:
        filename = 'example_1.nc'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="CDF.*'RIFF' and 'RIFX' supported"):
                wavfile.read(fp, mmap=mmap)

def test_read_unknown_riff_form_type():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'Transparent Busy.ani'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Not a WAV file.*ACON'):
                wavfile.read(fp, mmap=mmap)

def test_read_unknown_wave_format():
    if False:
        for i in range(10):
            print('nop')
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-1ch-1byte-ulaw.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unknown wave file format.*MULAW.*Supported formats'):
                wavfile.read(fp, mmap=mmap)

def test_read_early_eof_with_data():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof.wav'
        with open(datafile(filename), 'rb') as fp:
            with warns(wavfile.WavFileWarning, match='Reached EOF'):
                (rate, data) = wavfile.read(fp, mmap=mmap)
                assert data.size > 0
                assert rate == 44100
                data[0] = 0

def test_read_early_eof():
    if False:
        return 10
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unexpected end of file.'):
                wavfile.read(fp, mmap=mmap)

def test_read_incomplete_chunk():
    if False:
        while True:
            i = 10
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="Incomplete chunk ID.*b'f'"):
                wavfile.read(fp, mmap=mmap)

def test_read_inconsistent_header():
    if False:
        i = 10
        return i + 15
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-3ch-5S-24bit-inconsistent.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='header is invalid'):
                wavfile.read(fp, mmap=mmap)

@pytest.mark.parametrize('dt_str', ['<i2', '<i4', '<i8', '<f4', '<f8', '>i2', '>i4', '>i8', '>f4', '>f8', '|u1'])
@pytest.mark.parametrize('channels', [1, 2, 5])
@pytest.mark.parametrize('rate', [8000, 32000])
@pytest.mark.parametrize('mmap', [False, True])
@pytest.mark.parametrize('realfile', [False, True])
def test_write_roundtrip(realfile, mmap, rate, channels, dt_str, tmpdir):
    if False:
        return 10
    dtype = np.dtype(dt_str)
    if realfile:
        tmpfile = str(tmpdir.join('temp.wav'))
    else:
        tmpfile = BytesIO()
    data = np.random.rand(100, channels)
    if channels == 1:
        data = data[:, 0]
    if dtype.kind == 'f':
        data = data.astype(dtype)
    else:
        data = (data * 128).astype(dtype)
    wavfile.write(tmpfile, rate, data)
    (rate2, data2) = wavfile.read(tmpfile, mmap=mmap)
    assert_equal(rate, rate2)
    assert_(data2.dtype.byteorder in ('<', '=', '|'), msg=data2.dtype)
    assert_array_equal(data, data2)
    if realfile:
        data2[0] = 0
    else:
        with pytest.raises(ValueError, match='read-only'):
            data2[0] = 0
    if realfile and mmap and IS_PYPY and (sys.platform == 'win32'):
        break_cycles()
        break_cycles()