import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
IS_PYPY = 'PyPy' == platform.python_implementation()
FREQUENCIES = [11025, 22050, 44100, 48000]
SIZES = [-16, -8, 8, 16]
CHANNELS = [1, 2]
BUFFERS = [3024]
CONFIGS = [{'frequency': f, 'size': s, 'channels': c} for f in FREQUENCIES for s in SIZES for c in CHANNELS]
CONFIG = {'frequency': 44100, 'size': 32, 'channels': 2, 'allowedchanges': 0}

class InvalidBool:
    """To help test invalid bool values."""
    __bool__ = None

class MixerModuleTest(unittest.TestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        mixer.quit()
        mixer.pre_init(0, 0, 0, 0)

    def test_init__keyword_args(self):
        if False:
            while True:
                i = 10
        mixer.init(**CONFIG)
        mixer_conf = mixer.get_init()
        self.assertEqual(mixer_conf[0], CONFIG['frequency'])
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG['size']))
        self.assertGreaterEqual(mixer_conf[2], CONFIG['channels'])

    def test_pre_init__keyword_args(self):
        if False:
            print('Hello World!')
        mixer.pre_init(**CONFIG)
        mixer.init()
        mixer_conf = mixer.get_init()
        self.assertEqual(mixer_conf[0], CONFIG['frequency'])
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG['size']))
        self.assertGreaterEqual(mixer_conf[2], CONFIG['channels'])

    def test_pre_init__zero_values(self):
        if False:
            print('Hello World!')
        mixer.pre_init(22050, -8, 1)
        mixer.pre_init(0, 0, 0)
        mixer.init(allowedchanges=0)
        self.assertEqual(mixer.get_init()[0], 44100)
        self.assertEqual(mixer.get_init()[1], -16)
        self.assertGreaterEqual(mixer.get_init()[2], 2)

    def test_init__zero_values(self):
        if False:
            while True:
                i = 10
        mixer.pre_init(44100, 8, 1, allowedchanges=0)
        mixer.init(0, 0, 0)
        self.assertEqual(mixer.get_init(), (44100, 8, 1))

    def test_get_init__returns_None_if_mixer_not_initialized(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(mixer.get_init())

    def test_get_num_channels__defaults_eight_after_init(self):
        if False:
            i = 10
            return i + 15
        mixer.init()
        self.assertEqual(mixer.get_num_channels(), 8)

    def test_set_num_channels(self):
        if False:
            return 10
        mixer.init()
        default_num_channels = mixer.get_num_channels()
        for i in range(1, default_num_channels + 1):
            mixer.set_num_channels(i)
            self.assertEqual(mixer.get_num_channels(), i)

    def test_quit(self):
        if False:
            for i in range(10):
                print('nop')
        'get_num_channels() Should throw pygame.error if uninitialized\n        after mixer.quit()'
        mixer.init()
        mixer.quit()
        self.assertRaises(pygame.error, mixer.get_num_channels)

    @unittest.skipIf(sys.platform.startswith('win'), 'See github issue 892.')
    @unittest.skipIf(IS_PYPY, 'random errors here with pypy')
    def test_sound_args(self):
        if False:
            while True:
                i = 10

        def get_bytes(snd):
            if False:
                for i in range(10):
                    print('nop')
            return snd.get_raw()
        mixer.init()
        sample = b'\x00\xff' * 24
        wave_path = example_path(os.path.join('data', 'house_lo.wav'))
        uwave_path = str(wave_path)
        bwave_path = uwave_path.encode(sys.getfilesystemencoding())
        snd = mixer.Sound(file=wave_path)
        self.assertTrue(snd.get_length() > 0.5)
        snd_bytes = get_bytes(snd)
        self.assertTrue(len(snd_bytes) > 1000)
        self.assertEqual(get_bytes(mixer.Sound(wave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(file=uwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(uwave_path)), snd_bytes)
        arg_emsg = 'Sound takes either 1 positional or 1 keyword argument'
        with self.assertRaises(TypeError) as cm:
            mixer.Sound()
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, buffer=sample)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(foobar=sample)
        self.assertEqual(str(cm.exception), "Unrecognized keyword argument 'foobar'")
        snd = mixer.Sound(wave_path, **{})
        self.assertEqual(get_bytes(snd), snd_bytes)
        snd = mixer.Sound(*[], **{'file': wave_path})
        with self.assertRaises(TypeError) as cm:
            mixer.Sound([])
        self.assertEqual(str(cm.exception), 'Unrecognized argument (type list)')
        with self.assertRaises(TypeError) as cm:
            snd = mixer.Sound(buffer=[])
        emsg = 'Expected object with buffer interface: got a list'
        self.assertEqual(str(cm.exception), emsg)
        ufake_path = '12345678'
        self.assertRaises(IOError, mixer.Sound, ufake_path)
        self.assertRaises(IOError, mixer.Sound, '12345678')
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer='something')
        emsg = 'Unicode object not allowed as buffer object'
        self.assertEqual(str(cm.exception), emsg)
        self.assertEqual(get_bytes(mixer.Sound(buffer=sample)), sample)
        if type(sample) != str:
            somebytes = get_bytes(mixer.Sound(sample))
            self.assertEqual(somebytes, sample)
        self.assertEqual(get_bytes(mixer.Sound(file=bwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(bwave_path)), snd_bytes)
        snd = mixer.Sound(wave_path)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        snd2 = mixer.Sound(array=snd)
        self.assertEqual(snd.get_raw(), snd2.get_raw())

    def test_sound_unicode(self):
        if False:
            while True:
                i = 10
        'test non-ASCII unicode path'
        mixer.init()
        import shutil
        ep = example_path('data')
        temp_file = os.path.join(ep, '你好.wav')
        org_file = os.path.join(ep, 'house_lo.wav')
        shutil.copy(org_file, temp_file)
        try:
            with open(temp_file, 'rb') as f:
                pass
        except OSError:
            raise unittest.SkipTest('the path cannot be opened')
        try:
            sound = mixer.Sound(temp_file)
            del sound
        finally:
            os.remove(temp_file)

    @unittest.skipIf(os.environ.get('SDL_AUDIODRIVER') == 'disk', 'this test fails without real sound card')
    def test_array_keyword(self):
        if False:
            while True:
                i = 10
        try:
            from numpy import array, arange, zeros, int8, uint8, int16, uint16, int32, uint32
        except ImportError:
            self.skipTest('requires numpy')
        freq = 22050
        format_list = [-8, 8, -16, 16]
        channels_list = [1, 2]
        a_lists = {f: [] for f in format_list}
        a32u_mono = arange(0, 256, 1, uint32)
        a16u_mono = a32u_mono.astype(uint16)
        a8u_mono = a32u_mono.astype(uint8)
        au_list_mono = [(1, a) for a in [a8u_mono, a16u_mono, a32u_mono]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_mono)
        a32s_mono = arange(-128, 128, 1, int32)
        a16s_mono = a32s_mono.astype(int16)
        a8s_mono = a32s_mono.astype(int8)
        as_list_mono = [(1, a) for a in [a8s_mono, a16s_mono, a32s_mono]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_mono)
        a32u_stereo = zeros([a32u_mono.shape[0], 2], uint32)
        a32u_stereo[:, 0] = a32u_mono
        a32u_stereo[:, 1] = 255 - a32u_mono
        a16u_stereo = a32u_stereo.astype(uint16)
        a8u_stereo = a32u_stereo.astype(uint8)
        au_list_stereo = [(2, a) for a in [a8u_stereo, a16u_stereo, a32u_stereo]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_stereo)
        a32s_stereo = zeros([a32s_mono.shape[0], 2], int32)
        a32s_stereo[:, 0] = a32s_mono
        a32s_stereo[:, 1] = -1 - a32s_mono
        a16s_stereo = a32s_stereo.astype(int16)
        a8s_stereo = a32s_stereo.astype(int8)
        as_list_stereo = [(2, a) for a in [a8s_stereo, a16s_stereo, a32s_stereo]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_stereo)
        for format in format_list:
            for channels in channels_list:
                try:
                    mixer.init(freq, format, channels)
                except pygame.error:
                    continue
                try:
                    (__, f, c) = mixer.get_init()
                    if f != format or c != channels:
                        continue
                    for (c, a) in a_lists[format]:
                        self._test_array_argument(format, a, c == channels)
                finally:
                    mixer.quit()

    def _test_array_argument(self, format, a, test_pass):
        if False:
            i = 10
            return i + 15
        from numpy import array, all as all_
        try:
            snd = mixer.Sound(array=a)
        except ValueError:
            if not test_pass:
                return
            self.fail('Raised ValueError: Format %i, dtype %s' % (format, a.dtype))
        if not test_pass:
            self.fail('Did not raise ValueError: Format %i, dtype %s' % (format, a.dtype))
        a2 = array(snd)
        a3 = a.astype(a2.dtype)
        lshift = abs(format) - 8 * a.itemsize
        if lshift >= 0:
            a3 <<= lshift
        self.assertTrue(all_(a2 == a3), 'Format %i, dtype %s' % (format, a.dtype))

    def _test_array_interface_fail(self, a):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, mixer.Sound, array=a)

    def test_array_interface(self):
        if False:
            for i in range(10):
                print('nop')
        mixer.init(22050, -16, 1, allowedchanges=0)
        snd = mixer.Sound(buffer=b'\x00\x7f' * 20)
        d = snd.__array_interface__
        self.assertTrue(isinstance(d, dict))
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            typestr = '<i2'
        else:
            typestr = '>i2'
        self.assertEqual(d['typestr'], typestr)
        self.assertEqual(d['shape'], (20,))
        self.assertEqual(d['strides'], (2,))
        self.assertEqual(d['data'], (snd._samples_address, False))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf__one_channel(self):
        if False:
            i = 10
            return i + 15
        mixer.init(22050, -16, 1)
        self._NEWBUF_export_check()

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf__twho_channel(self):
        if False:
            print('Hello World!')
        mixer.init(22050, -16, 2)
        self._NEWBUF_export_check()

    def _NEWBUF_export_check(self):
        if False:
            while True:
                i = 10
        (freq, fmt, channels) = mixer.get_init()
        ndim = 1 if channels == 1 else 2
        itemsize = abs(fmt) // 8
        formats = {8: 'B', -8: 'b', 16: '=H', -16: '=h', 32: '=I', -32: '=i', 64: '=Q', -64: '=q'}
        format = formats[fmt]
        from pygame.tests.test_utils import buftools
        Exporter = buftools.Exporter
        Importer = buftools.Importer
        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        (fsys, frev) = ('<', '>') if is_lil_endian else ('>', '<')
        shape = (10, channels)[:ndim]
        strides = (channels * itemsize, itemsize)[2 - ndim:]
        exp = Exporter(shape, format=frev + 'i')
        snd = mixer.Sound(array=exp)
        buflen = len(exp) * itemsize * channels
        imp = Importer(snd, buftools.PyBUF_SIMPLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_WRITABLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FORMAT)
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_ND)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_STRIDES)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, 2)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, exp.shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        imp = Importer(snd, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        if ndim == 1:
            imp = Importer(snd, buftools.PyBUF_F_CONTIGUOUS)
            self.assertEqual(imp.ndim, 1)
            self.assertTrue(imp.format is None)
            self.assertEqual(imp.strides, strides)
        else:
            self.assertRaises(BufferError, Importer, snd, buftools.PyBUF_F_CONTIGUOUS)

    def test_fadeout(self):
        if False:
            while True:
                i = 10
        'Ensure pygame.mixer.fadeout() stops playback after fading out the sound.'
        if mixer.get_init() is None:
            mixer.init()
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        channel = pygame.mixer.find_channel()
        channel.play(sound)
        fadeout_time = 200
        channel.fadeout(fadeout_time)
        pygame.time.wait(fadeout_time + 30)
        self.assertFalse(channel.get_busy())

    def test_find_channel(self):
        if False:
            return 10
        mixer.init()
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound = mixer.Sound(file=filename)
        num_channels = mixer.get_num_channels()
        if num_channels > 0:
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)
            channels = []
            for channel_id in range(0, num_channels):
                channel = mixer.Channel(channel_id)
                channel.play(sound)
                channels.append(channel)
            found_channel = mixer.find_channel()
            self.assertIsNone(found_channel)
            found_channel = mixer.find_channel(True)
            self.assertIsNotNone(found_channel)
            found_channel = mixer.find_channel(force=True)
            self.assertIsNotNone(found_channel)
            for channel in channels:
                channel.stop()
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)

    @unittest.expectedFailure
    def test_pause(self):
        if False:
            return 10
        'Ensure pygame.mixer.pause() temporarily stops playback of all sound channels.'
        if mixer.get_init() is None:
            mixer.init()
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel = mixer.find_channel()
        channel.play(sound)
        mixer.pause()
        self.assertFalse(channel.get_busy())
        mixer.unpause()
        self.assertTrue(channel.get_busy())

    def test_set_reserved(self):
        if False:
            print('Hello World!')
        'Ensure pygame.mixer.set_reserved() reserves the given number of channels.'
        mixer.init()
        default_num_channels = mixer.get_num_channels()
        result = mixer.set_reserved(default_num_channels)
        self.assertEqual(result, default_num_channels)
        result = mixer.set_reserved(default_num_channels + 1)
        self.assertEqual(result, default_num_channels)
        result = mixer.set_reserved(0)
        self.assertEqual(result, 0)
        result = mixer.set_reserved(int(default_num_channels / 2))
        self.assertEqual(result, int(default_num_channels / 2))

    def test_stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stops playback of all active sound channels.'
        if mixer.get_init() is None:
            mixer.init()
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        channel = pygame.mixer.Channel(0)
        channel.play(sound)
        pygame.mixer.stop()
        for i in range(pygame.mixer.get_num_channels()):
            self.assertFalse(pygame.mixer.Channel(i).get_busy())

    def test_get_sdl_mixer_version(self):
        if False:
            i = 10
            return i + 15
        'Ensures get_sdl_mixer_version works correctly with no args.'
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        version = pygame.mixer.get_sdl_mixer_version()
        self.assertIsInstance(version, expected_type)
        self.assertEqual(len(version), expected_length)
        for item in version:
            self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__args(self):
        if False:
            while True:
                i = 10
        'Ensures get_sdl_mixer_version works correctly using args.'
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(value)
            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)
            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__kwargs(self):
        if False:
            return 10
        'Ensures get_sdl_mixer_version works correctly using kwargs.'
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(linked=value)
            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)
            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__invalid_args_kwargs(self):
        if False:
            return 10
        'Ensures get_sdl_mixer_version handles invalid args and kwargs.'
        invalid_bool = InvalidBool()
        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(invalid_bool)
        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(linked=invalid_bool)

    def test_get_sdl_mixer_version__linked_equals_compiled(self):
        if False:
            while True:
                i = 10
        "Ensures get_sdl_mixer_version's linked/compiled versions are equal."
        linked_version = pygame.mixer.get_sdl_mixer_version(linked=True)
        complied_version = pygame.mixer.get_sdl_mixer_version(linked=False)
        self.assertTupleEqual(linked_version, complied_version)

class ChannelTypeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        mixer.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        mixer.quit()

    def setUp(cls):
        if False:
            for i in range(10):
                print('nop')
        if mixer.get_init() is None:
            mixer.init()

    def test_channel(self):
        if False:
            return 10
        'Ensure Channel() creation works.'
        channel = mixer.Channel(0)
        self.assertIsInstance(channel, mixer.ChannelType)
        self.assertEqual(channel.__class__.__name__, 'Channel')

    def test_channel__without_arg(self):
        if False:
            return 10
        'Ensure exception for Channel() creation with no argument.'
        with self.assertRaises(TypeError):
            mixer.Channel()

    def test_channel__invalid_id(self):
        if False:
            while True:
                i = 10
        'Ensure exception for Channel() creation with an invalid id.'
        with self.assertRaises(IndexError):
            mixer.Channel(-1)

    def test_channel__before_init(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure exception for Channel() creation with non-init mixer.'
        mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            mixer.Channel(0)

    def test_fadeout(self):
        if False:
            while True:
                i = 10
        'Ensure Channel.fadeout() stops playback after fading out.'
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        fadeout_time = 1000
        channel.fadeout(fadeout_time)
        pygame.time.wait(fadeout_time + 100)
        self.assertFalse(channel.get_busy())

    def test_get_busy(self):
        if False:
            i = 10
            return i + 15
        "Ensure an idle channel's busy state is correct."
        expected_busy = False
        channel = mixer.Channel(0)
        busy = channel.get_busy()
        self.assertEqual(busy, expected_busy)

    def test_get_busy__active(self):
        if False:
            i = 10
            return i + 15
        "Ensure an active channel's busy state is correct."
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        self.assertTrue(channel.get_busy())

    def todo_test_get_endevent(self):
        if False:
            print('Hello World!')
        self.fail()

    def test_get_queue(self):
        if False:
            while True:
                i = 10
        'Ensure Channel.get_queue() returns any queued Sound.'
        channel = mixer.Channel(0)
        (frequency, format, channels) = mixer.get_init()
        sound_length_in_ms = 200
        sound_length_in_ms_2 = 400
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound1 = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        sound2 = mixer.Sound(b'\x00' * int(sound_length_in_ms_2 * bytes_per_ms))
        channel.play(sound1)
        channel.queue(sound2)
        self.assertEqual(channel.get_queue().get_length(), sound2.get_length())
        pygame.time.wait(sound_length_in_ms + 100)
        self.assertEqual(channel.get_sound().get_length(), sound2.get_length())
        pygame.time.wait(sound_length_in_ms_2 + 100)
        self.assertIsNone(channel.get_queue())

    def test_get_sound(self):
        if False:
            return 10
        'Ensure Channel.get_sound() returns the currently playing Sound.'
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        got_sound = channel.get_sound()
        self.assertEqual(got_sound, sound)
        channel.stop()
        got_sound = channel.get_sound()
        self.assertIsNone(got_sound)

    def test_get_volume(self):
        if False:
            print('Hello World!')
        "Ensure a channel's volume can be retrieved."
        expected_volume = 1.0
        channel = mixer.Channel(0)
        volume = channel.get_volume()
        self.assertAlmostEqual(volume, expected_volume)

    def test_pause_unpause(self):
        if False:
            return 10
        '\n        Test if the Channel can be paused and unpaused.\n        '
        if mixer.get_init() is None:
            mixer.init()
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        channel = sound.play()
        channel.pause()
        self.assertTrue(channel.get_busy(), msg="Channel should be paused but it's not.")
        channel.unpause()
        self.assertTrue(channel.get_busy(), msg="Channel should be unpaused but it's not.")
        sound.stop()

    def test_pause_unpause__before_init(self):
        if False:
            while True:
                i = 10
        '\n        Ensure exception for Channel.pause() with non-init mixer.\n        '
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel = sound.play()
        mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            channel.pause()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            channel.unpause()

    def todo_test_queue(self):
        if False:
            for i in range(10):
                print('nop')
        self.fail()

    def test_stop(self):
        if False:
            while True:
                i = 10
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        channel.stop()
        self.assertFalse(channel.get_busy())
        channel.queue(sound)
        channel.stop()
        self.assertFalse(channel.get_busy())
        channel.play(sound)
        self.assertTrue(channel.get_busy())

class ChannelSetVolumeTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        mixer.init()
        self.channel = pygame.mixer.Channel(0)
        self.sound = pygame.mixer.Sound(example_path('data/boom.wav'))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        mixer.quit()

    def test_set_volume_with_one_argument(self):
        if False:
            print('Hello World!')
        self.channel.play(self.sound)
        self.channel.set_volume(0.5)
        self.assertEqual(self.channel.get_volume(), 0.5)

    @unittest.expectedFailure
    def test_set_volume_with_two_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        self.channel.play(self.sound)
        self.channel.set_volume(0.3, 0.7)
        self.assertEqual(self.channel.get_volume(), (0.3, 0.7))

class ChannelEndEventTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pygame.display.init()
        pygame.display.set_mode((40, 40))
        if mixer.get_init() is None:
            mixer.init()

    def tearDown(self):
        if False:
            while True:
                i = 10
        pygame.display.quit()
        mixer.quit()

    def test_get_endevent(self):
        if False:
            while True:
                i = 10
        'Ensure Channel.get_endevent() returns the correct event type.'
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        END_EVENT = pygame.USEREVENT + 1
        channel.set_endevent(END_EVENT)
        got_end_event = channel.get_endevent()
        self.assertEqual(got_end_event, END_EVENT)
        channel.stop()
        while channel.get_busy():
            pygame.time.wait(10)
        events = pygame.event.get(got_end_event)
        self.assertTrue(len(events) > 0)

class TestSoundPlay(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        mixer.init()
        self.filename = example_path(os.path.join('data', 'house_lo.wav'))
        self.sound = mixer.Sound(file=self.filename)

    def tearDown(self):
        if False:
            return 10
        mixer.quit()

    def test_play_once(self):
        if False:
            while True:
                i = 10
        'Test playing a sound once.'
        channel = self.sound.play()
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())

    def test_play_multiple_times(self):
        if False:
            while True:
                i = 10
        'Test playing a sound multiple times.'
        (frequency, format, channels) = mixer.get_init()
        sound_length_in_ms = 100
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        self.assertAlmostEqual(sound.get_length(), sound_length_in_ms / 1000.0, places=2)
        num_loops = 5
        channel = sound.play(loops=num_loops)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        pygame.time.wait(sound_length_in_ms * num_loops - 100)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(sound_length_in_ms + 200)
        self.assertFalse(channel.get_busy())

    def test_play_indefinitely(self):
        if False:
            return 10
        'Test playing a sound indefinitely.'
        (frequency, format, channels) = mixer.get_init()
        sound_length_in_ms = 100
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        channel = sound.play(loops=-1)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        for _ in range(2):
            self.assertTrue(channel.get_busy())
            pygame.time.wait(sound_length_in_ms)

    def test_play_with_maxtime(self):
        if False:
            print('Hello World!')
        'Test playing a sound with maxtime.'
        channel = self.sound.play(maxtime=200)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(200 + 50)
        self.assertFalse(channel.get_busy())

    def test_play_with_fade_ms(self):
        if False:
            for i in range(10):
                print('nop')
        'Test playing a sound with fade_ms.'
        channel = self.sound.play(fade_ms=500)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(250)
        self.assertGreater(channel.get_volume(), 0.3)
        self.assertLess(channel.get_volume(), 0.8)
        pygame.time.wait(300)
        self.assertEqual(channel.get_volume(), 1.0)

    def test_play_with_invalid_loops(self):
        if False:
            return 10
        'Test playing a sound with invalid loops.'
        with self.assertRaises(TypeError):
            self.sound.play(loops='invalid')

    def test_play_with_invalid_maxtime(self):
        if False:
            print('Hello World!')
        'Test playing a sound with invalid maxtime.'
        with self.assertRaises(TypeError):
            self.sound.play(maxtime='invalid')

    def test_play_with_invalid_fade_ms(self):
        if False:
            return 10
        'Test playing a sound with invalid fade_ms.'
        with self.assertRaises(TypeError):
            self.sound.play(fade_ms='invalid')

class SoundTypeTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        mixer.quit()

    def setUp(cls):
        if False:
            for i in range(10):
                print('nop')
        if mixer.get_init() is None:
            mixer.init()

    def test_sound(self):
        if False:
            while True:
                i = 10
        'Ensure Sound() creation with a filename works.'
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound1 = mixer.Sound(filename)
        sound2 = mixer.Sound(file=filename)
        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def test_sound__from_file_object(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure Sound() creation with a file object works.'
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        with open(filename, 'rb') as file_obj:
            sound = mixer.Sound(file_obj)
            self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_sound_object(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure Sound() creation with a Sound() object works.'
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound_obj = mixer.Sound(file=filename)
        sound = mixer.Sound(sound_obj)
        self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_pathlib(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure Sound() creation with a pathlib.Path object works.'
        path = pathlib.Path(example_path(os.path.join('data', 'house_lo.wav')))
        sound1 = mixer.Sound(path)
        sound2 = mixer.Sound(file=path)
        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def todo_test_sound__from_buffer(self):
        if False:
            return 10
        'Ensure Sound() creation with a buffer works.'
        self.fail()

    def todo_test_sound__from_array(self):
        if False:
            i = 10
            return i + 15
        'Ensure Sound() creation with an array works.'
        self.fail()

    def test_sound__without_arg(self):
        if False:
            print('Hello World!')
        'Ensure exception raised for Sound() creation with no argument.'
        with self.assertRaises(TypeError):
            mixer.Sound()

    def test_sound__before_init(self):
        if False:
            return 10
        'Ensure exception raised for Sound() creation with non-init mixer.'
        mixer.quit()
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            mixer.Sound(file=filename)

    @unittest.skipIf(IS_PYPY, 'pypy skip')
    def test_samples_address(self):
        if False:
            print('Hello World!')
        'Test the _samples_address getter.'
        try:
            from ctypes import pythonapi, c_void_p, py_object
            Bytes_FromString = pythonapi.PyBytes_FromString
            Bytes_FromString.restype = c_void_p
            Bytes_FromString.argtypes = [py_object]
            samples = b'abcdefgh'
            sample_bytes = Bytes_FromString(samples)
            snd = mixer.Sound(buffer=samples)
            self.assertNotEqual(snd._samples_address, sample_bytes)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                snd._samples_address

    def test_get_length(self):
        if False:
            i = 10
            return i + 15
        'Tests if get_length returns a correct length.'
        try:
            for size in SIZES:
                pygame.mixer.quit()
                pygame.mixer.init(size=size)
                filename = example_path(os.path.join('data', 'punch.wav'))
                sound = mixer.Sound(file=filename)
                sound_bytes = sound.get_raw()
                (mix_freq, mix_bits, mix_channels) = pygame.mixer.get_init()
                mix_bytes = abs(mix_bits) / 8
                expected_length = float(len(sound_bytes)) / mix_freq / mix_bytes / mix_channels
                self.assertAlmostEqual(expected_length, sound.get_length())
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_length()

    def test_get_num_channels(self):
        if False:
            return 10
        '\n        Tests if Sound.get_num_channels returns the correct number\n        of channels playing a specific sound.\n        '
        try:
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            self.assertEqual(sound.get_num_channels(), 0)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 1)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 2)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), 0)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_num_channels()

    def test_get_volume(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensure a sound's volume can be retrieved."
        try:
            expected_volume = 1.0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            volume = sound.get_volume()
            self.assertAlmostEqual(volume, expected_volume)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_volume()

    def test_get_volume__while_playing(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensure a sound's volume can be retrieved while playing."
        try:
            expected_volume = 1.0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.play(-1)
            volume = sound.get_volume()
            self.assertAlmostEqual(volume, expected_volume)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_volume()

    def test_set_volume(self):
        if False:
            return 10
        "Ensure a sound's volume can be set."
        try:
            float_delta = 1.0 / 128
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            current_volume = sound.get_volume()
            volumes = ((-1, current_volume), (0, 0.0), (0.01, 0.01), (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.99, 0.99), (1, 1.0), (1.1, 1.0), (2.0, 1.0))
            for (volume_set_value, expected_volume) in volumes:
                sound.set_volume(volume_set_value)
                self.assertAlmostEqual(sound.get_volume(), expected_volume, delta=float_delta)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.set_volume(1)

    def test_set_volume__while_playing(self):
        if False:
            i = 10
            return i + 15
        "Ensure a sound's volume can be set while playing."
        try:
            float_delta = 1.0 / 128
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            current_volume = sound.get_volume()
            volumes = ((-1, current_volume), (0, 0.0), (0.01, 0.01), (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.99, 0.99), (1, 1.0), (1.1, 1.0), (2.0, 1.0))
            sound.play(loops=-1)
            for (volume_set_value, expected_volume) in volumes:
                sound.set_volume(volume_set_value)
                self.assertAlmostEqual(sound.get_volume(), expected_volume, delta=float_delta)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.set_volume(1)

    def test_stop(self):
        if False:
            print('Hello World!')
        'Ensure stop can be called while not playing a sound.'
        try:
            expected_channels = 0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), expected_channels)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.stop()

    def test_stop__while_playing(self):
        if False:
            i = 10
            return i + 15
        'Ensure stop stops a playing sound.'
        try:
            expected_channels = 0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.play(-1)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), expected_channels)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.stop()

    def test_get_raw(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure get_raw returns the correct bytestring.'
        try:
            samples = b'abcdefgh'
            snd = mixer.Sound(buffer=samples)
            raw = snd.get_raw()
            self.assertIsInstance(raw, bytes)
            self.assertEqual(raw, samples)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                snd.get_raw()

    def test_correct_subclassing(self):
        if False:
            print('Hello World!')

        class CorrectSublass(mixer.Sound):

            def __init__(self, file):
                if False:
                    print('Hello World!')
                super().__init__(file=file)
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        correct = CorrectSublass(filename)
        try:
            correct.get_volume()
        except Exception:
            self.fail('This should not raise an exception.')

    def test_incorrect_subclassing(self):
        if False:
            i = 10
            return i + 15

        class IncorrectSuclass(mixer.Sound):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass
        incorrect = IncorrectSuclass()
        self.assertRaises(RuntimeError, incorrect.get_volume)

class TestSoundFadeout(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if mixer.get_init() is None:
            pygame.mixer.init()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pygame.mixer.quit()

    def test_fadeout_with_valid_time(self):
        if False:
            i = 10
            return i + 15
        'Tests if fadeout stops sound playback after fading it out over the time argument in milliseconds.'
        filename = example_path(os.path.join('data', 'punch.wav'))
        sound = mixer.Sound(file=filename)
        channel = sound.play()
        channel.fadeout(1000)
        pygame.time.wait(2000)
        self.assertFalse(channel.get_busy())

class TestGetBusy(unittest.TestCase):
    """Test pygame.mixer.get_busy.

    |tags:slow|
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pygame.mixer.init()

    def tearDown(self):
        if False:
            return 10
        pygame.mixer.quit()

    def test_no_sound_playing(self):
        if False:
            return 10
        '\n        Test that get_busy returns False when no sound is playing.\n        '
        self.assertFalse(pygame.mixer.get_busy())

    def test_one_sound_playing(self):
        if False:
            return 10
        '\n        Test that get_busy returns True when one sound is playing.\n        '
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound.play()
        time.sleep(0.2)
        self.assertTrue(pygame.mixer.get_busy())
        sound.stop()

    def test_multiple_sounds_playing(self):
        if False:
            while True:
                i = 10
        '\n        Test that get_busy returns True when multiple sounds are playing.\n        '
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        self.assertTrue(pygame.mixer.get_busy())
        sound1.stop()
        sound2.stop()

    def test_all_sounds_stopped(self):
        if False:
            print('Hello World!')
        '\n        Test that get_busy returns False when all sounds are stopped.\n        '
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        sound1.stop()
        sound2.stop()
        time.sleep(0.2)
        self.assertFalse(pygame.mixer.get_busy())

    def test_all_sounds_stopped_with_fadeout(self):
        if False:
            while True:
                i = 10
        '\n        Test that get_busy returns False when all sounds are stopped with\n        fadeout.\n        '
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        sound1.fadeout(100)
        sound2.fadeout(100)
        time.sleep(0.3)
        self.assertFalse(pygame.mixer.get_busy())

    def test_sound_fading_out(self):
        if False:
            i = 10
            return i + 15
        'Tests that get_busy() returns True when a sound is fading out'
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound.play(fade_ms=1000)
        time.sleep(1.1)
        self.assertTrue(pygame.mixer.get_busy())
        sound.stop()
if __name__ == '__main__':
    unittest.main()