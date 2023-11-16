import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray

class SndarrayTest(unittest.TestCase):
    array_dtypes = {8: uint8, -8: int8, 16: uint16, -16: int16, 32: float32}

    def _assert_compatible(self, arr, size):
        if False:
            return 10
        dtype = self.array_dtypes[size]
        self.assertEqual(arr.dtype, dtype)

    def test_array(self):
        if False:
            while True:
                i = 10

        def check_array(size, channels, test_data):
            if False:
                while True:
                    i = 10
            try:
                pygame.mixer.init(22050, size, channels, allowedchanges=0)
            except pygame.error:
                return
            try:
                (__, sz, __) = pygame.mixer.get_init()
                if sz == size:
                    srcarr = array(test_data, self.array_dtypes[size])
                    snd = pygame.sndarray.make_sound(srcarr)
                    arr = pygame.sndarray.array(snd)
                    self._assert_compatible(arr, size)
                    self.assertTrue(alltrue(arr == srcarr), 'size: %i\n%s\n%s' % (size, arr, test_data))
            finally:
                pygame.mixer.quit()
        check_array(8, 1, [0, 15, 240, 255])
        check_array(8, 2, [[0, 128], [45, 65], [100, 161], [255, 64]])
        check_array(16, 1, [0, 255, 65280, 65535])
        check_array(16, 2, [[0, 65535], [65535, 0], [255, 65280], [3855, 61680]])
        check_array(-8, 1, [0, -128, 127, 100])
        check_array(-8, 2, [[0, -128], [-100, 100], [37, -80], [255, 0]])
        check_array(-16, 1, [0, 32767, -32767, -1])
        check_array(-16, 2, [[0, -32767], [-32767, 0], [32767, 0], [0, 32767]])

    def test_get_arraytype(self):
        if False:
            i = 10
            return i + 15
        array_type = pygame.sndarray.get_arraytype()
        self.assertEqual(array_type, 'numpy', f'unknown array type {array_type}')

    def test_get_arraytypes(self):
        if False:
            return 10
        arraytypes = pygame.sndarray.get_arraytypes()
        self.assertIn('numpy', arraytypes)
        for atype in arraytypes:
            self.assertEqual(atype, 'numpy', f'unknown array type {atype}')

    def test_make_sound(self):
        if False:
            while True:
                i = 10

        def check_sound(size, channels, test_data):
            if False:
                for i in range(10):
                    print('nop')
            try:
                pygame.mixer.init(22050, size, channels, allowedchanges=0)
            except pygame.error:
                return
            try:
                (__, sz, __) = pygame.mixer.get_init()
                if sz == size:
                    srcarr = array(test_data, self.array_dtypes[size])
                    snd = pygame.sndarray.make_sound(srcarr)
                    arr = pygame.sndarray.samples(snd)
                    self.assertTrue(alltrue(arr == srcarr), 'size: %i\n%s\n%s' % (size, arr, test_data))
            finally:
                pygame.mixer.quit()
        check_sound(8, 1, [0, 15, 240, 255])
        check_sound(8, 2, [[0, 128], [45, 65], [100, 161], [255, 64]])
        check_sound(16, 1, [0, 255, 65280, 65535])
        check_sound(16, 2, [[0, 65535], [65535, 0], [255, 65280], [3855, 61680]])
        check_sound(-8, 1, [0, -128, 127, 100])
        check_sound(-8, 2, [[0, -128], [-100, 100], [37, -80], [255, 0]])
        check_sound(-16, 1, [0, 32767, -32767, -1])
        check_sound(-16, 2, [[0, -32767], [-32767, 0], [32767, 0], [0, 32767]])
        check_sound(32, 2, [[0.0, -1.0], [-1.0, 0], [1.0, 0], [0, 1.0]])

    def test_samples(self):
        if False:
            i = 10
            return i + 15
        null_byte = b'\x00'

        def check_sample(size, channels, test_data):
            if False:
                for i in range(10):
                    print('nop')
            try:
                pygame.mixer.init(22050, size, channels, allowedchanges=0)
            except pygame.error:
                return
            try:
                (__, sz, __) = pygame.mixer.get_init()
                if sz == size:
                    zeroed = null_byte * (abs(size) // 8 * len(test_data) * channels)
                    snd = pygame.mixer.Sound(buffer=zeroed)
                    samples = pygame.sndarray.samples(snd)
                    self._assert_compatible(samples, size)
                    samples[...] = test_data
                    arr = pygame.sndarray.array(snd)
                    self.assertTrue(alltrue(samples == arr), 'size: %i\n%s\n%s' % (size, arr, test_data))
            finally:
                pygame.mixer.quit()
        check_sample(8, 1, [0, 15, 240, 255])
        check_sample(8, 2, [[0, 128], [45, 65], [100, 161], [255, 64]])
        check_sample(16, 1, [0, 255, 65280, 65535])
        check_sample(16, 2, [[0, 65535], [65535, 0], [255, 65280], [3855, 61680]])
        check_sample(-8, 1, [0, -128, 127, 100])
        check_sample(-8, 2, [[0, -128], [-100, 100], [37, -80], [255, 0]])
        check_sample(-16, 1, [0, 32767, -32767, -1])
        check_sample(-16, 2, [[0, -32767], [-32767, 0], [32767, 0], [0, 32767]])
        check_sample(32, 2, [[0.0, -1.0], [-1.0, 0], [1.0, 0], [0, 1.0]])

    def test_use_arraytype(self):
        if False:
            for i in range(10):
                print('nop')

        def do_use_arraytype(atype):
            if False:
                while True:
                    i = 10
            pygame.sndarray.use_arraytype(atype)
        pygame.sndarray.use_arraytype('numpy')
        self.assertEqual(pygame.sndarray.get_arraytype(), 'numpy')
        self.assertRaises(ValueError, do_use_arraytype, 'not an option')

    def test_float32(self):
        if False:
            while True:
                i = 10
        'sized arrays work with Sounds and 32bit float arrays.'
        try:
            pygame.mixer.init(22050, 32, 2, allowedchanges=0)
        except pygame.error:
            self.skipTest('unsupported mixer configuration')
        arr = array([[0.0, -1.0], [-1.0, 0], [1.0, 0], [0, 1.0]], float32)
        newsound = pygame.mixer.Sound(array=arr)
        pygame.mixer.quit()
if __name__ == '__main__':
    unittest.main()