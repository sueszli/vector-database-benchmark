from unittest import main
from numpy.testing import TestCase, assert_equal, assert_almost_equal
import aubio

class aubio_tempo_default(TestCase):

    def test_members(self):
        if False:
            return 10
        o = aubio.tempo()
        assert_equal([o.buf_size, o.hop_size, o.method, o.samplerate], [1024, 512, 'default', 44100])

class aubio_tempo_params(TestCase):
    samplerate = 44100

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.o = aubio.tempo(samplerate=self.samplerate)

    def test_get_delay(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.o.get_delay(), 0)

    def test_set_delay(self):
        if False:
            for i in range(10):
                print('nop')
        val = 256
        self.o.set_delay(val)
        assert_equal(self.o.get_delay(), val)

    def test_get_delay_s(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.o.get_delay_s(), 0.0)

    def test_set_delay_s(self):
        if False:
            while True:
                i = 10
        val = 0.05
        self.o.set_delay_s(val)
        assert_almost_equal(self.o.get_delay_s(), val)

    def test_get_delay_ms(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.o.get_delay_ms(), 0.0)

    def test_set_delay_ms(self):
        if False:
            i = 10
            return i + 15
        val = 50.0
        self.o.set_delay_ms(val)
        assert_almost_equal(self.o.get_delay_ms(), val)

    def test_get_threshold(self):
        if False:
            i = 10
            return i + 15
        assert_almost_equal(self.o.get_threshold(), 0.3)

    def test_set_threshold(self):
        if False:
            while True:
                i = 10
        val = 0.1
        self.o.set_threshold(val)
        assert_almost_equal(self.o.get_threshold(), val)

    def test_get_silence(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.o.get_silence(), -90.0)

    def test_set_silence(self):
        if False:
            print('Hello World!')
        val = -50.0
        self.o.set_silence(val)
        assert_almost_equal(self.o.get_silence(), val)

    def test_get_last(self):
        if False:
            return 10
        self.assertEqual(self.o.get_last(), 0.0)

    def test_get_last_s(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.o.get_last_s(), 0.0)

    def test_get_last_ms(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.o.get_last_ms(), 0.0)

    def test_get_period(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.o.get_period(), 0.0)

    def test_get_period_s(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.o.get_period_s(), 0.0)

    def test_get_last_tatum(self):
        if False:
            return 10
        self.assertEqual(self.o.get_last_tatum(), 0.0)

    def test_set_tatum_signature(self):
        if False:
            i = 10
            return i + 15
        self.o.set_tatum_signature(8)
        self.o.set_tatum_signature(64)
        self.o.set_tatum_signature(1)

    def test_set_wrong_tatum_signature(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self.o.set_tatum_signature(101)
        with self.assertRaises(ValueError):
            self.o.set_tatum_signature(0)
if __name__ == '__main__':
    main()