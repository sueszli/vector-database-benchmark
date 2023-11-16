from unittest import TestCase
from mycroft.client.speech.data_structures import RollingMean, CyclicAudioBuffer

class TestRollingMean(TestCase):

    def test_before_rolling(self):
        if False:
            print('Hello World!')
        mean = RollingMean(10)
        for i in range(5):
            mean.append_sample(i)
        self.assertEqual(mean.value, 2)
        for i in range(5):
            mean.append_sample(i)
        self.assertEqual(mean.value, 2)

    def test_during_rolling(self):
        if False:
            i = 10
            return i + 15
        mean = RollingMean(10)
        for _ in range(10):
            mean.append_sample(5)
        self.assertEqual(mean.value, 5)
        for _ in range(5):
            mean.append_sample(1)
        self.assertAlmostEqual(mean.value, 3)
        for _ in range(5):
            mean.append_sample(2)
        self.assertAlmostEqual(mean.value, 1.5)

class TestCyclicBuffer(TestCase):

    def test_init(self):
        if False:
            print('Hello World!')
        buff = CyclicAudioBuffer(16, b'abc')
        self.assertEqual(buff.get(), b'abc')
        self.assertEqual(len(buff), 3)

    def test_init_larger_inital_data(self):
        if False:
            while True:
                i = 10
        size = 16
        buff = CyclicAudioBuffer(size, b'a' * (size + 3))
        self.assertEqual(buff.get(), b'a' * size)

    def test_append_with_room_left(self):
        if False:
            for i in range(10):
                print('nop')
        buff = CyclicAudioBuffer(16, b'abc')
        buff.append(b'def')
        self.assertEqual(buff.get(), b'abcdef')

    def test_append_with_full(self):
        if False:
            print('Hello World!')
        buff = CyclicAudioBuffer(3, b'abc')
        buff.append(b'de')
        self.assertEqual(buff.get(), b'cde')
        self.assertEqual(len(buff), 3)

    def test_get_last(self):
        if False:
            print('Hello World!')
        buff = CyclicAudioBuffer(3, b'abcdef')
        self.assertEqual(buff.get_last(3), b'def')

    def test_get_item(self):
        if False:
            print('Hello World!')
        buff = CyclicAudioBuffer(6, b'abcdef')
        self.assertEqual(buff[:], b'abcdef')