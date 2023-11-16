import typing
import unittest
from hashlib import md5

class TestMD5(unittest.TestCase):

    def test_md5(self):
        if False:
            return 10
        raw_input: bytes = bytes([40, 191, 78, 94, 78, 117, 138, 65, 100, 0, 78, 86, 255, 250, 1, 8, 46, 46, 0, 182, 208, 104, 62, 128, 47, 12, 169, 254, 100, 83, 105, 122])
        h = md5()
        h.update(raw_input)
        raw_output: typing.List[int] = [x - 256 if x >= 128 else x for x in h.digest()]
        assert raw_output == [81, 33, 71, -71, -98, 113, -27, 117, 120, 7, 121, -95, -74, 69, 20, 72]

    def test_50_repeats_of_md5(self):
        if False:
            i = 10
            return i + 15
        raw_input: bytes = bytes([40, 191, 78, 94, 78, 117, 138, 65, 100, 0, 78, 86, 255, 250, 1, 8, 46, 46, 0, 182, 208, 104, 62, 128, 47, 12, 169, 254, 100, 83, 105, 122])
        h = md5()
        h.update(raw_input)
        raw_output: typing.List[int] = [x - 256 if x >= 128 else x for x in h.digest()]
        assert raw_output == [81, 33, 71, -71, -98, 113, -27, 117, 120, 7, 121, -95, -74, 69, 20, 72]
        prev: bytes = h.digest()
        for _ in range(0, 50):
            h = md5()
            h.update(prev)
            prev = h.digest()[0:16]
        raw_output: typing.List[int] = [x - 256 if x >= 128 else x for x in prev]
        assert raw_output == [90, 0, 52, 79, 64, -48, -91, -59, 43, 22, 11, -125, 14, 110, 8, 110]