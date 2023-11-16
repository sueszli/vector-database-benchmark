"""Tests for our estimation of compressed content."""
import zlib
import hashlib
from bzrlib import estimate_compressed_size, tests

class TestZLibEstimator(tests.TestCase):

    def get_slightly_random_content(self, length, seed=''):
        if False:
            while True:
                i = 10
        'We generate some hex-data that can be seeded.\n\n        The output should be deterministic, but the data stream is effectively\n        random.\n        '
        h = hashlib.md5(seed)
        hex_content = []
        count = 0
        while count < length:
            b = h.hexdigest()
            hex_content.append(b)
            h.update(b)
            count += len(b)
        return ''.join(hex_content)[:length]

    def test_adding_content(self):
        if False:
            return 10
        ze = estimate_compressed_size.ZLibEstimator(32000)
        raw_data = self.get_slightly_random_content(60000)
        block_size = 1000
        for start in xrange(0, len(raw_data), block_size):
            ze.add_content(raw_data[start:start + block_size])
            if ze.full():
                break
        self.assertTrue(54000 <= start <= 58000, 'Unexpected amount of raw data added: %d bytes' % (start,))
        raw_comp = zlib.compress(raw_data[:start])
        self.assertTrue(31000 < len(raw_comp) < 33000, 'Unexpected compressed size: %d bytes' % (len(raw_comp),))

    def test_adding_more_content(self):
        if False:
            while True:
                i = 10
        ze = estimate_compressed_size.ZLibEstimator(64000)
        raw_data = self.get_slightly_random_content(150000)
        block_size = 1000
        for start in xrange(0, len(raw_data), block_size):
            ze.add_content(raw_data[start:start + block_size])
            if ze.full():
                break
        self.assertTrue(110000 <= start <= 114000, 'Unexpected amount of raw data added: %d bytes' % (start,))
        raw_comp = zlib.compress(raw_data[:start])
        self.assertTrue(63000 < len(raw_comp) < 65000, 'Unexpected compressed size: %d bytes' % (len(raw_comp),))