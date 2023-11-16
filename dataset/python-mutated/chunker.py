from io import BytesIO
from ..chunker import ChunkerFixed, Chunker, get_chunker, buzhash, buzhash_update
from ..constants import *
from . import BaseTestCase

def cf(chunks):
    if False:
        i = 10
        return i + 15
    'chunk filter'

    def _cf(chunk):
        if False:
            return 10
        if chunk.meta['allocation'] == CH_DATA:
            assert len(chunk.data) == chunk.meta['size']
            return bytes(chunk.data)
        if chunk.meta['allocation'] in (CH_HOLE, CH_ALLOC):
            assert chunk.data is None
            return chunk.meta['size']
        assert False, 'unexpected allocation value'
    return [_cf(chunk) for chunk in chunks]

class ChunkerFixedTestCase(BaseTestCase):

    def test_chunkify_just_blocks(self):
        if False:
            while True:
                i = 10
        data = b'foobar' * 1500
        chunker = ChunkerFixed(4096)
        parts = cf(chunker.chunkify(BytesIO(data)))
        self.assert_equal(parts, [data[0:4096], data[4096:8192], data[8192:]])

    def test_chunkify_header_and_blocks(self):
        if False:
            return 10
        data = b'foobar' * 1500
        chunker = ChunkerFixed(4096, 123)
        parts = cf(chunker.chunkify(BytesIO(data)))
        self.assert_equal(parts, [data[0:123], data[123:123 + 4096], data[123 + 4096:123 + 8192], data[123 + 8192:]])

    def test_chunkify_just_blocks_fmap_complete(self):
        if False:
            return 10
        data = b'foobar' * 1500
        chunker = ChunkerFixed(4096)
        fmap = [(0, 4096, True), (4096, 8192, True), (8192, 99999999, True)]
        parts = cf(chunker.chunkify(BytesIO(data), fmap=fmap))
        self.assert_equal(parts, [data[0:4096], data[4096:8192], data[8192:]])

    def test_chunkify_header_and_blocks_fmap_complete(self):
        if False:
            print('Hello World!')
        data = b'foobar' * 1500
        chunker = ChunkerFixed(4096, 123)
        fmap = [(0, 123, True), (123, 4096, True), (123 + 4096, 4096, True), (123 + 8192, 4096, True)]
        parts = cf(chunker.chunkify(BytesIO(data), fmap=fmap))
        self.assert_equal(parts, [data[0:123], data[123:123 + 4096], data[123 + 4096:123 + 8192], data[123 + 8192:]])

    def test_chunkify_header_and_blocks_fmap_zeros(self):
        if False:
            while True:
                i = 10
        data = b'H' * 123 + b'_' * 4096 + b'X' * 4096 + b'_' * 4096
        chunker = ChunkerFixed(4096, 123)
        fmap = [(0, 123, True), (123, 4096, False), (123 + 4096, 4096, True), (123 + 8192, 4096, False)]
        parts = cf(chunker.chunkify(BytesIO(data), fmap=fmap))
        self.assert_equal(parts, [data[0:123], 4096, data[123 + 4096:123 + 8192], 4096])

    def test_chunkify_header_and_blocks_fmap_partial(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'H' * 123 + b'_' * 4096 + b'X' * 4096 + b'_' * 4096
        chunker = ChunkerFixed(4096, 123)
        fmap = [(0, 123, True), (123 + 4096, 4096, True)]
        parts = cf(chunker.chunkify(BytesIO(data), fmap=fmap))
        self.assert_equal(parts, [data[0:123], data[123 + 4096:123 + 8192]])

class ChunkerTestCase(BaseTestCase):

    def test_chunkify(self):
        if False:
            i = 10
            return i + 15
        data = b'0' * int(1.5 * (1 << CHUNK_MAX_EXP)) + b'Y'
        parts = cf(Chunker(0, 1, CHUNK_MAX_EXP, 2, 2).chunkify(BytesIO(data)))
        self.assert_equal(len(parts), 2)
        self.assert_equal(b''.join(parts), data)
        self.assert_equal(cf(Chunker(0, 1, CHUNK_MAX_EXP, 2, 2).chunkify(BytesIO(b''))), [])
        self.assert_equal(cf(Chunker(0, 1, CHUNK_MAX_EXP, 2, 2).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'fooba', b'rboobaz', b'fooba', b'rboobaz', b'fooba', b'rboobaz'])
        self.assert_equal(cf(Chunker(1, 1, CHUNK_MAX_EXP, 2, 2).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'fo', b'obarb', b'oob', b'azf', b'oobarb', b'oob', b'azf', b'oobarb', b'oobaz'])
        self.assert_equal(cf(Chunker(2, 1, CHUNK_MAX_EXP, 2, 2).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foob', b'ar', b'boobazfoob', b'ar', b'boobazfoob', b'ar', b'boobaz'])
        self.assert_equal(cf(Chunker(0, 2, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foobarboobaz' * 3])
        self.assert_equal(cf(Chunker(1, 2, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foobar', b'boobazfo', b'obar', b'boobazfo', b'obar', b'boobaz'])
        self.assert_equal(cf(Chunker(2, 2, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foob', b'arboobaz', b'foob', b'arboobaz', b'foob', b'arboobaz'])
        self.assert_equal(cf(Chunker(0, 3, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foobarboobaz' * 3])
        self.assert_equal(cf(Chunker(1, 3, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foobarbo', b'obazfoobar', b'boobazfo', b'obarboobaz'])
        self.assert_equal(cf(Chunker(2, 3, CHUNK_MAX_EXP, 2, 3).chunkify(BytesIO(b'foobarboobaz' * 3))), [b'foobarboobaz', b'foobarboobaz', b'foobarboobaz'])

    def test_buzhash(self):
        if False:
            while True:
                i = 10
        self.assert_equal(buzhash(b'abcdefghijklmnop', 0), 3795437769)
        self.assert_equal(buzhash(b'abcdefghijklmnop', 1), 3795400502)
        self.assert_equal(buzhash(b'abcdefghijklmnop', 1), buzhash_update(buzhash(b'Xabcdefghijklmno', 1), ord('X'), ord('p'), 16, 1))
        self.assert_equal(buzhash(b'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz', 0), 566521248)

    def test_small_reads(self):
        if False:
            while True:
                i = 10

        class SmallReadFile:
            input = b'a' * (20 + 1)

            def read(self, nbytes):
                if False:
                    print('Hello World!')
                self.input = self.input[:-1]
                return self.input[:1]
        chunker = get_chunker(*CHUNKER_PARAMS, seed=0)
        reconstructed = b''.join(cf(chunker.chunkify(SmallReadFile())))
        assert reconstructed == b'a' * 20