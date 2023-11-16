import base64
import hashlib
import io
import os
import tempfile
import zlib
from ..hashindex import NSIndex, ChunkIndex
from ..crypto.file_integrity import IntegrityCheckedFile, FileIntegrityError
from . import BaseTestCase, unopened_tempfile

def H(x):
    if False:
        return 10
    return bytes('%-0.32d' % x, 'ascii')

def H2(x):
    if False:
        return 10
    return hashlib.sha256(H(x)).digest()

class HashIndexTestCase(BaseTestCase):

    def _generic_test(self, cls, make_value, sha):
        if False:
            print('Hello World!')
        idx = cls()
        self.assert_equal(len(idx), 0)
        for x in range(100):
            idx[H(x)] = make_value(x)
        self.assert_equal(len(idx), 100)
        for x in range(100):
            self.assert_equal(idx[H(x)], make_value(x))
        for x in range(100):
            idx[H(x)] = make_value(x * 2)
        self.assert_equal(len(idx), 100)
        for x in range(100):
            self.assert_equal(idx[H(x)], make_value(x * 2))
        for x in range(50):
            del idx[H(x)]
        for x in range(50, 100):
            assert H(x) in idx
        for x in range(50):
            assert H(x) not in idx
        for x in range(50):
            self.assert_raises(KeyError, idx.__delitem__, H(x))
        self.assert_equal(len(idx), 50)
        with unopened_tempfile() as filepath:
            idx.write(filepath)
            del idx
            with open(filepath, 'rb') as fd:
                self.assert_equal(hashlib.sha256(fd.read()).hexdigest(), sha)
            idx = cls.read(filepath)
            self.assert_equal(len(idx), 50)
            for x in range(50, 100):
                self.assert_equal(idx[H(x)], make_value(x * 2))
            idx.clear()
            self.assert_equal(len(idx), 0)
            idx.write(filepath)
            del idx
            self.assert_equal(len(cls.read(filepath)), 0)
        idx = cls()
        idx.setdefault(H(0), make_value(42))
        assert H(0) in idx
        assert idx[H(0)] == make_value(42)
        idx.setdefault(H(0), make_value(23))
        assert H(0) in idx
        assert idx[H(0)] == make_value(42)
        assert idx.setdefault(H(1), make_value(23)) == make_value(23)
        assert idx.setdefault(H(0), make_value(23)) == make_value(42)
        del idx

    def test_nsindex(self):
        if False:
            return 10
        self._generic_test(NSIndex, lambda x: (x, x, x), '0d7880dbe02b64f03c471e60e193a1333879b4f23105768b10c9222accfeac5e')

    def test_chunkindex(self):
        if False:
            i = 10
            return i + 15
        self._generic_test(ChunkIndex, lambda x: (x, x), '5915fcf986da12e5f3ac68e05242b9c729e6101b0460b1d4e4a9e9f7cdf1b7da')

    def test_resize(self):
        if False:
            return 10
        n = 2000
        with unopened_tempfile() as filepath:
            idx = NSIndex()
            idx.write(filepath)
            initial_size = os.path.getsize(filepath)
            self.assert_equal(len(idx), 0)
            for x in range(n):
                idx[H(x)] = (x, x, x, x)
            idx.write(filepath)
            assert initial_size < os.path.getsize(filepath)
            for x in range(n):
                del idx[H(x)]
            self.assert_equal(len(idx), 0)
            idx.write(filepath)
            self.assert_equal(initial_size, os.path.getsize(filepath))

    def test_iteritems(self):
        if False:
            print('Hello World!')
        idx = NSIndex()
        for x in range(100):
            idx[H(x)] = (x, x, x, x)
        iterator = idx.iteritems()
        all = list(iterator)
        self.assert_equal(len(all), 100)
        self.assert_raises(StopIteration, next, iterator)
        second_half = list(idx.iteritems(marker=all[49][0]))
        self.assert_equal(len(second_half), 50)
        self.assert_equal(second_half, all[50:])

    def test_chunkindex_merge(self):
        if False:
            return 10
        idx1 = ChunkIndex()
        idx1[H(1)] = (1, 100)
        idx1[H(2)] = (2, 200)
        idx1[H(3)] = (3, 300)
        idx2 = ChunkIndex()
        idx2[H(1)] = (4, 100)
        idx2[H(2)] = (5, 200)
        idx2[H(4)] = (6, 400)
        idx1.merge(idx2)
        assert idx1[H(1)] == (5, 100)
        assert idx1[H(2)] == (7, 200)
        assert idx1[H(3)] == (3, 300)
        assert idx1[H(4)] == (6, 400)

    def test_chunkindex_summarize(self):
        if False:
            i = 10
            return i + 15
        idx = ChunkIndex()
        idx[H(1)] = (1, 1000)
        idx[H(2)] = (2, 2000)
        idx[H(3)] = (3, 3000)
        (size, unique_size, unique_chunks, chunks) = idx.summarize()
        assert size == 1000 + 2 * 2000 + 3 * 3000
        assert unique_size == 1000 + 2000 + 3000
        assert chunks == 1 + 2 + 3
        assert unique_chunks == 3

    def test_flags(self):
        if False:
            while True:
                i = 10
        idx = NSIndex()
        key = H(0)
        self.assert_raises(KeyError, idx.flags, key, 0)
        idx[key] = (0, 0, 0)
        self.assert_equal(idx.flags(key, mask=3), 0)
        idx.flags(key, mask=1, value=1)
        self.assert_equal(idx.flags(key, mask=1), 1)
        idx.flags(key, mask=2, value=2)
        self.assert_equal(idx.flags(key, mask=2), 2)
        self.assert_equal(idx.flags(key, mask=3), 3)
        idx.flags(key, mask=2, value=0)
        self.assert_equal(idx.flags(key, mask=2), 0)
        idx.flags(key, mask=1, value=0)
        self.assert_equal(idx.flags(key, mask=1), 0)
        self.assert_equal(idx.flags(key, mask=3), 0)

    def test_flags_iteritems(self):
        if False:
            for i in range(10):
                print('nop')
        idx = NSIndex()
        keys_flagged0 = {H(i) for i in (1, 2, 3, 42)}
        keys_flagged1 = {H(i) for i in (11, 12, 13, 142)}
        keys_flagged2 = {H(i) for i in (21, 22, 23, 242)}
        keys_flagged3 = {H(i) for i in (31, 32, 33, 342)}
        for key in keys_flagged0:
            idx[key] = (0, 0, 0)
            idx.flags(key, mask=3, value=0)
        for key in keys_flagged1:
            idx[key] = (0, 0, 0)
            idx.flags(key, mask=3, value=1)
        for key in keys_flagged2:
            idx[key] = (0, 0, 0)
            idx.flags(key, mask=3, value=2)
        for key in keys_flagged3:
            idx[key] = (0, 0, 0)
            idx.flags(key, mask=3, value=3)
        k_all = {k for (k, v) in idx.iteritems()}
        self.assert_equal(k_all, keys_flagged0 | keys_flagged1 | keys_flagged2 | keys_flagged3)
        k0 = {k for (k, v) in idx.iteritems(mask=3, value=0)}
        self.assert_equal(k0, keys_flagged0)
        k1 = {k for (k, v) in idx.iteritems(mask=3, value=1)}
        self.assert_equal(k1, keys_flagged1)
        k1 = {k for (k, v) in idx.iteritems(mask=3, value=2)}
        self.assert_equal(k1, keys_flagged2)
        k1 = {k for (k, v) in idx.iteritems(mask=3, value=3)}
        self.assert_equal(k1, keys_flagged3)
        k1 = {k for (k, v) in idx.iteritems(mask=1, value=1)}
        self.assert_equal(k1, keys_flagged1 | keys_flagged3)
        k1 = {k for (k, v) in idx.iteritems(mask=1, value=0)}
        self.assert_equal(k1, keys_flagged0 | keys_flagged2)

class HashIndexExtraTestCase(BaseTestCase):
    """These tests are separate because they should not become part of the selftest."""

    def test_chunk_indexer(self):
        if False:
            print('Hello World!')
        key_count = int(65537 * ChunkIndex.MAX_LOAD_FACTOR) - 10
        index = ChunkIndex(key_count)
        all_keys = [hashlib.sha256(H(k)).digest() for k in range(key_count)]
        (keys, to_delete_keys) = (all_keys[0:2 * key_count // 3], all_keys[2 * key_count // 3:])
        for (i, key) in enumerate(keys):
            index[key] = (i, i)
        for (i, key) in enumerate(to_delete_keys):
            index[key] = (i, i)
        for key in to_delete_keys:
            del index[key]
        for (i, key) in enumerate(keys):
            assert index[key] == (i, i)
        for key in to_delete_keys:
            assert index.get(key) is None
        for key in keys:
            del index[key]
        assert list(index.iteritems()) == []

class HashIndexSizeTestCase(BaseTestCase):

    def test_size_on_disk(self):
        if False:
            i = 10
            return i + 15
        idx = ChunkIndex()
        assert idx.size() == 1024 + 1031 * (32 + 2 * 4)

    def test_size_on_disk_accurate(self):
        if False:
            for i in range(10):
                print('nop')
        idx = ChunkIndex()
        for i in range(1234):
            idx[H(i)] = (i, i ** 2)
        with unopened_tempfile() as filepath:
            idx.write(filepath)
            size = os.path.getsize(filepath)
        assert idx.size() == size

class HashIndexRefcountingTestCase(BaseTestCase):

    def test_chunkindex_limit(self):
        if False:
            return 10
        idx = ChunkIndex()
        idx[H(1)] = (ChunkIndex.MAX_VALUE - 1, 1)
        for i in range(5):
            (refcount, *_) = idx.incref(H(1))
            assert refcount == ChunkIndex.MAX_VALUE
        for i in range(5):
            (refcount, *_) = idx.decref(H(1))
            assert refcount == ChunkIndex.MAX_VALUE

    def _merge(self, refcounta, refcountb):
        if False:
            while True:
                i = 10

        def merge(refcount1, refcount2):
            if False:
                return 10
            idx1 = ChunkIndex()
            idx1[H(1)] = (refcount1, 1)
            idx2 = ChunkIndex()
            idx2[H(1)] = (refcount2, 1)
            idx1.merge(idx2)
            (refcount, *_) = idx1[H(1)]
            return refcount
        result = merge(refcounta, refcountb)
        assert result == merge(refcountb, refcounta)
        return result

    def test_chunkindex_merge_limit1(self):
        if False:
            return 10
        half = ChunkIndex.MAX_VALUE // 2
        assert self._merge(half, half) == ChunkIndex.MAX_VALUE - 1

    def test_chunkindex_merge_limit2(self):
        if False:
            return 10
        assert self._merge(3000000000, 2000000000) == ChunkIndex.MAX_VALUE

    def test_chunkindex_merge_limit3(self):
        if False:
            i = 10
            return i + 15
        half = ChunkIndex.MAX_VALUE // 2
        assert self._merge(half + 1, half) == ChunkIndex.MAX_VALUE

    def test_chunkindex_merge_limit4(self):
        if False:
            i = 10
            return i + 15
        half = ChunkIndex.MAX_VALUE // 2
        assert self._merge(half + 2, half) == ChunkIndex.MAX_VALUE
        assert self._merge(half + 1, half + 1) == ChunkIndex.MAX_VALUE

    def test_chunkindex_add(self):
        if False:
            i = 10
            return i + 15
        idx1 = ChunkIndex()
        idx1.add(H(1), 5, 6)
        assert idx1[H(1)] == (5, 6)
        idx1.add(H(1), 1, 2)
        assert idx1[H(1)] == (6, 2)

    def test_incref_limit(self):
        if False:
            i = 10
            return i + 15
        idx1 = ChunkIndex()
        idx1[H(1)] = (ChunkIndex.MAX_VALUE, 6)
        idx1.incref(H(1))
        (refcount, *_) = idx1[H(1)]
        assert refcount == ChunkIndex.MAX_VALUE

    def test_decref_limit(self):
        if False:
            while True:
                i = 10
        idx1 = ChunkIndex()
        idx1[H(1)] = (ChunkIndex.MAX_VALUE, 6)
        idx1.decref(H(1))
        (refcount, *_) = idx1[H(1)]
        assert refcount == ChunkIndex.MAX_VALUE

    def test_decref_zero(self):
        if False:
            return 10
        idx1 = ChunkIndex()
        idx1[H(1)] = (0, 0)
        with self.assert_raises(AssertionError):
            idx1.decref(H(1))

    def test_incref_decref(self):
        if False:
            print('Hello World!')
        idx1 = ChunkIndex()
        idx1.add(H(1), 5, 6)
        assert idx1[H(1)] == (5, 6)
        idx1.incref(H(1))
        assert idx1[H(1)] == (6, 6)
        idx1.decref(H(1))
        assert idx1[H(1)] == (5, 6)

    def test_setitem_raises(self):
        if False:
            while True:
                i = 10
        idx1 = ChunkIndex()
        with self.assert_raises(AssertionError):
            idx1[H(1)] = (ChunkIndex.MAX_VALUE + 1, 0)

    def test_keyerror(self):
        if False:
            for i in range(10):
                print('nop')
        idx = ChunkIndex()
        with self.assert_raises(KeyError):
            idx.incref(H(1))
        with self.assert_raises(KeyError):
            idx.decref(H(1))
        with self.assert_raises(KeyError):
            idx[H(1)]
        with self.assert_raises(OverflowError):
            idx.add(H(1), -1, 0)

class HashIndexDataTestCase(BaseTestCase):
    HASHINDEX = b'eJzt0DEKgwAMQNFoBXsMj9DqDUQoToKTR3Hzwr2DZi+0HS19HwIZHhnST/OjHYeljIhLTl1FVDlN7teQ9M/tGcdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHMdxHPfqbu+7F2nKz67Nc9sX97r1+Rt/4TiO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO4ziO487lDoRvHEk='

    def _serialize_hashindex(self, idx):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tempdir:
            file = os.path.join(tempdir, 'idx')
            idx.write(file)
            with open(file, 'rb') as f:
                return self._pack(f.read())

    def _deserialize_hashindex(self, bytestring):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tempdir:
            file = os.path.join(tempdir, 'idx')
            with open(file, 'wb') as f:
                f.write(self._unpack(bytestring))
            return ChunkIndex.read(file)

    def _pack(self, bytestring):
        if False:
            return 10
        return base64.b64encode(zlib.compress(bytestring))

    def _unpack(self, bytestring):
        if False:
            for i in range(10):
                print('nop')
        return zlib.decompress(base64.b64decode(bytestring))

    def test_identical_creation(self):
        if False:
            print('Hello World!')
        idx1 = ChunkIndex()
        idx1[H(1)] = (1, 2)
        idx1[H(2)] = (2 ** 31 - 1, 0)
        idx1[H(3)] = (4294962296, 0)
        serialized = self._serialize_hashindex(idx1)
        assert self._unpack(serialized) == self._unpack(self.HASHINDEX)

    def test_read_known_good(self):
        if False:
            for i in range(10):
                print('nop')
        idx1 = self._deserialize_hashindex(self.HASHINDEX)
        assert idx1[H(1)] == (1, 2)
        assert idx1[H(2)] == (2 ** 31 - 1, 0)
        assert idx1[H(3)] == (4294962296, 0)
        idx2 = ChunkIndex()
        idx2[H(3)] = (2 ** 32 - 123456, 6)
        idx1.merge(idx2)
        assert idx1[H(3)] == (ChunkIndex.MAX_VALUE, 6)

class HashIndexIntegrityTestCase(HashIndexDataTestCase):

    def write_integrity_checked_index(self, tempdir):
        if False:
            while True:
                i = 10
        idx = self._deserialize_hashindex(self.HASHINDEX)
        file = os.path.join(tempdir, 'idx')
        with IntegrityCheckedFile(path=file, write=True) as fd:
            idx.write(fd)
        integrity_data = fd.integrity_data
        assert 'final' in integrity_data
        assert 'HashHeader' in integrity_data
        return (file, integrity_data)

    def test_integrity_checked_file(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tempdir:
            (file, integrity_data) = self.write_integrity_checked_index(tempdir)
            with open(file, 'r+b') as fd:
                fd.write(b'Foo')
            with self.assert_raises(FileIntegrityError):
                with IntegrityCheckedFile(path=file, write=False, integrity_data=integrity_data) as fd:
                    ChunkIndex.read(fd)

class HashIndexCompactTestCase(HashIndexDataTestCase):

    def index(self, num_entries, num_buckets, num_empty):
        if False:
            return 10
        index_data = io.BytesIO()
        index_data.write(b'BORG2IDX')
        index_data.write(2 .to_bytes(4, 'little'))
        index_data.write(num_entries.to_bytes(4, 'little'))
        index_data.write(num_buckets.to_bytes(4, 'little'))
        index_data.write(num_empty.to_bytes(4, 'little'))
        index_data.write(32 .to_bytes(4, 'little'))
        index_data.write((3 * 4).to_bytes(4, 'little'))
        index_data.write(bytes(1024 - 32))
        self.index_data = index_data

    def index_from_data(self):
        if False:
            i = 10
            return i + 15
        self.index_data.seek(0)
        index = ChunkIndex.read(self.index_data, permit_compact=True)
        return index

    def write_entry(self, key, *values):
        if False:
            print('Hello World!')
        self.index_data.write(key)
        for value in values:
            self.index_data.write(value.to_bytes(4, 'little'))

    def write_empty(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.write_entry(key, 4294967295, 0, 0)

    def write_deleted(self, key):
        if False:
            return 10
        self.write_entry(key, 4294967294, 0, 0)

    def compare_indexes(self, idx1, idx2):
        if False:
            print('Hello World!')
        'Check that the two hash tables contain the same data.  idx1\n        is allowed to have "mis-filed" entries, because we only need to\n        iterate over it.  But idx2 needs to support lookup.'
        for (k, v) in idx1.iteritems():
            assert v == idx2[k]
        assert len(idx1) == len(idx2)

    def compare_compact(self, layout):
        if False:
            print('Hello World!')
        "A generic test of a hashindex with the specified layout.  layout should\n        be a string consisting only of the characters '*' (filled), 'D' (deleted)\n        and 'E' (empty).\n        "
        num_buckets = len(layout)
        num_empty = layout.count('E')
        num_entries = layout.count('*')
        self.index(num_entries=num_entries, num_buckets=num_buckets, num_empty=num_empty)
        k = 0
        for c in layout:
            if c == 'D':
                self.write_deleted(H2(k))
            elif c == 'E':
                self.write_empty(H2(k))
            else:
                assert c == '*'
                self.write_entry(H2(k), 3 * k + 1, 3 * k + 2, 3 * k + 3)
            k += 1
        idx = self.index_from_data()
        cpt = self.index_from_data()
        cpt.compact()
        assert idx.size() == 1024 + num_buckets * (32 + 3 * 4)
        assert cpt.size() == 1024 + num_entries * (32 + 3 * 4)
        self.compare_indexes(idx, cpt)

    def test_simple(self):
        if False:
            while True:
                i = 10
        self.compare_compact('*DE**E')

    def test_first_empty(self):
        if False:
            return 10
        self.compare_compact('D*E**E')

    def test_last_used(self):
        if False:
            print('Hello World!')
        self.compare_compact('D*E*E*')

    def test_too_few_empty_slots(self):
        if False:
            return 10
        self.compare_compact('D**EE*')

    def test_empty(self):
        if False:
            print('Hello World!')
        self.compare_compact('DEDEED')

    def test_num_buckets_zero(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_compact('')

    def test_already_compact(self):
        if False:
            for i in range(10):
                print('nop')
        self.compare_compact('***')

    def test_all_at_front(self):
        if False:
            while True:
                i = 10
        self.compare_compact('*DEEED')
        self.compare_compact('**DEED')
        self.compare_compact('***EED')
        self.compare_compact('****ED')
        self.compare_compact('*****D')

    def test_all_at_back(self):
        if False:
            i = 10
            return i + 15
        self.compare_compact('EDEEE*')
        self.compare_compact('DEDE**')
        self.compare_compact('DED***')
        self.compare_compact('ED****')
        self.compare_compact('D*****')

    def test_merge(self):
        if False:
            return 10
        master = ChunkIndex()
        idx1 = ChunkIndex()
        idx1[H(1)] = (1, 100)
        idx1[H(2)] = (2, 200)
        idx1[H(3)] = (3, 300)
        idx1.compact()
        assert idx1.size() == 1024 + 3 * (32 + 2 * 4)
        master.merge(idx1)
        self.compare_indexes(idx1, master)

class NSIndexTestCase(BaseTestCase):

    def test_nsindex_segment_limit(self):
        if False:
            while True:
                i = 10
        idx = NSIndex()
        with self.assert_raises(AssertionError):
            idx[H(1)] = (NSIndex.MAX_VALUE + 1, 0, 0, 0)
        assert H(1) not in idx
        idx[H(2)] = (NSIndex.MAX_VALUE, 0, 0, 0)
        assert H(2) in idx

class AllIndexTestCase(BaseTestCase):

    def test_max_load_factor(self):
        if False:
            for i in range(10):
                print('nop')
        assert NSIndex.MAX_LOAD_FACTOR < 1.0
        assert ChunkIndex.MAX_LOAD_FACTOR < 1.0

class IndexCorruptionTestCase(BaseTestCase):

    def test_bug_4829(self):
        if False:
            return 10
        from struct import pack

        def HH(x, y, z):
            if False:
                print('Hello World!')
            return pack('<IIIIIIII', x, y, z, 0, 0, 0, 0, 0)
        idx = NSIndex()
        for y in range(700):
            idx[HH(0, y, 0)] = (0, y, 0)
        assert idx.size() == 1024 + 1031 * 48
        for y in range(400):
            del idx[HH(0, y, 0)]
        for y in range(330):
            idx[HH(600, y, 0)] = (600, y, 0)
        assert [idx.get(HH(0, y, 0)) for y in range(400, 700)] == [(0, y, 0) for y in range(400, 700)]
        assert [HH(0, y, 0) in idx for y in range(400)] == [False for y in range(400)]
        assert [idx.get(HH(600, y, 0)) for y in range(330)] == [(600, y, 0) for y in range(330)]