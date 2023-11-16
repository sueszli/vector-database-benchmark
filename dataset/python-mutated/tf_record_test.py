"""Tests for tf_record.TFRecordWriter and tf_record.tf_record_iterator."""
import gzip
import os
import random
import string
import zlib
import six
from tensorflow.python.framework import errors_impl
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat
TFRecordCompressionType = tf_record.TFRecordCompressionType
_TEXT = b"Gaily bedight,\n    A gallant knight,\n    In sunshine and in shadow,\n    Had journeyed long,\n    Singing a song,\n    In search of Eldorado.\n\n    But he grew old\n    This knight so bold\n    And o'er his heart a shadow\n    Fell as he found\n    No spot of ground\n    That looked like Eldorado.\n\n   And, as his strength\n   Failed him at length,\n   He met a pilgrim shadow\n   'Shadow,' said he,\n   'Where can it be\n   This land of Eldorado?'\n\n   'Over the Mountains\n    Of the Moon'\n    Down the Valley of the Shadow,\n    Ride, boldly ride,'\n    The shade replied,\n    'If you seek for Eldorado!'\n    "

class TFCompressionTestCase(test.TestCase):
    """TFCompression Test"""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TFCompressionTestCase, self).setUp()
        self._num_files = 2
        self._num_records = 7

    def _Record(self, f, r):
        if False:
            i = 10
            return i + 15
        return compat.as_bytes('Record %d of file %d' % (r, f))

    def _CreateFiles(self, options=None, prefix=''):
        if False:
            return 10
        filenames = []
        for i in range(self._num_files):
            name = prefix + 'tfrecord.%d.txt' % i
            records = [self._Record(i, j) for j in range(self._num_records)]
            fn = self._WriteRecordsToFile(records, name, options)
            filenames.append(fn)
        return filenames

    def _WriteRecordsToFile(self, records, name='tfrecord', options=None):
        if False:
            while True:
                i = 10
        fn = os.path.join(self.get_temp_dir(), name)
        with tf_record.TFRecordWriter(fn, options=options) as writer:
            for r in records:
                writer.write(r)
        return fn

    def _ZlibCompressFile(self, infile, name='tfrecord.z'):
        if False:
            i = 10
            return i + 15
        with open(infile, 'rb') as f:
            cdata = zlib.compress(f.read())
        zfn = os.path.join(self.get_temp_dir(), name)
        with open(zfn, 'wb') as f:
            f.write(cdata)
        return zfn

    def _GzipCompressFile(self, infile, name='tfrecord.gz'):
        if False:
            i = 10
            return i + 15
        with open(infile, 'rb') as f:
            cdata = f.read()
        gzfn = os.path.join(self.get_temp_dir(), name)
        with gzip.GzipFile(gzfn, 'wb') as f:
            f.write(cdata)
        return gzfn

    def _ZlibDecompressFile(self, infile, name='tfrecord'):
        if False:
            while True:
                i = 10
        with open(infile, 'rb') as f:
            cdata = zlib.decompress(f.read())
        fn = os.path.join(self.get_temp_dir(), name)
        with open(fn, 'wb') as f:
            f.write(cdata)
        return fn

    def _GzipDecompressFile(self, infile, name='tfrecord'):
        if False:
            while True:
                i = 10
        with gzip.GzipFile(infile, 'rb') as f:
            cdata = f.read()
        fn = os.path.join(self.get_temp_dir(), name)
        with open(fn, 'wb') as f:
            f.write(cdata)
        return fn

class TFRecordWriterTest(TFCompressionTestCase):
    """TFRecordWriter Test"""

    def _AssertFilesEqual(self, a, b, equal):
        if False:
            print('Hello World!')
        for (an, bn) in zip(a, b):
            with open(an, 'rb') as af, open(bn, 'rb') as bf:
                if equal:
                    self.assertEqual(af.read(), bf.read())
                else:
                    self.assertNotEqual(af.read(), bf.read())

    def _CompressionSizeDelta(self, records, options_a, options_b):
        if False:
            i = 10
            return i + 15
        'Validate compression with options_a and options_b and return size delta.\n\n    Compress records with options_a and options_b. Uncompress both compressed\n    files and assert that the contents match the original records. Finally\n    calculate how much smaller the file compressed with options_a was than the\n    file compressed with options_b.\n\n    Args:\n      records: The records to compress\n      options_a: First set of options to compress with, the baseline for size.\n      options_b: Second set of options to compress with.\n\n    Returns:\n      The difference in file size when using options_a vs options_b. A positive\n      value means options_a was a better compression than options_b. A negative\n      value means options_b had better compression than options_a.\n\n    '
        fn_a = self._WriteRecordsToFile(records, 'tfrecord_a', options=options_a)
        test_a = list(tf_record.tf_record_iterator(fn_a, options=options_a))
        self.assertEqual(records, test_a, options_a)
        fn_b = self._WriteRecordsToFile(records, 'tfrecord_b', options=options_b)
        test_b = list(tf_record.tf_record_iterator(fn_b, options=options_b))
        self.assertEqual(records, test_b, options_b)
        return os.path.getsize(fn_a) - os.path.getsize(fn_b)

    def testWriteReadZLibFiles(self):
        if False:
            print('Hello World!')
        'test Write Read ZLib Files'
        options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
        files = self._CreateFiles(options, prefix='uncompressed')
        zlib_files = [self._ZlibCompressFile(fn, 'tfrecord_%s.z' % i) for (i, fn) in enumerate(files)]
        self._AssertFilesEqual(files, zlib_files, False)
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        compressed_files = self._CreateFiles(options, prefix='compressed')
        self._AssertFilesEqual(compressed_files, zlib_files, True)
        uncompressed_files = [self._ZlibDecompressFile(fn, 'tfrecord_%s.z' % i) for (i, fn) in enumerate(compressed_files)]
        self._AssertFilesEqual(uncompressed_files, files, True)

    def testWriteReadGzipFiles(self):
        if False:
            while True:
                i = 10
        'test Write Read Gzip Files'
        options = tf_record.TFRecordOptions(TFRecordCompressionType.NONE)
        files = self._CreateFiles(options, prefix='uncompressed')
        gzip_files = [self._GzipCompressFile(fn, 'tfrecord_%s.gz' % i) for (i, fn) in enumerate(files)]
        self._AssertFilesEqual(files, gzip_files, False)
        options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
        compressed_files = self._CreateFiles(options, prefix='compressed')
        uncompressed_files = [self._GzipDecompressFile(fn, 'tfrecord_%s.gz' % i) for (i, fn) in enumerate(compressed_files)]
        self._AssertFilesEqual(uncompressed_files, files, True)

    def testNoCompressionType(self):
        if False:
            return 10
        'test No Compression Type'
        self.assertEqual('', tf_record.TFRecordOptions.get_compression_type_string(tf_record.TFRecordOptions()))
        self.assertEqual('', tf_record.TFRecordOptions.get_compression_type_string(tf_record.TFRecordOptions('')))
        with self.assertRaises(ValueError):
            tf_record.TFRecordOptions(5)
        with self.assertRaises(ValueError):
            tf_record.TFRecordOptions('BZ2')

    def testZlibCompressionType(self):
        if False:
            while True:
                i = 10
        'test Zlib Compression Type'
        zlib_t = tf_record.TFRecordCompressionType.ZLIB
        self.assertEqual('ZLIB', tf_record.TFRecordOptions.get_compression_type_string(tf_record.TFRecordOptions('ZLIB')))
        self.assertEqual('ZLIB', tf_record.TFRecordOptions.get_compression_type_string(tf_record.TFRecordOptions(zlib_t)))
        self.assertEqual('ZLIB', tf_record.TFRecordOptions.get_compression_type_string(tf_record.TFRecordOptions(tf_record.TFRecordOptions(zlib_t))))

    def testCompressionOptions(self):
        if False:
            print('Hello World!')
        'Create record with mix of random and repeated data to test compression on.'
        rnd = random.Random(123)
        random_record = compat.as_bytes(''.join((rnd.choice(string.digits) for _ in range(10000))))
        repeated_record = compat.as_bytes(_TEXT)
        for _ in range(10000):
            start_i = rnd.randint(0, len(_TEXT))
            length = rnd.randint(10, 200)
            repeated_record += _TEXT[start_i:start_i + length]
        records = [random_record, repeated_record, random_record]
        tests = [('compression_level', 2, 'LE'), ('compression_level', 6, 0), ('flush_mode', zlib.Z_FULL_FLUSH, 1), ('flush_mode', zlib.Z_NO_FLUSH, 0), ('input_buffer_size', 4096, 0), ('output_buffer_size', 4096, 0), ('window_bits', 8, -1), ('compression_strategy', zlib.Z_HUFFMAN_ONLY, -1), ('compression_strategy', zlib.Z_FILTERED, 'LE')]
        compression_type = tf_record.TFRecordCompressionType.ZLIB
        options_a = tf_record.TFRecordOptions(compression_type)
        for (prop, value, delta_sign) in tests:
            options_b = tf_record.TFRecordOptions(compression_type=compression_type, **{prop: value})
            delta = self._CompressionSizeDelta(records, options_a, options_b)
            if delta_sign == 'LE':
                self.assertLessEqual(delta, 0, "Setting {} = {}, file was {} smaller didn't match sign of {}".format(prop, value, delta, delta_sign))
            else:
                self.assertTrue(delta == 0 if delta_sign == 0 else delta // delta_sign > 0, "Setting {} = {}, file was {} smaller didn't match sign of {}".format(prop, value, delta, delta_sign))

class TFRecordWriterZlibTest(TFCompressionTestCase):
    """TFRecordWriter Zlib test"""

    def testZLibFlushRecord(self):
        if False:
            i = 10
            return i + 15
        'test ZLib Flush Record'
        original = [b'small record']
        fn = self._WriteRecordsToFile(original, 'small_record')
        with open(fn, 'rb') as h:
            buff = h.read()
        compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)
        output = b''
        for c in buff:
            if isinstance(c, int):
                c = six.int2byte(c)
            output += compressor.compress(c)
            output += compressor.flush(zlib.Z_FULL_FLUSH)
        output += compressor.flush(zlib.Z_FULL_FLUSH)
        output += compressor.flush(zlib.Z_FULL_FLUSH)
        output += compressor.flush(zlib.Z_FINISH)
        with open(fn, 'wb') as h:
            h.write(output)
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        actual = list(tf_record.tf_record_iterator(fn, options=options))
        self.assertEqual(actual, original)

    def testZlibReadWrite(self):
        if False:
            while True:
                i = 10
        'Verify that files produced are zlib compatible.'
        original = [b'foo', b'bar']
        fn = self._WriteRecordsToFile(original, 'zlib_read_write.tfrecord')
        zfn = self._ZlibCompressFile(fn, 'zlib_read_write.tfrecord.z')
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        actual = list(tf_record.tf_record_iterator(zfn, options=options))
        self.assertEqual(actual, original)

    def testZlibReadWriteLarge(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that writing large contents also works.'
        original = [_TEXT * 10240]
        fn = self._WriteRecordsToFile(original, 'zlib_read_write_large.tfrecord')
        zfn = self._ZlibCompressFile(fn, 'zlib_read_write_large.tfrecord.z')
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        actual = list(tf_record.tf_record_iterator(zfn, options=options))
        self.assertEqual(actual, original)

    def testGzipReadWrite(self):
        if False:
            print('Hello World!')
        'Verify that files produced are gzip compatible.'
        original = [b'foo', b'bar']
        fn = self._WriteRecordsToFile(original, 'gzip_read_write.tfrecord')
        gzfn = self._GzipCompressFile(fn, 'tfrecord.gz')
        options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
        actual = list(tf_record.tf_record_iterator(gzfn, options=options))
        self.assertEqual(actual, original)

class TFRecordIteratorTest(TFCompressionTestCase):
    """TFRecordIterator test"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TFRecordIteratorTest, self).setUp()
        self._num_records = 7

    def testIterator(self):
        if False:
            i = 10
            return i + 15
        'test Iterator'
        records = [self._Record(0, i) for i in range(self._num_records)]
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        fn = self._WriteRecordsToFile(records, 'compressed_records', options)
        reader = tf_record.tf_record_iterator(fn, options)
        for expected in records:
            record = next(reader)
            self.assertEqual(expected, record)
        with self.assertRaises(StopIteration):
            record = next(reader)

    def testWriteZlibRead(self):
        if False:
            while True:
                i = 10
        'Verify compression with TFRecordWriter is zlib library compatible.'
        original = [b'foo', b'bar']
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        fn = self._WriteRecordsToFile(original, 'write_zlib_read.tfrecord.z', options)
        zfn = self._ZlibDecompressFile(fn, 'write_zlib_read.tfrecord')
        actual = list(tf_record.tf_record_iterator(zfn))
        self.assertEqual(actual, original)

    def testWriteZlibReadLarge(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify compression for large records is zlib library compatible.'
        original = [_TEXT * 10240]
        options = tf_record.TFRecordOptions(TFRecordCompressionType.ZLIB)
        fn = self._WriteRecordsToFile(original, 'write_zlib_read_large.tfrecord.z', options)
        zfn = self._ZlibDecompressFile(fn, 'write_zlib_read_large.tfrecord')
        actual = list(tf_record.tf_record_iterator(zfn))
        self.assertEqual(actual, original)

    def testWriteGzipRead(self):
        if False:
            i = 10
            return i + 15
        original = [b'foo', b'bar']
        options = tf_record.TFRecordOptions(TFRecordCompressionType.GZIP)
        fn = self._WriteRecordsToFile(original, 'write_gzip_read.tfrecord.gz', options)
        gzfn = self._GzipDecompressFile(fn, 'write_gzip_read.tfrecord')
        actual = list(tf_record.tf_record_iterator(gzfn))
        self.assertEqual(actual, original)

    def testReadGrowingFile_preservesReadOffset(self):
        if False:
            return 10
        'Verify that tf_record_iterator preserves read offset even after EOF.\n\n    When a file is iterated to EOF, the iterator should raise StopIteration but\n    not actually close the reader. Then if later new data is appended, the\n    iterator should start returning that new data on the next call to next(),\n    preserving the read offset. This behavior is required by TensorBoard.\n    '
        fn = os.path.join(self.get_temp_dir(), 'file.tfrecord')
        with tf_record.TFRecordWriter(fn) as writer:
            writer.write(b'one')
            writer.write(b'two')
            writer.flush()
            iterator = tf_record.tf_record_iterator(fn)
            self.assertEqual(b'one', next(iterator))
            self.assertEqual(b'two', next(iterator))
            with self.assertRaises(StopIteration):
                next(iterator)
            with self.assertRaises(StopIteration):
                next(iterator)
            writer.write(b'three')
            writer.flush()
            self.assertEqual(b'three', next(iterator))
            with self.assertRaises(StopIteration):
                next(iterator)

    def testReadTruncatedFile_preservesReadOffset(self):
        if False:
            print('Hello World!')
        'Verify that tf_record_iterator throws an exception on bad TFRecords.\n\n    When a truncated record is completed, the iterator should return that new\n    record on the next attempt at iteration, preserving the read offset. This\n    behavior is required by TensorBoard.\n    '
        fn = os.path.join(self.get_temp_dir(), 'temp_file')
        with tf_record.TFRecordWriter(fn) as writer:
            writer.write(b'truncated')
        with open(fn, 'rb') as f:
            record_bytes = f.read()
        fn_truncated = os.path.join(self.get_temp_dir(), 'truncated_file')
        with tf_record.TFRecordWriter(fn_truncated) as writer:
            writer.write(b'good')
        with open(fn_truncated, 'ab', buffering=0) as f:
            f.write(record_bytes[:-1])
            iterator = tf_record.tf_record_iterator(fn_truncated)
            self.assertEqual(b'good', next(iterator))
            with self.assertRaises(errors_impl.DataLossError):
                next(iterator)
            with self.assertRaises(errors_impl.DataLossError):
                next(iterator)
            f.write(record_bytes[-1:])
            self.assertEqual(b'truncated', next(iterator))
            with self.assertRaises(StopIteration):
                next(iterator)

    def testReadReplacedFile_preservesReadOffset_afterReopen(self):
        if False:
            return 10
        'Verify that tf_record_iterator allows reopening at the same read offset.\n\n    In some cases, data will be logically "appended" to a file by replacing the\n    entire file with a new version that includes the additional data. For\n    example, this can happen with certain GCS implementations (since GCS has no\n    true append operation), or when using rsync without the `--inplace` option\n    to transfer snapshots of a growing file. Since the iterator retains a handle\n    to a stale version of the file, it won\'t return any of the new data.\n\n    To force this to happen, callers can check for a replaced file (e.g. via a\n    stat call that reflects an increased file size) and opt to close and reopen\n    the iterator. When iteration is next attempted, this should result in\n    reading from the newly opened file, while preserving the read offset. This\n    behavior is required by TensorBoard.\n    '

        def write_records_to_file(filename, records):
            if False:
                return 10
            writer = tf_record.TFRecordWriter(filename)
            for record in records:
                writer.write(record)
            writer.close()
        fn = os.path.join(self.get_temp_dir(), 'orig_file')
        write_records_to_file(fn, [b'one', b'two'])
        iterator = tf_record.tf_record_iterator(fn)
        self.assertEqual(b'one', next(iterator))
        self.assertEqual(b'two', next(iterator))
        with self.assertRaises(StopIteration):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)
        fn2 = os.path.join(self.get_temp_dir(), 'new_file')
        write_records_to_file(fn2, [b'one', b'two', b'three'])
        if os.name == 'nt':
            iterator.close()
        os.replace(fn2, fn)
        with self.assertRaises(StopIteration):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)
        iterator.close()
        iterator.reopen()
        self.assertEqual(b'three', next(iterator))
        with self.assertRaises(StopIteration):
            next(iterator)

class TFRecordRandomReaderTest(TFCompressionTestCase):

    def testRandomReaderReadingWorks(self):
        if False:
            return 10
        'Test read access to random offsets in the TFRecord file.'
        records = [self._Record(0, i) for i in range(self._num_records)]
        fn = self._WriteRecordsToFile(records, 'uncompressed_records')
        reader = tf_record.tf_record_random_reader(fn)
        offset = 0
        offsets = [offset]
        for i in range(self._num_records):
            (record, offset) = reader.read(offset)
            self.assertEqual(record, records[i])
            offsets.append(offset)
        with self.assertRaisesRegex(IndexError, 'Out of range.*offset'):
            reader.read(offset)
        for i in range(self._num_records - 1, 0, -1):
            (record, offset) = reader.read(offsets[i])
            self.assertEqual(offset, offsets[i + 1])
            self.assertEqual(record, records[i])

    def testRandomReaderThrowsErrorForInvalidOffset(self):
        if False:
            return 10
        records = [self._Record(0, i) for i in range(self._num_records)]
        fn = self._WriteRecordsToFile(records, 'uncompressed_records')
        reader = tf_record.tf_record_random_reader(fn)
        with self.assertRaisesRegex(errors_impl.DataLossError, 'corrupted record'):
            reader.read(1)

    def testClosingRandomReaderCausesErrorsForFurtherReading(self):
        if False:
            while True:
                i = 10
        records = [self._Record(0, i) for i in range(self._num_records)]
        fn = self._WriteRecordsToFile(records, 'uncompressed_records')
        reader = tf_record.tf_record_random_reader(fn)
        reader.close()
        with self.assertRaisesRegex(errors_impl.FailedPreconditionError, 'closed'):
            reader.read(0)

class TFRecordWriterCloseAndFlushTests(test.TestCase):
    """TFRecordWriter close and flush tests"""

    def setUp(self, compression_type=TFRecordCompressionType.NONE):
        if False:
            while True:
                i = 10
        super(TFRecordWriterCloseAndFlushTests, self).setUp()
        self._fn = os.path.join(self.get_temp_dir(), 'tf_record_writer_test.txt')
        self._options = tf_record.TFRecordOptions(compression_type)
        self._writer = tf_record.TFRecordWriter(self._fn, self._options)
        self._num_records = 20

    def _Record(self, r):
        if False:
            print('Hello World!')
        return compat.as_bytes('Record %d' % r)

    def testWriteAndLeaveOpen(self):
        if False:
            for i in range(10):
                print('nop')
        records = list(map(self._Record, range(self._num_records)))
        for record in records:
            self._writer.write(record)

    def testWriteAndRead(self):
        if False:
            for i in range(10):
                print('nop')
        records = list(map(self._Record, range(self._num_records)))
        for record in records:
            self._writer.write(record)
        self._writer.close()
        actual = list(tf_record.tf_record_iterator(self._fn, self._options))
        self.assertListEqual(actual, records)

    def testFlushAndRead(self):
        if False:
            i = 10
            return i + 15
        records = list(map(self._Record, range(self._num_records)))
        for record in records:
            self._writer.write(record)
        self._writer.flush()
        actual = list(tf_record.tf_record_iterator(self._fn, self._options))
        self.assertListEqual(actual, records)

    def testDoubleClose(self):
        if False:
            print('Hello World!')
        self._writer.write(self._Record(0))
        self._writer.close()
        self._writer.close()

    def testFlushAfterCloseIsError(self):
        if False:
            return 10
        self._writer.write(self._Record(0))
        self._writer.close()
        with self.assertRaises(errors_impl.FailedPreconditionError):
            self._writer.flush()

    def testWriteAfterCloseIsError(self):
        if False:
            i = 10
            return i + 15
        self._writer.write(self._Record(0))
        self._writer.close()
        with self.assertRaises(errors_impl.FailedPreconditionError):
            self._writer.write(self._Record(1))

class TFRecordWriterCloseAndFlushGzipTests(TFRecordWriterCloseAndFlushTests):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TFRecordWriterCloseAndFlushGzipTests, self).setUp(TFRecordCompressionType.GZIP)

class TFRecordWriterCloseAndFlushZlibTests(TFRecordWriterCloseAndFlushTests):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TFRecordWriterCloseAndFlushZlibTests, self).setUp(TFRecordCompressionType.ZLIB)
if __name__ == '__main__':
    test.main()