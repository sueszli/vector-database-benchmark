"""Tests for DecodeRaw op from parsing_ops."""
import gzip
import io
import zlib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

class DecodeCompressedOpTest(test.TestCase):

    def _compress(self, bytes_in, compression_type):
        if False:
            while True:
                i = 10
        if not compression_type:
            return bytes_in
        elif compression_type == 'ZLIB':
            return zlib.compress(bytes_in)
        else:
            out = io.BytesIO()
            with gzip.GzipFile(fileobj=out, mode='wb') as f:
                f.write(bytes_in)
            return out.getvalue()

    def testDecompressShapeInference(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            for compression_type in ['ZLIB', 'GZIP', '']:
                with self.cached_session():
                    in_bytes = array_ops.placeholder(dtypes.string, shape=[2])
                    decompressed = parsing_ops.decode_compressed(in_bytes, compression_type=compression_type)
                    self.assertEqual([2], decompressed.get_shape().as_list())

    def testDecompress(self):
        if False:
            for i in range(10):
                print('nop')
        for compression_type in ['ZLIB', 'GZIP', '']:
            with self.cached_session():

                def decode(in_bytes, compression_type=compression_type):
                    if False:
                        i = 10
                        return i + 15
                    return parsing_ops.decode_compressed(in_bytes, compression_type=compression_type)
                in_val = [self._compress(b'AaAA', compression_type), self._compress(b'bBbb', compression_type)]
                result = self.evaluate(decode(in_val))
                self.assertAllEqual([b'AaAA', b'bBbb'], result)

    def testDecompressWithRaw(self):
        if False:
            print('Hello World!')
        for compression_type in ['ZLIB', 'GZIP', '']:
            with self.cached_session():

                def decode(in_bytes, compression_type=compression_type):
                    if False:
                        print('Hello World!')
                    decompressed = parsing_ops.decode_compressed(in_bytes, compression_type)
                    return parsing_ops.decode_raw(decompressed, out_type=dtypes.int16)
                result = self.evaluate(decode([self._compress(b'AaBC', compression_type)]))
                self.assertAllEqual([[ord('A') + ord('a') * 256, ord('B') + ord('C') * 256]], result)
if __name__ == '__main__':
    test.main()