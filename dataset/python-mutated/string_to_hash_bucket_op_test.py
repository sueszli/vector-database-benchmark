"""Tests for StringToHashBucket op from string_ops."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class StringToHashBucketOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testStringToOneHashBucketFast(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            input_string = array_ops.placeholder(dtypes.string)
            output = string_ops.string_to_hash_bucket_fast(input_string, 1)
            result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})
            self.assertAllEqual([0, 0, 0], result)

    @test_util.run_deprecated_v1
    def testStringToHashBucketsFast(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            input_string = array_ops.placeholder(dtypes.string)
            output = string_ops.string_to_hash_bucket_fast(input_string, 10)
            result = output.eval(feed_dict={input_string: ['a', 'b', 'c', 'd']})
            self.assertAllEqual([9, 2, 2, 5], result)

    @test_util.run_deprecated_v1
    def testStringToOneHashBucketLegacyHash(self):
        if False:
            return 10
        with self.cached_session():
            input_string = array_ops.placeholder(dtypes.string)
            output = string_ops.string_to_hash_bucket(input_string, 1)
            result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})
            self.assertAllEqual([0, 0, 0], result)

    @test_util.run_deprecated_v1
    def testStringToHashBucketsLegacyHash(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            input_string = array_ops.placeholder(dtypes.string)
            output = string_ops.string_to_hash_bucket(input_string, 10)
            result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})
            self.assertAllEqual([8, 0, 7], result)

    def testStringToOneHashBucketStrongOneHashBucket(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            input_string = constant_op.constant(['a', 'b', 'c'])
            output = string_ops.string_to_hash_bucket_strong(input_string, 1, key=[123, 345])
            self.assertAllEqual([0, 0, 0], self.evaluate(output))

    def testStringToHashBucketsStrong(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            input_string = constant_op.constant(['a', 'b', 'c'])
            output = string_ops.string_to_hash_bucket_strong(input_string, 10, key=[98765, 132])
            self.assertAllEqual([4, 2, 8], self.evaluate(output))

    def testStringToHashBucketsStrongInvalidKey(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            input_string = constant_op.constant(['a', 'b', 'c'])
            with self.assertRaisesOpError('Key must have 2 elements'):
                string_ops.string_to_hash_bucket_strong(input_string, 10, key=[98765]).eval()
if __name__ == '__main__':
    test.main()