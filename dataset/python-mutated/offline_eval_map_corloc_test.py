"""Tests for utilities in offline_eval_map_corloc binary."""
import tensorflow as tf
from object_detection.metrics import offline_eval_map_corloc as offline_eval

class OfflineEvalMapCorlocTest(tf.test.TestCase):

    def test_generateShardedFilenames(self):
        if False:
            while True:
                i = 10
        test_filename = '/path/to/file'
        result = offline_eval._generate_sharded_filenames(test_filename)
        self.assertEqual(result, [test_filename])
        test_filename = '/path/to/file-00000-of-00050'
        result = offline_eval._generate_sharded_filenames(test_filename)
        self.assertEqual(result, [test_filename])
        result = offline_eval._generate_sharded_filenames('/path/to/@3.record')
        self.assertEqual(result, ['/path/to/-00000-of-00003.record', '/path/to/-00001-of-00003.record', '/path/to/-00002-of-00003.record'])
        result = offline_eval._generate_sharded_filenames('/path/to/abc@3')
        self.assertEqual(result, ['/path/to/abc-00000-of-00003', '/path/to/abc-00001-of-00003', '/path/to/abc-00002-of-00003'])
        result = offline_eval._generate_sharded_filenames('/path/to/@1')
        self.assertEqual(result, ['/path/to/-00000-of-00001'])

    def test_generateFilenames(self):
        if False:
            print('Hello World!')
        test_filenames = ['/path/to/file', '/path/to/@3.record']
        result = offline_eval._generate_filenames(test_filenames)
        self.assertEqual(result, ['/path/to/file', '/path/to/-00000-of-00003.record', '/path/to/-00001-of-00003.record', '/path/to/-00002-of-00003.record'])
if __name__ == '__main__':
    tf.test.main()