import time
import unittest
import apache_beam as beam
from apache_beam import coders
from apache_beam.runners.interactive import cache_manager as cache
from apache_beam.runners.interactive import interactive_beam as ib

class FileBasedCacheManagerTest(object):
    """Unit test for FileBasedCacheManager.

  Note that this set of tests focuses only the methods that interacts with
  the LOCAL file system. The idea is that once FileBasedCacheManager works well
  with the local file system, it should work with any file system with
  `apache_beam.io.filesystem` interface. Those tests that involve interactions
  with Beam pipeline (i.e. source(), sink(), ReadCache, and WriteCache) will be
  tested with InteractiveRunner as a part of integration tests instead.
  """
    cache_format = None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.cache_manager = cache.FileBasedCacheManager(cache_format=self.cache_format)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache_manager.cleanup()

    def mock_write_cache(self, values, prefix, cache_label):
        if False:
            for i in range(10):
                print('nop')
        'Cache the PCollection where cache.WriteCache would write to.'
        time.sleep(0.1)
        labels = [prefix, cache_label]
        self.cache_manager.write(values, *labels)

    def test_exists(self):
        if False:
            while True:
                i = 10
        'Test that CacheManager can correctly tell if the cache exists or not.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        self.assertTrue(self.cache_manager.exists(prefix, cache_label))
        self.cache_manager.cleanup()
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        self.assertTrue(self.cache_manager.exists(prefix, cache_label))

    def test_empty_label_not_exist(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        self.assertTrue(self.cache_manager.exists(prefix, cache_label))
        self.assertFalse(self.cache_manager.exists(prefix, ''))

    def test_size(self):
        if False:
            print('Hello World!')
        'Test getting the size of some cache label.'
        if self.cache_manager.cache_format != 'text':
            return
        prefix = 'full'
        cache_label = 'some-cache-label'
        self.assertEqual(self.cache_manager.size(prefix, cache_label), 0)
        value = 'a'
        self.mock_write_cache([value], prefix, cache_label)
        coder = self.cache_manager.load_pcoder(prefix, cache_label)
        encoded = coder.encode(value)
        self.assertEqual(self.cache_manager.size(prefix, cache_label), len(encoded) + 1)

    def test_clear(self):
        if False:
            return 10
        'Test that CacheManager can correctly tell if the cache exists or not.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        self.assertTrue(self.cache_manager.exists(prefix, cache_label))
        self.assertTrue(self.cache_manager.clear(prefix, cache_label))
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))

    def test_read_basic(self):
        if False:
            while True:
                i = 10
        'Test the condition where the cache is read once after written once.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.assertListEqual(pcoll_list, cache_version_one)
        self.assertEqual(version, 0)
        self.assertTrue(self.cache_manager.is_latest_version(version, prefix, cache_label))

    def test_read_version_update(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests if the version is properly updated after the files are updated.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        cache_version_two = ['cache', 'version', 'two']
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.mock_write_cache(cache_version_two, prefix, cache_label)
        self.assertFalse(self.cache_manager.is_latest_version(version, prefix, cache_label))
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.assertListEqual(pcoll_list, cache_version_two)
        self.assertEqual(version, 1)
        self.assertTrue(self.cache_manager.is_latest_version(version, prefix, cache_label))

    def test_read_before_write(self):
        if False:
            while True:
                i = 10
        'Test the behavior when read() is called before WriteCache completes.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        self.assertFalse(self.cache_manager.exists(prefix, cache_label))
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.assertListEqual(pcoll_list, [])
        self.assertEqual(version, -1)
        self.assertTrue(self.cache_manager.is_latest_version(version, prefix, cache_label))

    def test_read_over_cleanup(self):
        if False:
            i = 10
            return i + 15
        'Test the behavior of read() over cache cleanup.'
        prefix = 'full'
        cache_label = 'some-cache-label'
        cache_version_one = ['cache', 'version', 'one']
        cache_version_two = ['cache', 'version', 'two']
        self.mock_write_cache(cache_version_one, prefix, cache_label)
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.cache_manager.cleanup()
        self.assertTrue(self.cache_manager.is_latest_version(version, prefix, cache_label))
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.assertListEqual(pcoll_list, [])
        self.assertEqual(version, -1)
        self.assertFalse(self.cache_manager.is_latest_version(version, prefix, cache_label))
        self.mock_write_cache(cache_version_two, prefix, cache_label)
        self.assertFalse(self.cache_manager.is_latest_version(version, prefix, cache_label))
        (reader, version) = self.cache_manager.read(prefix, cache_label)
        pcoll_list = list(reader)
        self.assertListEqual(pcoll_list, cache_version_two)
        self.assertEqual(version, 1)
        self.assertTrue(self.cache_manager.is_latest_version(version, prefix, cache_label))

    def test_load_saved_pcoder(self):
        if False:
            return 10
        pipeline = beam.Pipeline()
        pcoll = pipeline | beam.Create([1, 2, 3])
        _ = pcoll | cache.WriteCache(self.cache_manager, 'a key')
        self.assertIs(type(self.cache_manager.load_pcoder('full', 'a key')), type(coders.registry.get_coder(int)))

    def test_cache_manager_uses_gcs_ib_cache_root(self):
        if False:
            print('Hello World!')
        '\n    Checks that FileBasedCacheManager._cache_dir is set to the\n    cache_root set under Interactive Beam for a GCS directory.\n    '
        ib.options.cache_root = 'gs://'
        cache_manager_with_ib_option = cache.FileBasedCacheManager(cache_dir=ib.options.cache_root)
        self.assertEqual(ib.options.cache_root, cache_manager_with_ib_option._cache_dir)
        ib.options.cache_root = None

    def test_cache_manager_uses_local_ib_cache_root(self):
        if False:
            print('Hello World!')
        '\n    Checks that FileBasedCacheManager._cache_dir is set to the\n    cache_root set under Interactive Beam for a local directory\n    and that the cached values are the same as the values of a\n    cache using default settings.\n    '
        prefix = 'full'
        cache_label = 'some-cache-label'
        cached_values = [1, 2, 3]
        self.mock_write_cache(cached_values, prefix, cache_label)
        (reader_one, _) = self.cache_manager.read(prefix, cache_label)
        pcoll_list_one = list(reader_one)
        ib.options.cache_root = '/tmp/it-test/'
        cache_manager_with_ib_option = cache.FileBasedCacheManager(cache_dir=ib.options.cache_root)
        self.assertEqual(ib.options.cache_root, cache_manager_with_ib_option._cache_dir)
        cache_manager_with_ib_option.write(cached_values, *[prefix, cache_label])
        (reader_two, _) = self.cache_manager.read(prefix, cache_label)
        pcoll_list_two = list(reader_two)
        self.assertEqual(pcoll_list_one, pcoll_list_two)
        ib.options.cache_root = None

class TextFileBasedCacheManagerTest(FileBasedCacheManagerTest, unittest.TestCase):
    cache_format = 'text'

class TFRecordBasedCacheManagerTest(FileBasedCacheManagerTest, unittest.TestCase):
    cache_format = 'tfrecord'
if __name__ == '__main__':
    unittest.main()