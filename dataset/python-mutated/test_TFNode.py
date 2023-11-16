import getpass
import os
import unittest
from tensorflowonspark import TFManager, TFNode

class TFNodeTest(unittest.TestCase):

    def test_hdfs_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Normalization of absolution & relative string paths depending on filesystem'
        cwd = os.getcwd()
        user = getpass.getuser()
        fs = ['file://', 'hdfs://', 'viewfs://']
        paths = {'hdfs://foo/bar': ['hdfs://foo/bar', 'hdfs://foo/bar', 'hdfs://foo/bar'], 'viewfs://foo/bar': ['viewfs://foo/bar', 'viewfs://foo/bar', 'viewfs://foo/bar'], 'file://foo/bar': ['file://foo/bar', 'file://foo/bar', 'file://foo/bar'], '/foo/bar': ['file:///foo/bar', 'hdfs:///foo/bar', 'viewfs:///foo/bar'], 'foo/bar': ['file://{}/foo/bar'.format(cwd), 'hdfs:///user/{}/foo/bar'.format(user), 'viewfs:///user/{}/foo/bar'.format(user)]}
        for i in range(len(fs)):
            ctx = type('MockContext', (), {'defaultFS': fs[i], 'working_dir': cwd})
            for (path, expected) in paths.items():
                final_path = TFNode.hdfs_path(ctx, path)
                self.assertEqual(final_path, expected[i], 'fs({}) + path({}) => {}, expected {}'.format(fs[i], path, final_path, expected[i]))

    def test_datafeed(self):
        if False:
            print('Hello World!')
        'TFNode.DataFeed basic operations'
        mgr = TFManager.start('abc'.encode('utf-8'), ['input', 'output'], 'local')
        q = mgr.get_queue('input')
        for i in range(10):
            q.put(i)
        q.put(None)
        feed = TFNode.DataFeed(mgr)
        self.assertFalse(feed.done_feeding)
        batch = feed.next_batch(2)
        self.assertEqual(len(batch), 2)
        self.assertEqual(sum(batch), 1)
        self.assertFalse(feed.done_feeding)
        batch = feed.next_batch(4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(sum(batch), 14)
        self.assertFalse(feed.done_feeding)
        batch = feed.next_batch(10)
        self.assertEqual(len(batch), 4)
        self.assertEqual(sum(batch), 30)
        self.assertTrue(feed.should_stop())
if __name__ == '__main__':
    unittest.main()