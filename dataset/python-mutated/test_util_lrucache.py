from test.picardtestcase import PicardTestCase
from picard.util.lrucache import LRUCache

class LRUCacheTest(PicardTestCase):

    def test_simple_getset(self):
        if False:
            return 10
        lrucache = LRUCache(3)
        lrucache['test'] = 1
        self.assertEqual(lrucache['test'], 1)
        self.assertIn('test', lrucache._ordered_keys)

    def test_simple_del(self):
        if False:
            return 10
        lrucache = LRUCache(3)
        lrucache['test'] = 1
        del lrucache['test']
        self.assertNotIn('test', lrucache)
        self.assertNotIn('test', lrucache._ordered_keys)

    def test_max_size(self):
        if False:
            return 10
        lrucache = LRUCache(3)
        lrucache['test1'] = 1
        lrucache['test2'] = 2
        lrucache['test3'] = 3
        lrucache['test4'] = 4
        self.assertNotIn('test1', lrucache)

    def test_lru(self):
        if False:
            return 10
        lrucache = LRUCache(3)
        lrucache['test1'] = 1
        lrucache['test2'] = 2
        lrucache['test3'] = 3
        self.assertEqual(len(lrucache._ordered_keys), 3)
        self.assertEqual('test3', lrucache._ordered_keys[0])
        self.assertEqual('test2', lrucache._ordered_keys[1])
        self.assertEqual('test1', lrucache._ordered_keys[2])
        self.assertEqual(2, lrucache['test2'])
        self.assertEqual('test2', lrucache._ordered_keys[0])
        self.assertEqual('test3', lrucache._ordered_keys[1])
        self.assertEqual('test1', lrucache._ordered_keys[2])
        lrucache['test1'] = 4
        self.assertEqual('test1', lrucache._ordered_keys[0])
        self.assertEqual('test2', lrucache._ordered_keys[1])
        self.assertEqual('test3', lrucache._ordered_keys[2])

    def test_dict_like_init(self):
        if False:
            for i in range(10):
                print('nop')
        lrucache = LRUCache(3, [('test1', 1), ('test2', 2)])
        self.assertEqual(lrucache['test1'], 1)
        self.assertEqual(lrucache['test2'], 2)

    def test_get_keyerror(self):
        if False:
            i = 10
            return i + 15
        lrucache = LRUCache(3)
        with self.assertRaises(KeyError):
            value = lrucache['notakey']

    def test_del_keyerror(self):
        if False:
            print('Hello World!')
        lrucache = LRUCache(3)
        with self.assertRaises(KeyError):
            del lrucache['notakey']