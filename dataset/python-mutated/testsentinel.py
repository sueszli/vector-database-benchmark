import unittest
import copy
import pickle
from unittest.mock import sentinel, DEFAULT

class SentinelTest(unittest.TestCase):

    def testSentinels(self):
        if False:
            while True:
                i = 10
        self.assertEqual(sentinel.whatever, sentinel.whatever, 'sentinel not stored')
        self.assertNotEqual(sentinel.whatever, sentinel.whateverelse, 'sentinel should be unique')

    def testSentinelName(self):
        if False:
            return 10
        self.assertEqual(str(sentinel.whatever), 'sentinel.whatever', 'sentinel name incorrect')

    def testDEFAULT(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(DEFAULT, sentinel.DEFAULT)

    def testBases(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(AttributeError, lambda : sentinel.__bases__)

    def testPickle(self):
        if False:
            while True:
                i = 10
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=proto):
                pickled = pickle.dumps(sentinel.whatever, proto)
                unpickled = pickle.loads(pickled)
                self.assertIs(unpickled, sentinel.whatever)

    def testCopy(self):
        if False:
            print('Hello World!')
        self.assertIs(copy.copy(sentinel.whatever), sentinel.whatever)
        self.assertIs(copy.deepcopy(sentinel.whatever), sentinel.whatever)
if __name__ == '__main__':
    unittest.main()