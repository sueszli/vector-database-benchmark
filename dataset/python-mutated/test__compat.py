from __future__ import absolute_import, print_function, division
import os
import unittest

class TestFSPath(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.__path = None

    def __fspath__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.__path is not None:
            return self.__path
        raise AttributeError('Accessing path data')

    def _callFUT(self, arg):
        if False:
            return 10
        from gevent._compat import _fspath
        return _fspath(arg)

    def test_text(self):
        if False:
            print('Hello World!')
        s = u'path'
        self.assertIs(s, self._callFUT(s))

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        s = b'path'
        self.assertIs(s, self._callFUT(s))

    def test_None(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self._callFUT(None)

    def test_working_path(self):
        if False:
            while True:
                i = 10
        self.__path = u'text'
        self.assertIs(self.__path, self._callFUT(self))
        self.__path = b'bytes'
        self.assertIs(self.__path, self._callFUT(self))

    def test_failing_path_AttributeError(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(self.__path)
        with self.assertRaises(AttributeError):
            self._callFUT(self)

    def test_fspath_non_str(self):
        if False:
            return 10
        self.__path = object()
        with self.assertRaises(TypeError):
            self._callFUT(self)

@unittest.skipUnless(hasattr(os, 'fspath'), 'Tests native os.fspath')
class TestNativeFSPath(TestFSPath):

    def _callFUT(self, arg):
        if False:
            while True:
                i = 10
        return os.fspath(arg)
if __name__ == '__main__':
    unittest.main()