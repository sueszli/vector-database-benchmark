import inspect
import unittest
from paddle.distributed.fleet.utils.fs import FS

class FSTest(unittest.TestCase):

    def _test_method(self, func):
        if False:
            for i in range(10):
                print('nop')
        args = inspect.getfullargspec(func).args
        a = None
        try:
            if len(args) == 1:
                func()
            elif len(args) == 2:
                func(a)
            elif len(args) == 3:
                func(a, a)
            elif len(args) == 5:
                func(a, a, a, a)
            print('args:', args, len(args), 'func:', func)
            self.assertFalse(True)
        except NotImplementedError as e:
            pass

    def test(self):
        if False:
            i = 10
            return i + 15
        fs = FS()
        for (name, func) in inspect.getmembers(fs, predicate=inspect.ismethod):
            self._test_method(func)
if __name__ == '__main__':
    unittest.main()