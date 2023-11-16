from .. import util
machinery = util.import_importlib('importlib.machinery')
import unittest

class PathHookTest:
    """Test the path hook for source."""

    def path_hook(self):
        if False:
            i = 10
            return i + 15
        return self.machinery.FileFinder.path_hook((self.machinery.SourceFileLoader, self.machinery.SOURCE_SUFFIXES))

    def test_success(self):
        if False:
            print('Hello World!')
        with util.create_modules('dummy') as mapping:
            self.assertTrue(hasattr(self.path_hook()(mapping['.root']), 'find_spec'))

    def test_success_legacy(self):
        if False:
            for i in range(10):
                print('nop')
        with util.create_modules('dummy') as mapping:
            self.assertTrue(hasattr(self.path_hook()(mapping['.root']), 'find_module'))

    def test_empty_string(self):
        if False:
            while True:
                i = 10
        self.assertTrue(hasattr(self.path_hook()(''), 'find_spec'))

    def test_empty_string_legacy(self):
        if False:
            return 10
        self.assertTrue(hasattr(self.path_hook()(''), 'find_module'))
(Frozen_PathHookTest, Source_PathHooktest) = util.test_both(PathHookTest, machinery=machinery)
if __name__ == '__main__':
    unittest.main()