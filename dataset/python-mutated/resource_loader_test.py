from tensorflow.python.platform import googletest
from tensorflow.python.platform import resource_loader

class ResourceLoaderTest(googletest.TestCase):

    def test_exception(self):
        if False:
            return 10
        with self.assertRaises(IOError):
            resource_loader.load_resource('/fake/file/path/dne')

    def test_exists(self):
        if False:
            i = 10
            return i + 15
        contents = resource_loader.load_resource('python/platform/resource_loader.py')
        self.assertIn(b'tensorflow', contents)
if __name__ == '__main__':
    googletest.main()