"""Tests for resources."""
from tensorflow.python.platform import googletest
from syntaxnet.util import resources

class ResourcesTest(googletest.TestCase):
    """Testing rig."""

    def testInvalidResource(self):
        if False:
            return 10
        for path in ['bad/path/to/no/file', 'syntaxnet/testdata', 'syntaxnet/testdata/context.pbtxt']:
            with self.assertRaises(IOError):
                resources.GetSyntaxNetResource(path)
            with self.assertRaises(IOError):
                resources.GetSyntaxNetResourceAsFile(path)

    def testValidResource(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'syntaxnet/testdata/hello.txt'
        self.assertEqual('hello world\n', resources.GetSyntaxNetResource(path))
        with resources.GetSyntaxNetResourceAsFile(path) as resource_file:
            self.assertEqual('hello world\n', resource_file.read())
if __name__ == '__main__':
    googletest.main()