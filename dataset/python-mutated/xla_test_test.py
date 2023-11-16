"""Tests for the XLATestCase test fixture base class."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import test

class XlaTestCaseTestCase(test.TestCase):

    def testManifestEmptyLineDoesNotCatchAll(self):
        if False:
            while True:
                i = 10
        manifest = '\ntestCaseOne\n'
        (disabled_regex, _) = xla_test.parse_disabled_manifest(manifest)
        self.assertEqual(disabled_regex, 'testCaseOne')

    def testManifestWholeLineCommentDoesNotCatchAll(self):
        if False:
            for i in range(10):
                print('nop')
        manifest = '# I am a comment\ntestCaseOne\ntestCaseTwo\n'
        (disabled_regex, _) = xla_test.parse_disabled_manifest(manifest)
        self.assertEqual(disabled_regex, 'testCaseOne|testCaseTwo')
if __name__ == '__main__':
    test.main()