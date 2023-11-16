"""Tests for chunks_to_lines."""
from bzrlib import tests
from bzrlib.tests import features

def load_tests(standard_tests, module, loader):
    if False:
        while True:
            i = 10
    (suite, _) = tests.permute_tests_for_extension(standard_tests, loader, 'bzrlib._chunks_to_lines_py', 'bzrlib._chunks_to_lines_pyx')
    return suite
compiled_chunkstolines_feature = features.ModuleAvailableFeature('bzrlib._chunks_to_lines_pyx')

class TestChunksToLines(tests.TestCase):
    module = None

    def assertChunksToLines(self, lines, chunks, alreadly_lines=False):
        if False:
            for i in range(10):
                print('nop')
        result = self.module.chunks_to_lines(chunks)
        self.assertEqual(lines, result)
        if alreadly_lines:
            self.assertIs(chunks, result)

    def test_fulltext_chunk_to_lines(self):
        if False:
            return 10
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz\n'], ['foo\nbar\r\nba\rz\n'])
        self.assertChunksToLines(['foobarbaz\n'], ['foobarbaz\n'], alreadly_lines=True)
        self.assertChunksToLines(['foo\n', 'bar\n', '\n', 'baz\n', '\n', '\n'], ['foo\nbar\n\nbaz\n\n\n'])
        self.assertChunksToLines(['foobarbaz'], ['foobarbaz'], alreadly_lines=True)
        self.assertChunksToLines(['foobarbaz'], ['foo', 'bar', 'baz'])

    def test_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertChunksToLines(['\n'], ['\n'], alreadly_lines=True)
        self.assertChunksToLines(['\n'], ['', '\n', ''])
        self.assertChunksToLines(['\n'], ['\n', ''])
        self.assertChunksToLines(['\n'], ['', '\n'])
        self.assertChunksToLines(['\n', '\n', '\n'], ['\n\n\n'])
        self.assertChunksToLines(['\n', '\n', '\n'], ['\n', '\n', '\n'], alreadly_lines=True)

    def test_lines_to_lines(self):
        if False:
            return 10
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz\n'], ['foo\n', 'bar\r\n', 'ba\rz\n'], alreadly_lines=True)

    def test_no_final_newline(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\nbar\r\nba\rz'])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\n', 'bar\r\n', 'ba\rz'], alreadly_lines=True)
        self.assertChunksToLines(('foo\n', 'bar\r\n', 'ba\rz'), ('foo\n', 'bar\r\n', 'ba\rz'), alreadly_lines=True)
        self.assertChunksToLines([], [], alreadly_lines=True)
        self.assertChunksToLines(['foobarbaz'], ['foobarbaz'], alreadly_lines=True)
        self.assertChunksToLines([], [''])

    def test_mixed(self):
        if False:
            return 10
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\n', 'bar\r\nba\r', 'z'])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\nb', 'a', 'r\r\nba\r', 'z'])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\nbar\r\nba', '\r', 'z'])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz'], ['foo\n', '', 'bar\r\nba', '\r', 'z'])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz\n'], ['foo\n', 'bar\r\n', 'ba\rz\n', ''])
        self.assertChunksToLines(['foo\n', 'bar\r\n', 'ba\rz\n'], ['foo\n', 'bar', '\r\n', 'ba\rz\n'])

    def test_not_lines(self):
        if False:
            return 10
        self.assertRaises(TypeError, self.module.chunks_to_lines, object())
        self.assertRaises(TypeError, self.module.chunks_to_lines, [object()])
        self.assertRaises(TypeError, self.module.chunks_to_lines, ['foo', object()])