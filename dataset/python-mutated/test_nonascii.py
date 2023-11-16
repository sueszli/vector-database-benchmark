"""Test that various operations work in a non-ASCII environment."""
import os
import sys
from unicodedata import normalize
from bzrlib import osutils
from bzrlib.osutils import pathjoin
from bzrlib.tests import TestCase, TestCaseWithTransport, TestSkipped

class NonAsciiTest(TestCaseWithTransport):

    def test_add_in_nonascii_branch(self):
        if False:
            return 10
        'Test adding in a non-ASCII branch.'
        br_dir = u'ሴ'
        try:
            wt = self.make_branch_and_tree(br_dir)
        except UnicodeEncodeError:
            raise TestSkipped("filesystem can't accomodate nonascii names")
            return
        with file(pathjoin(br_dir, 'a'), 'w') as f:
            f.write('hello')
        wt.add(['a'], ['a-id'])
a_circle_c = u'å'
a_circle_d = u'å'
a_dots_c = u'ä'
a_dots_d = u'ä'
z_umlat_c = u'Ž'
z_umlat_d = u'Ž'
squared_c = u'¼'
squared_d = u'¼'
quarter_c = u'²'
quarter_d = u'²'

class TestNormalization(TestCase):
    """Verify that we have our normalizations correct."""

    def test_normalize(self):
        if False:
            print('Hello World!')
        self.assertEqual(a_circle_d, normalize('NFD', a_circle_c))
        self.assertEqual(a_circle_c, normalize('NFC', a_circle_d))
        self.assertEqual(a_dots_d, normalize('NFD', a_dots_c))
        self.assertEqual(a_dots_c, normalize('NFC', a_dots_d))
        self.assertEqual(z_umlat_d, normalize('NFD', z_umlat_c))
        self.assertEqual(z_umlat_c, normalize('NFC', z_umlat_d))
        self.assertEqual(squared_d, normalize('NFC', squared_c))
        self.assertEqual(squared_c, normalize('NFD', squared_d))
        self.assertEqual(quarter_d, normalize('NFC', quarter_c))
        self.assertEqual(quarter_c, normalize('NFD', quarter_d))

class NormalizedFilename(TestCaseWithTransport):
    """Test normalized_filename and associated helpers"""

    def test__accessible_normalized_filename(self):
        if False:
            print('Hello World!')
        anf = osutils._accessible_normalized_filename
        self.assertEqual((u'ascii', True), anf('ascii'))
        self.assertEqual((a_circle_c, True), anf(a_circle_c))
        self.assertEqual((a_circle_c, True), anf(a_circle_d))
        self.assertEqual((a_dots_c, True), anf(a_dots_c))
        self.assertEqual((a_dots_c, True), anf(a_dots_d))
        self.assertEqual((z_umlat_c, True), anf(z_umlat_c))
        self.assertEqual((z_umlat_c, True), anf(z_umlat_d))
        self.assertEqual((squared_c, True), anf(squared_c))
        self.assertEqual((squared_c, True), anf(squared_d))
        self.assertEqual((quarter_c, True), anf(quarter_c))
        self.assertEqual((quarter_c, True), anf(quarter_d))

    def test__inaccessible_normalized_filename(self):
        if False:
            return 10
        inf = osutils._inaccessible_normalized_filename
        self.assertEqual((u'ascii', True), inf('ascii'))
        self.assertEqual((a_circle_c, True), inf(a_circle_c))
        self.assertEqual((a_circle_c, False), inf(a_circle_d))
        self.assertEqual((a_dots_c, True), inf(a_dots_c))
        self.assertEqual((a_dots_c, False), inf(a_dots_d))
        self.assertEqual((z_umlat_c, True), inf(z_umlat_c))
        self.assertEqual((z_umlat_c, False), inf(z_umlat_d))
        self.assertEqual((squared_c, True), inf(squared_c))
        self.assertEqual((squared_c, True), inf(squared_d))
        self.assertEqual((quarter_c, True), inf(quarter_c))
        self.assertEqual((quarter_c, True), inf(quarter_d))

    def test_functions(self):
        if False:
            for i in range(10):
                print('nop')
        if osutils.normalizes_filenames():
            self.assertEqual(osutils.normalized_filename, osutils._accessible_normalized_filename)
        else:
            self.assertEqual(osutils.normalized_filename, osutils._inaccessible_normalized_filename)

    def test_platform(self):
        if False:
            while True:
                i = 10
        files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3']
        try:
            self.build_tree(files)
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        if sys.platform == 'darwin':
            expected = sorted([a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3'])
        else:
            expected = sorted(files)
        present = sorted(os.listdir(u'.'))
        self.assertEqual(expected, present)

    def test_access_normalized(self):
        if False:
            print('Hello World!')
        files = [a_circle_c + '.1', a_dots_c + '.2', z_umlat_c + '.3', squared_c + '.4', quarter_c + '.5']
        try:
            self.build_tree(files, line_endings='native')
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        for fname in files:
            (path, can_access) = osutils.normalized_filename(fname)
            self.assertEqual(path, fname)
            self.assertTrue(can_access)
            f = open(path, 'rb')
            try:
                shouldbe = 'contents of %s%s' % (path.encode('utf8'), os.linesep)
                actual = f.read()
            finally:
                f.close()
            self.assertEqual(shouldbe, actual, 'contents of %r is incorrect: %r != %r' % (path, shouldbe, actual))

    def test_access_non_normalized(self):
        if False:
            print('Hello World!')
        files = [a_circle_d + '.1', a_dots_d + '.2', z_umlat_d + '.3']
        try:
            self.build_tree(files)
        except UnicodeError:
            raise TestSkipped('filesystem cannot create unicode files')
        for fname in files:
            (path, can_access) = osutils.normalized_filename(fname)
            self.assertNotEqual(path, fname)
            f = open(fname, 'rb')
            f.close()
            if can_access:
                f = open(path, 'rb')
                f.close()
            else:
                self.assertRaises(IOError, open, path, 'rb')