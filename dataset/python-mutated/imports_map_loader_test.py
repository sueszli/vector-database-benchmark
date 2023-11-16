"""Tests for imports_map_loader.py."""
import textwrap
from pytype import file_utils
from pytype import imports_map_loader
from pytype.platform_utils import tempfile as compatible_tempfile
from pytype.tests import test_utils
import unittest

class FakeOptions:
    """Fake options."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.open_function = open

class ImportMapLoaderTest(unittest.TestCase):
    """Tests for imports_map_loader.py."""

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.builder = imports_map_loader.ImportsMapBuilder(FakeOptions())

    def build_imports_map(self, path):
        if False:
            while True:
                i = 10
        return self.builder.build_from_file(path)

    def test_read_imports_info(self):
        if False:
            print('Hello World!')
        'Test reading an imports_info file into ImportsInfo.'
        with compatible_tempfile.NamedTemporaryFile() as fi:
            fi.write(textwrap.dedent(file_utils.replace_separator('\n        a/b/__init__.py prefix/1/a/b/__init__.py~\n        a/b/b.py prefix/1/a/b/b.py~suffix\n        a/b/c.pyi prefix/1/a/b/c.pyi~\n        a/b/d.py prefix/1/a/b/d.py~\n        a/b/e.py 2/a/b/e1.py~\n        a/b/e 2/a/b/e2.py~\n        a/b/e 2/a/b/foo/#2.py~\n      ')).encode('utf-8'))
            fi.seek(0)
            expected = [('a/b/__init__', ['prefix/1/a/b/__init__.py~']), ('a/b/b', ['prefix/1/a/b/b.py~suffix']), ('a/b/c', ['prefix/1/a/b/c.pyi~']), ('a/b/d', ['prefix/1/a/b/d.py~']), ('a/b/e', ['2/a/b/foo/#2.py~', '2/a/b/e1.py~', '2/a/b/e2.py~'])]
            f = file_utils.replace_separator
            expected = [(f(k), list(map(f, v))) for (k, v) in expected]
            items = self.builder._read_from_file(fi.name)
            actual = self.builder._build_multimap(items).items()
            self.assertCountEqual(actual, expected)

    def test_build_imports_info(self):
        if False:
            while True:
                i = 10
        'Test building an ImportsInfo from an imports_info tuple.'
        items = [('a/b/__init__.py', 'prefix/1/a/b/__init__.py~'), ('a/b/b.py', 'prefix/1/a/b/b.py~suffix'), ('a/b/c.pyi', 'prefix/1/a/b/c.pyi~'), ('a/b/d.py', 'prefix/1/a/b/d.py~'), ('a/b/e.py', '2/a/b/e1.py~'), ('a/b/e', '2/a/b/e2.py~'), ('a/b/e', '2/a/b/foo/#2.py~')]
        expected = [('a/b/__init__', ['prefix/1/a/b/__init__.py~']), ('a/b/b', ['prefix/1/a/b/b.py~suffix']), ('a/b/c', ['prefix/1/a/b/c.pyi~']), ('a/b/d', ['prefix/1/a/b/d.py~']), ('a/b/e', ['2/a/b/foo/#2.py~', '2/a/b/e1.py~', '2/a/b/e2.py~'])]
        f = file_utils.replace_separator
        expected = [(f(k), list(map(f, v))) for (k, v) in expected]
        actual = self.builder._build_multimap(items).items()
        self.assertCountEqual(actual, expected)

    def test_do_not_filter(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('a/b/c.pyi'))
            imports_info = f"{file_utils.replace_separator('a/b/c.pyi')} " + f"{d[file_utils.replace_separator('a/b/c.pyi')]}\n"
            d.create_file('imports_info', imports_info)
            imports_map = self.build_imports_map(d['imports_info'])
            self.assertEqual(imports_map[file_utils.replace_separator('a/b/c')], d[file_utils.replace_separator('a/b/c.pyi')])

    def test_invalid_map_entry(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            imports_info = f"{file_utils.replace_separator('a/b/c.pyi')}{d[file_utils.replace_separator('a/b/c.pyi')]}\n"
            d.create_file('imports_info', imports_info)
            with self.assertRaises(ValueError):
                self.build_imports_map(d['imports_info'])
if __name__ == '__main__':
    unittest.main()