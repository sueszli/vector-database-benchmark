import unittest
from r2.lib import js

def concat_sources(sources):
    if False:
        for i in range(10):
            print('nop')
    return ';'.join(sources)

class TestFileSource(js.FileSource):

    def get_source(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.name

class TestModule(js.Module):

    def get_default_source(self, source):
        if False:
            print('Hello World!')
        return TestFileSource(source)

    def build(self, *args, **kwargs):
        if False:
            print('Hello World!')
        sources = self.get_flattened_sources([])
        sources = [s.get_source() for s in sources]
        return concat_sources(sources)

class TestModuleGetFlattenedSources(unittest.TestCase):

    def test_flat_modules_include_all_sources(self):
        if False:
            i = 10
            return i + 15
        test_files = ['foo.js', 'bar.js', 'baz.js', 'qux.js']
        test_module = TestModule('test_module', *test_files)
        self.assertEqual(test_module.build(), concat_sources(test_files))

    def test_nested_modules_include_all_sources(self):
        if False:
            for i in range(10):
                print('nop')
        test_files_a = ['foo.js', 'bar.js']
        test_module_a = TestModule('test_module_a', *test_files_a)
        test_files_b = ['baz.js', 'qux.js']
        test_module_b = TestModule('test_module_b', *test_files_b)
        test_module = TestModule('test_mobule', test_module_a, test_module_b)
        self.assertEqual(test_module.build(), concat_sources(test_files_a + test_files_b))

    def test_flat_modules_only_include_sources_once(self):
        if False:
            for i in range(10):
                print('nop')
        test_files = ['foo.js', 'bar.js', 'baz.js', 'qux.js']
        test_files_dup = test_files * 2
        test_module = TestModule('test_module', *test_files_dup)
        self.assertEqual(test_module.build(), concat_sources(test_files))

    def test_nested_modules_only_include_sources_once(self):
        if False:
            print('Hello World!')
        test_files = ['foo.js', 'bar.js', 'baz.js', 'qux.js']
        test_module_a = TestModule('test_module_a', *test_files)
        test_module_b = TestModule('test_module_b', *test_files)
        test_module = TestModule('test_mobule', test_module_a, test_module_b)
        self.assertEqual(test_module.build(), concat_sources(test_files))

    def test_filtered_modules_do_not_include_filtered_sources(self):
        if False:
            for i in range(10):
                print('nop')
        test_files = ['foo.js', 'bar.js']
        filtered_files = ['baz.js', 'qux.js']
        all_files = test_files + filtered_files
        filter_module = TestModule('filter_module', *filtered_files)
        test_module = TestModule('test_module', *all_files, filter_module=filter_module)
        self.assertEqual(test_module.build(), concat_sources(test_files))