import os
import unittest
from coala_utils.ContextManagers import make_temp, prepare_file
from coalib.io.FileProxy import FileProxy, FileProxyMap, FileDictGenerator

class FileProxyTest(unittest.TestCase):

    def test_fileproxy_relative_name(self):
        if False:
            i = 10
            return i + 15
        with prepare_file(['coala'], None) as (_, file):
            relative_url = os.path.relpath(file, __file__)
            with self.assertRaises(ValueError) as context:
                FileProxy(relative_url)
            self.assertEqual('expecting absolute filename', str(context.exception))

    def test_fileproxy_init(self):
        if False:
            for i in range(10):
                print('nop')
        with prepare_file([], None) as (_, file):
            url = os.path.normcase(os.getcwd())
            fileproxy = FileProxy(file, url, 'coala')
            self.assertEqual(fileproxy.version, -1)
            self.assertEqual(fileproxy.contents(), 'coala')
            self.assertEqual(fileproxy.workspace, url)

    def test_fileproxy_normcase(self):
        if False:
            print('Hello World!')
        with prepare_file([], None) as (_, file):
            fileproxy = FileProxy(file, None, 'coala')
            self.assertEqual(fileproxy.filename, os.path.normcase(file))

    def test_fileproxy_str(self):
        if False:
            print('Hello World!')
        with prepare_file([], None) as (_, file):
            empty_fileproxy = FileProxy(file)
            gen_str = f'<FileProxy {empty_fileproxy.filename}, {empty_fileproxy.version}>'
            self.assertEqual(gen_str, str(empty_fileproxy))

    def test_fileproxy_from_file(self):
        if False:
            return 10
        with prepare_file(['coala\n'], None) as (_, file):
            fileproxy = FileProxy.from_file(file, None)
            self.assertEqual(fileproxy.version, -1)
            self.assertEqual(fileproxy.workspace, None)
            self.assertEqual(fileproxy.contents(), 'coala\n')
            self.assertEqual(fileproxy.filename, os.path.normcase(file))

    def test_file_from_file_missing_file(self):
        if False:
            return 10
        with prepare_file([], None) as (_, file):
            with self.assertRaises(OSError):
                FileProxy.from_file(file + 'coala', '.')

    def test_fileproxy_clear(self):
        if False:
            print('Hello World!')
        with prepare_file(['coala'], None) as (_, file):
            fileproxy = FileProxy.from_file(file, None)
            fileproxy.clear()
            self.assertEqual(fileproxy.contents(), '')

    def test_fileproxy_replace(self):
        if False:
            i = 10
            return i + 15
        with prepare_file(['coala\n'], None) as (_, file):
            fileproxy = FileProxy.from_file(file, '.')
            self.assertEqual(fileproxy.version, -1)
            self.assertEqual(fileproxy.contents(), 'coala\n')
            self.assertTrue(fileproxy.replace('coala-rocks', 1))
            self.assertEqual(fileproxy.contents(), 'coala-rocks')
            self.assertFalse(fileproxy.replace('bears-rocks', 1))
            self.assertEqual(fileproxy.contents(), 'coala-rocks')
            self.assertFalse(fileproxy.replace('coala-mountains', 0))
            self.assertEqual(fileproxy.contents(), 'coala-rocks')

    def test_fileproxy_get_disk_contents(self):
        if False:
            while True:
                i = 10
        with prepare_file(['coala\n'], None) as (_, file):
            proxy = FileProxy(file)
            contents = proxy.get_disk_contents()
            self.assertEqual(contents, 'coala\n')

    def test_fileproxy_lines(self):
        if False:
            for i in range(10):
                print('nop')
        with prepare_file(['coala\n', 'bears\n'], None) as (lines, file):
            proxy = FileProxy.from_file(file, None)
            self.assertEqual(proxy.lines(), tuple(lines))

    def test_fileproxy_hash(self):
        if False:
            return 10
        with prepare_file(['coala\n', 'bears\n'], None) as (_, file):
            proxy = FileProxy.from_file(file, None)
            self.assertEqual(hash(proxy), hash(os.path.normcase(file)))

    def test_fileproxy_binary_file(self):
        if False:
            i = 10
            return i + 15
        with make_temp() as filename:
            data = bytearray([120, 3, 255, 0, 100])
            with open(filename, 'wb') as file:
                file.write(data)
            proxy = FileProxy.from_file(filename, None, binary=True)
            self.assertEqual(proxy.filename, os.path.normcase(filename))
            self.assertEqual(proxy.contents(), data)
            self.assertEqual(proxy.lines(), (data,))

class FileProxyMapTest(unittest.TestCase):

    def random_proxy(self, contents=['coala\n']):
        if False:
            print('Hello World!')
        with prepare_file(contents, None) as (_, file):
            return FileProxy.from_file(file, None)

    def empty_proxymap(self):
        if False:
            print('Hello World!')
        return FileProxyMap()

    def test_proxymap_add(self):
        if False:
            return 10
        proxymap = self.empty_proxymap()
        with self.assertRaises(TypeError):
            proxymap.add(123) is False
        with self.assertRaises(TypeError):
            proxymap.add('coala') is False
        proxy_one = self.random_proxy()
        self.assertTrue(proxymap.add(proxy_one))
        proxy_two = FileProxy(proxy_one.filename, '.', 'coala-rocks')
        self.assertFalse(proxymap.add(proxy_two, replace=False))
        self.assertTrue(proxymap.add(proxy_two, replace=True))

    def test_proxymap_remove(self):
        if False:
            i = 10
            return i + 15
        random = self.random_proxy()
        proxymap = self.empty_proxymap()
        proxymap.add(random)
        self.assertEqual(len(proxymap._map), 1)
        proxymap.remove(random.filename)
        self.assertEqual(len(proxymap._map), 0)
        with prepare_file([], None) as (_, file):
            self.assertEqual(proxymap.remove(file), None)

    def test_proxymap_get(self):
        if False:
            print('Hello World!')
        proxymap = self.empty_proxymap()
        with prepare_file([], None) as (_, file):
            assert proxymap.get(file) is None
        proxy = self.random_proxy()
        proxymap.add(proxy)
        self.assertEqual(proxymap.get(proxy.filename), proxy)

    def test_proxymap_resolve_finds(self):
        if False:
            while True:
                i = 10
        proxy = self.random_proxy()
        proxymap = self.empty_proxymap()
        proxymap.add(proxy)
        self.assertEqual(proxymap.resolve(proxy.filename), proxy)

    def test_proxymap_resolve_creates(self):
        if False:
            print('Hello World!')
        with prepare_file(['coala-rocks\n'], None) as (lines, file):
            proxy = self.empty_proxymap().resolve(file)
            self.assertEqual(proxy.lines(), tuple(lines))

    def test_proxymap_resolve_creates_binary(self):
        if False:
            while True:
                i = 10
        with make_temp() as filename:
            data = bytearray([120, 3, 255, 0, 100])
            with open(filename, 'wb') as file:
                file.write(data)
            proxy = self.empty_proxymap().resolve(filename, binary=True)
            self.assertEqual(proxy.lines(), (data,))

    def test_proxymap_resolve_not_finds_hard(self):
        if False:
            return 10
        with prepare_file(['coala'], None) as (_, file):
            with self.assertRaises(OSError):
                self.empty_proxymap().resolve(file + 'coala', hard_sync=True)

    def test_proxymap_resolve_create_soft_relative_name(self):
        if False:
            print('Hello World!')
        with prepare_file([], None) as (_, file):
            relative_url = os.path.relpath(file, __file__)
            with self.assertRaises(ValueError) as context:
                self.empty_proxymap().resolve(relative_url, hard_sync=False)
            self.assertEqual('expecting absolute filename', str(context.exception))

    def test_proxymap_resolve_not_finds_soft(self):
        if False:
            return 10
        with prepare_file(['coala\n'], None) as (_, file):
            missing_file = file + 'coala'
            proxy = self.empty_proxymap().resolve(missing_file, None, hard_sync=False)
            normcased = os.path.normcase(missing_file)
            self.assertEqual(proxy.filename, normcased)
            self.assertEqual(proxy.contents(), '')

class FileDictGeneratorTest(unittest.TestCase):

    def test_get_file_dict_not_implemented(self):
        if False:
            return 10
        generator = FileDictGenerator()
        with self.assertRaises(NotImplementedError):
            generator.get_file_dict([])