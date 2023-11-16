import contextlib
import importlib
import os
import sys
import tempfile
import unittest
import warnings
from test.test_importlib import util

@contextlib.contextmanager
def sys_modules_context():
    if False:
        i = 10
        return i + 15
    "\n    Make sure sys.modules is the same object and has the same content\n    when exiting the context as when entering.\n\n    Similar to importlib.test.util.uncache, but doesn't require explicit\n    names.\n    "
    sys_modules_saved = sys.modules
    sys_modules_copy = sys.modules.copy()
    try:
        yield
    finally:
        sys.modules = sys_modules_saved
        sys.modules.clear()
        sys.modules.update(sys_modules_copy)

@contextlib.contextmanager
def namespace_tree_context(**kwargs):
    if False:
        print('Hello World!')
    "\n    Save import state and sys.modules cache and restore it on exit.\n    Typical usage:\n\n    >>> with namespace_tree_context(path=['/tmp/xxyy/portion1',\n    ...         '/tmp/xxyy/portion2']):\n    ...     pass\n    "
    kwargs.setdefault('meta_path', sys.meta_path)
    kwargs.setdefault('path_hooks', sys.path_hooks)
    import_context = util.import_state(**kwargs)
    with import_context, sys_modules_context():
        yield

class NamespacePackageTest(unittest.TestCase):
    """
    Subclasses should define self.root and self.paths (under that root)
    to be added to sys.path.
    """
    root = os.path.join(os.path.dirname(__file__), 'namespace_pkgs')

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.resolved_paths = [os.path.join(self.root, path) for path in self.paths]
        self.ctx = namespace_tree_context(path=self.resolved_paths)
        self.ctx.__enter__()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.ctx.__exit__(None, None, None)

class SingleNamespacePackage(NamespacePackageTest):
    paths = ['portion1']

    def test_simple_package(self):
        if False:
            for i in range(10):
                print('nop')
        import foo.one
        self.assertEqual(foo.one.attr, 'portion1 foo one')

    def test_cant_import_other(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ImportError):
            import foo.two

    def test_module_repr(self):
        if False:
            while True:
                i = 10
        import foo.one
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(foo.__spec__.loader.module_repr(foo), "<module 'foo' (namespace)>")

class DynamicPathNamespacePackage(NamespacePackageTest):
    paths = ['portion1']

    def test_dynamic_path(self):
        if False:
            i = 10
            return i + 15
        import foo.one
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        with self.assertRaises(ImportError):
            import foo.two
        sys.path.append(os.path.join(self.root, 'portion2'))
        import foo.two
        self.assertEqual(foo.two.attr, 'portion2 foo two')

class CombinedNamespacePackages(NamespacePackageTest):
    paths = ['both_portions']

    def test_imports(self):
        if False:
            return 10
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'both_portions foo one')
        self.assertEqual(foo.two.attr, 'both_portions foo two')

class SeparatedNamespacePackages(NamespacePackageTest):
    paths = ['portion1', 'portion2']

    def test_imports(self):
        if False:
            i = 10
            return i + 15
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        self.assertEqual(foo.two.attr, 'portion2 foo two')

class SeparatedNamespacePackagesCreatedWhileRunning(NamespacePackageTest):
    paths = ['portion1']

    def test_invalidate_caches(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as temp_dir:
            sys.path.append(temp_dir)
            import foo.one
            self.assertEqual(foo.one.attr, 'portion1 foo one')
            with self.assertRaises(ImportError):
                import foo.just_created
            namespace_path = os.path.join(temp_dir, 'foo')
            os.mkdir(namespace_path)
            module_path = os.path.join(namespace_path, 'just_created.py')
            with open(module_path, 'w', encoding='utf-8') as file:
                file.write('attr = "just_created foo"')
            with self.assertRaises(ImportError):
                import foo.just_created
            importlib.invalidate_caches()
            import foo.just_created
            self.assertEqual(foo.just_created.attr, 'just_created foo')

class SeparatedOverlappingNamespacePackages(NamespacePackageTest):
    paths = ['portion1', 'both_portions']

    def test_first_path_wins(self):
        if False:
            print('Hello World!')
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        self.assertEqual(foo.two.attr, 'both_portions foo two')

    def test_first_path_wins_again(self):
        if False:
            for i in range(10):
                print('nop')
        sys.path.reverse()
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'both_portions foo one')
        self.assertEqual(foo.two.attr, 'both_portions foo two')

    def test_first_path_wins_importing_second_first(self):
        if False:
            for i in range(10):
                print('nop')
        import foo.two
        import foo.one
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        self.assertEqual(foo.two.attr, 'both_portions foo two')

class SingleZipNamespacePackage(NamespacePackageTest):
    paths = ['top_level_portion1.zip']

    def test_simple_package(self):
        if False:
            for i in range(10):
                print('nop')
        import foo.one
        self.assertEqual(foo.one.attr, 'portion1 foo one')

    def test_cant_import_other(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ImportError):
            import foo.two

class SeparatedZipNamespacePackages(NamespacePackageTest):
    paths = ['top_level_portion1.zip', 'portion2']

    def test_imports(self):
        if False:
            i = 10
            return i + 15
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        self.assertEqual(foo.two.attr, 'portion2 foo two')
        self.assertIn('top_level_portion1.zip', foo.one.__file__)
        self.assertNotIn('.zip', foo.two.__file__)

class SingleNestedZipNamespacePackage(NamespacePackageTest):
    paths = ['nested_portion1.zip/nested_portion1']

    def test_simple_package(self):
        if False:
            i = 10
            return i + 15
        import foo.one
        self.assertEqual(foo.one.attr, 'portion1 foo one')

    def test_cant_import_other(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ImportError):
            import foo.two

class SeparatedNestedZipNamespacePackages(NamespacePackageTest):
    paths = ['nested_portion1.zip/nested_portion1', 'portion2']

    def test_imports(self):
        if False:
            for i in range(10):
                print('nop')
        import foo.one
        import foo.two
        self.assertEqual(foo.one.attr, 'portion1 foo one')
        self.assertEqual(foo.two.attr, 'portion2 foo two')
        fn = os.path.join('nested_portion1.zip', 'nested_portion1')
        self.assertIn(fn, foo.one.__file__)
        self.assertNotIn('.zip', foo.two.__file__)

class LegacySupport(NamespacePackageTest):
    paths = ['not_a_namespace_pkg', 'portion1', 'portion2', 'both_portions']

    def test_non_namespace_package_takes_precedence(self):
        if False:
            print('Hello World!')
        import foo.one
        with self.assertRaises(ImportError):
            import foo.two
        self.assertIn('__init__', foo.__file__)
        self.assertNotIn('namespace', str(foo.__loader__).lower())

class DynamicPathCalculation(NamespacePackageTest):
    paths = ['project1', 'project2']

    def test_project3_fails(self):
        if False:
            return 10
        import parent.child.one
        self.assertEqual(len(parent.__path__), 2)
        self.assertEqual(len(parent.child.__path__), 2)
        import parent.child.two
        self.assertEqual(len(parent.__path__), 2)
        self.assertEqual(len(parent.child.__path__), 2)
        self.assertEqual(parent.child.one.attr, 'parent child one')
        self.assertEqual(parent.child.two.attr, 'parent child two')
        with self.assertRaises(ImportError):
            import parent.child.three
        self.assertEqual(len(parent.__path__), 2)
        self.assertEqual(len(parent.child.__path__), 2)

    def test_project3_succeeds(self):
        if False:
            print('Hello World!')
        import parent.child.one
        self.assertEqual(len(parent.__path__), 2)
        self.assertEqual(len(parent.child.__path__), 2)
        import parent.child.two
        self.assertEqual(len(parent.__path__), 2)
        self.assertEqual(len(parent.child.__path__), 2)
        self.assertEqual(parent.child.one.attr, 'parent child one')
        self.assertEqual(parent.child.two.attr, 'parent child two')
        with self.assertRaises(ImportError):
            import parent.child.three
        sys.path.append(os.path.join(self.root, 'project3'))
        import parent.child.three
        self.assertEqual(len(parent.__path__), 3)
        self.assertEqual(len(parent.child.__path__), 3)
        self.assertEqual(parent.child.three.attr, 'parent child three')

class ZipWithMissingDirectory(NamespacePackageTest):
    paths = ['missing_directory.zip']

    @unittest.expectedFailure
    def test_missing_directory(self):
        if False:
            print('Hello World!')
        import foo.one

    def test_present_directory(self):
        if False:
            while True:
                i = 10
        import bar.two
        self.assertEqual(bar.two.attr, 'missing_directory foo two')

class ModuleAndNamespacePackageInSameDir(NamespacePackageTest):
    paths = ['module_and_namespace_package']

    def test_module_before_namespace_package(self):
        if False:
            return 10
        import a_test
        self.assertEqual(a_test.attr, 'in module')

class ReloadTests(NamespacePackageTest):
    paths = ['portion1']

    def test_simple_package(self):
        if False:
            return 10
        import foo.one
        foo = importlib.reload(foo)
        self.assertEqual(foo.one.attr, 'portion1 foo one')

    def test_cant_import_other(self):
        if False:
            return 10
        import foo
        with self.assertRaises(ImportError):
            import foo.two
        foo = importlib.reload(foo)
        with self.assertRaises(ImportError):
            import foo.two

    def test_dynamic_path(self):
        if False:
            while True:
                i = 10
        import foo.one
        with self.assertRaises(ImportError):
            import foo.two
        sys.path.append(os.path.join(self.root, 'portion2'))
        foo = importlib.reload(foo)
        import foo.two
        self.assertEqual(foo.two.attr, 'portion2 foo two')

class LoaderTests(NamespacePackageTest):
    paths = ['portion1']

    def test_namespace_loader_consistency(self):
        if False:
            i = 10
            return i + 15
        import foo
        self.assertEqual(foo.__loader__, foo.__spec__.loader)
        self.assertIsNotNone(foo.__loader__)

    def test_namespace_origin_consistency(self):
        if False:
            i = 10
            return i + 15
        import foo
        self.assertIsNone(foo.__spec__.origin)
        self.assertIsNone(foo.__file__)

    def test_path_indexable(self):
        if False:
            return 10
        import foo
        expected_path = os.path.join(self.root, 'portion1', 'foo')
        self.assertEqual(foo.__path__[0], expected_path)
if __name__ == '__main__':
    unittest.main()