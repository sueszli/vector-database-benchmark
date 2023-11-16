import importlib
from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf
import torch.nn
from torch.package import EmptyMatchError, Importer, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
try:
    from .common import PackageTestCase
except ImportError:
    from common import PackageTestCase

class TestDependencyAPI(PackageTestCase):
    """Dependency management API tests.
    - mock()
    - extern()
    - deny()
    """

    def test_extern(self):
        if False:
            i = 10
            return i + 15
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(['package_a.subpackage', 'module_a'])
            he.save_source_string('foo', 'import package_a.subpackage; import module_a')
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage
        module_a_im = hi.import_module('module_a')
        hi.import_module('package_a.subpackage')
        package_a_im = hi.import_module('package_a')
        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob(self):
        if False:
            for i in range(10):
                print('nop')
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(['package_a.*', 'module_*'])
            he.save_module('package_a')
            he.save_source_string('test_module', dedent('                    import package_a.subpackage\n                    import module_a\n                    '))
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage
        module_a_im = hi.import_module('module_a')
        hi.import_module('package_a.subpackage')
        package_a_im = hi.import_module('package_a')
        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob_allow_empty(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that an error is thrown when a extern glob is specified with allow_empty=True\n        and no matching module is required during packaging.\n        '
        import package_a.subpackage
        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, 'did not match any modules'):
            with PackageExporter(buffer) as exporter:
                exporter.extern(include=['package_b.*'], allow_empty=False)
                exporter.save_module('package_a.subpackage')

    def test_deny(self):
        if False:
            while True:
                i = 10
        '\n        Test marking packages as "deny" during export.\n        '
        buffer = BytesIO()
        with self.assertRaisesRegex(PackagingError, 'denied'):
            with PackageExporter(buffer) as exporter:
                exporter.deny(['package_a.subpackage', 'module_a'])
                exporter.save_source_string('foo', 'import package_a.subpackage')

    def test_deny_glob(self):
        if False:
            print('Hello World!')
        '\n        Test marking packages as "deny" using globs instead of package names.\n        '
        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as exporter:
                exporter.deny(['package_a.*', 'module_*'])
                exporter.save_source_string('test_module', dedent('                        import package_a.subpackage\n                        import module_a\n                        '))

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_mock(self):
        if False:
            return 10
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(['package_a.subpackage', 'module_a'])
            he.save_source_string('foo', 'import package_a.subpackage')
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage
        _ = package_a.subpackage
        import module_a
        _ = module_a
        m = hi.import_module('package_a.subpackage')
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, 'was mocked out'):
            r()

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_mock_glob(self):
        if False:
            for i in range(10):
                print('nop')
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(['package_a.*', 'module*'])
            he.save_module('package_a')
            he.save_source_string('test_module', dedent('                    import package_a.subpackage\n                    import module_a\n                    '))
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage
        _ = package_a.subpackage
        import module_a
        _ = module_a
        m = hi.import_module('package_a.subpackage')
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, 'was mocked out'):
            r()

    def test_mock_glob_allow_empty(self):
        if False:
            print('Hello World!')
        '\n        Test that an error is thrown when a mock glob is specified with allow_empty=True\n        and no matching module is required during packaging.\n        '
        import package_a.subpackage
        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, 'did not match any modules'):
            with PackageExporter(buffer) as exporter:
                exporter.mock(include=['package_b.*'], allow_empty=False)
                exporter.save_module('package_a.subpackage')

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_pickle_mocked(self):
        if False:
            while True:
                i = 10
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as he:
                he.mock(include='package_a.subpackage')
                he.intern('**')
                he.save_pickle('obj', 'obj.pkl', obj2)

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_pickle_mocked_all(self):
        if False:
            print('Hello World!')
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.intern(include='package_a.**')
            he.mock('**')
            he.save_pickle('obj', 'obj.pkl', obj2)

    def test_allow_empty_with_error(self):
        if False:
            for i in range(10):
                print('nop')
        'If an error occurs during packaging, it should not be shadowed by the allow_empty error.'
        buffer = BytesIO()
        with self.assertRaises(ModuleNotFoundError):
            with PackageExporter(buffer) as pe:
                pe.extern('foo', allow_empty=False)
                pe.save_module('aodoifjodisfj')
                pe.save_source_string('bar', 'import foo\n')

    def test_implicit_intern(self):
        if False:
            for i in range(10):
                print('nop')
        'The save_module APIs should implicitly intern the module being saved.'
        import package_a
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_module('package_a')

    def test_intern_error(self):
        if False:
            return 10
        'Failure to handle all dependencies should lead to an error.'
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as he:
                he.save_pickle('obj', 'obj.pkl', obj2)
        self.assertEqual(str(e.exception), dedent('\n                * Module did not match against any action pattern. Extern, mock, or intern it.\n                    package_a\n                    package_a.subpackage\n\n                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!\n                '))
        with PackageExporter(buffer) as he:
            he.intern(['package_a', 'package_a.subpackage'])
            he.save_pickle('obj', 'obj.pkl', obj2)

    @skipIf(IS_WINDOWS, 'extension modules have a different file extension on windows')
    def test_broken_dependency(self):
        if False:
            return 10
        'A unpackageable dependency should raise a PackagingError.'

        def create_module(name):
            if False:
                return 10
            spec = importlib.machinery.ModuleSpec(name, self, is_package=False)
            module = importlib.util.module_from_spec(spec)
            ns = module.__dict__
            ns['__spec__'] = spec
            ns['__loader__'] = self
            ns['__file__'] = f'{name}.so'
            ns['__cached__'] = None
            return module

        class BrokenImporter(Importer):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.modules = {'foo': create_module('foo'), 'bar': create_module('bar')}

            def import_module(self, module_name):
                if False:
                    while True:
                        i = 10
                return self.modules[module_name]
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer, importer=BrokenImporter()) as exporter:
                exporter.intern(['foo', 'bar'])
                exporter.save_source_string('my_module', 'import foo; import bar')
        self.assertEqual(str(e.exception), dedent('\n                * Module is a C extension module. torch.package supports Python modules only.\n                    foo\n                    bar\n\n                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!\n                '))

    def test_invalid_import(self):
        if False:
            while True:
                i = 10
        'An incorrectly-formed import should raise a PackagingError.'
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as exporter:
                exporter.save_source_string('foo', 'from ........ import lol')
        self.assertEqual(str(e.exception), dedent('\n                * Dependency resolution failed.\n                    foo\n                      Context: attempted relative import beyond top-level package\n\n                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!\n                '))

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_repackage_mocked_module(self):
        if False:
            print('Hello World!')
        'Re-packaging a package that contains a mocked module should work correctly.'
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.mock('package_a')
            exporter.save_source_string('foo', 'import package_a')
        buffer.seek(0)
        importer = PackageImporter(buffer)
        foo = importer.import_module('foo')
        with self.assertRaises(NotImplementedError):
            foo.package_a.get_something()
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.intern('package_a')
            exporter.mock('**')
            exporter.save_source_string('foo', 'import package_a')
        buffer2.seek(0)
        importer2 = PackageImporter(buffer2)
        foo2 = importer2.import_module('foo')
        with self.assertRaises(NotImplementedError):
            foo2.package_a.get_something()

    def test_externing_c_extension(self):
        if False:
            while True:
                i = 10
        'Externing c extensions modules should allow us to still access them especially those found in torch._C.'
        buffer = BytesIO()
        model = torch.nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64, dropout=1.0, batch_first=True, activation='gelu', norm_first=True)
        with PackageExporter(buffer) as e:
            e.extern('torch.**')
            e.intern('**')
            e.save_pickle('model', 'model.pkl', model)
        buffer.seek(0)
        imp = PackageImporter(buffer)
        imp.load_pickle('model', 'model.pkl')
if __name__ == '__main__':
    run_tests()