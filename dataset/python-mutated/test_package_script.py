from io import BytesIO
from textwrap import dedent
from unittest import skipIf
import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests
try:
    from .common import PackageTestCase
except ImportError:
    from common import PackageTestCase
try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, 'no torchvision')

class TestPackageScript(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    def test_package_interface(self):
        if False:
            return 10
        'Packaging an interface class should work correctly.'
        import package_a.fake_interface as fake
        uses_interface = fake.UsesInterface()
        scripted = torch.jit.script(uses_interface)
        scripted.proxy_mod = torch.jit.script(fake.NewModule())
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', uses_interface)
        buffer.seek(0)
        package_importer = PackageImporter(buffer)
        loaded = package_importer.load_pickle('model', 'model.pkl')
        scripted_loaded = torch.jit.script(loaded)
        scripted_loaded.proxy_mod = torch.jit.script(fake.NewModule())
        input = torch.tensor(1)
        self.assertEqual(scripted(input), scripted_loaded(input))

    def test_different_package_interface(self):
        if False:
            while True:
                i = 10
        'Test a case where the interface defined in the package is\n        different than the one defined in the loading environment, to make\n        sure TorchScript can distinguish between the two.\n        '
        import package_a.fake_interface as fake
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string(fake.__name__, dedent('                    import torch\n                    from torch import Tensor\n\n                    @torch.jit.interface\n                    class ModuleInterface(torch.nn.Module):\n                        def one(self, inp1: Tensor) -> Tensor:\n                            pass\n\n                    class ImplementsInterface(torch.nn.Module):\n                        def one(self, inp1: Tensor) -> Tensor:\n                            return inp1 + 1\n\n                    class UsesInterface(torch.nn.Module):\n                        proxy_mod: ModuleInterface\n\n                        def __init__(self):\n                            super().__init__()\n                            self.proxy_mod = ImplementsInterface()\n\n                        def forward(self, input: Tensor) -> Tensor:\n                            return self.proxy_mod.one(input)\n                    '))
        buffer.seek(0)
        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        torch.jit.script(diff_fake.UsesInterface())

    def test_package_script_class(self):
        if False:
            print('Hello World!')
        import package_a.fake_script_class as fake
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(fake.__name__)
        buffer.seek(0)
        package_importer = PackageImporter(buffer)
        loaded = package_importer.import_module(fake.__name__)
        input = torch.tensor(1)
        self.assertTrue(torch.allclose(fake.uses_script_class(input), loaded.uses_script_class(input)))

    def test_package_script_class_referencing_self(self):
        if False:
            for i in range(10):
                print('nop')
        import package_a.fake_script_class as fake
        obj = fake.UsesIdListFeature()
        torch.jit.script(obj)
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern('**')
            exporter.save_pickle('obj', 'obj.pkl', obj)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        obj_loaded = importer.load_pickle('obj', 'obj.pkl')
        scripted_obj_loaded = torch.jit.script(obj_loaded)
        buffer2 = scripted_obj_loaded.save_to_buffer()
        torch.jit.load(BytesIO(buffer2))

    def test_different_package_script_class(self):
        if False:
            while True:
                i = 10
        'Test a case where the script class defined in the package is\n        different than the one defined in the loading environment, to make\n        sure TorchScript can distinguish between the two.\n        '
        import package_a.fake_script_class as fake
        buffer = BytesIO()
        with PackageExporter(buffer) as pe2:
            pe2.save_source_string(fake.__name__, dedent('                    import torch\n\n                    @torch.jit.script\n                    class MyScriptClass:\n                        def __init__(self, x):\n                            self.bar = x\n                    '))
        buffer.seek(0)
        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        input = torch.rand(2, 3)
        loaded_script_class = diff_fake.MyScriptClass(input)
        orig_script_class = fake.MyScriptClass(input)
        self.assertEqual(loaded_script_class.bar, orig_script_class.foo)

    def test_save_scriptmodule(self):
        if False:
            print('Hello World!')
        '\n        Test basic saving of ScriptModule.\n        '
        from package_a.test_module import ModWithTensor
        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'mod.pkl', scripted_mod)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle('res', 'mod.pkl', map_location='cpu')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    @skipIf(IS_FBCODE or IS_SANDCASTLE, 'Tests that use temporary files are disabled in fbcode')
    def test_save_scriptmodule_file(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test basic saving of ScriptModule in file.\n        '
        from package_a.test_module import ModWithTensor
        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        filename = self.temp()
        with PackageExporter(filename) as e:
            e.save_pickle('res', 'mod.pkl', scripted_mod)
        importer = PackageImporter(filename)
        loaded_mod = importer.load_pickle('res', 'mod.pkl')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    def test_save_scriptmodule_with_submods(self):
        if False:
            i = 10
            return i + 15
        '\n        Test basic saving of ScriptModule with submodule.\n        '
        from package_a.test_module import ModWithSubmod, ModWithTensor
        scripted_mod = torch.jit.script(ModWithSubmod(ModWithTensor(torch.rand(1, 2, 3))))
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'mod.pkl', scripted_mod)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle('res', 'mod.pkl', map_location='cpu')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    def test_save_scriptmodules_submod_redefinition(self):
        if False:
            while True:
                i = 10
        '\n        Test to verify saving multiple ScriptModules with same top module\n        but different submodules works. Submodule is redefined to between\n        the defintion of the top module to check that the different concrete\n        types of the modules are thoroughly recognized by serializaiton code.\n        '

        class Submod(torch.nn.Module):

            def forward(self, input: str):
                if False:
                    print('Hello World!')
                input = input + '_submod'
                return input

        class TopMod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.modB = Submod()

            def forward(self, input: str):
                if False:
                    return 10
                return self.modB(input)
        scripted_mod_0 = torch.jit.script(TopMod())

        class Submod(torch.nn.Module):

            def forward(self, input: str):
                if False:
                    print('Hello World!')
                input = input + '_submod(changed)'
                return input
        scripted_mod_1 = torch.jit.script(TopMod())
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_0)
            e.save_pickle('res', 'mod2.pkl', scripted_mod_1)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle('res', 'mod1.pkl')
        loaded_mod_1 = importer.load_pickle('res', 'mod2.pkl')
        self.assertEqual(loaded_mod_0('input'), scripted_mod_0('input'))
        self.assertEqual(loaded_mod_1('input'), scripted_mod_1('input'))
        self.assertNotEqual(loaded_mod_0('input'), loaded_mod_1('input'))

    def test_save_independent_scriptmodules(self):
        if False:
            return 10
        '\n        Test to verify saving multiple ScriptModules with completely\n        separate code works.\n        '
        from package_a.test_module import ModWithTensor, SimpleTest
        scripted_mod_0 = torch.jit.script(SimpleTest())
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_0)
            e.save_pickle('res', 'mod2.pkl', scripted_mod_1)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle('res', 'mod1.pkl')
        loaded_mod_1 = importer.load_pickle('res', 'mod2.pkl')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))

    def test_save_repeat_scriptmodules(self):
        if False:
            print('Hello World!')
        "\n        Test to verify saving multiple different modules and\n        repeats of same scriptmodule in package works. Also tests that\n        PyTorchStreamReader isn't having code hidden from\n        PyTorchStreamWriter writing ScriptModule code files multiple times.\n        "
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor, SimpleTest
        scripted_mod_0 = torch.jit.script(SimpleTest())
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_2 = torch.jit.script(ModWithSubmodAndTensor(torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))))
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'mod0.pkl', scripted_mod_0)
            e.save_pickle('res', 'mod1.pkl', scripted_mod_1)
            e.save_pickle('res', 'mod2.pkl', scripted_mod_0)
            e.save_pickle('res', 'mod3.pkl', scripted_mod_1)
            e.save_pickle('res', 'mod4.pkl', scripted_mod_2)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle('res', 'mod0.pkl')
        loaded_mod_1 = importer.load_pickle('res', 'mod3.pkl')
        loaded_mod_2 = importer.load_pickle('res', 'mod4.pkl')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))
        self.assertEqual(loaded_mod_2(input), scripted_mod_2(input))

    def test_scriptmodules_repeat_save(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test to verify saving and loading same ScriptModule object works\n        across multiple packages.\n        '
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor
        scripted_mod_0 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_1 = torch.jit.script(ModWithSubmodAndTensor(torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))))
        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_0)
        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_module_0 = importer_0.load_pickle('res', 'mod1.pkl')
        buffer_1 = BytesIO()
        with PackageExporter(buffer_1) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_1)
            e.save_pickle('res', 'mod2.pkl', loaded_module_0)
        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        loaded_module_1 = importer_1.load_pickle('res', 'mod1.pkl')
        reloaded_module_0 = importer_1.load_pickle('res', 'mod2.pkl')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_module_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_module_0(input), reloaded_module_0(input))
        self.assertEqual(loaded_module_1(input), scripted_mod_1(input))

    @skipIfNoTorchVision
    def test_save_scriptmodule_only_necessary_code(self):
        if False:
            return 10
        "\n        Test to verify when saving multiple packages with same CU\n        that packages don't include unnecessary torchscript code files.\n        The TorchVision code should only be saved in the package that\n        relies on it.\n        "
        from package_a.test_module import ModWithTensor

        class ModWithTorchVision(torch.nn.Module):

            def __init__(self, name: str):
                if False:
                    return 10
                super().__init__()
                self.tvmod = resnet18()

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input * 4
        scripted_mod_0 = torch.jit.script(ModWithTorchVision('foo'))
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_0)
        buffer_0.seek(0)
        importer_0 = importer = PackageImporter(buffer_0)
        buffer_1 = BytesIO()
        with PackageExporter(buffer_1) as e:
            e.save_pickle('res', 'mod1.pkl', scripted_mod_1)
        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        self.assertTrue('torchvision' in str(importer_0.file_structure()))
        self.assertFalse('torchvision' in str(importer_1.file_structure()))

    def test_save_scriptmodules_in_container(self):
        if False:
            while True:
                i = 10
        '\n        Test saving of ScriptModules inside of container. Checks that relations\n        between shared modules are upheld.\n        '
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor
        scripted_mod_a = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_b = torch.jit.script(ModWithSubmodAndTensor(torch.rand(1, 2, 3), scripted_mod_a))
        script_mods_list = [scripted_mod_a, scripted_mod_b]
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('res', 'list.pkl', script_mods_list)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_list = importer.load_pickle('res', 'list.pkl')
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_list[0](input), scripted_mod_a(input))
        self.assertEqual(loaded_mod_list[1](input), scripted_mod_b(input))

    def test_save_eager_mods_sharing_scriptmodule(self):
        if False:
            return 10
        '\n        Test saving of single ScriptModule shared by multiple\n        eager modules (ScriptModule should be saved just once\n        even though is contained in multiple pickles).\n        '
        from package_a.test_module import ModWithSubmod, SimpleTest
        scripted_mod = torch.jit.script(SimpleTest())
        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern('**')
            e.save_pickle('res', 'mod1.pkl', mod1)
            e.save_pickle('res', 'mod2.pkl', mod2)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        file_structure = importer.file_structure()
        self.assertTrue(file_structure.has_file('.data/ts_code/0'))
        self.assertFalse(file_structure.has_file('.data/ts_code/1'))

    def test_load_shared_scriptmodules(self):
        if False:
            return 10
        '\n        Test loading of single ScriptModule shared by multiple eager\n        modules in single pickle (ScriptModule objects should be the same).\n        '
        from package_a.test_module import ModWithMultipleSubmods, ModWithSubmod, SimpleTest
        scripted_mod = torch.jit.script(SimpleTest())
        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)
        mod_parent = ModWithMultipleSubmods(mod1, mod2)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern('**')
            e.save_pickle('res', 'mod.pkl', mod_parent)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle('res', 'mod.pkl')
        self.assertTrue(id(loaded_mod.mod1.script_mod) == id(loaded_mod.mod2.script_mod))

    def test_save_shared_tensors(self):
        if False:
            print('Hello World!')
        '\n        Test tensors shared across eager and ScriptModules are serialized once.\n        '
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor
        shared_tensor = torch.rand(2, 3, 4)
        scripted_mod = torch.jit.script(ModWithTensor(shared_tensor))
        mod1 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)
        mod2 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern('**')
            e.save_pickle('res', 'tensor', shared_tensor)
            e.save_pickle('res', 'mod1.pkl', mod1)
            e.save_pickle('res', 'mod2.pkl', mod2)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_1 = importer.load_pickle('res', 'mod1.pkl')
        file_structure = importer.file_structure(include='.data/*.storage')
        self.assertTrue(len(file_structure.children['.data'].children) == 1)
        input = torch.rand(2, 3, 4)
        self.assertEqual(loaded_mod_1(input), mod1(input))

    def test_load_shared_tensors(self):
        if False:
            return 10
        '\n        Test tensors shared across eager and ScriptModules on load\n        are the same.\n        '
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor
        shared_tensor = torch.ones(3, 3)
        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))
        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)
        self.assertEqual(shared_tensor.storage()._cdata, scripted_mod_0.tensor.storage()._cdata)
        self.assertEqual(shared_tensor.storage()._cdata, scripted_mod_1.tensor.storage()._cdata)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern('**')
            e.save_pickle('res', 'mod1.pkl', mod1)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_1 = importer.load_pickle('res', 'mod1.pkl')
        self.assertEqual(loaded_mod_1.tensor.storage()._cdata, loaded_mod_1.sub_mod_0.tensor.storage()._cdata)
        self.assertEqual(loaded_mod_1.tensor.storage()._cdata, loaded_mod_1.sub_mod_1.tensor.storage()._cdata)
        loaded_mod_1.tensor.add_(torch.ones(3, 3))
        self.assertTrue(torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor))
        self.assertTrue(torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor))

    def test_load_shared_tensors_repackaged(self):
        if False:
            i = 10
            return i + 15
        '\n        Test tensors shared across eager and ScriptModules on load\n        are the same across multiple package saves and loads. This is\n        an important test because not all of the tensor information is restored\n        in python between packages. The python identity is not maintained, but\n        the backing cpp TensorImpl is. We load/save storages based off of this\n        cpp TensorImpl and not the python identity.\n        '
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor
        shared_tensor = torch.ones(3, 3)
        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))
        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)
        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.intern('**')
            e.save_pickle('res', 'mod1.pkl', mod1)
        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_mod_0 = importer_0.load_pickle('res', 'mod1.pkl')
        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, importer=importer_0) as e:
            e.intern('**')
            e.save_pickle('res', 'mod1.pkl', loaded_mod_0)
        buffer_1.seek(0)
        importer = PackageImporter(buffer_1)
        loaded_mod_1 = importer.load_pickle('res', 'mod1.pkl')
        self.assertEqual(loaded_mod_1.tensor.storage()._cdata, loaded_mod_1.sub_mod_0.tensor.storage()._cdata)
        self.assertEqual(loaded_mod_1.tensor.storage()._cdata, loaded_mod_1.sub_mod_1.tensor.storage()._cdata)
        loaded_mod_1.tensor.add_(torch.ones(3, 3))
        self.assertTrue(torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor))
        self.assertTrue(torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor))

    def test_saving_and_scripting_packaged_mod(self):
        if False:
            print('Hello World!')
        '\n        Test scripting a module loaded from a package\n        and saving it in a new package as a script object.\n        '
        from package_a.test_module import SimpleTest
        orig_mod = SimpleTest()
        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.intern('**')
            e.save_pickle('model', 'model.pkl', orig_mod)
        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_mod = importer_0.load_pickle('model', 'model.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded_mod(input), orig_mod(input))
        scripted_mod = torch.jit.script(loaded_mod)
        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, importer=importer_0) as e:
            e.intern('**')
            e.save_pickle('res', 'scripted_mod.pkl', scripted_mod)
        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        loaded_mod_scripted = importer_1.load_pickle('res', 'scripted_mod.pkl')
        self.assertEqual(loaded_mod_scripted(input), orig_mod(input))

    def test_mixing_packaged_and_inline_modules(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test saving inline and imported modules in same package with\n        independent code.\n        '

        class InlineMod(torch.nn.Module):

            def __init__(self, name: str):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 2, 3)

            def forward(self, input: str):
                if False:
                    print('Hello World!')
                input = input + '_modInline:' + self.name
                return (input, self.tensor * 4)
        inline_mod = InlineMod('inline')
        scripted_inline = torch.jit.script(inline_mod)
        from package_a.test_module import SimpleTest
        imported_mod = SimpleTest()
        scripted_imported = torch.jit.script(imported_mod)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('model', 'inline.pkl', scripted_inline)
            e.save_pickle('model', 'imported.pkl', scripted_imported)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_inline = importer.load_pickle('model', 'inline.pkl')
        loaded_imported = importer.load_pickle('model', 'imported.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded_imported(input), imported_mod(input))
        self.assertEqual(loaded_inline('input'), inline_mod('input'))

    @skipIfNoTorchVision
    def test_mixing_packaged_and_inline_modules_shared_code(self):
        if False:
            i = 10
            return i + 15
        '\n        Test saving inline and imported modules in same package that\n        share code.\n        '

        class TorchVisionTestInline(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.tvmod = resnet18()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = a_non_torch_leaf(x, x)
                return torch.relu(x + 3.0)

        def a_non_torch_leaf(a, b):
            if False:
                i = 10
                return i + 15
            return a + b
        inline_mod = TorchVisionTestInline()
        scripted_inline = torch.jit.script(inline_mod)
        from package_c.test_module import TorchVisionTest
        imported_mod = TorchVisionTest()
        scripted_imported = torch.jit.script(imported_mod)
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle('model', 'inline.pkl', scripted_inline)
            e.save_pickle('model', 'imported.pkl', scripted_imported)
        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_inline = importer.load_pickle('model', 'inline.pkl')
        loaded_imported = importer.load_pickle('model', 'imported.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded_imported(input), imported_mod(input))
        self.assertEqual(loaded_inline(input), inline_mod(input))

    def test_tensor_sharing_pickle(self):
        if False:
            i = 10
            return i + 15
        'Test that saving a ScriptModule and a separately saving a tensor\n        object causes no issues.\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.foo = torch.ones(2, 3)

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo
        scripted_m = torch.jit.script(M())
        original_tensor = torch.ones(0)
        f = BytesIO()
        with torch.package.PackageExporter(f) as exporter:
            exporter.save_pickle('model', 'model.pkl', scripted_m)
            exporter.save_pickle('model', 'input.pkl', original_tensor)
        f.seek(0)
        importer = PackageImporter(f)
        loaded_m = importer.load_pickle('model', 'model.pkl')
        loaded_tensor = importer.load_pickle('model', 'input.pkl')
        self.assertEqual(scripted_m.foo, loaded_m.foo)
        self.assertEqual(original_tensor, loaded_tensor)
if __name__ == '__main__':
    run_tests()