from io import BytesIO
import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.package import ObjMismatchError, PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests
try:
    from .common import PackageTestCase
except ImportError:
    from common import PackageTestCase
torch.fx.wrap('len')
torch.fx.wrap('len')

class TestPackageFX(PackageTestCase):
    """Tests for compatibility with FX."""

    def test_package_fx_simple(self):
        if False:
            print('Hello World!')

        class SimpleTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.relu(x + 3.0)
        st = SimpleTest()
        traced = symbolic_trace(st)
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.save_pickle('model', 'model.pkl', traced)
        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle('model', 'model.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))

    def test_package_then_fx(self):
        if False:
            return 10
        from package_a.test_module import SimpleTest
        model = SimpleTest()
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', model)
        f.seek(0)
        pi = PackageImporter(f)
        loaded = pi.load_pickle('model', 'model.pkl')
        traced = symbolic_trace(loaded)
        input = torch.rand(2, 3)
        self.assertEqual(loaded(input), traced(input))

    def test_package_fx_package(self):
        if False:
            i = 10
            return i + 15
        from package_a.test_module import SimpleTest
        model = SimpleTest()
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', model)
        f.seek(0)
        pi = PackageImporter(f)
        loaded = pi.load_pickle('model', 'model.pkl')
        traced = symbolic_trace(loaded)
        f2 = BytesIO()
        with self.assertRaises(ObjMismatchError):
            with PackageExporter(f2) as pe:
                pe.intern('**')
                pe.save_pickle('model', 'model.pkl', traced)
        f2.seek(0)
        with PackageExporter(f2, importer=(pi, sys_importer)) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', traced)
        f2.seek(0)
        pi2 = PackageImporter(f2)
        loaded2 = pi2.load_pickle('model', 'model.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded(input), loaded2(input))

    def test_package_fx_with_imports(self):
        if False:
            for i in range(10):
                print('nop')
        import package_a.subpackage
        graph = Graph()
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        c = graph.call_function(package_a.subpackage.leaf_function, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', gm)
        f.seek(0)
        pi = PackageImporter(f)
        loaded_gm = pi.load_pickle('model', 'model.pkl')
        input_x = torch.rand(2, 3)
        input_y = torch.rand(2, 3)
        self.assertTrue(torch.allclose(loaded_gm(input_x, input_y), gm(input_x, input_y)))
        packaged_dependency = pi.import_module('package_a.subpackage')
        self.assertTrue(packaged_dependency is not package_a.subpackage)

    def test_package_fx_custom_tracer(self):
        if False:
            while True:
                i = 10
        from package_a.test_all_leaf_modules_tracer import TestAllLeafModulesTracer
        from package_a.test_module import ModWithTwoSubmodsAndTensor, SimpleTest

        class SpecialGraphModule(torch.fx.GraphModule):

            def __init__(self, root, graph, info):
                if False:
                    print('Hello World!')
                super().__init__(root, graph)
                self.info = info
        sub_module = SimpleTest()
        module = ModWithTwoSubmodsAndTensor(torch.ones(3), sub_module, sub_module)
        tracer = TestAllLeafModulesTracer()
        graph = tracer.trace(module)
        self.assertEqual(graph._tracer_cls, TestAllLeafModulesTracer)
        gm = SpecialGraphModule(module, graph, 'secret')
        self.assertEqual(gm._tracer_cls, TestAllLeafModulesTracer)
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern('**')
            pe.save_pickle('model', 'model.pkl', gm)
        f.seek(0)
        pi = PackageImporter(f)
        loaded_gm = pi.load_pickle('model', 'model.pkl')
        self.assertEqual(type(loaded_gm).__class__.__name__, SpecialGraphModule.__class__.__name__)
        self.assertEqual(loaded_gm.info, 'secret')
        input_x = torch.randn(3)
        self.assertEqual(loaded_gm(input_x), gm(input_x))

    def test_package_fx_wrap(self):
        if False:
            print('Hello World!')

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, a):
                if False:
                    return 10
                return len(a)
        traced = torch.fx.symbolic_trace(TestModule())
        f = BytesIO()
        with torch.package.PackageExporter(f) as pe:
            pe.save_pickle('model', 'model.pkl', traced)
        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle('model', 'model.pkl')
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))
if __name__ == '__main__':
    run_tests()